# engines/exit_engine.py
import time
from datetime import date, datetime

from option_bot_spreads.config.config import SETTINGS, DTEConfig
from option_bot_spreads.helpers.utils import utc_now_iso
from option_bot_spreads.helpers.state_utils import ML_COLLECTOR


# Failsafe profit exit parameters (override via SETTINGS if present)
FORCED_TP_PCT = getattr(SETTINGS, "FORCED_TP_PCT", 0.60)  # 60% of max profit
MIN_LIFE_USED_FOR_FORCED_TP = getattr(
    SETTINGS,
    "MIN_LIFE_USED_FOR_FORCED_TP",
    0.30,
)  # 30% of contract life


def run_exit_pass(
    state,
    d,
    o,
    positions_cache,
    spy_price_cached,
    vixy_cached,
    sys_log,
    trade_log,
    record_close_spread,
    make_spread_key,
    min_dte,
    max_dte,
    reconcile_persistent_spreads,
):
    """
    Evaluate all open spreads and submit exit orders if any exit conditions are met.
    """

    # ============================
    # MARKET HOURS GUARD
    # ============================
    if not d.is_market_open():
        time.sleep(10)
        return

    # ============================
    # FIX #A: positions_cache None-guard
    # ============================
    if positions_cache is None:
        sys_log.warning("[EXIT] positions_cache is None; skipping exit pass.")
        time.sleep(5)
        return

    # ============================
    # GROUP POSITIONS
    # ============================
    pos_groups = {}
    for p in positions_cache:
        if len(p.symbol) < 9:
            continue
        exp = p.symbol[3:9]
        pos_groups.setdefault(exp, []).append(
            type("OpenLeg", (), {"symbol": p.symbol, "qty": float(p.qty)})()
        )

    all_symbols_to_check = []
    for exp_key, legs in pos_groups.items():
        for leg in legs:
            all_symbols_to_check.append(leg.symbol)

    # Single batch fetch
    all_quotes = d.latest_quotes(all_symbols_to_check) if all_symbols_to_check else {}

    for exp_key, legs in pos_groups.items():
        if len(legs) < 2:
            continue

        short_leg = next((p for p in legs if p.qty < 0), None)
        long_leg = next((p for p in legs if p.qty > 0), None)
        if not short_leg or not long_leg:
            continue

        skey = make_spread_key(short_leg.symbol, long_leg.symbol, exp_key)

        # ============================
        # FIX #B: Pending close guard w/lock
        # ============================
        with state.lock:
            pending_close = state.get(f"pending_close_{skey}")
            if pending_close:
                try:
                    order = d.trading.get_order_by_id(pending_close["order_id"])
                    if order.status in [
                        "new",
                        "accepted",
                        "pending_new",
                        "partially_filled",
                    ]:
                        continue
                    state.delete(f"pending_close_{skey}")
                except Exception:
                    state.delete(f"pending_close_{skey}")

        open_spreads_cache = state.get("open_spreads", {})
        spread_state = open_spreads_cache.get(skey)
        if not spread_state:
            continue

        # ============================
        # QUOTES
        # ============================
        sb, sa = all_quotes.get(short_leg.symbol, (0.0, 0.0))
        lb, la = all_quotes.get(long_leg.symbol, (0.0, 0.0))
        if (sb == 0 and sa == 0) or (lb == 0 and la == 0):
            continue

        short_mid = (sb + sa) / 2 if sb and sa else max(sb, sa)
        long_mid = (lb + la) / 2 if lb and la else max(lb, la)
        current_spread = short_mid - long_mid

        # ============================
        # DTE handling
        # ============================
        entry_dte = spread_state.get("entry_dte")
        if entry_dte is None:
            entry_dte = (min_dte + max_dte) // 2

        exp_str = spread_state.get("expiration")
        if exp_str:
            expiration_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        else:
            expiration_date = datetime.strptime(exp_key, "%y%m%d").date()

        days_left = (expiration_date - date.today()).days

        entry_regime = spread_state.get("entry_regime", "MID_VOL")
        threshold_dte = DTEConfig.get_exit_dte(entry_regime)

        short_strike = d.parse_strike(short_leg.symbol)
        side = "put" if short_leg.symbol[9] == "P" else "call"
        entry_credit = spread_state.get("credit", None)

        # Ensure profit_pct is always defined before ML logging
        profit_pct = None

        exit_reasons = []
        use_market_order = False

        # ======================================================
        # PHASE 3: HYBRID EXIT MODE (45/21)
        # ======================================================
        if SETTINGS.DURATION_MODE.lower() == "45_21":
            # Compute hybrid-specific thresholds
            tp_pct = SETTINGS.DURATION_45_TP_PCT       # usually 0.50
            exit_dte = SETTINGS.DURATION_45_EXIT_DTE   # usually 21
            sl_mult = SETTINGS.DURATION_45_EMERGENCY_SL_MULT  # default 3x

            # PROFIT PCT
            if entry_credit and entry_credit > 0:
                max_profit = entry_credit
                profit_now = entry_credit - current_spread
                profit_pct = profit_now / max_profit if max_profit > 0 else 0.0
            else:
                profit_pct = 0.0

            # ---------------------------------------
            # 1) 50% PROFIT TARGET
            # ---------------------------------------
            if profit_pct >= tp_pct:
                exit_reasons.append("TP_45_21")

            # ---------------------------------------
            # 2) EXIT AT 21 DTE
            # ---------------------------------------
            elif days_left <= exit_dte and days_left > 2:   # still avoid expiration-day logic
                exit_reasons.append("TIME_21DTE")

            # ---------------------------------------
            # 3) EMERGENCY STOP LOSS
            # ---------------------------------------
            else:
                emergency_sl = entry_credit * sl_mult if entry_credit else None
                if emergency_sl and current_spread >= emergency_sl:
                    exit_reasons.append("EMERGENCY_SL_45_21")

            # If hybrid mode fired an exit condition → handle immediately
            if exit_reasons:
                use_market_order = False  # 45/21 always uses limit, except expiration-day rules
                # Execution identical to later block, so jump directly to execution
                pass
            else:
                # No hybrid exit → skip ALL adaptive exit logic
                continue


        # ============================
        # PRIORITY 1 EMERGENCY
        # ============================
        if days_left <= SETTINGS.EMERGENCY_EXIT_DTE:
            exit_reasons.append("EMERGENCY_DTE_2_OR_LESS")
            use_market_order = SETTINGS.USE_MARKET_ORDER_EMERGENCY
            trade_log.critical(
                "[EMERGENCY EXIT] %s at %d DTE - MUST CLOSE to avoid assignment",
                skey,
                days_left,
            )

        # PIN RISK
        elif days_left > 2:
            spy_price = spy_price_cached
            if spy_price:
                pin_risk_dte = DTEConfig.get_pin_risk_dte(entry_regime)
                pin_risk_moneyness = DTEConfig.get_pin_risk_moneyness(entry_regime)

                if days_left <= pin_risk_dte:
                    if side == "put":
                        itm_threshold = short_strike * (1 - pin_risk_moneyness)
                        if spy_price < itm_threshold:
                            exit_reasons.append("PIN_RISK_ITM")
                            use_market_order = days_left <= 3
                    else:
                        itm_threshold = short_strike * (1 + pin_risk_moneyness)
                        if spy_price > itm_threshold:
                            exit_reasons.append("PIN_RISK_ITM")
                            use_market_order = days_left <= 3

        # ============================
        # PRIORITY 2 RISK MGMT
        # ============================
        if not exit_reasons:
            vixy_current = vixy_cached
            if vixy_current > SETTINGS.VIXY_MAX_EXIT:
                exit_reasons.append("VOL_SPIKE")

        if not exit_reasons and entry_credit is not None:
            if SETTINGS.DURATION_MODE.lower() != "45_21":
                stop_loss_level = entry_credit * SETTINGS.SL_MULT
                if current_spread >= stop_loss_level:
                    exit_reasons.append("STOP_LOSS")


        # ============================
        # PRIORITY 3 TIME EXIT
        # ============================
        if SETTINGS.DURATION_MODE.lower() != "45_21":
            if not exit_reasons and days_left > 2:
                if days_left <= threshold_dte:
                    exit_reasons.append("DTE_THRESHOLD")



        # ============================
        # PRIORITY 4 PROFIT MGMT
        # ============================
        if SETTINGS.DURATION_MODE.lower() != "45_21":
            if not exit_reasons and entry_credit:
                max_profit = entry_credit
                profit_so_far = entry_credit - current_spread
                profit_pct = profit_so_far / max_profit

                if SETTINGS.DURATION_MODE.lower() == "45_21":
                    profit_target_pct = SETTINGS.DURATION_45_TP_PCT
                else:
                    profit_target_pct = SETTINGS.PROFIT_TARGETS.get(
                        entry_regime,
                        SETTINGS.TP_PCT,
                    )

                if "peak_profit_pct" not in spread_state:
                    spread_state["peak_profit_pct"] = profit_pct

                if profit_pct > spread_state["peak_profit_pct"]:
                    spread_state["peak_profit_pct"] = profit_pct
                    state.set("open_spreads", open_spreads_cache)

                peak_profit_pct = spread_state["peak_profit_pct"]

                if profit_pct >= profit_target_pct:
                    if peak_profit_pct > profit_target_pct + SETTINGS.TRAILING_STOP_ACTIVATION:
                        excess_profit = (peak_profit_pct - profit_target_pct) * max_profit
                        target_profit = profit_target_pct * max_profit
                        locked_in_excess = excess_profit * SETTINGS.TRAILING_STOP_LOCKUP
                        trailing_stop = entry_credit - target_profit - locked_in_excess

                        if current_spread >= trailing_stop:
                            exit_reasons.append("TRAILING_STOP")
                    else:
                        exit_reasons.append("TAKE_PROFIT")

        # ============================================
        # PRIORITY 5: Failsafe "ripe" profit exit
        # ============================================
        if SETTINGS.DURATION_MODE.lower() != "45_21":
            if not exit_reasons and entry_credit is not None and entry_credit > 0:
                max_profit = entry_credit
                profit_so_far = entry_credit - current_spread
                profit_pct = profit_so_far / max_profit if max_profit > 0 else 0.0

                if entry_dte and entry_dte > 0:
                    life_used = (entry_dte - days_left) / float(entry_dte)
                    if life_used < 0.0:
                        life_used = 0.0
                    elif life_used > 1.0:
                        life_used = 1.0
                else:
                    life_used = 0.0

                if profit_pct >= FORCED_TP_PCT and life_used >= MIN_LIFE_USED_FOR_FORCED_TP:
                    exit_reasons.append("FORCED_TAKE_PROFIT")
                    trade_log.info(
                        "[FORCED TP] %.1f%% profit with %.0f%% of life used | Regime=%s",
                        profit_pct * 100.0,
                        life_used * 100.0,
                        entry_regime,
                    )

        # ============================
        # NO EXIT → CONTINUE
        # ============================
        if not exit_reasons:
            continue

        # ============================
        # EXECUTE EXIT ORDER
        # ============================
        close_qty = int(abs(short_leg.qty))

        if use_market_order or days_left <= SETTINGS.EMERGENCY_EXIT_DTE:
            limit_price = None
        elif "PIN_RISK_ITM" in exit_reasons and days_left <= 3:
            limit_price = None
        else:
            limit_price = max(current_spread * 1.05, 0.05)

        # ============================
        # ML: record exit decision
        # ============================
        main_reason = exit_reasons[0]
        try:
            ML_COLLECTOR.record_exit_decision(
                spread_key=skey,
                decision_engine="exit_engine",
                reason=main_reason,
                score=None,
                regime=entry_regime,
                extra={
                    "all_reasons": exit_reasons,
                    "days_left": days_left,
                    "entry_dte": entry_dte,
                    "entry_credit": entry_credit,
                    "current_spread": current_spread,
                    "profit_pct": profit_pct,
                },
            )
        except Exception as e:
            trade_log.debug(
                "[ML] Failed to record exit decision for %s: %s",
                skey,
                e,
            )

        order_id = o.close_spread(
            short_leg.symbol,
            long_leg.symbol,
            close_qty,
            limit_price=limit_price,
        )

        if order_id:
            state.set(
                f"pending_close_{skey}",
                {
                    "order_id": order_id,
                    "submitted_at": utc_now_iso(),
                    "reasons": ",".join(exit_reasons),
                    "days_left": days_left,
                    "limit_price": limit_price,
                    "use_market": use_market_order,
                },
            )

            # DO NOT RECONCILE HERE — LET MAIN LOOP HANDLE IT
            record_close_spread(state, skey, close_price=current_spread)
            state.set("last_close_ts", utc_now_iso())


class ExitEngine:
    """
    Thin OO wrapper around run_exit_pass.
    """

    def __init__(
        self,
        data_layer,
        order_manager,
        sys_log,
        trade_log,
        record_close_spread,
        make_spread_key,
        reconcile_persistent_spreads,
    ):
        self.d = data_layer
        self.o = order_manager
        self.sys_log = sys_log
        self.trade_log = trade_log
        self.record_close_spread = record_close_spread
        self.make_spread_key = make_spread_key
        self.reconcile_persistent_spreads = reconcile_persistent_spreads

    def run(
        self,
        state,
        positions_cache,
        spy_price_cached,
        vixy_cached,
        min_dte,
        max_dte,
    ):
        return run_exit_pass(
            state=state,
            d=self.d,
            o=self.o,
            positions_cache=positions_cache,
            spy_price_cached=spy_price_cached,
            vixy_cached=vixy_cached,
            sys_log=self.sys_log,
            trade_log=self.trade_log,
            record_close_spread=self.record_close_spread,
            make_spread_key=self.make_spread_key,
            min_dte=min_dte,
            max_dte=max_dte,
            reconcile_persistent_spreads=self.reconcile_persistent_spreads,
        )
