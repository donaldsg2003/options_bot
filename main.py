import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytz
from alpaca.trading.requests import GetOrdersRequest

# Config + Keyring Support
from option_bot_spreads.config.config import SETTINGS
from option_bot_spreads.utils.check_keys import verify_keyring_keys

# Core modules
from option_bot_spreads.core.data_layer import DataLayer
from option_bot_spreads.core.db import init_db, log_market_features
from option_bot_spreads.core.order_manager import OrderManager
from option_bot_spreads.core.persistent_state import PersistentState
from option_bot_spreads.core.strategy_engine import StrategyEngine
from option_bot_spreads.core.vixy_regime import get_adaptive_parameters
from option_bot_spreads.core.orphan_detector import OrphanDetector

# Engines
from option_bot_spreads.engines.adaptive_engine import run_adaptive_refresh
from option_bot_spreads.engines.db_engine import DBEngine, maybe_run_db_maintenance
from option_bot_spreads.engines.entry_engine import EntryEngine
from option_bot_spreads.engines.exit_engine import ExitEngine
from option_bot_spreads.engines.orphan_engine import run_orphan_cycle
from option_bot_spreads.engines.reconcile_engine import reconcile_persistent_spreads
from option_bot_spreads.engines.summary_engine import log_hourly_summary, log_startup_summary

# Paths
from option_bot_spreads.paths import LOGS_DIR, SESSION_STATE

# Risk Manager
from option_bot_spreads.risk.risk_manager import RiskManager

# Helpers
from option_bot_spreads.helpers.state_utils import (
    PendingOrders,
    can_open_spread,
    make_spread_key,
    pair_exists_live,
    record_close_spread,
    record_open_spread,
)
from option_bot_spreads.helpers.utils import force_utf8_output, utc_now_iso
force_utf8_output()


# ======================================================
# Logging Setup
# ======================================================
def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_file = Path(LOGS_DIR) / "system.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            TimedRotatingFileHandler(
                log_file, when="midnight", interval=1, backupCount=14
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )


def ensure_valid_credentials():
    """
    Validates Alpaca keys via keyring, not config file.
    """
    verify_keyring_keys()  # your project’s correct method


# ======================================================
# Main Bot
# ======================================================
def main():
    
    setup_logging()
    ensure_valid_credentials()
    init_db()  # ensures options_bot.db schema exists

    # NEW: centralized DB engine for maintenance
    db_engine = DBEngine()

    sys_log = logging.getLogger()
    data_log = logging.getLogger("data")
    trade_log = logging.getLogger("trades")

    sys_log.info("Starting options spread bot | TEST_MODE=%s", SETTINGS.TEST_MODE)
    log_startup_summary(SETTINGS)

    # Persistent state file
    state = PersistentState(str(SESSION_STATE))
    sys_log.info(
        "Persistent state loaded (%d tracked pairs)",
        len(state.get("open_spreads", {})),
    )

    # Core components
    d = DataLayer()
    s = StrategyEngine(d)
    r = RiskManager(d)
    o = OrderManager(d, trade_log=trade_log)

    # Pending order tracking
    pending_orders = PendingOrders(state)
    pending_orders.reset()

    # Engines
    exit_engine = ExitEngine(
        data_layer=d,
        order_manager=o,
        sys_log=sys_log,
        trade_log=trade_log,
        record_close_spread=record_close_spread,
        make_spread_key=make_spread_key,
        reconcile_persistent_spreads=reconcile_persistent_spreads,
    )

    entry_engine = EntryEngine(
        data_layer=d,
        strategy_engine=s,
        risk_manager=r,
        order_manager=o,
        data_log=data_log,
        trade_log=trade_log,
        pending_orders=pending_orders,
        make_spread_key=make_spread_key,
        record_open_spread=record_open_spread,
        can_open_spread=can_open_spread,
        pair_exists_live=pair_exists_live,
        new_trade=getattr(d, "new_trade", None),
        add_leg=getattr(d, "add_leg", None),
    )

    # Orphan detector
    orphan_detector = OrphanDetector(state, d.trading)
    orphan_detector.AUTO_CLOSE_ENABLED = SETTINGS.AUTO_CLOSE_ORPHANS
    orphan_detector.MIN_VERIFICATIONS = SETTINGS.ORPHAN_MIN_VERIFICATIONS
    orphan_detector.VERIFICATION_INTERVAL = SETTINGS.ORPHAN_VERIFICATION_INTERVAL
    orphan_detector.ORPHAN_AGE_MIN = SETTINGS.ORPHAN_AGE_MIN

    # Cached positions
    positions_cache = None
    positions_cache_time = None
    positions_cache_ttl = 10

    # Adaptive parameters
    (
        regime,
        delta,
        width,
        min_credit,
        COOLDOWN_SEC,
        BEAR_MODE,
        min_dte,
        max_dte,
        delta_window,
    ) = get_adaptive_parameters()

    # Apply to strategy
    s.target_delta = delta
    s.spread_width = width
    s.min_net_credit = min_credit
    s.delta_window = delta_window
    r.spread_width = width
    s.regime = regime

    # FIXED: assign these for EntryEngine call
    target_delta = delta
    spread_width = width

    state.set("last_vixy_regime", regime)

    # Initial reconciliation
    reconcile_persistent_spreads(state, o, orphan_detector)

    last_summary_hour = None
    last_closed_log = None

    # ============================================================
    # Main Loop
    # ============================================================
    while True:
        try:
            # Market closed?
            if not d.is_market_open():
                now = datetime.now()
                if not last_closed_log or (now - last_closed_log).seconds > 3600:
                    eastern = datetime.now(pytz.timezone("US/Eastern"))
                    sys_log.info("Market closed (ET %s) — pausing...", eastern.strftime("%H:%M"))
                    last_closed_log = now
                time.sleep(300)
                continue

            # Weekly maintenance
            maybe_run_db_maintenance(db_engine)

            # Refresh prices
            spy_price_cached = d.get_spy_price(force_refresh=True)
            vixy_cached = d.get_vixy_close(force_refresh=True)
            regime_latest = state.get("last_vixy_regime", regime)
            vixy_pct_latest = state.get("last_vixy_percentile", None)

            # Log to data file
            data_log.info(
                "Equity=$%.2f | SPY=%.2f | VIXY=%.2f | VIXY_pct=%s | Regime=%s",
                d.get_account_equity(),
                spy_price_cached or 0.0,
                vixy_cached or 0.0,
                f"{(vixy_pct_latest or 0.0) * 100:.2f}%" if vixy_pct_latest else "NA",
                regime_latest,
            )

            # Persist market snapshot
            try:
                log_market_features(
                    spy=spy_price_cached,
                    vixy=vixy_cached,
                    vixy_pct=vixy_pct_latest,
                    regime=regime_latest,
                )
            except Exception as e:
                sys_log.warning("[DB] Failed to log market_features: %s", e)

            # Reconciliation timer
            pending_order_count = len(
                d.trading.get_orders(filter=GetOrdersRequest(status="open"))
            )
            reconcile_interval = 300 if pending_order_count > 0 else 900

            last_reconcile_ts = state.get("last_reconcile_ts", "")
            now_utc = datetime.now(timezone.utc)

            if last_reconcile_ts:
                try:
                    last_reconcile_dt = datetime.fromisoformat(
                        last_reconcile_ts.replace("Z", "+00:00")
                    )
                    seconds_since = (now_utc - last_reconcile_dt).total_seconds()
                except Exception:
                    seconds_since = 9999
            else:
                seconds_since = 9999

            if seconds_since > reconcile_interval:
                sys_log.info(
                    "[RECONCILE] Running periodic sync (interval %ds, pending=%d)",
                    reconcile_interval,
                    pending_order_count,
                )
                reconcile_persistent_spreads(state, o, orphan_detector)
                state.set("last_reconcile_ts", utc_now_iso())

                run_orphan_cycle(
                    state=state,
                    data_layer=d,
                    orphan_detector=orphan_detector,
                    sys_log=sys_log,
                    trade_log=trade_log,
                )

            # Hourly summary
            hour_key = now_utc.strftime("%Y-%m-%d %H")
            if hour_key != last_summary_hour and now_utc.minute == 0:
                log_hourly_summary(state, d)
                last_summary_hour = hour_key

            # Adaptive refresh
            refresh = run_adaptive_refresh(state, s, r, SETTINGS)
            if refresh:
                regime = refresh["regime"]
                COOLDOWN_SEC = refresh["cooldown"]
                BEAR_MODE = refresh["bear_mode"]
                min_dte = refresh["min_dte"]
                max_dte = refresh["max_dte"]
                s.regime = regime

                # FIX: update target_delta and spread_width on adaptive refresh
                target_delta = s.target_delta
                spread_width = s.spread_width
                

            # Refresh cached positions
            now_ts = time.time()
            if (
                positions_cache is None
                or positions_cache_time is None
                or (now_ts - positions_cache_time) > positions_cache_ttl
            ):
                try:
                    positions_cache = d.trading.get_all_positions()
                    positions_cache_time = now_ts
                except Exception as e:
                    sys_log.warning("Failed to refresh positions cache: %s", e)
                    positions_cache = []

            # Exit engine
            exit_engine.run(
                state=state,
                positions_cache=positions_cache,
                spy_price_cached=spy_price_cached,
                vixy_cached=vixy_cached,
                min_dte=min_dte,
                max_dte=max_dte,
            )

            # Entry engine
            signal = s.market_signal()

            entry_engine.run(
                state=state,
                positions_cache=positions_cache,
                spy_price_cached=spy_price_cached,
                vixy_cached=vixy_cached,
                regime_latest=regime,
                min_dte=min_dte,
                max_dte=max_dte,
                bear_mode=BEAR_MODE,
                cooldown_sec=COOLDOWN_SEC,
                signal=signal,
                target_delta=target_delta,
                spread_width=spread_width
            )

        except Exception as e:
            sys_log.exception("Loop error: %s", e)

        time.sleep(10)

if __name__ == "__main__":
    main()
