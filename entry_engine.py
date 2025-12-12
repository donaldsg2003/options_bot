# engines/entry_engine.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from option_bot_spreads.config.config import SETTINGS
from option_bot_spreads.core.data_layer import DataLayer
from option_bot_spreads.core.strategy_engine import StrategyEngine, SpreadCandidate
from option_bot_spreads.core.persistent_state import PersistentState
from option_bot_spreads.core.order_manager import OrderManager
from option_bot_spreads.helpers.state_utils import PendingOrders, make_spread_key
from option_bot_spreads.helpers.state_utils import ML_COLLECTOR
from option_bot_spreads.helpers.spread_utils import extract_exp_from_symbol
from option_bot_spreads.helpers.utils import utc_now_iso
from option_bot_spreads.risk.risk_manager import RiskManager

log = logging.getLogger(__name__)


@dataclass
class EntryResult:
    """Represents the final outcome of an entry attempt."""
    success: bool
    reason: str = ""
    trade_id: Optional[str] = None
    spread_key: Optional[str] = None
    credit_received: Optional[float] = None
    qty: int = 0


class EntryEngine:
    """
    Orchestrates the opening of new spreads.

    - StrategyEngine: finds candidate spreads.
    - RiskManager: decides if we can trade and how big.
    - OrderManager: submits the order.
    - PersistentState + PendingOrders: enforce cooldowns and position limits.

    PATCH: ML candidate logging happens BEFORE risk / position-limit checks,
    so ML data is collected even when entries are blocked.
    """

    def __init__(
        self,
        data_layer: DataLayer,
        strategy_engine: StrategyEngine,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        data_log: Optional[logging.Logger],
        trade_log: Optional[logging.Logger],
        pending_orders: PendingOrders,
        make_spread_key: Callable[[str, str, str], str],
        record_open_spread: Callable[..., None],
        can_open_spread: Callable[[PersistentState, str, int], bool],
        pair_exists_live: Optional[Callable[..., bool]] = None,
        new_trade: Optional[Callable[..., None]] = None,
        add_leg: Optional[Callable[..., None]] = None,
    ) -> None:
        self.data_layer = data_layer
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.data_log = data_log or logging.getLogger("data")
        self.trade_log = trade_log or logging.getLogger("trades")
        self.pending_orders = pending_orders
        self.make_spread_key = make_spread_key
        self.record_open_spread = record_open_spread
        self.can_open_spread_fn = can_open_spread
        self.pair_exists_live_fn = pair_exists_live  # reserved for later wiring
        self.new_trade = new_trade
        self.add_leg = add_leg

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    @staticmethod
    def _market_side_from_signal(signal: str) -> Optional[str]:
        """
        Map high-level signal to option side for StrategyEngine.

        BULL  -> put credit spread (bullish)
        BEAR  -> call credit spread (bearish)
        other -> no trade
        """
        if signal == "BULL":
            return "put"
        if signal == "BEAR":
            return "call"
        return None

    @staticmethod
    def _spread_already_open(
        positions_cache: Optional[Sequence],
        short_sym: str,
        long_sym: str,
    ) -> bool:
        """
        Duplicate guard using the positions cache.

        We treat the spread as existing if either leg is already present
        in positions. This is conservative and prevents repeat entries
        when reconciliation has not yet updated state.
        """
        if not positions_cache:
            return False

        try:
            symbols = {getattr(p, "symbol", "") for p in positions_cache}
        except Exception:
            return False

        return short_sym in symbols or long_sym in symbols

    @staticmethod
    def _open_spreads_in_regime(state: PersistentState, regime: str) -> int:
        """Count currently open spreads for a given regime."""
        opened = state.get("open_spreads", {}) or {}
        return sum(
            1
            for info in opened.values()
            if isinstance(info, dict)
            and info.get("status") == "open"
            and info.get("entry_regime") == regime
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run(
        self,
        *,
        state: PersistentState,
        positions_cache,
        spy_price_cached: Optional[float],
        vixy_cached: Optional[float],
        regime_latest: str,
        min_dte: int,
        max_dte: int,
        bear_mode: bool,
        cooldown_sec: int,
        signal: Optional[str],
        target_delta: float,
        spread_width: float,
    ) -> Optional[EntryResult]:
        """
        Single entry decision step, called once per main loop iteration.
        """

        # --------------------------------------------------
        # 0) Basic validation and signal gating
        # --------------------------------------------------
        if signal is None:
            log.debug("[ENTRY] No signal provided.")
            return None

        if signal not in ("BULL", "BEAR"):
            self.data_log.debug("[ENTRY] Signal %s not actionable.", signal)
            return None

        side = self._market_side_from_signal(signal)
        if side is None:
            self.data_log.debug("[ENTRY] Signal %s maps to no trade.", signal)
            return None

        if bear_mode and signal == "BEAR" and not SETTINGS.ALLOW_BEAR_CALLS:
            self.trade_log.info(
                "[ENTRY BLOCKED] Bear mode active and bear calls disabled."
            )
            return EntryResult(success=False, reason="Bear calls disabled in bear mode")

        # --------------------------------------------------
        # 1) Find candidate spreads (ALWAYS for ML, even if entry blocked)
        # --------------------------------------------------
        self.data_log.info(
            "[ENTRY] Searching spreads | signal=%s | side=%s | regime=%s | "
            "target_delta=%.2f | width=%.2f | DTE=%d-%d",
            signal,
            side,
            regime_latest,
            target_delta,
            spread_width,
            min_dte,
            max_dte,
        )

        candidates = self.strategy_engine.find_spreads(
            side=side, min_dte=min_dte, max_dte=max_dte
        )
        if not candidates:
            self.data_log.info("[ENTRY] No candidates found for side=%s.", side)
            return EntryResult(success=False, reason="No candidates found")

        # ML: record all candidate spreads for this signal
        try:
            for cand in candidates:
                ML_COLLECTOR.record_spread_candidate(
                    signal=signal,
                    regime=regime_latest,
                    short_symbol=cand.short_sym,
                    long_symbol=cand.long_sym,
                    credit=cand.net_credit,
                    delta=cand.delta,
                    width=abs(cand.long_strike - cand.short_strike),
                    dte=cand.dte,
                    chosen=0,
                    extra={"side": cand.side},
                )
        except Exception as e:
            self.data_log.debug("[ML] Failed to record spread candidates: %s", e)

        # Simple selection rule for now: pick highest net credit.
        c: SpreadCandidate = max(candidates, key=lambda sc: sc.net_credit)

        spread_key = self.make_spread_key(c.short_sym, c.long_sym, c.expiration)

        # ML: mark the chosen candidate and log entry signal
        try:
            ML_COLLECTOR.record_spread_candidate(
                signal=signal,
                regime=regime_latest,
                short_symbol=c.short_sym,
                long_symbol=c.long_sym,
                credit=c.net_credit,
                delta=c.delta,
                width=abs(c.long_strike - c.short_strike),
                dte=c.dte,
                chosen=1,
                extra={"side": c.side, "spread_key": spread_key},
            )
            ML_COLLECTOR.record_entry_signal(
                signal=signal,
                regime=regime_latest,
                spread_key=spread_key,
                chosen=1,
                extra={
                    "target_delta": target_delta,
                    "width": spread_width,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                },
            )
        except Exception as e:
            self.data_log.debug("[ML] Failed to record entry signal: %s", e)

        # --------------------------------------------------
        # 2) Risk and position limits (APPLIED AFTER ML LOGGING)
        # --------------------------------------------------
        # Risk manager veto
        if not self.risk_manager.entry_allowed(signal):
            self.trade_log.info(
                "[ENTRY BLOCKED] Risk manager disallowed new entry for signal %s",
                signal,
            )
            return EntryResult(success=False, reason="Risk manager blocked entry")

        # Global position limit
        open_spreads = state.get("open_spreads", {}) or {}
        if len(open_spreads) >= SETTINGS.MAX_CONCURRENT_SPREADS:
            self.trade_log.info(
                "[ENTRY BLOCKED] Max concurrent spreads reached (%d).",
                SETTINGS.MAX_CONCURRENT_SPREADS,
            )
            return EntryResult(success=False, reason="Max concurrent spreads reached")

        # Per-regime position limit (open + pending)
        open_in_regime = self._open_spreads_in_regime(state, regime_latest)
        pending_in_regime = self.pending_orders.get(regime_latest)
        if open_in_regime + pending_in_regime >= SETTINGS.MAX_POSITIONS_PER_REGIME:
            self.trade_log.info(
                "[ENTRY BLOCKED] Regime %s at limit (open=%d, pending=%d, max=%d).",
                regime_latest,
                open_in_regime,
                pending_in_regime,
                SETTINGS.MAX_POSITIONS_PER_REGIME,
            )
            return EntryResult(success=False, reason="Per-regime limit reached")

        # --------------------------------------------------
        # 3) Cooldown & duplicate-guard checks
        # --------------------------------------------------
        # Spread-specific cooldown
        if not self.can_open_spread_fn(state, spread_key, cooldown_sec):
            self.trade_log.info(
                "[ENTRY BLOCKED] Cooldown active for %s (cooldown=%ds).",
                spread_key,
                cooldown_sec,
            )
            return EntryResult(success=False, reason="Cooldown active")

        # L1: Duplicate guard using positions cache
        if self._spread_already_open(positions_cache, c.short_sym, c.long_sym):
            self.trade_log.info(
                "[ENTRY BLOCKED] Spread %s already present in positions cache.",
                spread_key,
            )
            return EntryResult(success=False, reason="Spread already open")

        # L2: Live broker duplicate guard
        if self.order_manager.spread_exists_live(c.short_sym, c.long_sym):
            self.trade_log.info(
                "[ENTRY BLOCKED] Spread %s exists live at broker.",
                spread_key,
            )
            return EntryResult(success=False, reason="Spread exists at broker")

        # --------------------------------------------------
        # 4) Risk-based sizing
        # --------------------------------------------------
        equity = self.data_layer.get_account_equity()
        sizing = self.risk_manager.size_contracts(equity, net_credit=c.net_credit)

        if not sizing.allowed or sizing.qty <= 0:
            self.trade_log.info(
                "[ENTRY BLOCKED] Sizing veto. Qty=%d Reason=%s",
                sizing.qty,
                sizing.reason,
            )
            return EntryResult(success=False, reason=sizing.reason or "Sizing veto")

        qty = sizing.qty

        # --------------------------------------------------
        # 5) Submit order with pending-order protection
        # --------------------------------------------------
        new_pending = self.pending_orders.increment(regime_latest)
        try:
            if open_in_regime + new_pending > SETTINGS.MAX_POSITIONS_PER_REGIME:
                # Roll back the increment and block
                self.pending_orders.decrement(regime_latest)
                self.trade_log.info(
                    "[ENTRY BLOCKED] Pending-order limit reached for regime %s.",
                    regime_latest,
                )
                return EntryResult(
                    success=False, reason="Pending-order regime limit reached"
                )

            trade_id = self.order_manager.open_spread(signal, c, qty)
            if not trade_id:
                # Submission failed, roll back pending count
                self.pending_orders.decrement(regime_latest)
                self.trade_log.error(
                    "[ENTRY FAILED] Order submission failed for %s.", spread_key
                )
                return EntryResult(success=False, reason="Order submission failed")

        except Exception as e:
            # Ensure we always roll back pending count on error
            self.pending_orders.decrement(regime_latest)
            self.trade_log.exception("[ENTRY FAILED] Exception submitting order: %s", e)
            return EntryResult(success=False, reason=str(e))

        # --------------------------------------------------
        # 6) Persist state / ML hooks
        # --------------------------------------------------
        exp_date = extract_exp_from_symbol(c.short_sym)
        credit = float(c.net_credit)

        self.record_open_spread(
            state=state,
            spread_key=spread_key,
            qty=qty,
            credit=credit,
            regime=regime_latest,
            trade_id=trade_id,
            short_leg_id=None,
            long_leg_id=None,
            expiration=exp_date,
        )

        # Optional ML hook
        if self.new_trade is not None:
            try:
                self.new_trade(
                    trade_id=trade_id,
                    opened_at=utc_now_iso(),
                    regime=regime_latest,
                    spread_key=spread_key,
                    qty=qty,
                    credit=credit,
                    dte=c.dte,
                    side=c.side,
                )
            except Exception as e:
                self.data_log.warning(
                    "[ML] new_trade hook failed for %s: %s", spread_key, e
                )

        self.trade_log.info(
            "[ENTRY SUCCESS] %s | Qty=%d | Credit=%.2f | TradeID=%s",
            spread_key,
            qty,
            credit,
            trade_id,
        )

        return EntryResult(
            success=True,
            trade_id=trade_id,
            spread_key=spread_key,
            credit_received=credit,
            qty=qty,
        )
