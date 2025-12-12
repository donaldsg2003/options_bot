import logging
import math
from dataclasses import dataclass
from typing import Optional

from option_bot_spreads.config.config import SETTINGS
from option_bot_spreads.core.data_layer import DataLayer

log = logging.getLogger(__name__)

@dataclass
class SizingDecision:
    qty: int
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, data: DataLayer, trade_log: Optional[logging.Logger] = None):
        self.data_layer = data
        self.trade_log = trade_log or logging.getLogger("trades")
        self.spread_width = SETTINGS.SPREAD_WIDTH

    # ======================================================
    # Entry Permission Logic
    # ======================================================
    def entry_allowed(self, signal: str) -> bool:
        """
        Determine whether an entry is allowed based on volatility regime and settings.
        Logs any blocks to the trades log.
        """
        vixy = self.data_layer.get_vixy_close()
        allow = False
        reason = ""

        if vixy < SETTINGS.VIXY_MAX_ENTRY:
            if signal == "BULL":
                allow = True
            elif signal == "BEAR":
                allow = SETTINGS.ALLOW_BEAR_CALLS
                if not allow:
                    reason = "Bear call entries disabled"
            elif signal == "NEUTRAL":
                allow = SETTINGS.ALLOW_NEUTRAL_PUTS
                if not allow:
                    reason = "Neutral put entries disabled"
        else:
            reason = f"VIXY {vixy:.2f} above entry limit {SETTINGS.VIXY_MAX_ENTRY:.2f}"

        logging.debug(
            "[RISK] EntryAllowed=%s | Signal=%s | VIXY=%.2f | MaxEntry=%.2f",
            allow, signal, vixy, SETTINGS.VIXY_MAX_ENTRY
        )

        if not allow:
            msg = f"BLOCKED ENTRY ({signal}) → {reason or 'disallowed by risk rules'}"
            logging.info(msg)
            if self.trade_log:
                self.trade_log.warning(msg)

        return allow

    # ======================================================
    # Position Sizing
    # ======================================================
    def size_contracts(self, equity: float, net_credit: float | None = None) -> SizingDecision:
        """
        Compute contract quantity based on equity and risk percentage.

        net_credit is accepted for compatibility with EntryEngine.run(),
        but is not yet used in the sizing logic.
        """
        spread_risk = self.spread_width * 100.0
        max_risk_dollars = equity * SETTINGS.MAX_RISK_PCT
        qty_by_cap = max(1, int(max_risk_dollars // spread_risk))
        qty = max(1, min(qty_by_cap, 10))  # cap at 10 contracts

        logging.debug(
            "[RISK] Equity=%.2f | SpreadRisk=%.2f | MaxRiskPct=%.2f | Qty=%d",
            equity, spread_risk, SETTINGS.MAX_RISK_PCT, qty
        )

        decision = SizingDecision(qty=qty, allowed=True, reason="OK")

        # Optionally log large size warnings
        if qty >= 10 and self.trade_log:
            self.trade_log.warning(
                "RISK: max size reached (%d contracts) | Equity=$%.2f",
                qty,
                equity,
            )

        return decision

    # ======================================================
    # Exit Threshold
    # ======================================================
    def exit_dte_threshold(self, entry_dte: int) -> int:
        """
        Compute when to begin closing spreads based on DTE fraction rule.
        """
        raw = math.floor(entry_dte * SETTINGS.EXIT_FRACTION)
        # Ensure at least MIN_EXIT_DTE days, but no fixed "3 days" magic.
        threshold = max(SETTINGS.MIN_EXIT_DTE, raw)
        logging.debug(
            "[RISK] ExitThreshold=%d days | EntryDTE=%d | ExitFraction=%.2f",
            threshold, entry_dte, SETTINGS.EXIT_FRACTION
        )
        return threshold

