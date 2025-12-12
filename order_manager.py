import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING, Any

from alpaca.trading.enums import (
    OrderType,
    OrderSide,
    OrderClass,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import LimitOrderRequest

from option_bot_spreads.core.data_layer import DataLayer
from option_bot_spreads.core.strategy_engine import SpreadCandidate
from option_bot_spreads.helpers.utils import utc_now_iso  # ✅ UTC timestamp helper

trade_log = logging.getLogger("trades")

if TYPE_CHECKING:
    # Only exists for static analysis; does nothing at runtime
    class OptionLegRequest:
        symbol: str
        side: Any
        ratio_qty: int
        position_intent: Any
else:
    # Real runtime import
    from alpaca.trading.requests import OptionLegRequest


@dataclass
class OpenLeg:
    symbol: str
    qty: float


class OrderManager:
    def __init__(
        self,
        data_layer: DataLayer,
        csv_path: Optional[str] = None,
        trade_log: Optional[logging.Logger] = None,
    ):
        self.data_layer = data_layer
        self.csv_path = csv_path
        self.trade_log = trade_log or logging.getLogger("trades")

    # ======================================================
    # Internal Helpers
    # ======================================================
    def _log_csv(self, row: List[str]):
        """Append a trade row to the daily CSV trade journal."""
        if not self.csv_path:
            return

        hdr = [
            "timestamp",
            "signal",
            "side",
            "dte",
            "short",
            "long",
            "qty",
            "limit",
            "order_id",
            "status",
            "net_credit",
        ]
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(hdr)
            writer.writerow(row)

        def spread_exists_live(self, short_sym: str, long_sym: str) -> bool:
            """
            Returns True if either option leg exists in the live positions at Alpaca.
            Prevents duplicate spread entries at broker level.
            """
            try:
                # Use the same source as group_positions()
                positions = self.data_layer.trading.get_all_positions()
                symbols = {getattr(p, "symbol", "") for p in positions}

            except Exception as e:
                self.trade_log.error("[ORDER] Failed to fetch live positions: %s", e)
                return False

            if short_sym in symbols or long_sym in symbols:
                self.trade_log.info(
                    "[ORDER] Live duplicate detected: short=%s long=%s found in broker positions.",
                    short_sym,
                    long_sym,
                )
                return True

            return False



    # ======================================================
    # Open Spread
    # ======================================================
    def open_spread(
        self, signal: str, c: SpreadCandidate, qty: int
    ) -> Optional[str]:
        """Submit a new multi-leg order to open a vertical spread."""
        legs = [
            OptionLegRequest(
                symbol=c.short_sym,
                side=OrderSide.SELL,
                ratio_qty=1,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=c.long_sym,
                side=OrderSide.BUY,
                ratio_qty=1,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
        ]

        # Conservative open: 95% of modelled credit
        limit = round(c.net_credit * 0.95, 2)

        req = LimitOrderRequest(
            qty=int(qty),
            type=OrderType.LIMIT,
            limit_price=limit,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
        )

        try:
            order = self.data_layer.trading.submit_order(req)
            trade_log.info(
                "OPEN MLEG %s %s/%s QTY=%d LIMIT=%.2f ID=%s",
                c.side.upper(),
                c.short_sym,
                c.long_sym,
                qty,
                limit,
                order.id,
            )

            if self.trade_log:
                self.trade_log.info(
                    "OPEN %s|%s|%s Qty=%d | Side=%s | Delta=%s | Credit=%.2f | Limit=%.2f",
                    c.short_sym,
                    c.long_sym,
                    getattr(c, "expiration", "NA"),
                    qty,
                    signal,
                    getattr(c, "delta", "N/A"),
                    c.net_credit,
                    limit,
                )

            self._log_csv(
                [
                    utc_now_iso(),
                    signal,
                    c.side,
                    str(c.dte),
                    c.short_sym,
                    c.long_sym,
                    str(qty),
                    f"{limit:.2f}",
                    order.id,
                    order.status,
                    f"{c.net_credit:.2f}",
                ]
            )
            return order.id

        except Exception as e:
            trade_log.exception("Open spread failed: %s", e)
            if self.trade_log:
                self.trade_log.error(
                    "FAILED OPEN %s|%s | %s", c.short_sym, c.long_sym, e
                )
            return None

    # ======================================================
    # Group Positions
    # ======================================================
    def group_positions(self) -> Dict[str, List[OpenLeg]]:
        """Return current option positions grouped by expiration date (YYMMDD)."""
        out: Dict[str, List[OpenLeg]] = {}
        for p in self.data_layer.trading.get_all_positions():
            # Example symbol: "SPY251121P00658000"
            exp = p.symbol[3:9]  # "251121"
            out.setdefault(exp, []).append(
                OpenLeg(symbol=p.symbol, qty=float(p.qty))
            )
        return out

    # ======================================================
    # Close Spread
    # ======================================================
    def close_spread(
        self,
        short_sym: str,
        long_sym: str,
        qty: int,
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """Close an existing vertical spread."""
        legs = [
            OptionLegRequest(
                symbol=short_sym,
                side=OrderSide.BUY,
                ratio_qty=1,
                position_intent=PositionIntent.BUY_TO_CLOSE,
            ),
            OptionLegRequest(
                symbol=long_sym,
                side=OrderSide.SELL,
                ratio_qty=1,
                position_intent=PositionIntent.SELL_TO_CLOSE,
            ),
        ]

        if limit_price is None:
            limit_price = 0.10

        req = LimitOrderRequest(
            qty=int(qty),
            type=OrderType.LIMIT,
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
        )

        try:
            order = self.data_layer.trading.submit_order(req)
            trade_log.info(
                "CLOSE MLEG %s/%s QTY=%d LIMIT=%.2f ID=%s",
                short_sym,
                long_sym,
                qty,
                limit_price,
                order.id,
            )

            if self.trade_log:
                self.trade_log.info(
                    "CLOSE %s|%s | Qty=%d | Limit=%.2f | ID=%s",
                    short_sym,
                    long_sym,
                    qty,
                    limit_price,
                    order.id,
                )

            return order.id

        except Exception as e:
            trade_log.exception("Close spread failed: %s", e)
            if self.trade_log:
                self.trade_log.error(
                    "FAILED CLOSE %s|%s | %s", short_sym, long_sym, e
                )
            return None
