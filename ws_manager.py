"""
ws_manager.py

Managed WebSocket controller for Alpaca Options Trades.

Responsibilities:
- Start/stop WS automatically based on market hours
- Subscribe to a SMALL rolling set of option symbols
- Handle reconnects, throttling, and backoff
- Persist last trade prints into options_snapshots
- Never block market_data_collector

WS is used ONLY for:
- last_trade_price
- last_trade_size
- last_trade_ts
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from alpaca.data.live import OptionDataStream

log = logging.getLogger(__name__)


class WSManager:
    """
    Lightweight stateful WS controller.

    This class should be instantiated ONCE and
    ticked from market_data_collector.
    """

    # ----------------------------
    # Rate-limit safety
    # ----------------------------
    MAX_SYMBOLS = 12
    ROTATION_INTERVAL_SEC = 600   # rotate symbols every 10 minutes
    BACKOFF_STEPS = [5, 15, 60, 300]

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        db_writer,               # callable(symbol, price, size, ts)
        symbol_provider,         # callable() -> list[str]
        market_open_fn,          # callable() -> bool
    ):
        self.api_key = api_key
        self.secret_key = secret_key

        self.db_writer = db_writer
        self.symbol_provider = symbol_provider
        self.market_open_fn = market_open_fn

        self.ws: Optional[OptionDataStream] = None
        self.running = False

        self.subscribed_symbols: List[str] = []
        self.last_rotation_ts = 0.0
        self.backoff_idx = 0

    # ======================================================
    # Public control
    # ======================================================

    def tick(self):
        """
        Called periodically (e.g. every loop iteration).

        Decides whether WS should be running and healthy.
        """
        if not self.market_open_fn():
            self.stop()
            return

        if not self.running:
            self.start()
            return

        if time.time() - self.last_rotation_ts > self.ROTATION_INTERVAL_SEC:
            self.rotate_symbols()

    def start(self):
        """
        Start WS if not running.
        """
        if self.running:
            return

        symbols = self._select_symbols()
        if not symbols:
            log.info("[WS] No symbols available — skipping WS start")
            return

        log.info("[WS] Starting with %d symbols", len(symbols))

        self.ws = OptionDataStream(
            self.api_key,
            self.secret_key,
        )

        for sym in symbols:
            self.ws.subscribe_trades(self._on_trade, sym)

        self.subscribed_symbols = symbols
        self.last_rotation_ts = time.time()
        self.running = True

        asyncio.create_task(self._run())

    def stop(self):
        """
        Graceful shutdown.
        """
        if not self.running:
            return

        log.info("[WS] Stopping")
        try:
            if self.ws:
                self.ws.stop()
        except Exception:
            pass

        self.ws = None
        self.running = False
        self.subscribed_symbols = []
        self.backoff_idx = 0

    # ======================================================
    # Symbol management
    # ======================================================

    def rotate_symbols(self):
        """
        Unsubscribe old symbols and subscribe new ones.
        """
        if not self.running:
            return

        new_symbols = self._select_symbols()
        if not new_symbols:
            return

        if set(new_symbols) == set(self.subscribed_symbols):
            return

        log.info(
            "[WS] Rotating symbols (%d → %d)",
            len(self.subscribed_symbols),
            len(new_symbols),
        )

        try:
            self.stop()
            self.start()
        except Exception as e:
            log.error("[WS] Rotation failed: %s", e)
            self._apply_backoff()

    def _select_symbols(self) -> List[str]:
        """
        Pull a SMALL prioritized symbol list from DB or logic layer.
        """
        try:
            symbols = self.symbol_provider()
            symbols = list(dict.fromkeys(symbols))  # dedupe, preserve order
            return symbols[: self.MAX_SYMBOLS]
        except Exception as e:
            log.error("[WS] Symbol selection failed: %s", e)
            return []

    # ======================================================
    # WebSocket internals
    # ======================================================

    async def _run(self):
        """
        WS event loop with backoff handling.
        """
        try:
            await self.ws.run()
        except Exception as e:
            log.error("[WS] Error: %s", e)
            self._apply_backoff()

    async def _on_trade(self, trade):
        """
        Trade callback.
        """
        try:
            self.db_writer(
                symbol=trade.symbol,
                price=trade.price,
                size=trade.size,
                ts=trade.timestamp,
            )
        except Exception as e:
            log.error("[WS] DB write failed: %s", e)

    # ======================================================
    # Backoff logic
    # ======================================================

    def _apply_backoff(self):
        """
        Exponential-ish backoff on WS failure.
        """
        self.stop()

        delay = self.BACKOFF_STEPS[
            min(self.backoff_idx, len(self.BACKOFF_STEPS) - 1)
        ]
        self.backoff_idx += 1

        log.warning("[WS] Backing off for %ds", delay)
        time.sleep(delay)
