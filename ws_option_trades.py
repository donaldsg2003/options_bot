#!/usr/bin/env python3
"""
ws_option_trades.py

Listen to Alpaca Options trades via WebSocket and persist last-trade info into
ml_training_data.db options_snapshots rows.

Design goals:
- Never block other components
- Minimal DB writes (single UPDATE per trade)
- No assumptions about bid/ask (often None without SIP)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from alpaca.data.live import OptionDataStream

from option_bot_spreads.paths import ML_TRAINING_DB, MARKET_DATA_LOG


load_dotenv()

DB_PATH = ML_TRAINING_DB

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# ------------------------------------------------------------
# Logging (lazy formatting)
# ------------------------------------------------------------
log = logging.getLogger("ws_option_trades")

if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(MARKET_DATA_LOG, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_api_keys():
    """Try keyring first, then fall back to env."""
    api_key = ""
    secret_key = ""

    if KEYRING_AVAILABLE:
        try:
            api_key = keyring.get_password("alpaca", "API_KEY") or ""
            secret_key = keyring.get_password("alpaca", "SECRET_KEY") or ""
        except Exception:
            pass

    if not api_key:
        api_key = os.getenv("APCA_API_KEY_ID")
    if not secret_key:
        secret_key = os.getenv("APCA_SECRET_KEY")

    return api_key, secret_key


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _load_latest_snapshot_symbols(limit: int = 200) -> List[str]:
    """
    Pull symbols from options_snapshots at the latest timestamp (the most recent batch).
    Typically this is the 50 symbols your collector writes.
    """
    try:
        with _db_connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(timestamp) FROM options_snapshots")
            row = cur.fetchone()
            if not row or not row[0]:
                return []
            latest_ts = row[0]

            cur.execute(
                """
                SELECT DISTINCT symbol
                FROM options_snapshots
                WHERE timestamp = ?
                ORDER BY symbol
                LIMIT ?
                """,
                (latest_ts, limit),
            )
            syms = [r[0] for r in cur.fetchall() if r and r[0]]
            return syms
    except Exception as e:
        log.warning("[WS] Failed to load latest snapshot symbols: %s", e)
        return []


def _update_last_trade(symbol: str, trade_price: float, trade_size: float, trade_ts_iso: str) -> None:
    """
    Update last trade fields for the most recent snapshot row of this symbol.

    We update ONLY the latest timestamp row for that symbol to keep state clean:
      WHERE symbol = ? AND timestamp = (SELECT MAX(timestamp) FROM options_snapshots WHERE symbol = ?)
    """
    try:
        with _db_connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE options_snapshots
                SET last_trade_price = ?,
                    last_trade_size  = ?,
                    last_trade_ts    = ?
                WHERE symbol = ?
                  AND timestamp = (
                      SELECT MAX(timestamp)
                      FROM options_snapshots
                      WHERE symbol = ?
                  )
                """,
                (float(trade_price), float(trade_size), trade_ts_iso, symbol, symbol),
            )
            conn.commit()
    except Exception as e:
        log.warning("[WS] DB update failed for %s: %s", symbol, e)


async def run_stream(refresh_symbols_seconds: int = 120) -> None:
    api_key, secret = get_api_keys()
    if not api_key or not secret:
        raise RuntimeError("Missing Alpaca API keys (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")

    stream = OptionDataStream(api_key, secret)

    subscribed: set[str] = set()
    last_refresh = 0.0

    async def on_trade(t) -> None:
        # Trade model fields vary slightly by SDK version; use getattr safely
        sym = getattr(t, "symbol", None)
        price = getattr(t, "price", None)
        size = getattr(t, "size", None)

        if not sym or price is None or size is None:
            return

        # Timestamp: use message ts if present, otherwise now
        ts = getattr(t, "timestamp", None)
        if ts is None:
            trade_ts_iso = _utc_now_iso()
        else:
            # pydantic datetime -> ISO
            try:
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                trade_ts_iso = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                trade_ts_iso = _utc_now_iso()

        _update_last_trade(sym, float(price), float(size), trade_ts_iso)

        log.info("[TRADE] %s price=%s size=%s ts=%s", sym, price, size, trade_ts_iso)

    while True:
        now = time.time()
        if (now - last_refresh) >= refresh_symbols_seconds or not subscribed:
            syms = _load_latest_snapshot_symbols(limit=200)
            new_set = set(syms)

            if not new_set:
                log.info("[WS] No symbols available yet. Waiting...")
                await asyncio.sleep(5)
                last_refresh = now
                continue

            if new_set != subscribed:
                # Clear and resubscribe
                try:
                    stream.stop()
                except Exception:
                    pass

                stream = OptionDataStream(api_key, secret)
                stream.subscribe_trades(on_trade, *sorted(new_set))

                subscribed = new_set
                log.info("[WS] Subscribed to %d option symbols (latest snapshot batch).", len(subscribed))

            last_refresh = now

        # Keep the WS running in short increments so we can refresh subscriptions
        try:
            await asyncio.wait_for(stream._run_forever(), timeout=refresh_symbols_seconds)
        except asyncio.TimeoutError:
            # normal: refresh loop
            continue
        except Exception as e:
            log.warning("[WS] Stream error: %s", e)
            await asyncio.sleep(2)


def main() -> None:
    log.info("Starting ws_option_trades. DB=%s", DB_PATH)
    try:
        asyncio.run(run_stream(refresh_symbols_seconds=120))
    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down.")
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
