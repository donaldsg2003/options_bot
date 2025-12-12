#!/usr/bin/env python3
# ml_data_collector.py - Lightweight ML lifecycle logger

"""
Lightweight ML data collection for the options bot.

This module focuses on TRACK 3:
    data/ml_training_data.db

It records:
- One row per trade in `ml_trades`
- Entry/exit events in `ml_trade_events`

It is intentionally simple and side-effect-free so it can be safely
imported by core modules like helpers/state_utils.py.
"""

from __future__ import annotations
from option_bot_spreads.paths import ML_TRAINING_DB

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import json


def _utc_now_iso() -> str:
    """Return UTC timestamp as ISO string with Z suffix (no micros)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


@dataclass
class TradeEntrySnapshot:
    trade_id: str
    spread_key: str
    opened_at: Optional[str]
    qty: int
    entry_credit: float
    entry_dte: Optional[int]
    regime: Optional[str]


@dataclass
class TradeExitSnapshot:
    trade_id: str
    spread_key: str
    opened_at: Optional[str]
    closed_at: Optional[str]
    qty: int
    entry_credit: float
    close_price: Optional[float]
    realized_pl: Optional[float]
    return_pct: Optional[float]
    entry_dte: Optional[int]
    regime: Optional[str]


class MLDataCollector:
    """
    MLDataCollector manages the dedicated ML DB:
        data/ml_training_data.db

    Tables:
        ml_trades:
            trade_id TEXT PRIMARY KEY
            spread_key TEXT
            opened_at TEXT
            closed_at TEXT
            qty INTEGER
            entry_credit REAL
            close_price REAL
            realized_pl REAL
            return_pct REAL
            entry_dte INTEGER
            regime_entry TEXT

        ml_trade_events:
            id INTEGER PRIMARY KEY AUTOINCREMENT
            trade_id TEXT NOT NULL
            event_type TEXT NOT NULL  -- 'entry' or 'exit'
            timestamp TEXT NOT NULL
            details TEXT               -- optional JSON / free-form
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        if db_path is None:
            self.db_path = ML_TRAINING_DB
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = logging.getLogger("ml")

        self._init_db()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        cur = conn.cursor()

        # One row per trade
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_trades (
                trade_id TEXT PRIMARY KEY,
                spread_key TEXT,
                opened_at TEXT,
                closed_at TEXT,
                qty INTEGER,
                entry_credit REAL,
                close_price REAL,
                realized_pl REAL,
                return_pct REAL,
                entry_dte INTEGER,
                regime_entry TEXT
            )
            """
        )

        # Entry/exit events
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                event_type TEXT NOT NULL,    -- 'entry' or 'exit'
                timestamp TEXT NOT NULL,
                details TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_entry_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                signal TEXT,
                regime TEXT,
                spread_key TEXT,
                chosen INTEGER DEFAULT 0,
                details TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_spread_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                signal TEXT,
                regime TEXT,
                short_symbol TEXT,
                long_symbol TEXT,
                credit REAL,
                delta REAL,
                width REAL,
                dte INTEGER,
                chosen INTEGER DEFAULT 0,
                details TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_exit_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                spread_key TEXT,
                decision_engine TEXT,
                reason TEXT,
                score REAL,
                regime TEXT,
                details TEXT
            )
        """
        )


        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Public API: Entry / Exit snapshots
    # --------------------------------------------------
    def record_entry(
        self,
        trade_id: str,
        spread_key: str,
        opened_at: Optional[str],
        qty: int,
        entry_credit: float,
        entry_dte: Optional[int],
        regime: Optional[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a trade ENTRY snapshot into ml_trades + ml_trade_events.
        Safe to call multiple times for the same trade_id; it will upsert.
        """
        if not trade_id:
            # Fallback to spread_key as synthetic ID, but log it
            trade_id = spread_key
            self._log.warning(
                "[ML] record_entry called without trade_id, using spread_key=%s",
                spread_key,
            )

        if opened_at is None:
            opened_at = _utc_now_iso()

        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Upsert into ml_trades
            cur.execute(
                """
                INSERT INTO ml_trades (
                    trade_id, spread_key, opened_at, qty,
                    entry_credit, entry_dte, regime_entry
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    spread_key=excluded.spread_key,
                    opened_at=excluded.opened_at,
                    qty=excluded.qty,
                    entry_credit=excluded.entry_credit,
                    entry_dte=excluded.entry_dte,
                    regime_entry=excluded.regime_entry
                """,
                (
                    trade_id,
                    spread_key,
                    opened_at,
                    int(qty),
                    float(entry_credit),
                    entry_dte,
                    regime,
                ),
            )

            # Insert an "entry" event
            details_str = None
            if extra:
                try:
                    import json

                    details_str = json.dumps(extra, default=str)
                except Exception:
                    details_str = str(extra)

            cur.execute(
                """
                INSERT INTO ml_trade_events (
                    trade_id, event_type, timestamp, details
                ) VALUES (?, ?, ?, ?)
                """,
                (trade_id, "entry", opened_at, details_str),
            )

            conn.commit()
        finally:
            conn.close()

    def record_exit(
        self,
        trade_id: str,
        spread_key: str,
        opened_at: Optional[str],
        closed_at: Optional[str],
        qty: int,
        entry_credit: float,
        close_price: Optional[float],
        realized_pl: Optional[float],
        return_pct: Optional[float],
        entry_dte: Optional[int],
        regime: Optional[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a trade EXIT snapshot into ml_trades + ml_trade_events.

        This does NOT enforce that an entry was previously logged.
        If called first, it will create the row and fill what it can.
        """
        if not trade_id:
            trade_id = spread_key
            self._log.warning(
                "[ML] record_exit called without trade_id, using spread_key=%s",
                spread_key,
            )

        if closed_at is None:
            closed_at = _utc_now_iso()

        conn = self._get_conn()
        try:
            cur = conn.cursor()

            # Upsert / update ml_trades
            cur.execute(
                """
                INSERT INTO ml_trades (
                    trade_id, spread_key, opened_at, closed_at, qty,
                    entry_credit, close_price, realized_pl, return_pct,
                    entry_dte, regime_entry
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    spread_key=excluded.spread_key,
                    opened_at=COALESCE(ml_trades.opened_at, excluded.opened_at),
                    closed_at=excluded.closed_at,
                    qty=excluded.qty,
                    entry_credit=COALESCE(ml_trades.entry_credit, excluded.entry_credit),
                    close_price=excluded.close_price,
                    realized_pl=excluded.realized_pl,
                    return_pct=excluded.return_pct,
                    entry_dte=COALESCE(ml_trades.entry_dte, excluded.entry_dte),
                    regime_entry=COALESCE(ml_trades.regime_entry, excluded.regime_entry)
                """,
                (
                    trade_id,
                    spread_key,
                    opened_at,
                    closed_at,
                    int(qty),
                    float(entry_credit),
                    close_price,
                    realized_pl,
                    return_pct,
                    entry_dte,
                    regime,
                ),
            )

            # Insert an "exit" event
            details_str = None
            if extra:
                try:
                    import json

                    details_str = json.dumps(extra, default=str)
                except Exception:
                    details_str = str(extra)

            cur.execute(
                """
                INSERT INTO ml_trade_events (
                    trade_id, event_type, timestamp, details
                ) VALUES (?, ?, ?, ?)
                """,
                (trade_id, "exit", closed_at, details_str),
            )

            conn.commit()
        finally:
            conn.close()

    # --------------------------------------------------
    # Public API: signals, candidates, exit decisions
    # --------------------------------------------------
    def record_entry_signal(
        self,
        signal: Optional[str],
        regime: Optional[str],
        spread_key: Optional[str],
        chosen: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a high level entry signal (chosen or not)."""
        ts = _utc_now_iso()
        details_str = json.dumps(extra, sort_keys=True) if extra else None

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ml_entry_signals (
                    timestamp, signal, regime, spread_key, chosen, details
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, signal, regime, spread_key, int(bool(chosen)), details_str),
            )
            conn.commit()
        finally:
            conn.close()

    def record_spread_candidate(
        self,
        signal: Optional[str],
        regime: Optional[str],
        short_symbol: str,
        long_symbol: str,
        credit: float,
        delta: float,
        width: float,
        dte: int,
        chosen: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a candidate spread considered during entry logic."""
        ts = _utc_now_iso()
        details_str = json.dumps(extra, sort_keys=True) if extra else None

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ml_spread_candidates (
                    timestamp, signal, regime,
                    short_symbol, long_symbol,
                    credit, delta, width, dte,
                    chosen, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    signal,
                    regime,
                    short_symbol,
                    long_symbol,
                    float(credit),
                    float(delta),
                    float(width),
                    int(dte),
                    int(bool(chosen)),
                    details_str,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_exit_decision(
        self,
        spread_key: str,
        decision_engine: str,
        reason: str,
        score: Optional[float],
        regime: Optional[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an exit decision (even if no order is sent)."""
        ts = _utc_now_iso()
        details_str = json.dumps(extra, sort_keys=True) if extra else None

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ml_exit_decisions (
                    timestamp, spread_key, decision_engine,
                    reason, score, regime, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    spread_key,
                    decision_engine,
                    reason,
                    score if score is not None else None,
                    regime,
                    details_str,
                ),
            )
            conn.commit()
        finally:
            conn.close()