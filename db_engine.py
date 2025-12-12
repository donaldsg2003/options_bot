"""
db_engine.py
Centralized database manager for the options bot.

Creates all operational tables, handles safe inserts,
maintenance (prune + vacuum), and ensures schema integrity.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from option_bot_spreads.paths import TRADING_DB

log = logging.getLogger(__name__)


# =============================================================================
# DB ENGINE
# =============================================================================


class DBEngine:
    """
    Centralized DB engine for the operational database.

    - Ensures full schema creation (Option 2 ML-ready)
    - Provides safe insert/update helpers
    - Handles maintenance (vacuum/prune)
    """

    def __init__(self, db_path: Path = TRADING_DB) -> None:
        self.db_path = Path(db_path)
        self._ensure_directory()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_directory(self) -> None:
        """Ensure DB directory exists."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        """Open a SQLite connection with WAL enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    # ------------------------------------------------------------------
    # Schema Initialization
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create all required operational tables for Option 2 schema."""
        conn = self._connect()
        cur = conn.cursor()

        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS open_spreads (
                spread_key TEXT PRIMARY KEY,
                short_symbol TEXT,
                long_symbol TEXT,
                qty INTEGER,
                credit REAL,
                entry_timestamp TEXT,
                entry_regime TEXT,
                expiration TEXT,
                trade_id TEXT,
                status TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS closed_spreads (
                spread_key TEXT PRIMARY KEY,
                short_symbol TEXT,
                long_symbol TEXT,
                qty INTEGER,
                credit REAL,
                debit REAL,
                pnl REAL,
                pnl_pct REAL,
                entry_timestamp TEXT,
                exit_timestamp TEXT,
                entry_regime TEXT,
                exit_reason TEXT,
                expiration TEXT
            );

            CREATE TABLE IF NOT EXISTS market_features (
                timestamp TEXT PRIMARY KEY,
                dte_bucket INTEGER,
                spy_range REAL,
                spy_body REAL,
                spy_change REAL,
                spy_volatility REAL,
                zscore_range REAL,
                zscore_body REAL,
                percentile_vol REAL,
                percentile_range REAL,
                percentile_body REAL
            );

            CREATE TABLE IF NOT EXISTS orphan_orders (
                order_id TEXT PRIMARY KEY,
                detected_at TEXT,
                status TEXT,
                reason TEXT
            );

            CREATE TABLE IF NOT EXISTS trade_summary (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                avg_pnl REAL,
                avg_pnl_pct REAL,
                gross_pnl REAL,
                gross_pnl_pct REAL
            );
            """
        )

        conn.commit()
        conn.close()
        log.info("[DB] Schema initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------

    def insert(self, table: str, data: Dict[str, Any]) -> None:
        """Generic safe insert."""
        keys = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        values = list(data.values())

        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("[DB] Insert failed (%s): %s", table, e)

    def insert_many(self, table: str, rows: Iterable[Dict[str, Any]]) -> None:
        """Efficient batch insert."""
        rows = list(rows)
        if not rows:
            return

        keys = ", ".join(rows[0].keys())
        placeholders = ", ".join(["?"] * len(rows[0]))

        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.executemany(
                f"INSERT OR REPLACE INTO {table} ({keys}) VALUES ({placeholders})",
                [list(r.values()) for r in rows],
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.error("[DB] insert_many failed (%s): %s", table, e)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        try:
            conn = self._connect()
            conn.execute("VACUUM;")
            conn.close()
            log.info("[DB] Vacuum completed")
        except Exception as e:
            log.error("[DB] Vacuum failed: %s", e)

    def prune_table(self, table: str, older_than_days: int) -> None:
        """
        Delete old rows based on timestamp column.

        Assumes timestamps are stored in UTC ISO-8601 form, e.g. '2025-11-11T21:41:33Z'.
        """
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cutoff = (
            cutoff_dt.replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            conn.close()
            log.info(
                "[DB] Pruned table %s older than %d days",
                table,
                older_than_days,
            )
        except Exception as e:
            log.error("[DB] prune_table failed (%s): %s", table, e)

# ------------------------------------------------------------------
# Legacy compatibility wrapper for main.py
# ------------------------------------------------------------------
def maybe_run_db_maintenance(db: DBEngine, days: int = 30) -> None:
    """
    Compatibility wrapper for the old interface used by main.py.
    Runs a prune on market tables and a vacuum.
    """
    try:
        db.prune_table("market_features", older_than_days=days)
        db.vacuum()
    except Exception as e:
        log.error("[DB] Maintenance failed: %s", e)
