# db.py
# =====================================================
# SQLite database engine for tracking trades, legs,
# fills, P/L, and market features for ML training.
# 
# TRACK 1: Your Paper/Live Trading Database
# Database: data/options_bot.db
# =====================================================

import sqlite3
import uuid
import threading
from datetime import datetime, timezone, timedelta
import os
import csv
import shutil

from option_bot_spreads.helpers.utils import utc_now_iso

# -----------------------------------------------------
# DB path: data/options_bot.db (TRACK 1: Your paper/live trading)
# This is separate from:
#   - market_data.db (Track 2: Pure market data for virtual backtesting)
#   - ml_training_data.db (Track 3: Live trade snapshots for ML)
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to import from paths.py (organized structure)
try:
    from option_bot_spreads.paths import TRADING_DB
    DB_PATH = str(TRADING_DB)
except ImportError:
    # Fallback to data/ folder
    DB_PATH = os.path.join(BASE_DIR, "data", "options_bot.db")
    # Ensure data directory exists
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

_lock = threading.Lock()


# -----------------------------------------------------
# Connection helper
# -----------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = 1")
    return conn


# -----------------------------------------------------
# Initialize database + tables
# -----------------------------------------------------
def init_db() -> None:
    """Create tables if they don't exist."""
    with _lock:
        conn = get_conn()
        cur = conn.cursor()

        # Trades table (one row per spread / strategy instance)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp_open TEXT,
                timestamp_close TEXT,
                underlying TEXT,
                strategy_type TEXT,
                regime_entry TEXT,
                regime_exit TEXT,
                vixy_pct_entry REAL,
                vixy_pct_exit REAL,
                credit REAL,
                debit REAL,
                pnl REAL,
                return_pct REAL,
                dte_entry INTEGER,
                dte_exit INTEGER,
                exit_reason TEXT,
                notes TEXT
            )
            """
        )

        # Legs table (each option leg of the trade)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS legs (
                leg_id TEXT PRIMARY KEY,
                trade_id TEXT,
                symbol TEXT,
                side TEXT,
                qty REAL,
                entry_price REAL,
                exit_price REAL,
                expiration TEXT,
                strike REAL,
                right TEXT,
                FOREIGN KEY(trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE
            )
            """
        )

        # Fills table (optional, per-leg executions)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                trade_id TEXT,
                leg_id TEXT,
                timestamp TEXT,
                price REAL,
                qty REAL,
                side TEXT,
                order_id TEXT,
                FOREIGN KEY(trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE,
                FOREIGN KEY(leg_id) REFERENCES legs(leg_id) ON DELETE CASCADE
            )
            """
        )

        # Market features (for ML / analysis later)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_features (
                time TEXT PRIMARY KEY,
                spy REAL,
                vixy REAL,
                vixy_pct REAL,
                regime TEXT,
                iv_rank REAL,
                expected_move REAL,
                skew REAL
            )
            """
        )

        # Simple useful indexes
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(timestamp_open)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_legs_trade_id ON legs(trade_id)"
        )

        conn.commit()
        conn.close()
        
        # Log initialization via logging system (already UTF-8 safe)
        # Note: Don't use print() here - it runs before force_utf8_output()
        # Logging is configured in main.py with UTF-8 encoding


# -----------------------------------------------------
# Trade helpers
# -----------------------------------------------------
def new_trade(
    underlying: str,
    strategy_type: str,
    regime_entry: str | None = None,
    vixy_pct_entry: float | None = None,
    credit: float | None = None,
    dte_entry: int | None = None,
    notes: str | None = None,
) -> str:
    """
    Insert a new trade row and return its trade_id.
    This is spread-level, not per-leg.
    """
    trade_id = str(uuid.uuid4())
    ts = utc_now_iso()

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trades (
                trade_id,
                timestamp_open,
                underlying,
                strategy_type,
                regime_entry,
                vixy_pct_entry,
                credit,
                dte_entry,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                ts,
                underlying,
                strategy_type,
                regime_entry,
                vixy_pct_entry,
                credit,
                dte_entry,
                notes,
            ),
        )
        conn.commit()
        conn.close()

    return trade_id


def close_trade(
    trade_id: str,
    regime_exit: str | None = None,
    vixy_pct_exit: float | None = None,
    debit: float | None = None,
    pnl: float | None = None,
    return_pct: float | None = None,
    dte_exit: int | None = None,
    exit_reason: str | None = None,
) -> None:
    """
    Update an existing trade with exit information and P/L.
    Safe to call multiple times; it just overwrites.
    """
    ts_close = utc_now_iso()

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE trades SET
                timestamp_close = ?,
                regime_exit = COALESCE(?, regime_exit),
                vixy_pct_exit = COALESCE(?, vixy_pct_exit),
                debit = COALESCE(?, debit),
                pnl = COALESCE(?, pnl),
                return_pct = COALESCE(?, return_pct),
                dte_exit = COALESCE(?, dte_exit),
                exit_reason = COALESCE(?, exit_reason)
            WHERE trade_id = ?
            """,
            (
                ts_close,
                regime_exit,
                vixy_pct_exit,
                debit,
                pnl,
                return_pct,
                dte_exit,
                exit_reason,
                trade_id,
            ),
        )
        conn.commit()
        conn.close()


# -----------------------------------------------------
# Legs helpers (future multi-leg support)
# -----------------------------------------------------
def add_leg(
    trade_id: str,
    symbol: str,
    side: str,
    qty: float,
    entry_price: float | None,
    expiration: str,
    strike: float,
    right: str,
) -> str:
    """
    Insert a new leg row. Entry price can be None if unknown at open time.
    """
    leg_id = str(uuid.uuid4())

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO legs (
                leg_id,
                trade_id,
                symbol,
                side,
                qty,
                entry_price,
                expiration,
                strike,
                right
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                leg_id,
                trade_id,
                symbol,
                side,
                qty,
                entry_price,
                expiration,
                strike,
                right,
            ),
        )
        conn.commit()
        conn.close()

    return leg_id


def record_leg_exit(
    leg_id: str,
    exit_price: float,
) -> None:
    """
    Record final exit_price for a leg.
    Typically called when the spread is closed.
    """
    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE legs SET exit_price = ? WHERE leg_id = ?",
            (exit_price, leg_id),
        )
        conn.commit()
        conn.close()


# -----------------------------------------------------
# Fills helpers (for multi-part executions)
# -----------------------------------------------------
def add_fill(
    trade_id: str,
    leg_id: str,
    price: float,
    qty: float,
    side: str,
    order_id: str | None = None,
) -> str:
    """
    Record a partial fill or full fill for a given leg.
    This is optional if you want detailed fill tracking.
    """
    fill_id = str(uuid.uuid4())
    ts = utc_now_iso()

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fills (
                fill_id,
                trade_id,
                leg_id,
                timestamp,
                price,
                qty,
                side,
                order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill_id,
                trade_id,
                leg_id,
                ts,
                price,
                qty,
                side,
                order_id,
            ),
        )
        conn.commit()
        conn.close()

    return fill_id


# -----------------------------------------------------
# Market features (for ML later)
# -----------------------------------------------------
def log_market_features(
    spy: float | None,
    vixy: float | None,
    vixy_pct: float | None,
    regime: str | None,
    iv_rank: float | None = None,
    expected_move: float | None = None,
    skew: float | None = None,
) -> None:
    """
    Log a single snapshot of market state.
    Called from the main loop if you want to collect features.
    """
    ts = utc_now_iso()

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO market_features (
                time,
                spy,
                vixy,
                vixy_pct,
                regime,
                iv_rank,
                expected_move,
                skew
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, spy, vixy, vixy_pct, regime, iv_rank, expected_move, skew),
        )
        conn.commit()
        conn.close()


# -----------------------------------------------------
# Query helpers (for metrics & dashboards)
# -----------------------------------------------------
def query_trades(
    underlying: str | None = None,
    strategy_type: str | None = None,
    open_only: bool | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Generic trade query.
    - open_only=True  -> only open trades
    - open_only=False -> only closed trades
    - open_only=None  -> all trades
    """
    where: list[str] = []
    params: list[object] = []

    if underlying:
        where.append("underlying = ?")
        params.append(underlying)
    if strategy_type:
        where.append("strategy_type = ?")
        params.append(strategy_type)
    if open_only is True:
        where.append("timestamp_close IS NULL")
    elif open_only is False:
        where.append("timestamp_close IS NOT NULL")

    sql = "SELECT * FROM trades"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY timestamp_open DESC"
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()

    return rows


def query_open_positions() -> list[dict]:
    """
    Return open trades joined with their legs.
    This is DB-level view; real positions still come from Alpaca.
    """
    sql = """
        SELECT
            t.trade_id,
            t.timestamp_open,
            t.underlying,
            t.strategy_type,
            t.regime_entry,
            t.credit,
            l.leg_id,
            l.symbol,
            l.side,
            l.qty,
            l.expiration,
            l.strike,
            l.right
        FROM trades t
        JOIN legs l ON t.trade_id = l.trade_id
        WHERE t.timestamp_close IS NULL
        ORDER BY t.timestamp_open DESC
    """
    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
    return rows


def query_strategy_results() -> list[dict]:
    """
    Aggregate performance by strategy_type and regime_entry.
    Useful for dashboards and ML feature inspection.
    """
    sql = """
        SELECT
            strategy_type,
            regime_entry,
            COUNT(*) AS trades,
            AVG(pnl) AS avg_pnl,
            AVG(return_pct) AS avg_return_pct,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
        FROM trades
        WHERE timestamp_close IS NOT NULL
        GROUP BY strategy_type, regime_entry
        ORDER BY strategy_type, regime_entry
    """
    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
    return rows


# -----------------------------------------------------
# ML / CSV export helpers
# -----------------------------------------------------
def _export_table_to_csv(table: str, csv_path: str) -> None:
    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table}")
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        conn.close()

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)


def export_trades_csv(csv_path: str = "exports/trades.csv") -> None:
    _export_table_to_csv("trades", csv_path)


def export_legs_csv(csv_path: str = "exports/legs.csv") -> None:
    _export_table_to_csv("legs", csv_path)


def export_market_features_csv(csv_path: str = "exports/market_features.csv") -> None:
    _export_table_to_csv("market_features", csv_path)


def export_training_dataset(csv_path: str = "exports/training_dataset.csv") -> None:
    """
    Simple flat dataset: one row per *closed* trade with its entry features.
    More complex feature engineering can be done in notebooks.
    """
    sql = """
        SELECT
            t.trade_id,
            t.timestamp_open,
            t.timestamp_close,
            t.underlying,
            t.strategy_type,
            t.regime_entry,
            t.regime_exit,
            t.vixy_pct_entry,
            t.vixy_pct_exit,
            t.credit,
            t.debit,
            t.pnl,
            t.return_pct,
            t.dte_entry,
            t.dte_exit,
            t.exit_reason,
            t.notes
        FROM trades t
        WHERE t.timestamp_close IS NOT NULL
        ORDER BY t.timestamp_open
    """

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        conn.close()

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)


# -----------------------------------------------------
# Simple maintenance (manual / scheduled)
# -----------------------------------------------------
def vacuum_db() -> None:
    """Run VACUUM to compact the database."""
    with _lock:
        conn = get_conn()
        conn.execute("VACUUM")
        conn.close()


def prune_old_features(days: int = 365) -> None:
    """
    Delete market_features rows older than `days`.
    Keeps DB small while preserving enough history for ML.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    with _lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM market_features WHERE time < ?",
            (cutoff_iso,),
        )
        conn.commit()
        conn.close()


def snapshot_db(snapshot_dir: str = "db_snapshots") -> str:
    """
    Create a timestamped copy of the DB file.
    Useful before schema changes or major upgrades.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(snapshot_dir, f"options_bot_{ts}.db")
    shutil.copy2(DB_PATH, snapshot_path)
    return snapshot_path