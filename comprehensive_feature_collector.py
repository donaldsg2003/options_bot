"""
Comprehensive ML Feature Collector

This module reads market features + trade metadata from the existing SQLite DB,
validates inputs, enriches with derived ML features, and produces training-
friendly feature dictionaries.

This file is fully self-contained and does NOT modify DB schema.
It only *reads* from the DB and emits ML-ready Python dict rows.

Lazy logging formatting is used everywhere for performance.
"""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, field_validator

DB_PATH = Path("data/options_bot.db")
log = logging.getLogger(__name__)

# ============================================================
# Pydantic-Validated Market Snapshot
# ============================================================


class MarketSnapshot(BaseModel):
    timestamp: str
    spy: float = Field(gt=0, lt=2000)
    vixy: float = Field(gt=0, lt=500)
    vixy_pct: float | None = Field(ge=0, le=1)
    regime: str | None = None
    iv_rank: float | None = None
    expected_move: float | None = None
    skew: float | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """
        Validate that timestamp is valid ISO format and allow a 'Z' suffix.
        """
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"Invalid timestamp format: {v}")
        return v


# ============================================================
# DB Helpers
# ============================================================


def _get_conn() -> sqlite3.Connection:
    """Create SQLite connection safely."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_market_features() -> Dict[str, MarketSnapshot]:
    """
    Load all validated market snapshots keyed by ISO timestamp.
    Matches core.db schema: market_features(time, spy, vixy, vixy_pct, ...).
    """
    conn = _get_conn()
    cur = conn.cursor()

    # NOTE: core.db uses 'time' as the primary key column for market_features,
    # so we alias it as 'timestamp' here to match MarketSnapshot.
    rows = cur.execute(
        """
        SELECT
            time AS timestamp,
            spy,
            vixy,
            vixy_pct,
            regime,
            iv_rank,
            expected_move,
            skew
        FROM market_features
        ORDER BY time ASC
        """
    ).fetchall()

    out: Dict[str, MarketSnapshot] = {}

    for r in rows:
        try:
            snap = MarketSnapshot(
                timestamp=r["timestamp"],
                spy=r["spy"],
                vixy=r["vixy"],
                vixy_pct=r["vixy_pct"],
                regime=r["regime"],
                iv_rank=r["iv_rank"],
                expected_move=r["expected_move"],
                skew=r["skew"],
            )
            out[snap.timestamp] = snap

        except Exception as e:
            log.error("[MARKET SNAPSHOT] Invalid row skipped: %s", e)

    conn.close()
    return out


def load_trades() -> List[sqlite3.Row]:
    """
    Load trades from the unified trades table defined in core.db.

    Schema (from core.db):
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
    """
    conn = _get_conn()
    cur = conn.cursor()

    # Alias columns so the rest of this module can use the older
    # names (id / opened_at / closed_at) without change.
    rows = cur.execute(
        """
        SELECT
            trade_id        AS id,
            timestamp_open  AS opened_at,
            timestamp_close AS closed_at,
            underlying,
            strategy_type,
            regime_entry,
            regime_exit,
            vixy_pct_entry,
            vixy_pct_exit,
            credit,
            debit,
            dte_entry,
            dte_exit,
            pnl,
            return_pct,
            exit_reason,
            notes
        FROM trades
        ORDER BY timestamp_open ASC
        """
    ).fetchall()

    conn.close()
    return rows


# ============================================================
# Helper: nearest market snapshot
# ============================================================


def nearest_snapshot(
    snapshots: Dict[str, MarketSnapshot], ts: Optional[str]
) -> Optional[MarketSnapshot]:

    if not ts:
        return None

    try:
        target = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

    best_snap = None
    best_delta = timedelta(days=9999)

    for s in snapshots.values():
        t2 = datetime.fromisoformat(s.timestamp.replace("Z", "+00:00"))
        delta = abs(t2 - target)
        if delta < best_delta:
            best_delta = delta
            best_snap = s

    return best_snap


# ============================================================
# Feature Derivation
# ============================================================


def derive_trade_features(
    trade_row: sqlite3.Row,
    mkt_open: Optional[MarketSnapshot],
    mkt_close: Optional[MarketSnapshot],
) -> Dict[str, Any]:
    """
    Derive ML-friendly features for each trade.
    """
    out: Dict[str, Any] = {}

    # Raw columns
    out["trade_id"] = trade_row["id"]
    out["opened_at"] = trade_row["opened_at"]
    out["closed_at"] = trade_row["closed_at"]
    out["strategy"] = trade_row["strategy_type"]
    out["underlying"] = trade_row["underlying"]
    out["regime_entry"] = trade_row["regime_entry"]
    out["regime_exit"] = trade_row["regime_exit"]
    out["credit"] = trade_row["credit"]
    out["debit"] = trade_row["debit"]
    out["pnl"] = trade_row["pnl"]
    out["return_pct"] = trade_row["return_pct"]
    out["dte_entry"] = trade_row["dte_entry"]
    out["dte_exit"] = trade_row["dte_exit"]
    out["exit_reason"] = trade_row["exit_reason"]

    # Expand notes JSON (if exists and JSON-encoded)
    notes = trade_row["notes"]
    if notes:
        try:
            import json

            n = json.loads(notes)
            for k, v in n.items():
                out[f"note_{k}"] = v
        except Exception as e:
            # Not fatal; just log and move on
            log.error("[NOTES] JSON decode failed: %s", e)

    # Market features at entry
    if mkt_open:
        out["spy_open"] = mkt_open.spy
        out["vixy_open"] = mkt_open.vixy
        out["vixy_pct_open"] = mkt_open.vixy_pct
        out["regime_mkt_open"] = mkt_open.regime
        out["iv_rank_open"] = mkt_open.iv_rank
        out["expected_move_open"] = mkt_open.expected_move
        out["skew_open"] = mkt_open.skew

    # Market features at exit
    if mkt_close:
        out["spy_close"] = mkt_close.spy
        out["vixy_close"] = mkt_close.vixy
        out["vixy_pct_close"] = mkt_close.vixy_pct
        out["regime_mkt_close"] = mkt_close.regime
        out["iv_rank_close"] = mkt_close.iv_rank
        out["expected_move_close"] = mkt_close.expected_move
        out["skew_close"] = mkt_close.skew

    # Derived metrics
    try:
        if out.get("spy_open") and out.get("spy_close"):
            out["spy_change_pct"] = out["spy_close"] / out["spy_open"] - 1
    except Exception:
        pass

    try:
        if out.get("vixy_open") and out.get("vixy_close"):
            out["vixy_change_pct"] = out["vixy_close"] / out["vixy_open"] - 1
    except Exception:
        pass

    return out


# ============================================================
# MAIN API: Produce all ML training rows
# ============================================================


def collect_all_features() -> List[Dict[str, Any]]:
    """
    High-level ML pipeline:
    - load trades
    - load market snapshots
    - match nearest market data to each trade entry/exit
    - output unified ML feature rows
    """
    log.info("[ML] Loading market snapshots...")
    snapshots = load_market_features()

    log.info("[ML] Loading trades...")
    trades = load_trades()

    out: List[Dict[str, Any]] = []
    log.info("[ML] Deriving features for %d trades", len(trades))

    for tr in trades:
        mkt_open = nearest_snapshot(snapshots, tr["opened_at"])
        mkt_close = nearest_snapshot(snapshots, tr["closed_at"])
        feat = derive_trade_features(tr, mkt_open, mkt_close)
        out.append(feat)

    log.info("[ML] Completed feature extraction: %d rows", len(out))
    return out


# ============================================================
# Save as CSV (optional)
# ============================================================


def export_csv(path: str = "ml_training_dataset.csv") -> None:
    rows = collect_all_features()
    if not rows:
        log.warning("[ML] No rows to export")
        return

    import csv

    keys = sorted(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)

    log.info("[ML] Exported %d rows → %s", len(rows), path)
