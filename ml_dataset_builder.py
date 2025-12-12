from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from option_bot_spreads.paths import ML_TRAINING_DB, DATA_DIR

log = logging.getLogger("data")


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    """
    Resolve the ML training DB path from the central paths module.

    We rely on option_bot_spreads.paths.ML_TRAINING_DB, which points to:
        option_bot_spreads/data/ml_training_data.db
    """
    return ML_TRAINING_DB


@dataclass
class MLDatasetBuilder:
    """
    Utility for building ML-ready datasets from ml_training_data.db.

    Phase 1:
        - Load all spread candidates from ml_spread_candidates
        - Flatten JSON 'details' field into feature columns
        - Derive a 'source' flag (live vs synthetic)
        - Save CSV snapshots for training.

    Later phases:
        - Join entries / exits / outcomes for lifecycle P&L labels.
    """

    db_path: Path = _default_db_path()
    candidate_table: str = "ml_spread_candidates"
    entry_table: str = "ml_entry_signals"
    exit_table: str = "ml_exit_decisions"
    trades_table: str = "ml_trades"

    # ======================================================================
    # Core utilities
    # ======================================================================
    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the ML DB."""
        db = self.db_path

        if not db.exists():
            raise FileNotFoundError(f"ML training DB not found at {db}")

        log.info("[ML DATA] Opening ML DB at %s", db)
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        return conn

    def list_tables(self) -> Dict[str, Dict[str, Any]]:
        """
        Introspect the SQLite DB and return table names + column info.

        Helpful for debugging schema when we extend to lifecycle datasets.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [r[0] for r in cursor.fetchall()]
            info: Dict[str, Dict[str, Any]] = {}

            for t in tables:
                col_cursor = conn.execute(f"PRAGMA table_info('{t}')")
                cols = [
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": row[3],
                        "default": row[4],
                        "pk": row[5],
                    }
                    for row in col_cursor.fetchall()
                ]
                info[t] = {"columns": cols, "n_columns": len(cols)}
            return info
        finally:
            conn.close()

    # ======================================================================
    # Candidate dataset
    # ======================================================================
    def load_candidates(self) -> pd.DataFrame:
        """
        Load all rows from ml_spread_candidates.

        Schema from ml_data_collector.py:

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
                details TEXT    -- JSON blob
            )

        We will:
            - Parse timestamp → datetime
            - Flatten JSON from 'details' into 'details_*' columns
            - Derive a 'source' column (live/synthetic).
        """
        conn = self._connect()
        try:
            # Ensure table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                (self.candidate_table,),
            )
            row = cursor.fetchone()
            if not row:
                raise RuntimeError(
                    f"Candidate table '{self.candidate_table}' not found in {self.db_path}"
                )

            df = pd.read_sql_query(
                f"SELECT * FROM {self.candidate_table}", conn
            )
        finally:
            conn.close()

        if df.empty:
            log.warning(
                "[ML DATA] Candidate table '%s' is empty in %s",
                self.candidate_table,
                self.db_path,
            )
            return df

        log.info(
            "[ML DATA] Loaded %d candidate rows from table '%s'",
            len(df),
            self.candidate_table,
        )

        # Normalize timestamp
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce", utc=True
                )
                log.info("[ML DATA] Parsed 'timestamp' as datetime")
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "[ML DATA] Failed to parse 'timestamp' column: %s",
                    exc,
                )

        # Flatten JSON 'details' column if present
        if "details" in df.columns:
            df = self._flatten_details(df)

        # Derive 'source' column:
        #   - if details_source exists, use that
        #   - else default to 'live' for backward compatibility
        if "details_source" in df.columns:
            df["source"] = df["details_source"].fillna("live")
        else:
            df["source"] = "live"

        return df

    def _flatten_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten the 'details' JSON column into top-level details_* columns.

        Keeps original 'details' column for traceability.
        """
        if "details" not in df.columns:
            return df

        parsed: Dict[int, Dict[str, Any]] = {}
        for idx, raw in df["details"].items():
            if raw is None or raw == "":
                continue
            try:
                if isinstance(raw, str):
                    obj = json.loads(raw)
                else:
                    obj = raw
                if isinstance(obj, dict):
                    parsed[idx] = obj
            except Exception as exc:  # noqa: BLE001
                log.debug("[ML DATA] Failed to parse 'details' JSON at row %s: %s", idx, exc)

        if not parsed:
            return df

        details_df = pd.DataFrame.from_dict(parsed, orient="index")

        # Prefix to avoid collisions
        details_df = details_df.add_prefix("details_")

        # Align indexes and concat
        details_df = details_df.reindex(df.index)
        merged = pd.concat([df, details_df], axis=1)

        log.info(
            "[ML DATA] Flattened 'details' into %d details_* feature columns",
            details_df.shape[1],
        )
        return merged

    def build_candidate_dataset(
        self,
        save: bool = True,
        out_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Build a candidate-level dataset and optionally persist to CSV.

        Returns:
            pandas.DataFrame with:
                - raw columns from ml_spread_candidates
                - details_* feature columns from JSON 'details'
                - 'source' column indicating 'live' or 'synthetic'
        """
        df = self.load_candidates()

        if df.empty:
            log.warning("[ML DATA] Candidate dataset is empty; nothing to save.")
            return df

        if not save:
            return df

        # Resolve output directory
        if out_dir is None:
            out_dir = DATA_DIR / "ml_datasets"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"ml_candidates_{ts_str}.csv"
        out_path = out_dir / filename

        df.to_csv(out_path, index=False)
        log.info(
            "[ML DATA] Saved candidate dataset (%d rows, %d cols) to %s",
            len(df),
            df.shape[1],
            out_path,
        )

        return df

    # ======================================================================
    # Lifecycle dataset (stub for now)
    # ======================================================================
    def build_lifecycle_dataset(self) -> pd.DataFrame:
        """
        Placeholder for a full lifecycle dataset builder.

        Target output: one row per *completed* spread with columns such as:
            - features at entry (delta, width, credit, regime, etc.)
            - next-bar / next-day P&L
            - full lifecycle P&L
            - win/loss flag
            - max drawdown
            - regime features over life of trade
            - volatility features over life of trade

        This will eventually join:
            - ml_spread_candidates
            - ml_entry_signals
            - ml_exit_decisions
            - ml_trades (and/or spread_lifecycle)

        For now, this is a stub so we do not guess your schemas.
        """
        raise NotImplementedError(
            "Lifecycle dataset building will be implemented once we lock in the "
            "ml_training_data.db table schemas for trades and exits."
        )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Convenience entry point:

        python -m option_bot_spreads.ml_dataset_builder

    Will:
        - connect to the ML DB
        - build candidate dataset
        - save a timestamped CSV under data/ml_datasets/
        - log a small preview.
    """
    builder = MLDatasetBuilder()
    df = builder.build_candidate_dataset(save=True)

    try:
        preview = df.head()
        log.info("[ML DATA] Candidate dataset preview:\n%s", preview)
    except Exception as exc:  # noqa: BLE001
        log.debug("[ML DATA] Could not log preview: %s", exc)


if __name__ == "__main__":
    main()
