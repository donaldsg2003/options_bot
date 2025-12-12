#!/usr/bin/env python3
"""
market_data_collector.py
Production-grade continuous market data collector for SPY/VIXY.

- Uses ONLY IEX feed (no SIP subscription required)
- Writes base snapshot row first, then injects IV/skew/indicators into the same row
- Regime logic matches trading bot (VIXY percentile)
"""

import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

from alpaca.data.historical import (OptionHistoricalDataClient,
                                    StockHistoricalDataClient)
from alpaca.data.requests import (OptionChainRequest, StockBarsRequest,
                                  StockLatestQuoteRequest)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from option_bot_spreads.paths import MARKET_DATA_LOG, ML_TRAINING_DB
from option_bot_spreads.core.iv_provider import MissingIVProvider
from option_bot_spreads.core.options_pricing import (
    OptionContract,
    compute_surface_features_for_expiration,
    implied_vol,
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    tte_years,
)

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# ============================================================
# Paths / Config
# ============================================================

load_dotenv()

# Use centralized project paths
DB_PATH = ML_TRAINING_DB
LOG_PATH = MARKET_DATA_LOG

def get_last_valid_iv_snapshot():
    """
    Returns the most recent market_snapshot row with valid IV
    that was computed live (is_iv_stale = 0).
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    atm_iv,
                    iv_skew_25d,
                    iv_skew_10d,
                    iv_call_put_skew
                FROM market_snapshots
                WHERE atm_iv IS NOT NULL
                  AND is_iv_stale = 0
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        log.error("[IV] Failed to fetch last valid IV snapshot: %s", e)
        return None


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


API_KEY, SECRET_KEY = get_api_keys()
if not API_KEY or not SECRET_KEY:
    print("[ERROR] Missing Alpaca API credentials")
    sys.exit(1)

# ============================================================
# Logging (lazy formatting)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# Alpaca Clients
# ============================================================

stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
option_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ============================================================
# DB Setup
# ============================================================


def init_db():
    """
    Create market_snapshots / options_snapshots tables with the
    columns this collector expects. Safe to call on every startup.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS market_snapshots (
            timestamp TEXT PRIMARY KEY,

            -- Core prices
            spy_price REAL,
            spy_bid REAL,
            spy_ask REAL,
            vixy_price REAL,
            vixy_bid REAL,
            vixy_ask REAL,
            vix_close REAL,

            -- Regime
            vixy_percentile REAL,
            vixy_1yr_low REAL,
            vixy_1yr_high REAL,
            regime TEXT,

            -- Simple stats
            spy_day_change_pct REAL,
            vixy_day_change_pct REAL,

            -- Technical indicators
            spy_ema_20 REAL,
            spy_ema_50 REAL,
            spy_ema_200 REAL,
            spy_rsi_14 REAL,
            spy_macd REAL,
            spy_macd_signal REAL,
            spy_macd_hist REAL,
            spy_bbands_upper REAL,
            spy_bbands_middle REAL,
            spy_bbands_lower REAL,
            spy_bbands_width REAL,
            spy_atr_14 REAL,
            spy_adx_14 REAL,
            spy_roc_10 REAL,
            spy_williams_r REAL,
            spy_stoch_k REAL,
            spy_stoch_d REAL,
            spy_cci_20 REAL,
            spy_volume REAL,
            spy_volume_sma_20 REAL,
            spy_obv REAL,
            spy_vwap REAL,
            spy_realized_vol_10d REAL,
            spy_realized_vol_20d REAL,
            spy_realized_vol_30d REAL,
            spy_vixy_correlation_20d REAL,

            -- IV / skew (daily snapshot based on upcoming Friday)
            atm_iv REAL,
            iv_skew_25d REAL,
            iv_skew_10d REAL,
            iv_call_put_skew REAL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS options_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            underlying TEXT,
            expiration TEXT,
            option_type TEXT,
            bid REAL,
            ask REAL,
            mid REAL,
            volume INTEGER,
            open_interest INTEGER,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            implied_vol REAL
        )
        """
    )

    conn.commit()
    conn.close()
    log.info("Database initialized at %s", DB_PATH)


# ============================================================
# VIXY regime logic (matches bot)
# ============================================================

_VIXY_CACHE_TS = None
_VIXY_CACHE_VAL = None
VIXY_CACHE_SECONDS = 300


def _filter_outliers(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    if len(series) < 10:
        return series
    z = np.abs((series - series.mean()) / (series.std() + 1e-9))
    return series[z < z_thresh]


def get_vixy_percentile(lookback_days: int = 365):
    """
    Compute VIXY price percentile over the past year using ONLY IEX daily bars.
    Returns: (current_price, percentile, 1yr_low, 1yr_high)
    """
    global _VIXY_CACHE_TS, _VIXY_CACHE_VAL

    now = time.time()
    if _VIXY_CACHE_TS is not None and (now - _VIXY_CACHE_TS) < VIXY_CACHE_SECONDS:
        return _VIXY_CACHE_VAL

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    try:
        req = StockBarsRequest(
            symbol_or_symbols="VIXY",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
        )
        resp = stock_client.get_stock_bars(req)
        df = resp.df

        if df is None or df.empty:
            log.warning("[VIXY] No VIXY bars returned")
            return None, None, None, None

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        closes = df["close"]
        current = float(closes.iloc[-1])

        closes_filtered = _filter_outliers(closes)
        low = float(closes_filtered.min())
        high = float(closes_filtered.max())

        if high == low:
            pct = 0.5
        else:
            pct = (current - low) / (high - low)

        _VIXY_CACHE_TS = now
        _VIXY_CACHE_VAL = (current, pct, low, high)
        return _VIXY_CACHE_VAL

    except Exception as e:
        log.error("[VIXY] Percentile calc failed: %s", e)
        return None, None, None, None


def classify_regime(vixy_percentile: float | None) -> str:
    """
    Match your bot's regime logic:
    - LOW_VOL: VIXY pct < 33%
    - MID_VOL: 33–66%
    - HIGH_VOL: > 66%
    """
    if vixy_percentile is None:
        return "MID_VOL"

    if vixy_percentile < 0.33:
        return "LOW_VOL"
    if vixy_percentile > 0.66:
        return "HIGH_VOL"
    return "MID_VOL"


# ============================================================
# Technical indicators (SPY) via daily bars (IEX only)
# ============================================================


def collect_technical_indicators(symbol: str = "SPY", lookback_days: int = 200) -> dict:
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
        )
        bars = stock_client.get_stock_bars(req).df

        if bars is None or bars.empty:
            log.warning("[INDICATORS] No bars returned for %s", symbol)
            return {}

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.droplevel(0)

        df = bars.copy()
        df["ret"] = df["close"].pct_change()

        # EMA
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)

        # RSI
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None:
            df["bb_upper"] = bb["BBU_20_2.0"]
            df["bb_middle"] = bb["BBM_20_2.0"]
            df["bb_lower"] = bb["BBL_20_2.0"]
            df["bb_width"] = bb["BBB_20_2.0"]

        # ATR, ADX
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

        # Rate of change, Williams %R, Stochastics, CCI
        df["roc_10"] = ta.roc(df["close"], length=10)
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        if stoch is not None:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]

        df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)

        # Volume-based indicators
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # Realized vol (annualized) over 10/20/30 days
        for window in (10, 20, 30):
            vol = df["ret"].tail(window).std() * np.sqrt(252)
            df[f"realized_vol_{window}d"] = vol

        # Correlation SPY vs VIXY (optional)
        try:
            start = end - timedelta(days=60)
            vixy_req = StockBarsRequest(
                symbol_or_symbols="VIXY",
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed="iex",
            )
            vixy_df = stock_client.get_stock_bars(vixy_req).df
            if vixy_df is not None and not vixy_df.empty:
                if isinstance(vixy_df.index, pd.MultiIndex):
                    vixy_df = vixy_df.droplevel(0)
                vixy_ret = vixy_df["close"].pct_change().tail(20)
                spy_ret = df["ret"].tail(20)
                if len(vixy_ret) == len(spy_ret):
                    df["spy_vixy_correlation_20d"] = spy_ret.corr(vixy_ret)
        except Exception as e:
            log.warning("[INDICATORS] VIXY correlation calc failed: %s", e)

        latest = df.iloc[-1]

        out = {
            "spy_ema_20": latest.get("ema_20"),
            "spy_ema_50": latest.get("ema_50"),
            "spy_ema_200": latest.get("ema_200"),
            "spy_rsi_14": latest.get("rsi_14"),
            "spy_macd": latest.get("macd"),
            "spy_macd_signal": latest.get("macd_signal"),
            "spy_macd_hist": latest.get("macd_hist"),
            "spy_bbands_upper": latest.get("bb_upper"),
            "spy_bbands_middle": latest.get("bb_middle"),
            "spy_bbands_lower": latest.get("bb_lower"),
            "spy_bbands_width": latest.get("bb_width"),
            "spy_atr_14": latest.get("atr_14"),
            "spy_adx_14": latest.get("adx_14"),
            "spy_roc_10": latest.get("roc_10"),
            "spy_williams_r": latest.get("williams_r"),
            "spy_stoch_k": latest.get("stoch_k"),
            "spy_stoch_d": latest.get("stoch_d"),
            "spy_cci_20": latest.get("cci_20"),
            "spy_volume": latest.get("volume"),
            "spy_volume_sma_20": latest.get("volume_sma_20"),
            "spy_obv": latest.get("obv"),
            "spy_vwap": latest.get("vwap"),
            "spy_realized_vol_10d": latest.get("realized_vol_10d"),
            "spy_realized_vol_20d": latest.get("realized_vol_20d"),
            "spy_realized_vol_30d": latest.get("realized_vol_30d"),
            "spy_vixy_correlation_20d": latest.get("spy_vixy_correlation_20d"),
        }

        return out

    except Exception as e:
        log.error("[INDICATORS] Failed: %s", e)
        return {}


# ============================================================
# IV / Skew estimation from option chain
# ============================================================


def _next_friday(target: datetime | None = None) -> datetime:
    if target is None:
        target = datetime.now(timezone.utc)
    # Next Friday from today (or given date)
    weekday = target.weekday()
    days_to_fri = (4 - weekday) % 7
    if days_to_fri == 0:
        days_to_fri = 7
    return target + timedelta(days=days_to_fri)


def collect_iv_metrics() -> dict:
    """
    Hybrid IV logic:
    - Use LIVE IV during market hours if option bid/ask exists
    - After hours, reuse last valid DB snapshot and mark stale
    """
    
    log.info("[IV-DEBUG] collect_iv_metrics() called")

    try:
        target = datetime.now(timezone.utc)

        # Determine expiration (next Friday)
        days_to_fri = (4 - target.weekday()) % 7
        if days_to_fri == 0:
            days_to_fri = 7
        exp_date = target + timedelta(days=days_to_fri)
        exp_str = exp_date.strftime("%Y-%m-%d")

        # Try live SPY quote
        try:
            quote = stock_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols="SPY")
            )["SPY"]
            spot = (quote.bid_price + quote.ask_price) / 2.0
        except Exception:
            spot = None

        live_possible = spot is not None and spot > 0

        contracts = []

        if live_possible:
            try:
                put_chain = option_client.get_option_chain(
                    OptionChainRequest(
                        underlying_symbol="SPY",
                        expiration_date=exp_str,
                        contract_type="put",
                    )
                )
                call_chain = option_client.get_option_chain(
                    OptionChainRequest(
                        underlying_symbol="SPY",
                        expiration_date=exp_str,
                        contract_type="call",
                    )
                )

                def _add(chain, opt_type):
                    right = "C" if opt_type == "call" else "P"
                    for sym, c in chain.items():
                        try:
                            strike = float(sym[-8:]) / 1000.0
                        except Exception:
                            continue

                        if abs(strike - spot) / spot > 0.20:
                            continue

                        bid = getattr(c, "bid_price", None)
                        ask = getattr(c, "ask_price", None)

                        if bid is None or ask is None or bid <= 0 or ask <= 0:
                            continue

                        contracts.append(
                            OptionContract(
                                symbol=sym,
                                strike=strike,
                                expiration=exp_date.date(),
                                option_type=right,
                                bid=float(bid),
                                ask=float(ask),
                            )
                        )

                _add(put_chain, "put")
                _add(call_chain, "call")

            except Exception:
                contracts = []

        # =========================
        # LIVE PATH
        # =========================
        if contracts:
            features = compute_surface_features_for_expiration(
                contracts,
                spot=spot,
                expiration=exp_date.date(),
                as_of_dt=target,
                max_strikes=15,
            )

            if features and features.get("atm_iv") is not None:
                return {
                    "atm_iv": features.get("atm_iv"),
                    "iv_skew_25d": features.get("skew_25d"),
                    "iv_skew_10d": features.get("skew_10d"),
                    "iv_call_put_skew": features.get("call_put_skew"),
                    "is_iv_stale": 0,
                }

        # =========================
        # STALE / DB FALLBACK PATH
        # =========================
        last = get_last_valid_iv_snapshot()
        if last:
            log.info("[IV] Using DB-backed stale IV snapshot")
            return {
                "atm_iv": last["atm_iv"],
                "iv_skew_25d": last["iv_skew_25d"],
                "iv_skew_10d": last["iv_skew_10d"],
                "iv_call_put_skew": last["iv_call_put_skew"],
                "is_iv_stale": 1,
            }

        return {}

    except Exception as e:
        log.error("[IV] Hybrid IV computation failed: %s", e)
        return {}

# ============================================================
# Market open / scheduling (no SIP required)
# ============================================================


def is_market_open() -> bool:
    """
    Uses Alpaca Calendar API when available.
    Fully compatible with BOTH:
        - cal.open / cal.close = datetime.time
        - cal.open / cal.close = datetime.datetime
    Falls back to simple ET time check only when needed.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()

        cals = trading_client.get_calendar(
            GetCalendarRequest(start=today, end=today)
        )

        if not cals:
            return False  # weekend or holiday

        cal = cals[0]

        # Normalize open
        if isinstance(cal.open, datetime):
            open_dt = cal.open.astimezone(timezone.utc)
        else:
            open_dt = datetime.combine(today, cal.open, tzinfo=timezone.utc)

        # Normalize close
        if isinstance(cal.close, datetime):
            close_dt = cal.close.astimezone(timezone.utc)
        else:
            close_dt = datetime.combine(today, cal.close, tzinfo=timezone.utc)

        return open_dt <= now_utc <= close_dt

    except Exception as e:
        # Fallback ONLY if the calendar API is unavailable
        log.info("[MARKET] Calendar lookup failed, using fallback schedule: %s", e)
        now_et = datetime.now().astimezone()
        t = now_et.time()
        return dt_time(9, 30) <= t <= dt_time(16, 0)


def wait_for_market_open(poll_seconds: int = 60):
    """
    Sleep until market is open.
    """
    while not is_market_open():
        log.info("[MARKET] Closed. Sleeping %ds...", poll_seconds)
        time.sleep(poll_seconds)


# ============================================================
# DB Helpers
# ============================================================


def insert_snapshot_row(row: dict):
    """
    Insert a new row into market_snapshots. Overwrites if timestamp already exists.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cols = list(row.keys())
    vals = [row[c] for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)

    sql = f"""
        INSERT OR REPLACE INTO market_snapshots ({col_list})
        VALUES ({placeholders})
    """

    cur.execute(sql, vals)
    conn.commit()
    conn.close()


def update_snapshot(timestamp: str, updates: dict):
    """
    Update specific columns of an existing snapshot row.
    """
    if not updates:
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    vals = list(updates.values())
    vals.append(timestamp)

    sql = f"""
        UPDATE market_snapshots
        SET {set_clause}
        WHERE timestamp = ?
    """

    cur.execute(sql, vals)
    conn.commit()
    conn.close()


# ============================================================
# Core snapshot cycle
# ============================================================


def collect_market_snapshot():
    """
    Insert a base snapshot row (prices + regime) and return its timestamp.
    """
    try:
        spy_quote = stock_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols="SPY")
        )["SPY"]
        vixy_quote = stock_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols="VIXY")
        )["VIXY"]

        spy_mid = (spy_quote.bid_price + spy_quote.ask_price) / 2
        vixy_mid = (vixy_quote.bid_price + vixy_quote.ask_price) / 2

        vixy_current, vixy_pct, vixy_low, vixy_high = get_vixy_percentile()
        if vixy_pct is None:
            vixy_pct = 0.5
            vixy_low = None
            vixy_high = None

        regime = classify_regime(vixy_pct)

        ts = (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        row = {
            "timestamp": ts,
            "spy_price": float(spy_mid),
            "spy_bid": float(spy_quote.bid_price),
            "spy_ask": float(spy_quote.ask_price),
            "vixy_price": float(vixy_mid),
            "vixy_bid": float(vixy_quote.bid_price),
            "vixy_ask": float(vixy_quote.ask_price),
            "vix_close": float(vixy_current) if vixy_current is not None else None,
            "vixy_percentile": float(vixy_pct),
            "vixy_1yr_low": float(vixy_low) if vixy_low is not None else None,
            "vixy_1yr_high": float(vixy_high) if vixy_high is not None else None,
            "regime": regime,
        }

        insert_snapshot_row(row)
        log.info(
            "[SNAPSHOT] %s | SPY=%.2f VIXY=%.2f pct=%.1f%% regime=%s",
            ts,
            row["spy_price"],
            row["vixy_price"],
            vixy_pct * 100.0,
            regime,
        )

        return True, ts

    except Exception as e:
        log.error("[SNAPSHOT] Failed to collect market snapshot: %s", e)
        return False, None


def enrich_snapshot_with_indicators(timestamp: str):
    """
    Compute and inject technical indicators for the snapshot's timestamp.
    """
    try:
        indicators = collect_technical_indicators("SPY")
        if not indicators:
            log.warning("[SNAPSHOT] No indicators computed for %s", timestamp)
            return False

        update_snapshot(timestamp, indicators)
        log.info("[SNAPSHOT] Indicators updated for %s", timestamp)
        return True
    except Exception as e:
        log.error("[SNAPSHOT] Failed to enrich snapshot with indicators: %s", e)
        return False


_IV_PROVIDER = MissingIVProvider()

def enrich_snapshot_with_iv(timestamp: str):
    """
    Hybrid IV: provider-driven.
    For now MissingIVProvider returns 'missing' and we no-op without spamming logs.
    Later: swap provider implementation without touching collector logic.
    """
    try:
        iv = _IV_PROVIDER.get_iv_snapshot(timestamp)
        if iv.source == "missing":
            return False

        update_snapshot(timestamp, iv.as_db_dict())
        log.info("[SNAPSHOT] IV metrics updated for %s (source=%s stale=%s)", timestamp, iv.source, iv.is_stale)
        return True

    except Exception as e:
        log.warning("[SNAPSHOT] IV provider failed for %s: %s", timestamp, e)
        return False


def collect_option_snapshots(timestamp: str, max_symbols: int = 50):
    """
    Collect a small sample of SPY options around ATM, record bid/ask/greeks.
    No SIP; we rely on Alpaca's indicative data.
    """
    try:
        # Use near-term Friday expiration
        exp_date = _next_friday()
        exp_str = exp_date.strftime("%Y-%m-%d")

        put_chain = option_client.get_option_chain(
            OptionChainRequest(
                underlying_symbol="SPY",
                expiration_date=exp_str,
                contract_type="put",
            )
        )
        call_chain = option_client.get_option_chain(
            OptionChainRequest(
                underlying_symbol="SPY",
                expiration_date=exp_str,
                contract_type="call",
            )
        )

        if not put_chain and not call_chain:
            log.warning("[OPTIONS] Empty chains at %s", exp_str)
            return 0

        # SPY spot (for ATM distance)
        quote = stock_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols="SPY")
        )["SPY"]
        spot = (quote.bid_price + quote.ask_price) / 2

        records = []

        def process_chain(chain: dict, opt_type: str):
            local_records = []
            for sym, c in chain.items():
                try:
                    strike = float(sym[-8:]) / 1000.0
                except Exception:
                    continue

                if abs(strike - spot) / spot > 0.10:
                    # Only keep strikes within 10% of spot
                    continue

                bid = getattr(c, "bid_price", None)
                ask = getattr(c, "ask_price", None)
                mid = None
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0

                g = getattr(c, "greeks", None)
                delta = getattr(g, "delta", None) if g else None
                gamma = getattr(g, "gamma", None) if g else None
                theta = getattr(g, "theta", None) if g else None
                vega = getattr(g, "vega", None) if g else None
                iv = getattr(g, "iv", None) if g else None

                vol = getattr(c, "volume", None)
                oi = getattr(c, "open_interest", None)

                local_records.append(
                    (
                        timestamp,
                        sym,
                        "SPY",
                        exp_str,
                        opt_type,
                        bid,
                        ask,
                        mid,
                        vol,
                        oi,
                        delta,
                        gamma,
                        theta,
                        vega,
                        iv,
                    )
                )
            return local_records

        records.extend(process_chain(put_chain, "put"))
        records.extend(process_chain(call_chain, "call"))

        if not records:
            log.warning("[OPTIONS] No near-ATM records at %s", exp_str)
            return 0

        # Limit count to max_symbols for DB size
        records = sorted(
            records, key=lambda r: abs(float(r[7] or 0.0)) if r[7] is not None else 999
        )
        records = records[:max_symbols]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.executemany(
            """
            INSERT INTO options_snapshots (
                timestamp, symbol, underlying, expiration, option_type,
                bid, ask, mid, volume, open_interest,
                delta, gamma, theta, vega, implied_vol
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        conn.commit()
        conn.close()

        log.info("[OPTIONS] Recorded %d option snapshots for %s", len(records), timestamp)
        return len(records)

    except Exception as e:
        log.error("[OPTIONS] Failed to collect option snapshots: %s", e)
        return 0


# ============================================================
# Main loop
# ============================================================


def main():
    init_db()

    snapshot_interval_sec = 60      # base snapshot every 60 seconds
    indicators_interval_sec = 300   # full technicals every 5 minutes
    options_interval_sec = 600      # option snapshot every 10 minutes

    last_snapshot_timestamp = None
    last_indicators_ts = 0.0
    last_options_ts = 0.0

    snapshot_count = 0
    feature_count = 0
    options_count = 0

    log.info("Starting market_data_collector. DB=%s", DB_PATH)

    try:
        while True:
            if not is_market_open():
                log.info("[MARKET CLOSED] Waiting for open...")
                wait_for_market_open()
                continue

            # --------------------------------------------------
            # 1) Base market snapshot (prices + regime)
            # --------------------------------------------------
            ok, ts = collect_market_snapshot()
            if ok:
                snapshot_count += 1
                last_snapshot_timestamp = ts

            now = time.time()

            # 1) Options first (so rows exist for this timestamp)
            if (
                last_snapshot_timestamp is not None
                and now - last_options_ts >= options_interval_sec
            ):
                n = collect_option_snapshots(last_snapshot_timestamp, max_symbols=50)
                options_count += n
                last_options_ts = now

            # 2) Indicators + IV next
            if (
                last_snapshot_timestamp is not None
                and now - last_indicators_ts >= indicators_interval_sec
            ):
                if enrich_snapshot_with_indicators(last_snapshot_timestamp):
                    feature_count += 1

                # This must come AFTER options are recorded for that timestamp
                if enrich_snapshot_with_iv(last_snapshot_timestamp):
                    feature_count += 1

                last_indicators_ts = now

            time.sleep(snapshot_interval_sec)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down.")
        log.info(
            "Final stats: snapshots=%d | features=%d | options=%d",
            snapshot_count,
            feature_count,
            options_count,
        )
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
