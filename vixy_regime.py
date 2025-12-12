# =====================================================
# vixy_regime.py - Enhanced Adaptive Regime System
# Now returns DTE ranges based on volatility
# =====================================================

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from option_bot_spreads.core.persistent_state import PersistentState
from option_bot_spreads.config import SETTINGS, DTEConfig
from option_bot_spreads.paths import SESSION_STATE
from option_bot_spreads.helpers.utils import force_utf8_output
force_utf8_output()

data_log = logging.getLogger("data")

# Caching
_last_vixy_fetch = None
_last_vixy_result = None
VIXY_CACHE_SECONDS = 300  # 5 min

# Load environment
load_dotenv()

# Secure key retrieval
API_KEY = (keyring.get_password("alpaca", "API_KEY") if KEYRING_AVAILABLE else None) \
          or os.getenv("APCA_API_KEY_ID")
SECRET_KEY = (keyring.get_password("alpaca", "SECRET_KEY") if KEYRING_AVAILABLE else None) \
             or os.getenv("APCA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca credentials")

# Alpaca client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Persistent state
STATE_FILE = str(SESSION_STATE)
MAX_HISTORY = 5
_state = PersistentState(STATE_FILE)


def _filter_outliers(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Remove statistical outliers using Z-score."""
    if len(series) < 10:
        return series
    z = np.abs((series - series.mean()) / (series.std() + 1e-9))
    return series[z < z_thresh]


def _smooth_percentile(p: float, history: list, alpha: float = 0.5) -> float:
    """EMA smoothing for percentile stability."""
    if not history:
        return p
    last = history[-1]
    return (alpha * p) + ((1 - alpha) * last)


def _load_state():
    """Load rolling percentile history + last regime."""
    history = _state.get("recent_vixy_percentiles", [])
    last_regime = _state.get("last_vixy_regime")
    last_update = _state.get("last_vixy_update")
    return history, last_regime, last_update


def _save_state(history, regime):
    """Persist state."""
    try:
        _state.set("recent_vixy_percentiles", history[-MAX_HISTORY:])
        _state.set("last_vixy_regime", regime)
        _state.set(
            "last_vixy_update",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    except Exception as e:
        data_log.warning("[WARN] Could not persist VIXY state: %s", e)


def get_vixy_percentile(days_back=365) -> Tuple[float | None, float | None, pd.Series | None]:
    """
    Fetch VIXY closes, filter outliers, smooth percentile.
    Returns: (current_vixy, percentile, closes_series)
    """
    global _last_vixy_fetch, _last_vixy_result

    now = datetime.now()

    # Cache check
    if _last_vixy_fetch and (now - _last_vixy_fetch).total_seconds() < VIXY_CACHE_SECONDS:
        return _last_vixy_result

    end = now
    start = end - timedelta(days=days_back)

    try:
        req = StockBarsRequest(
            symbol_or_symbols="VIXY",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        resp = client.get_stock_bars(req)
        df = resp.df

        if df is None or df.empty:
            data_log.warning("[VIXY] No bars returned.")
            return None, None, None

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        if "close" not in df.columns:
            data_log.warning("[VIXY] Missing close data.")
            return None, None, None

        closes = df["close"].dropna()

        if len(closes) < 10:
            data_log.warning("[VIXY] Insufficient VIXY bars.")
            return None, None, None

        # Filter outliers
        closes = _filter_outliers(closes)

        current = float(closes.iloc[-1])
        low = float(closes.min())
        high = float(closes.max())

        if high == low:
            percentile = 0.50
        else:
            percentile = (current - low) / (high - low)

        # Smooth with EMA
        history, _, _ = _load_state()
        percentile = _smooth_percentile(percentile, history)

        data_log.info(
            "[VIXY] Close=%.2f | Low=%.2f | High=%.2f | Percentile=%.1f%% | Bars=%d",
            current, low, high, percentile * 100, len(closes)
        )

        result = (current, percentile, closes)
        _last_vixy_fetch = now
        _last_vixy_result = result

        return result

    except Exception as e:
        data_log.error("[VIXY] Error fetching percentile: %s", e)
        return None, None, None


# =====================================================
# ENHANCED: Adaptive parameters now include DTE range
# =====================================================
def get_adaptive_parameters() -> Tuple[str, float, float, float, int, bool, int, int, float]:
    """
    Return adaptive parameters with regime-based DTE ranges.

    Returns:
        (regime, target_delta, spread_width, min_credit, cooldown_sec,
         bear_mode, min_dte, max_dte, delta_window)
    """
    _, percentile, _ = get_vixy_percentile()
    
    if percentile is None:
        data_log.warning("No valid VIXY percentile — using defaults.")
        return (
            "UNKNOWN",
            SETTINGS.TARGET_DELTA,
            SETTINGS.SPREAD_WIDTH,
            SETTINGS.MIN_NET_CREDIT,
            90,
            False,
            SETTINGS.MIN_DTE,
            SETTINGS.MAX_DTE,
            SETTINGS.DELTA_WINDOW  # Add default delta_window
        )

    # Load previous state
    recent, last_regime, _ = _load_state()
    recent.append(round(percentile, 4))
    recent = recent[-MAX_HISTORY:]

    # Persist latest percentile for DB pipeline
    try:
        _state.set("last_vixy_percentile", float(percentile))
    except Exception as e:
        data_log.warning("[WARN] Could not persist last_vixy_percentile: %s", e)

    # Count regime days (3-day persistence)
    high_days = sum(p > 0.75 for p in recent[-3:])
    low_days = sum(p < 0.25 for p in recent[-3:])

    # Regime classification with persistence
    if high_days >= 3:
        regime = "HIGH_VOL"
        delta = -0.15
        width = 5
        bear_mode = True
    elif low_days >= 3:
        regime = "LOW_VOL"
        delta = -0.30
        width = 10
        bear_mode = False
    else:
        regime = "MID_VOL"
        delta = -0.25
        width = 8
        bear_mode = False

    # Persist last regime if insufficient confirmation
    if last_regime and regime != last_regime:
        if (last_regime == "HIGH_VOL" and high_days < 2) or \
           (last_regime == "LOW_VOL" and low_days < 2):
            regime = last_regime
            data_log.info("[PERSIST] Holding previous regime: %s", regime)

    # Adaptive multipliers
    if regime == "LOW_VOL":
        min_credit = round(SETTINGS.MIN_NET_CREDIT * 0.9, 2)
        cooldown_sec = 90
    elif regime == "MID_VOL":
        min_credit = round(SETTINGS.MIN_NET_CREDIT * 1.0, 2)
        cooldown_sec = 75
    else:  # HIGH_VOL
        min_credit = round(SETTINGS.MIN_NET_CREDIT * 1.1, 2)
        cooldown_sec = 60

    # NEW: Get DTE range from regime
    if SETTINGS.USE_ADAPTIVE_DTE:
        min_dte, max_dte = DTEConfig.get_dte_range(regime)
    else:
        min_dte, max_dte = SETTINGS.MIN_DTE, SETTINGS.MAX_DTE

    # Check 45-21 settings
    if SETTINGS.DURATION_MODE.lower() == "45_21":
        # Use a narrow band around the desired entry DTE
        entry = SETTINGS.DURATION_45_ENTRY_DTE
        min_dte = entry - 3
        max_dte = entry + 3


    # Get delta window from regime
    delta_window = DTEConfig.get_delta_window(regime)

    # Save state
    _save_state(recent, regime)

    data_log.info(
        "[VIXY REGIME] %s | Pct=%.1f%% | Delta=%.2f±%.2f | Width=%.1f | CreditX=%.2f | Cooldown=%ds | BearMode=%s | DTE=%d-%d | History=%s",
        regime, percentile * 100, delta, delta_window, width, min_credit, cooldown_sec, bear_mode,
        min_dte, max_dte,
        [round(p * 100, 1) for p in recent],
    )

    return regime, delta, width, min_credit, cooldown_sec, bear_mode, min_dte, max_dte, delta_window


if __name__ == "__main__":
    current, percentile, _ = get_vixy_percentile()
    if current is None:
        print("Failed to retrieve VIXY data.")
    else:
        regime, delta, width, credit, cooldown, bear, min_dte, max_dte, delta_window = get_adaptive_parameters()
        print("\n" + "="*50)
        print(f"VIXY Close: {current:.2f}")
        print(f"Percentile (1-yr): {percentile:.2%}")
        print(f"Volatility Regime: {regime}")
        print(f"TARGET_DELTA: {delta}")
        print(f"SPREAD_WIDTH: {width}")
        print(f"DTE RANGE: {min_dte}-{max_dte} days")
        print(f"Delta Window: ±{delta_window:.2f}")
        print(f"MinCredit: {credit} | Cooldown: {cooldown}s | BearMode: {bear}")
        print("="*50 + "\n")