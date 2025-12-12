# ======================================================
# config.py Enhanced Adaptive Configuration
# ======================================================

import os
import sys
import logging
from dataclasses import dataclass
from typing import Dict, Tuple
from dotenv import load_dotenv

# ------------------------------------------------------
# Optional import: python-keyring for encrypted secrets
# ------------------------------------------------------
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# ------------------------------------------------------
# Load .env fallback
# ------------------------------------------------------
load_dotenv()


def _b(name: str, default="false") -> bool:
    """Safely parse boolean environment variables."""
    val = os.getenv(name, default).strip().lower()
    return val in ("1", "true", "yes", "y", "on")


# ------------------------------------------------------
# Secure key retrieval
# ------------------------------------------------------
def get_secure_key(service: str, key_name: str, env_fallback: str | None = None) -> str:
    """
    Retrieve a secret from keyring if available, otherwise from environment.
    """
    if KEYRING_AVAILABLE:
        try:
            stored = keyring.get_password(service, key_name)
            if stored:
                return stored
        except Exception:
            pass

    if env_fallback:
        env_val = os.getenv(env_fallback)
        if env_val:
            return env_val

    return ""


# ------------------------------------------------------
# Settings dataclass with Adaptive DTE
# ------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    # API Keys
    API_KEY: str = get_secure_key("alpaca", "API_KEY", "APCA_API_KEY_ID")
    SECRET_KEY: str = get_secure_key("alpaca", "SECRET_KEY", "APCA_SECRET_KEY")
    PAPER: bool = _b("APCA_PAPER", "true")

    # Strategy parameters
    SYMBOL: str = os.getenv("SYMBOL", "SPY")
    USE_VIXY: bool = _b("USE_VIXY", "true")
    
    # ADAPTIVE DTE - Regime-based ranges
    USE_ADAPTIVE_DTE: bool = _b("USE_ADAPTIVE_DTE", "true")

    # config.py – inside Settings dataclass
    DURATION_MODE: str = os.getenv("DURATION_MODE", "adaptive")
    # allowed: "adaptive", "45_21", "ml"

    @property
    def IS_45_21(self) -> bool:
        return self.DURATION_MODE.lower() == "45_21"

    # 45/21 specific knobs
    DURATION_45_ENTRY_DTE: int = int(os.getenv("DURATION_45_ENTRY_DTE", "45"))
    DURATION_45_EXIT_DTE: int = int(os.getenv("DURATION_45_EXIT_DTE", "21"))
    DURATION_45_TP_PCT: float = float(os.getenv("DURATION_45_TP_PCT", "0.50"))
    DURATION_45_EMERGENCY_SL_MULT: float = float(
        os.getenv("DURATION_45_EMERGENCY_SL_MULT", "3.0")
    )


    
    # Legacy single-range DTE (fallback when adaptive disabled)
    MIN_DTE: int = int(os.getenv("MIN_DTE", "14"))
    MAX_DTE: int = int(os.getenv("MAX_DTE", "21"))
    
    # Delta & spread geometry
    TARGET_DELTA: float = float(os.getenv("TARGET_DELTA", "-0.20"))
    DELTA_WINDOW: float = float(os.getenv("DELTA_WINDOW", "0.15"))
    SPREAD_WIDTH: float = float(os.getenv("SPREAD_WIDTH", "5.0"))
    DEFAULT_TARGET_DELTA: float = TARGET_DELTA
    DEFAULT_SPREAD_WIDTH: float = SPREAD_WIDTH

    # Entry & exit criteria
    MIN_NET_CREDIT: float = float(os.getenv("MIN_NET_CREDIT", "0.80"))
    EXIT_FRACTION: float = float(os.getenv("EXIT_FRACTION", "0.25"))
    TP_PCT: float = float(os.getenv("TP_PCT", "0.65"))
    SL_MULT: float = float(os.getenv("SL_MULT", "2.0"))
    MIN_EXIT_DTE: int = 1
    TRAILING_STOP_TRIGGER: float = float(os.getenv("TRAILING_STOP_TRIGGER", "0.5"))
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "0.3"))
    FORCED_TP_PCT = 0.55          # or 0.50 if you want very aggressive harvesting
    MIN_LIFE_USED_FOR_FORCED_TP = 0.20  # close even earlier in the lifecycle

    
    # ======================================================
    # PROFIT TARGET TUNING RECOMMENDATIONS
    # Based on your zero-assignment-risk requirement
    # ======================================================

    # Current profit targets in config.py are good:
    # HIGH_VOL: 50%  - Good (fast moves, early exit)
    # MID_VOL: 65%   - Good (standard target)
    # LOW_VOL: 75%   - Good (grind it out)

    # Consider these adjustments if you want to be MORE aggressive:
    # HIGH_VOL: 40%  - Exit earlier in volatile conditions
    # MID_VOL: 60%   - Slightly more conservative
    # LOW_VOL: 70%   - Still good for theta decay

    # Or LESS aggressive (let winners run more):
    # HIGH_VOL: 60%  - Wait for bigger moves
    # MID_VOL: 70%   - Higher targets
    # LOW_VOL: 80%   - Maximize theta decay
    PROFIT_TARGETS = {
        "HIGH_VOL": 0.50,  # Take 50% profit in high vol (moves fast)
        "MID_VOL": 0.70,   # Take 65% profit in normal conditions
        "LOW_VOL": 0.80,   # Take 75% profit in low vol (grind it out)
    }

    # Enhanced exit configuration
    EMERGENCY_EXIT_DTE: int = int(os.getenv("EMERGENCY_EXIT_DTE", "2"))
    TRAILING_STOP_LOCKUP: float = float(os.getenv("TRAILING_STOP_LOCKUP", "0.90"))
    TRAILING_STOP_ACTIVATION: float = float(os.getenv("TRAILING_STOP_ACTIVATION", "0.01"))
        
    # Exit order types by scenario
    USE_MARKET_ORDER_EMERGENCY: bool = _b("USE_MARKET_ORDER_EMERGENCY", "true")
    USE_MARKET_ORDER_PIN_RISK: bool = _b("USE_MARKET_ORDER_PIN_RISK", "true")

    # Volatility regime filters
    VIXY_MAX_ENTRY: float = float(os.getenv("VIXY_MAX_ENTRY", "55"))
    VIXY_MAX_EXIT: float = float(os.getenv("VIXY_MAX_EXIT", "70"))
    VIXY_LOOKBACK_DAYS: int = int(os.getenv("VIXY_LOOKBACK_DAYS", "365"))

    # Risk & sizing
    MAX_RISK_PCT: float = float(os.getenv("MAX_RISK_PCT", "0.05"))
    MAX_CONCURRENT_SPREADS: int = int(os.getenv("MAX_CONCURRENT_SPREADS", "6"))
    MAX_POSITIONS_PER_REGIME: int = 2  # Max 2 positions per DTE bucket
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.5"))

    # Orphaned leg detection
    AUTO_CLOSE_ORPHANS: bool = _b("AUTO_CLOSE_ORPHANS", "false")
    ORPHAN_MIN_VERIFICATIONS: int = int(os.getenv("ORPHAN_MIN_VERIFICATIONS", "3"))
    ORPHAN_VERIFICATION_INTERVAL: int = int(os.getenv("ORPHAN_VERIFICATION_INTERVAL", "300"))
    ORPHAN_AGE_MIN: int = int(os.getenv("ORPHAN_AGE_MIN", "900"))

    # Behavioral flags
    ALLOW_NEUTRAL_PUTS: bool = _b("ALLOW_NEUTRAL_PUTS", "true")
    ALLOW_BEAR_CALLS: bool = _b("ALLOW_BEAR_CALLS", "true")

    # Modes & debug
    ADAPTIVE_MODE: bool = True
    DEBUG_MODE: bool = _b("DEBUG_MODE", "false")
    TEST_MODE: bool = _b("TEST_MODE", "true")

    # Logging
    LOG_FILE: str = os.getenv("LOG_FILE", "options_bot_FULL.log")
    LOG_RETENTION_TIME: int = int(os.getenv("LOG_RETENTION_TIME", "30"))
    
    # ML Configuration
    ML_MIN_TRADES: int = 35  # Start ML after 35 completed trades
    ML_RETRAIN_INTERVAL: int = 10  # Retrain every 10 new trades
    ML_ENABLED: bool = _b("ML_ENABLED", "false")  # Manual toggle


# ------------------------------------------------------
# DTE Regime Buckets (not frozen, managed separately)
# ------------------------------------------------------
class DTEConfig:
    """
    Adaptive DTE ranges based on volatility regime.
    Shorter DTE in high vol = faster data collection + reduced gamma risk.
    Longer DTE in low vol = optimal theta efficiency.
    """
    # config.py - Add to DTEConfig
    BUCKETS: Dict[str, Dict[str, int]] = {
        "HIGH_VOL": {
            "min_dte": 10,
            "max_dte": 14,
            "max_positions": 0,
            "exit_dte": 3,
            "pin_risk_dte": 2,  # Pin risk: close ITM at 2 DTE
            "pin_risk_moneyness": 0.01,  # 1% ITM triggers close
            "delta_window": 0.12,   # Wider in high vol
        },
        "MID_VOL": {
            "min_dte": 14,
            "max_dte": 21,
            "max_positions": 6,
            "exit_dte": 5,
            "pin_risk_dte": 4,  # Pin risk: close ITM at 4 DTE
            "pin_risk_moneyness": 0.015,  # 1.5% ITM
            "delta_window": 0.10,  # Balanced
        },
        "LOW_VOL": {
            "min_dte": 30,
            "max_dte": 45,
            "max_positions": 6,
            "exit_dte": 10,
            "pin_risk_dte": 7,  # Pin risk: close ITM at 7 DTE
            "pin_risk_moneyness": 0.02,  # 2% ITM
            "delta_window": 0.08,  # Tighter in low vol
        },
    }

    @classmethod
    def get_exit_dte(cls, regime: str) -> int:
        """Get exit DTE threshold for regime."""
        # 45/21 mode: override with the global exit DTE
        if SETTINGS.DURATION_MODE.lower() == "45_21":
            return SETTINGS.DURATION_45_EXIT_DTE

        bucket = cls.BUCKETS.get(regime, cls.BUCKETS["MID_VOL"])
        return bucket.get("exit_dte", 7)
        
    @classmethod
    def get_pin_risk_dte(cls, regime: str) -> int:
        """Get pin risk DTE threshold for regime (earlier than exit_dte)."""
        bucket = cls.BUCKETS.get(regime, cls.BUCKETS["MID_VOL"])
        return bucket.get("pin_risk_dte", 4)  # Default to 4 DTE
    
    @classmethod
    def get_pin_risk_moneyness(cls, regime: str) -> float:
        """Get moneyness threshold (% ITM) that triggers pin risk exit."""
        bucket = cls.BUCKETS.get(regime, cls.BUCKETS["MID_VOL"])
        return bucket.get("pin_risk_moneyness", 0.015)  # Default 1.5%
        
    @classmethod
    def get_dte_range(cls, regime: str) -> Tuple[int, int]:
        """Get min/max DTE for a given regime."""
        bucket = cls.BUCKETS.get(regime, cls.BUCKETS["MID_VOL"])
        return bucket["min_dte"], bucket["max_dte"]
    
    @classmethod
    def get_max_positions(cls, regime: str) -> int:
        """Get position limit for a regime."""
        bucket = cls.BUCKETS.get(regime, cls.BUCKETS["MID_VOL"])
        return bucket["max_positions"]
    
    @classmethod
    def classify_dte_to_regime(cls, dte: int) -> str:
        """
        Reverse lookup: given a DTE, determine which regime bucket it belongs to.
        Used for position tracking.
        """
        for regime, config in cls.BUCKETS.items():
            if config["min_dte"] <= dte <= config["max_dte"]:
                return regime
        return "MID_VOL"  # default

    @classmethod
    def get_delta_window(cls, regime: str) -> float:
        """Get delta window for regime."""
        bucket = cls.BUCKETS.get(regime, cls.BUCKETS["MID_VOL"])
        return bucket.get("delta_window", 0.10)

# ------------------------------------------------------
# Initialize and verify
# ------------------------------------------------------
SETTINGS = Settings()


def _check_credentials():
    """Fail fast if API keys are missing."""
    if not SETTINGS.API_KEY or not SETTINGS.SECRET_KEY:
        logging.error(
            "Alpaca credentials are missing. Set them via keyring:\n"
            "  python setup_keys.py\n"
            "or in .env as fallback:\n"
            "  APCA_API_KEY_ID=...\n"
            "  APCA_SECRET_KEY=...\n"
        )
        sys.exit(1)
    else:
        logging.info(
            "Credentials OK loaded via %s",
            "keyring" if KEYRING_AVAILABLE else ".env"
        )


# Perform health check on import
_check_credentials()