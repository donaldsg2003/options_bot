# engines/adaptive_engine.py

import logging
from datetime import datetime
import pytz

from option_bot_spreads.core.vixy_regime import get_adaptive_parameters
from option_bot_spreads.helpers.utils import utc_now_iso


def run_adaptive_refresh(
    state,
    strategy_engine,
    risk_manager,
    SETTINGS,
):
    """
    Executes exactly at 9:30 ET:
    - refresh regime
    - update target delta, width, min credit, delta_window
    - sets last_adaptive_date
    """
    sys_log = logging.getLogger()

    eastern = pytz.timezone("US/Eastern")
    now_et = datetime.now(eastern)

    if now_et.hour != 9 or now_et.minute != 30:
        return False  # no-op

    today = now_et.strftime("%Y-%m-%d")
    last = state.get("last_adaptive_date", "")

    if today == last:
        return False  # already refreshed today

    try:
        regime, delta, width, min_credit, cooldown_sec, bear_mode, min_dte, max_dte, delta_window = (
            get_adaptive_parameters()
        )

        # Apply to strategy + risk engines
        strategy_engine.target_delta = delta
        strategy_engine.spread_width = width
        strategy_engine.min_net_credit = min_credit
        strategy_engine.delta_window = delta_window

        risk_manager.spread_width = width

        state.set("last_vixy_regime", regime)
        state.set("last_adaptive_date", today)

        sys_log.info(
            "[ADAPTIVE INIT] Regime=%s | Δ=%.2f±%.2f | Width=%.1f | DTE=%d-%d | MinCredit=%.2f | BearMode=%s",
            regime,
            delta,
            delta_window,
            width,
            min_dte,
            max_dte,
            min_credit,
            bear_mode,
        )

        return {
            "regime": regime,
            "delta": delta,
            "width": width,
            "min_credit": min_credit,
            "cooldown": cooldown_sec,
            "bear_mode": bear_mode,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "delta_window": delta_window,
        }

    except Exception as e:
        sys_log.warning("Adaptive refresh failed: %s", e)
        return False
