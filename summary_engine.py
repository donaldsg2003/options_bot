# engines/summary_engine.py

import logging
from datetime import datetime, timezone
import pytz
from option_bot_spreads.helpers.utils import utc_now_iso


def log_startup_summary(SETTINGS):
    sys_log = logging.getLogger()
    sys_log.info(
        "CONFIG: %s | TP=%.2f | SL=%.2f | AdaptiveDTE=%s",
        SETTINGS.SYMBOL,
        SETTINGS.TP_PCT,
        SETTINGS.SL_MULT,
        SETTINGS.USE_ADAPTIVE_DTE,
    )


def log_hourly_summary(state, data_layer):
    try:
        now = datetime.now(timezone.utc)

        last = state.get("last_summary_ts", "")
        if last:
            try:
                dt_last = datetime.fromisoformat(last.replace("Z", "+00:00"))
                if (now - dt_last).total_seconds() < 3600:
                    return
            except Exception:
                pass

        equity = float(data_layer.trading.get_account().equity)
        spy = data_layer.get_spy_price()
        vixy = data_layer.get_vixy_close()
        open_spreads = len(state.get("open_spreads", {}))

        sys_log = logging.getLogger()
        sys_log.info(
            "[HOURLY %s] Equity=$%.2f | SPY=%.2f | VIXY=%.2f | Open=%d",
            now.strftime("%Y-%m-%d %H:%M"),
            equity, spy or 0.0, vixy or 0.0, open_spreads,
        )

        state.set("last_summary_ts", utc_now_iso())

    except Exception as e:
        logging.getLogger().warning("Hourly summary failed: %s", e)
