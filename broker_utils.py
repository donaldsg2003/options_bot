# helpers/broker_utils.py

import logging
from datetime import date, datetime

from option_bot_spreads.config import DTEConfig


def strike_from_sym(sym: str) -> float:
    """
    Extract strike price from OCC symbol.
    Symbols end with a strike * 1000, e.g. 00500000 (for 500.000)
    """
    try:
        return float(sym[-8:]) / 1000.0
    except Exception:
        return 0.0


# ======================================================
# Direct Live Position Count (from Alpaca)
# ======================================================
def get_live_position_count_direct(trading_client) -> int:
    """
    Count number of paired spreads directly from Alpaca positions.
    THIS IS THE SLOW VERSION (calls API), but most accurate.
    """
    try:
        positions = trading_client.get_all_positions()
        exp_groups = {}

        for p in positions:
            if len(p.symbol) < 9:
                continue
            exp = p.symbol[3:9]    # YYMMDD
            exp_groups.setdefault(exp, []).append(p)

        total_spreads = 0

        for exp_key, legs in exp_groups.items():
            shorts = [p for p in legs if float(p.qty) < 0]
            longs  = [p for p in legs if float(p.qty) > 0]

            short_count = sum(abs(int(float(p.qty))) for p in shorts)
            long_count  = sum(abs(int(float(p.qty))) for p in longs)

            paired = min(short_count, long_count)
            total_spreads += paired

            if short_count != long_count:
                logging.warning(
                    "[POSITION CHECK] Unbalanced in %s: %d shorts vs %d longs",
                    exp_key, short_count, long_count
                )

        return total_spreads

    except Exception as e:
        logging.error("[POSITION CHECK] Alpaca direct count failed: %s", e)
        return 999


# ======================================================
# Cached Live Position Count
# ======================================================
def get_live_position_count_from_cache(positions) -> int:
    """
    Count spreads based on cached Alpaca positions.
    MUCH faster than direct call.
    """
    try:
        exp_groups = {}
        for p in positions:
            if len(p.symbol) < 9:
                continue
            exp = p.symbol[3:9]
            exp_groups.setdefault(exp, []).append(p)

        total_spreads = 0

        for exp_key, legs in exp_groups.items():
            shorts = [p for p in legs if float(p.qty) < 0]
            longs  = [p for p in legs if float(p.qty) > 0]

            short_count = sum(abs(int(float(p.qty))) for p in shorts)
            long_count  = sum(abs(int(float(p.qty))) for p in longs)

            paired = min(short_count, long_count)
            total_spreads += paired

            if short_count != long_count:
                logging.warning(
                    "[POSITION CHECK] Unbalanced in %s: %d shorts vs %d longs",
                    exp_key, short_count, long_count
                )

        return total_spreads

    except Exception as e:
        logging.error("[POSITION CHECK] Cached count failed: %s", e)
        return 999


# ======================================================
# Cached Regime Count
# ======================================================
def get_live_regime_counts_from_cache(positions, state) -> dict:
    """
    Determine regime counts by matching cached Alpaca positions
    to persisted open_spreads in state.
    """
    exp_groups = {}

    for p in positions:
        if len(p.symbol) < 9:
            continue
        exp = p.symbol[3:9]
        exp_groups.setdefault(exp, []).append(p)

    open_spreads_state = state.get("open_spreads", {})

    regime_counts = {
        "HIGH_VOL": 0,
        "MID_VOL":  0,
        "LOW_VOL":  0,
        "UNKNOWN":  0,
    }

    for exp_key, legs in exp_groups.items():
        if len(legs) < 2:
            continue

        shorts = sorted(
            [l for l in legs if float(l.qty) < 0],
            key=lambda l: strike_from_sym(l.symbol)
        )
        longs = sorted(
            [l for l in legs if float(l.qty) > 0],
            key=lambda l: strike_from_sym(l.symbol)
        )

        expanded_shorts, expanded_longs = [], []

        for leg in shorts:
            for _ in range(int(abs(float(leg.qty)))):
                expanded_shorts.append(leg)

        for leg in longs:
            for _ in range(int(abs(float(leg.qty)))):
                expanded_longs.append(leg)

        for short_leg, long_leg in zip(expanded_shorts, expanded_longs):
            spread_key = f"{short_leg.symbol}|{long_leg.symbol}|{exp_key}"

            stored = open_spreads_state.get(spread_key)

            if stored:
                regime = stored.get("entry_regime", "UNKNOWN")
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            else:
                # Must infer regime from DTE
                try:
                    exp_date = datetime.strptime(exp_key, "%y%m%d").date()
                    dte = (exp_date - date.today()).days
                    regime = DTEConfig.classify_dte_to_regime(dte)
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                except Exception:
                    regime_counts["UNKNOWN"] += 1

    return regime_counts


# ======================================================
# Direct Count Wrapper
# ======================================================
def get_live_position_count(order_mgr) -> int:
    """Wrapper to count via the order manager."""
    return get_live_position_count_direct(order_mgr.data_layer.trading)
