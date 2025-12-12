# engines/reconcile_engine.py (FIXED STRUCTURE)

import logging
from datetime import datetime, date
from option_bot_spreads.helpers.utils import utc_now_iso
from option_bot_spreads.helpers.spread_utils import strike_from_symbol, extract_exp_from_symbol
from option_bot_spreads.config import DTEConfig


def reconcile_persistent_spreads(state, order_mgr, orphan_detector=None):
    sys_log = logging.getLogger()
    trade_log = logging.getLogger("trades")

    sys_log.info("[STATE] Reconciling with live positions...")

    open_spreads = state.get("open_spreads", {})
    pos_groups = order_mgr.group_positions()
    live_spread_keys: dict[str, dict] = {}
    orphaned_legs = []

    # ======================================================
    # Helper to expand multi-qty legs
    # ======================================================
    def expand_legs(shorts, longs):
        expanded_shorts, expanded_longs = [], []
        for leg in shorts:
            expanded_shorts.extend([leg] * abs(int(leg.qty)))
        for leg in longs:
            expanded_longs.extend([leg] * abs(int(leg.qty)))
        return expanded_shorts, expanded_longs

    # ======================================================
    # Build live spreads
    # ======================================================
    for exp_key, legs in pos_groups.items():
        if len(legs) < 2:
            continue

        puts = [l for l in legs if "P" in l.symbol]
        calls = [l for l in legs if "C" in l.symbol]

        def build_side_spreads(side_legs):
            shorts = sorted([l for l in side_legs if l.qty < 0], key=lambda l: strike_from_symbol(l.symbol))
            longs = sorted([l for l in side_legs if l.qty > 0], key=lambda l: strike_from_symbol(l.symbol))

            expanded_shorts, expanded_longs = expand_legs(shorts, longs)

            # Handle orphans
            if len(expanded_shorts) != len(expanded_longs):
                diff = len(expanded_shorts) - len(expanded_longs)
                orphaned_legs.extend(
                    expanded_shorts[len(expanded_longs):] if diff > 0 else expanded_longs[len(expanded_shorts):]
                )

            # Pair legs
            for short_leg, long_leg in zip(expanded_shorts, expanded_longs):
                exp_date = extract_exp_from_symbol(short_leg.symbol)
                dte = (exp_date - date.today()).days if exp_date else None
                regime = DTEConfig.classify_dte_to_regime(dte) if dte is not None else "UNKNOWN"

                spread_key = f"{short_leg.symbol}|{long_leg.symbol}|{exp_key}"
                if spread_key not in live_spread_keys:
                    live_spread_keys[spread_key] = {
                        "opened_at": utc_now_iso(),
                        "qty": 1,
                        "credit": None,
                        "status": "open",
                        "entry_regime": regime,
                        "entry_dte": dte,
                        "expiration": exp_date.isoformat() if exp_date else None,
                    }
                    trade_log.info(
                        "[STATE] Recovered live spread: %s | regime=%s | dte=%s",
                        spread_key, regime, dte
                    )

        build_side_spreads(puts)
        build_side_spreads(calls)

    # ======================================================
    # Remove stale spreads (runs ONCE)
    # ======================================================
    removed = 0
    to_remove = [k for k in open_spreads if k not in live_spread_keys]

    for k in to_remove:
        removed += 1
        sys_log.info("[STATE] Removing stale spread: %s", k)
        open_spreads.pop(k, None)
        recent = state.get("recent_spreads", {})
        recent[k] = {"closed_at": utc_now_iso(), "reason": "reconciliation_cleanup"}
        state.set("recent_spreads", recent)

    # ======================================================
    # Add newly-discovered live spreads (ONCE)
    # ======================================================
    added = 0
    for key, meta in live_spread_keys.items():
        if key not in open_spreads:
            open_spreads[key] = meta
            added += 1

    # ======================================================
    # Diagnostics — RUNS ONCE
    # ======================================================
    state_keys = set(open_spreads.keys())
    live_keys = set(live_spread_keys.keys())

    if state_keys != live_keys:
        only_state = state_keys - live_keys
        only_live = live_keys - state_keys

        if only_state:
            sys_log.warning("[RECONCILE] %d spreads exist only in state", len(only_state))
        if only_live:
            sys_log.warning("[RECONCILE] %d spreads exist only live", len(only_live))
    else:
        sys_log.info("[RECONCILE] State/live spread keys in sync (%d spreads)", len(state_keys))

    # Per-expiry summary
    summary: dict[str, list[tuple]] = {}

    for skey, sval in open_spreads.items():
        short_sym, long_sym, exp_key = skey.split("|")
        qty = int(sval.get("qty", 1))
        summary.setdefault(exp_key, []).append(
            (strike_from_symbol(short_sym), strike_from_symbol(long_sym), qty)
        )

    for exp_key in sorted(summary.keys()):
        sys_log.info("[RECONCILE] %s: %d spreads", exp_key, len(summary[exp_key]))
        for short_k, long_k, qty in sorted(summary[exp_key]):
            sys_log.info("[RECONCILE]   %.1f/%.1f x%d", short_k, long_k, qty)

    # FINAL persistent write
    state.set("open_spreads", open_spreads)

    sys_log.info(
        "[RECONCILE] Summary: open=%d | added=%d | removed=%d | orphaned=%d",
        len(open_spreads), added, removed, len(orphaned_legs)
    )
