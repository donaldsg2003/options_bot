# engines/orphan_engine.py

import logging
from datetime import datetime, timezone

from option_bot_spreads.helpers.utils import utc_now_iso


def run_orphan_cycle(
    state,
    data_layer,
    orphan_detector,
    sys_log,
    trade_log,
    verification_interval=300,
):
    """
    Perform a full orphan-detection cycle:

    1. Detect orphan candidates from live positions
    2. Verify orphans using orphan detector’s multi-pass logic
    3. Auto-close or request manual close depending on settings

    The logic here mirrors what was previously embedded inside main.py,
    but is now centralized and safer.
    """

    try:
        # ----------------------------------------------
        # 1. Retrieve all live positions from Alpaca
        # ----------------------------------------------
        positions = data_layer.trading.get_all_positions()
        potential = orphan_detector.detect_orphans(positions)

        if not potential:
            return

        sys_log.warning(
            "[ORPHAN] Detected %d potentially orphaned legs", len(potential)
        )

        # ----------------------------------------------
        # 2. Multi-pass verification:
        #    Only confirm orphans that remain unmatched
        # ----------------------------------------------
        verified = orphan_detector.verify_orphans(potential)

        if not verified:
            sys_log.info(
                "[ORPHAN] No verified orphans after verification step."
            )
            return

        sys_log.critical(
            "[ORPHAN] %d orphaned legs VERIFIED for closure",
            len(verified),
        )

        # ----------------------------------------------
        # 3. Close each orphan according to config
        # ----------------------------------------------
        for orphan in verified:
            mode = (
                "AUTO"
                if orphan_detector.AUTO_CLOSE_ENABLED
                else "MANUAL"
            )

            order_id = orphan_detector.close_orphan(orphan, mode=mode)

            if order_id:
                trade_log.critical(
                    "[ORPHAN CLOSED] %s | OrderID=%s | Qty=%d",
                    orphan.symbol,
                    order_id,
                    orphan.qty,
                )
            else:
                trade_log.error(
                    "[ORPHAN ERROR] Failed to close orphan %s (qty=%d)",
                    orphan.symbol,
                    orphan.qty,
                )

        # ----------------------------------------------
        # 4. Update state with timestamp
        # ----------------------------------------------
        state.set("last_orphan_cycle", utc_now_iso())

    except Exception as e:
        sys_log.error("[ORPHAN] Orphan cycle failed: %s", e)
