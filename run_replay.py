"""
Synthetic Replay Runner (FINAL VERSION with anchored expirations)

This integrates:
- RegimeSwitchingSimulator
- ChainGenerator (patched to accept external expirations)
- ReplayEngine
- Anchored expiration dates so DTE decreases each day
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from option_bot_spreads.backtesting.simulator import RegimeSwitchingSimulator
from option_bot_spreads.backtesting.chain_generator import ChainGenerator
from option_bot_spreads.backtesting.replay_engine import ReplayEngine

# Simulation settings
N_DAYS = 60
START_PRICE = 480.0
RANDOM_SEED = 42

LOG_DIR = Path("logs/sim")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# Logging Setup
# ======================================================
def setup_logging():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"synthetic_replay_{ts}.log"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch.setLevel(logging.INFO)

    root.addHandler(fh)
    root.addHandler(ch)

    root.info("Logging initialized → %s", logfile)


# ======================================================
# Main
# ======================================================
def main():
    setup_logging()
    logging.info("=== Synthetic Replay Backtest START ===")

    # --------------------------------------------------
    # 1. Generate Synthetic Price Path
    # --------------------------------------------------
    sim = RegimeSwitchingSimulator(seed=RANDOM_SEED)
    logging.info("Simulating %d days ...", N_DAYS)

    path_df = sim.simulate_days(
        n_days=N_DAYS,
        start_price=START_PRICE,
        start_regime="MID_VOL",
    )

    path_df = path_df.sort_values("date").reset_index(drop=True)

    logging.info(
        "[SIM] Generated synthetic path: %d days | start=%.2f end=%.2f",
        len(path_df),
        path_df.iloc[0]["close"],
        path_df.iloc[-1]["close"],
    )

    # --------------------------------------------------
    # 2. ANCHORED Expiration Dates (DTE will age correctly!)
    # --------------------------------------------------
    base_date = path_df.iloc[0]["date"].date()
    dtes = [7, 14, 21, 30, 45]

    anchored_exps = {
        d: base_date + timedelta(days=d)
        for d in dtes
    }

    # Example: anchored_exps[14] = 2025-01-15 always

    # --------------------------------------------------
    # 3. Chain Generator with anchored expirations
    # --------------------------------------------------
    cg = ChainGenerator(expirations=anchored_exps)

    chains_by_day = {}
    for row in path_df.itertuples(index=False):
        today = row.date.date()

        # DTE = expiration - today (ages naturally)
        chains = cg.generate_chain_for_day(
            date=row.date,
            underlying_price=row.close,
            atm_iv=row.atm_iv,
            regime=row.regime,
        )

        chains_by_day[today] = chains

    logging.info("Built %d synthetic chain days", len(chains_by_day))

    # --------------------------------------------------
    # 4. Run Replay Engine
    # --------------------------------------------------
    replayer = ReplayEngine(
        path_df=path_df,
        chains_by_day=chains_by_day,
    )

    result = replayer.run()

    # --------------------------------------------------
    # 5. Save Results
    # --------------------------------------------------
    out_dir = Path("synthetic_results")
    out_dir.mkdir(exist_ok=True)

    # Equity curve
    eq_df = pd.DataFrame({
        "date": result.dates,
        "equity": result.equity_curve,
    })
    eq_file = out_dir / "equity_curve.csv"
    eq_df.to_csv(eq_file, index=False)
    logging.info("Saved equity curve → %s", eq_file)

    # Trades
    trades_list = []
    for t in result.trades:
        trades_list.append({
            "short": t.short_sym,
            "long": t.long_sym,
            "qty": t.qty,
            "entry": t.entry_time,
            "exit": t.close_time,
            "credit": t.entry_credit,
            "debit": t.close_debit,
            "pnl": t.realized_pl,
            "reason": t.exit_reason,
        })

    trades_df = pd.DataFrame(trades_list)
    t_file = out_dir / "trades.csv"
    trades_df.to_csv(t_file, index=False)
    logging.info("Saved trades → %s", t_file)

    logging.info(
        "=== Replay Complete | Final Equity %.2f ===",
        result.final_equity,
    )


if __name__ == "__main__":
    main()
