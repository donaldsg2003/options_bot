from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Literal
import logging
import time
import pandas_ta as ta
import pytz
import sys

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from option_bot_spreads.core.data_layer import DataLayer
from option_bot_spreads.config import SETTINGS
from option_bot_spreads.helpers.state_utils import ML_COLLECTOR  # NEW ML logging

data_log = logging.getLogger("data")
sys_log = logging.getLogger()
Side = Literal["put", "call"]


@dataclass
class SpreadCandidate:
    short_sym: str
    long_sym: str
    short_strike: float
    long_strike: float
    delta: float
    net_credit: float
    dte: int
    side: Side
    expiration: str


class StrategyEngine:
    def __init__(self, data: DataLayer):
        self.d = data

        # Runtime adaptive parameters (updated daily in main)
        self.target_delta = SETTINGS.TARGET_DELTA
        self.spread_width = SETTINGS.SPREAD_WIDTH
        self.min_net_credit = SETTINGS.MIN_NET_CREDIT
        self.delta_window = SETTINGS.DELTA_WINDOW

        # Cache for market signal
        self._signal_cache = None
        self._signal_cache_time = None
        self._signal_cache_ttl = 300  # 5 min

    # =====================================================================
    # SIGNAL ENGINE (cached)
    # =====================================================================
    def market_signal(self) -> str:
        now = time.time()

        if (
            self._signal_cache is not None
            and self._signal_cache_time is not None
            and now - self._signal_cache_time < self._signal_cache_ttl
        ):
            return self._signal_cache

        signal = self._fetch_market_signal()
        self._signal_cache = signal
        self._signal_cache_time = now

        data_log.info(
            "[SIGNAL CACHE] Updated market signal: %s (cached %d sec)",
            signal,
            self._signal_cache_ttl,
        )
        return signal

    def _fetch_market_signal(self) -> str:
        start = datetime.now(pytz.utc) - timedelta(days=180)
        bars_req = StockBarsRequest(
            symbol_or_symbols=SETTINGS.SYMBOL,
            timeframe=TimeFrame.Day,
            start=start,
            feed=DataFeed.IEX,
        )

        bars = self.d.stock.get_stock_bars(bars_req).df
        if len(bars) < 50:
            return "NEUTRAL"

        df = bars.copy()
        df["ema20"] = ta.ema(df["close"], 20)
        df["rsi"] = ta.rsi(df["close"], 14)

        macd = ta.macd(df["close"])
        if macd is None:
            return "NEUTRAL"

        df["macd"] = macd["MACD_12_26_9"]
        df["macds"] = macd["MACDs_12_26_9"]

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        def is_bullish():
            return (
                latest.close > latest.ema20
                and latest.rsi > 45
                and latest.macd > latest.macds
            )

        def is_bearish():
            return latest.macd < latest.macds and prev.macd >= prev.macds

        if is_bullish():
            return "BULL"
        elif is_bearish():
            return "BEAR"
        return "NEUTRAL"

    print(">>> calling find_spreads()", file=sys.stdout)
    with open("replay_debug_trace.txt", "a") as f:
        f.write("calling find_spreads()\n")

    # =====================================================================
    # SPREAD DISCOVERY ENGINE (ML-enhanced)
    # =====================================================================
    def find_spreads(
        self,
        side: Side,
        min_dte: int | None = None,
        max_dte: int | None = None,
    ) -> List[SpreadCandidate]:

        min_dte = min_dte if min_dte is not None else SETTINGS.MIN_DTE
        max_dte = max_dte if max_dte is not None else SETTINGS.MAX_DTE

        exp, dte = self.d.next_expiration(min_dte, max_dte)
        if not exp:
            data_log.warning("No expiration found in DTE range %d-%d", min_dte, max_dte)
            return []

        right = "put" if side == "put" else "call"
        chain = self.d.option_chain(exp, right=right)
        if not chain:
            return []

        symbols = list(chain.keys())
        deltas = self.d.snapshot_deltas(symbols)
        strikes = self.d.parse_strikes_bulk(symbols)

        if side == "put":
            lo = self.target_delta - self.delta_window
            hi = self.target_delta + self.delta_window
        else:
            lo = abs(self.target_delta) - self.delta_window
            hi = abs(self.target_delta) + self.delta_window

        yyyymmdd = exp[2:].replace("-", "")
        right_letter = "P" if side == "put" else "C"

        short_to_long = {}
        long_syms = []

        # ===================================================
        # BUILD SHORT → LONG STRIKE PAIRS
        # ===================================================
        for s in symbols:
            short_strike = strikes.get(s, 0.0)
            if short_strike == 0.0:
                continue

            long_strike = (
                short_strike - self.spread_width
                if side == "put"
                else short_strike + self.spread_width
            )

            long_sym = self.d.build_symbol(
                SETTINGS.SYMBOL, yyyymmdd, right_letter, long_strike
            )

            spread_width = self.spread_width
            regime = getattr(self, "regime", "MID_VOL")

            if long_strike not in chain:
                available = sorted(strikes.values())

                if not available:
                    continue

                nearest = min(available, key=lambda x: abs(x - long_strike))

                if len(available) > 1:
                    chain_spacing = min(
                        abs(available[i + 1] - available[i])
                        for i in range(len(available) - 1)
                    )
                else:
                    chain_spacing = spread_width

                base_gap = spread_width * 0.50

                regime_mult = {
                    "LOW_VOL": 0.75,
                    "MID_VOL": 1.25,
                    "HIGH_VOL": 1.75,
                }.get(regime, 1.0)

                liquidity_padding = chain_spacing * 1.5

                max_allowed_drift = max(base_gap * regime_mult, liquidity_padding)

                if abs(nearest - long_strike) > max_allowed_drift:
                    continue

                long_strike = nearest
                long_sym = self.d.build_symbol(
                    SETTINGS.SYMBOL, yyyymmdd, right_letter, long_strike
                )
                if long_sym not in chain:
                    continue

            short_to_long[s] = (long_sym, short_strike, long_strike)
            long_syms.append(long_sym)

        if not short_to_long:
            return []

        # ===================================================
        # BULK-QUOTE CALL
        # ===================================================
        all_syms = list(short_to_long.keys()) + long_syms
        quotes = self.d.latest_quotes(all_syms, feed="indicative")

        if not self._quotes_are_fresh(quotes):
            data_log.warning("Stale quotes detected - market closed or data frozen.")
            return []

        candidates: List[SpreadCandidate] = []

        # ===================================================
        # BUILD CANDIDATES + ML LOGGING
        # ===================================================
        for s, (long_sym, short_strike, long_strike) in short_to_long.items():
            delta = deltas.get(s)
            if delta is None:
                continue

            comp_delta = delta if side == "put" else abs(delta)
            if not (lo <= comp_delta <= hi):
                continue

            sb, sa = quotes.get(s, (0.0, 0.0))
            lb, la = quotes.get(long_sym, (0.0, 0.0))

            if (sb == 0 and sa == 0) or (lb == 0 and la == 0):
                continue

            short_mid = (sb + sa) / 2 if sb and sa else max(sb, sa)
            long_mid = (lb + la) / 2 if lb and la else max(lb, la)

            net = round(max(sb, short_mid) - max(la, long_mid), 2)

            if net < self.min_net_credit:
                continue

            cand = SpreadCandidate(
                short_sym=s,
                long_sym=long_sym,
                short_strike=short_strike,
                long_strike=long_strike,
                delta=delta,
                net_credit=net,
                dte=dte,
                side=side,
                expiration=yyyymmdd,
            )
            candidates.append(cand)

            # ===================================================
            # ML LOGGING FOR EACH CANDIDATE
            # ===================================================
            try:
                ML_COLLECTOR.record_spread_candidate(
                    signal=side.upper(),
                    regime=None,
                    short_symbol=s,
                    long_symbol=long_sym,
                    credit=net,
                    delta=delta,
                    width=abs(long_strike - short_strike),
                    dte=dte,
                    chosen=0,
                    extra={
                        "expiration": yyyymmdd,
                        "target_delta": self.target_delta,
                        "delta_window": self.delta_window,
                    },
                )
            except Exception as e:
                data_log.debug("[ML] Failed to record strategy candidate: %s", e)

        # ===================================================
        # SORT + RETURN
        # ===================================================
        candidates.sort(key=lambda c: c.net_credit, reverse=True)

        data_log.info(
            "[STRATEGY] Found %d %s spreads | DTE=%d | TargetDelta=%.2f | Window=%.2f | MinCredit=%.2f",
            len(candidates),
            side.upper(),
            dte,
            self.target_delta,
            SETTINGS.DELTA_WINDOW,
            self.min_net_credit,
        )
        data_log.info("[DIAG] StrategyEngine produced %d candidates", len(candidates))

        print(f">>> find_spreads() returned {len(candidates)} candidates", file=sys.stdout)
        with open("replay_debug_trace.txt", "a") as f:
            f.write(f"find_spreads() returned {len(candidates)}\n")

        return candidates

    


    # =====================================================================
    # Quote freshness checker
    # =====================================================================
    def _quotes_are_fresh(self, quotes: dict) -> bool:
        if not quotes:
            return False

        valid = sum(1 for (bid, ask) in quotes.values() if bid > 0 and ask > 0)
        return valid / len(quotes) >= 0.5
