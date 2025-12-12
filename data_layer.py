import logging
import time
from datetime import datetime, date, timedelta, time as dt_time
from typing import Dict, Tuple, Iterable, Set, List
from functools import wraps
import pytz
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data import StockHistoricalDataClient, StockLatestQuoteRequest
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, OptionChainRequest, OptionSnapshotRequest, BaseOptionLatestDataRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from option_bot_spreads.config import SETTINGS

data_log = logging.getLogger("data")


def retry_with_backoff(max_retries=3, base_delay=0.1):
    """
    Decorator for exponential backoff retry logic.
    Only sleeps on errors, not on success.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    data_log.warning(
                        "%s failed (attempt %d/%d), retrying in %.2fs: %s",
                        func.__name__, attempt + 1, max_retries, delay, e
                    )
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class DataLayer:
    def __init__(self):
        self.trading = TradingClient(SETTINGS.API_KEY, SETTINGS.SECRET_KEY, paper=SETTINGS.PAPER)
        self.stock = StockHistoricalDataClient(SETTINGS.API_KEY, SETTINGS.SECRET_KEY)
        self.opt = OptionHistoricalDataClient(SETTINGS.API_KEY, SETTINGS.SECRET_KEY)
        self.debug = SETTINGS.DEBUG_MODE
        
        # OPTIMIZATION: Price cache with timestamps
        self._price_cache = {}  # {symbol: (price, timestamp)}
        self._price_cache_ttl = 30  # 30 seconds TTL

    def _debug(self, msg, *args):
        """Debug log only when DEBUG_MODE=True."""
        if self.debug:
            data_log.debug(msg, *args)

    @staticmethod
    def parse_strike(sym: str) -> float:
        """
        Extract strike from option symbol.
        Example: 'SPY251121P00640000' -> 640.0
        """
        try:
            return float(sym[-8:]) / 1000.0
        except Exception:
            return 0.0

    @staticmethod
    def parse_strikes_bulk(symbols: List[str]) -> Dict[str, float]:
        """
        Parse strikes for multiple symbols at once (10x faster than loop).
        
        Args:
            symbols: List of option symbols
        
        Returns:
            Dict mapping symbol -> strike price
        """
        return {
            s: float(s[-8:]) / 1000.0 
            for s in symbols 
            if len(s) >= 8 and s[-8:].isdigit()
        }

    @staticmethod
    def build_symbol(root: str, yyyymmdd: str, right: str, strike: float) -> str:
        """
        Build OCC-format option symbol.
        Example: ('SPY', '251121', 'P', 640.0) -> 'SPY251121P00640000'
        """
        return f"{root}{yyyymmdd}{right.upper()}{int(round(strike * 1000)):08d}"

    def is_market_open(self):
        """
        Holiday-aware market hours using Alpaca calendar API.
        Handles weekends, full holidays, and half-days.
        """
        # ⭐ TEST MODE: Override for after-hours testing
        if os.getenv("FORCE_MARKET_HOURS_OVERRIDE", "false").lower() == "true":
            logging.warning("Market hours check OVERRIDDEN for testing!")
            return True  # Pretend market is always open

        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)
        today = now.date()

        try:
            req = GetCalendarRequest(start=today, end=today)
            cals = self.trading.get_calendar(req)
            if not cals:
                return False  # Holiday or weekend

            cal = cals[0]
            open_t = datetime.combine(today, cal.open, tzinfo=eastern)
            close_t = datetime.combine(today, cal.close, tzinfo=eastern)

            return open_t <= now <= close_t

        except Exception:
            # Fallback to normal market hours
            return dt_time(9, 30) <= now.time() <= dt_time(16, 0)

    def next_expiration(self, min_dte: int, max_dte: int) -> Tuple[str, int]:
        """
        Find next Friday expiration within DTE range.
        
        Returns:
            (expiration_date_str, dte) or ("", 0) if none found
        """
        today = date.today()
        cals = self.trading.get_calendar(
            GetCalendarRequest(start=today, end=today + timedelta(days=60))
        )
        
        for c in cals:
            d = c.date
            if d.weekday() != 4:  # Friday
                continue
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                return d.strftime("%Y-%m-%d"), dte
        
        return "", 0

    def get_market_snapshot(self) -> tuple[float | None, float | None]:
        """
        Fetch SPY and VIXY prices together (single batch optimization).
        ⚡ OPTIMIZED: One API call instead of two.
        
        Returns:
            (spy_price, vixy_price)
        """
        try:
            quotes = self.stock.get_stock_latest_quote(
                StockLatestQuoteRequest(
                    symbol_or_symbols=["SPY", "VIXY"],
                    feed=DataFeed.IEX
                )
            )
            
            spy_q = quotes.get("SPY")
            vixy_q = quotes.get("VIXY")
            
            spy_price = None
            vixy_price = None
            
            if spy_q:
                for field in ("ask_price", "bid_price", "last_price"):
                    v = getattr(spy_q, field, None)
                    if v and v > 0:
                        spy_price = float(v)
                        break
            
            if vixy_q:
                bid = getattr(vixy_q, "bid_price", None)
                ask = getattr(vixy_q, "ask_price", None)
                if bid and ask and bid > 0 and ask > 0:
                    vixy_price = round((bid + ask) / 2, 3)
                else:
                    last = getattr(vixy_q, "last_price", None)
                    if last and last > 0:
                        vixy_price = round(float(last), 3)
            
            return spy_price, vixy_price
            
        except Exception as e:
            data_log.warning("Market snapshot failed: %s", e)
            # Fall back to individual fetches
            return self.get_spy_price(), self.get_vixy_close()

    @retry_with_backoff(max_retries=3, base_delay=0.2)
    def get_vixy_close(self, force_refresh: bool = False) -> float:
        """
        Return latest VIXY price with fallback chain.
        OPTIMIZED: Cached for 30 seconds to reduce API calls.
        """
        cache_key = "VIXY"
        now = time.time()
        
        # Check cache (unless force refresh)
        if not force_refresh and cache_key in self._price_cache:
            price, timestamp = self._price_cache[cache_key]
            if now - timestamp < self._price_cache_ttl:
                return price
        
        # Cache miss - fetch fresh price
        price = self._fetch_vixy_close()
        
        # Handle None return (shouldn't happen, but safety check)
        if price is None or price == 0:
            data_log.error("[VIXY] Fetch returned invalid value, using fallback")
            price = 16.0
        
        if price > 0:
            self._price_cache[cache_key] = (price, now)
            self._debug("[PRICE CACHE] Updated VIXY: %.2f", price)
        
        return price

    def _fetch_vixy_close(self) -> float:
        """Internal method to actually fetch VIXY price from API."""
        try:
            q = self.stock.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols="VIXY", feed=DataFeed.IEX)
            )["VIXY"]
            
            bid = getattr(q, "bid_price", None)
            ask = getattr(q, "ask_price", None)
            last = getattr(q, "last_price", None)
            
            # Accept bid OR ask (after hours one side often zero)
            if bid and ask and bid > 0 and ask > 0:
                return round((bid + ask) / 2, 3)
            elif bid and bid > 0:
                return round(float(bid), 3)
            elif ask and ask > 0:
                return round(float(ask), 3)
            elif last and last > 0:
                return round(float(last), 3)
                
        except Exception as e:
            data_log.warning("Live VIXY quote failed: %s", e)

        # Fallback to bars
        try:
            from datetime import datetime, timedelta
            import pytz
            
            end = datetime.now(pytz.utc)
            start = end - timedelta(days=5)
            
            bars = self.stock.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols="VIXY", 
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end
                )
            ).df
            
            if not bars.empty:
                return float(bars["close"].iloc[-1])
                
        except Exception as e:
            data_log.warning("VIXY bars fallback failed: %s", e)

        return 16.0  # Emergency fallback

    @retry_with_backoff(max_retries=3, base_delay=0.2)
    def get_spy_price(self, force_refresh: bool = False) -> float | None:
        """
        Get most reliable SPY price using Alpaca only.
        OPTIMIZED: Cached for 30 seconds to reduce API calls.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            Float price or None if unavailable
        """
        cache_key = SETTINGS.SYMBOL
        now = time.time()
        
        # Check cache (unless force refresh)
        if not force_refresh and cache_key in self._price_cache:
            price, timestamp = self._price_cache[cache_key]
            if now - timestamp < self._price_cache_ttl:
                # Cache hit
                return price
        
        # Cache miss - fetch fresh price
        price = self._fetch_spy_price()
        
        if price is not None:
            self._price_cache[cache_key] = (price, now)
            self._debug("[PRICE CACHE] Updated SPY: %.2f", price)
        
        return price
    
    def _fetch_spy_price(self) -> float | None:
        """Internal method to actually fetch SPY price from API."""
        try:
            q = self.stock.get_stock_latest_quote(
                StockLatestQuoteRequest(
                    symbol_or_symbols=SETTINGS.SYMBOL, 
                    feed=DataFeed.IEX
                )
            )[SETTINGS.SYMBOL]

            for field in ("ask_price", "bid_price", "last_price"):
                v = getattr(q, field, None)
                if v and v > 0:
                    return float(v)

        except Exception as e:
            data_log.warning("Live SPY quote failed: %s", e)

        # Fallback to previous close
        try:
            bars = self.stock.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=SETTINGS.SYMBOL,
                    timeframe=TimeFrame.Day,
                    limit=1
                )
            ).df
            if not bars.empty:
                return float(bars["close"].iloc[-1])

        except Exception as e:
            data_log.warning("SPY fallback failed: %s", e)

        data_log.warning("SPY price unavailable.")
        return None

    # Add to DataLayer class in data_layer.py

    def get_account_equity(self) -> float:
        """
        Get account equity with 30-second caching.
        OPTIMIZED: Equity doesn't change rapidly, safe to cache.
        """
        if not hasattr(self, '_equity_cache'):
            self._equity_cache = None
            self._equity_cache_time = None
            self._equity_cache_ttl = 30  # 30 seconds
        
        now = time.time()
        
        if (self._equity_cache is None or 
            self._equity_cache_time is None or 
            now - self._equity_cache_time > self._equity_cache_ttl):
            
            try:
                account = self.trading.get_account()
                self._equity_cache = float(account.equity)
                self._equity_cache_time = now
                
                if self.debug:
                    data_log.debug(
                        "[CACHE] Refreshed equity cache: $%.2f", 
                        self._equity_cache
                    )
            except Exception as e:
                data_log.error("[CACHE] Failed to fetch equity: %s", e)
                # Return cached value if available, otherwise fail
                if self._equity_cache is not None:
                    return self._equity_cache
                raise
        
        return self._equity_cache

    def clear_price_cache(self):
        """Clear all cached prices (useful for testing or manual refresh)."""
        self._price_cache.clear()
        data_log.info("[PRICE CACHE] Cleared all cached prices")
    
    def get_cache_stats(self):
        return {
            "equity_cache": self._equity_cache,
            "equity_cache_age": (
                time.time() - self._equity_cache_time
                if self._equity_cache_time else None
            ),
            "equity_cache_ttl": self._equity_cache_ttl,
        }


    def option_chain(self, exp: str, right: str) -> Dict[str, object]:
        """
        Fetch option chain for given expiration and type (put/call).
        
        Args:
            exp: Expiration date string (YYYY-MM-DD)
            right: "put" or "call"
        
        Returns:
            Dict of option symbols -> contract objects
        """
        req = OptionChainRequest(
            underlying_symbol=SETTINGS.SYMBOL,
            expiration_date=exp,
            contract_type=right.lower()
        )
        return self.opt.get_option_chain(req)

    def _snapshot_quote_batch(self, chunk: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Fallback: pull latest_quote from snapshots for a batch.
        Used when primary quote source fails.
        """
        out: Dict[str, Tuple[float, float]] = {}
        try:
            snaps = self.opt.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=chunk)
            )
            for sym, snap in snaps.items():
                lq = getattr(snap, "latest_quote", None)
                if lq:
                    bid = float(getattr(lq, "bid_price", 0) or 0)
                    ask = float(getattr(lq, "ask_price", 0) or 0)
                    out[sym] = (bid, ask)
        except Exception as e:
            data_log.warning("Snapshot fallback failed for batch: %s", e)
        return out

    @retry_with_backoff(max_retries=2, base_delay=0.15)
    def latest_quotes(
        self, 
        symbols: Iterable[str], 
        feed: str = "indicative"
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get latest bid/ask for option symbols (optimized, no unnecessary delays).
        
        Args:
            symbols: Option symbols to fetch
            feed: Quote feed ('indicative' or 'sip')
        
        Returns:
            Dict mapping symbol -> (bid, ask)
        """

        symbols = list(dict.fromkeys(symbols))  # De-dup
        quotes: Dict[str, Tuple[float, float]] = {}
        batch_size = 100

        for i in range(0, len(symbols), batch_size):
            chunk = symbols[i:i + batch_size]
            
            try:
                req = BaseOptionLatestDataRequest(symbol_or_symbols=chunk, feed=feed)
                resp = self.opt.get_option_latest_quote(req)
                
                for sym in chunk:
                    q = resp.get(sym)
                    if q:
                        bid = float(getattr(q, "bid_price", 0) or 0)
                        ask = float(getattr(q, "ask_price", 0) or 0)
                        quotes[sym] = (bid, ask)
                
                self._debug(
                    "Fetched batch %d/%d (%d symbols)",
                    i // batch_size + 1,
                    (len(symbols) + batch_size - 1) // batch_size,
                    len(chunk)
                )

            except Exception as e:
                data_log.warning("Quote batch %d failed: %s", i // batch_size + 1, e)

            # Fallback for missing symbols in this chunk
            missing = [s for s in chunk if s not in quotes or quotes[s] == (0.0, 0.0)]
            if missing:
                snap_q = self._snapshot_quote_batch(missing)
                for sym, pair in snap_q.items():
                    if sym not in quotes or quotes[sym] == (0.0, 0.0):
                        quotes[sym] = pair

        return quotes

    def ensure_quote_union(
        self, 
        base_symbols: Iterable[str], 
        long_syms: Iterable[str]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Helper to fetch quotes for short & long legs in one call.
        """
        union_syms: Set[str] = set(base_symbols) | set(long_syms)
        return self.latest_quotes(list(union_syms), feed="indicative")

    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def snapshot_deltas(self, symbols: Iterable[str]) -> Dict[str, float]:
        """
        Fetch delta greeks for option symbols (chunked for rate limits).
        
        Returns:
            Dict mapping symbol -> delta value
        """
        symbols = list(dict.fromkeys(symbols))
        out: Dict[str, float] = {}
        step = 100
        
        for i in range(0, len(symbols), step):
            chunk = symbols[i:i + step]
            try:
                snaps = self.opt.get_option_snapshot(
                    OptionSnapshotRequest(symbol_or_symbols=chunk)
                )
                for s, snap in snaps.items():
                    d = getattr(getattr(snap, "greeks", None), "delta", None)
                    if d is not None:
                        out[s] = float(d)
            except Exception as e:
                data_log.warning("Snapshot greek batch %d failed: %s", i // step + 1, e)
        
        return out

    def diagnostic_snapshot(self, exp: str, right: str = "put"):
        """
        Diagnostic: verify deltas and quotes are being pulled correctly.
        Run once at startup for sanity check.
        """
        data_log.info("[DIAG] Fetching diagnostic snapshot for %s %s", SETTINGS.SYMBOL, exp)

        chain = self.option_chain(exp, right)
        if not chain:
            data_log.warning("[DIAG] No chain returned")
            return

        symbols = list(chain.keys())[:50]
        data_log.info("[DIAG] Pulled %d symbols", len(symbols))

        deltas = self.snapshot_deltas(symbols)
        valid_deltas = {k: v for k, v in deltas.items() if v is not None}
        data_log.info("[DIAG] Got %d deltas | Sample: %s", len(valid_deltas), list(valid_deltas.items())[:5])

        quotes = self.latest_quotes(symbols)
        valid_quotes = {k: v for k, v in quotes.items() if v != (0.0, 0.0)}
        data_log.info("[DIAG] Got %d quotes | Sample: %s", len(valid_quotes), list(valid_quotes.items())[:5])

        if not valid_deltas:
            data_log.warning("[DIAG] No valid deltas returned")
        if not valid_quotes:
            data_log.warning("[DIAG] No valid quotes returned")
        if valid_deltas and valid_quotes:
            data_log.info("[DIAG] Deltas & quotes look valid")