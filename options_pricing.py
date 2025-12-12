"""
options_pricing.py

Unified options pricing and volatility computation engine.

This module centralizes:
    - Black–Scholes pricing
    - Implied volatility solving
    - Greeks (delta, gamma, theta, vega)
    - Volatility surface sampling (ATM, 25Δ, 10Δ)
    - Skew calculations
    - Term structure inputs
    - Spread theoretical pricing
    - ML-ready volatility feature generation

All other components (entry/exit engines, data collectors,
ML pipelines, replay engine) should rely on this module for
any option math, IV solving, or volatility analytics.

This makes the system:
    - More maintainable
    - More accurate
    - Easier to test
    - Fully independent of IV from external APIs
    - Ready for future multi-leg strategies
    - ML-friendly with consistent volatility features

Use this module as the single source of truth for all option
math and volatility analytics throughout the bot.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable, List, Optional, Tuple, Dict
import math
import logging

log = logging.getLogger(__name__)


# ======================================================
# Basic math helpers
# ======================================================

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ======================================================
# Core data structure
# ======================================================

@dataclass
class OptionContract:
    symbol: str
    strike: float
    expiration: date
    option_type: str  # 'C' or 'P'
    bid: float
    ask: float
    last: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 or self.ask <= 0:
            return None
        return 0.5 * (self.bid + self.ask)


# ======================================================
# Black–Scholes pricing & greeks
# ======================================================

def _bs_d1(s, k, t, r, q, sigma):
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("Invalid inputs for BS d1")
    return (math.log(s/k) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))


def _bs_d2(d1, sigma, t):
    return d1 - sigma*math.sqrt(t)


def bs_price(s, k, t, r, q, sigma, is_call):
    d1 = _bs_d1(s, k, t, r, q, sigma)
    d2 = _bs_d2(d1, sigma, t)
    if is_call:
        return math.exp(-q*t)*s*_norm_cdf(d1) - math.exp(-r*t)*k*_norm_cdf(d2)
    return math.exp(-r*t)*k*_norm_cdf(-d2) - math.exp(-q*t)*s*_norm_cdf(-d1)


def bs_delta(s, k, t, r, q, sigma, is_call):
    d1 = _bs_d1(s, k, t, r, q, sigma)
    if is_call:
        return math.exp(-q*t) * _norm_cdf(d1)
    return -math.exp(-q*t) * _norm_cdf(-d1)


def bs_gamma(s, k, t, r, q, sigma):
    d1 = _bs_d1(s, k, t, r, q, sigma)
    return math.exp(-q*t) * _norm_pdf(d1) / (s * sigma * math.sqrt(t))


def bs_vega(s, k, t, r, q, sigma):
    d1 = _bs_d1(s, k, t, r, q, sigma)
    return s * math.exp(-q*t) * _norm_pdf(d1) * math.sqrt(t)


def bs_theta(s, k, t, r, q, sigma, is_call):
    d1 = _bs_d1(s, k, t, r, q, sigma)
    d2 = _bs_d2(d1, sigma, t)
    term1 = -(
        s * math.exp(-q*t) * _norm_pdf(d1) * sigma / (2*math.sqrt(t))
    )
    if is_call:
        term2 = q*s*math.exp(-q*t)*_norm_cdf(d1)
        term3 = -r*k*math.exp(-r*t)*_norm_cdf(d2)
        return term1 + term2 + term3
    term2 = -q*s*math.exp(-q*t)*_norm_cdf(-d1)
    term3 = r*k*math.exp(-r*t)*_norm_cdf(-d2)
    return term1 + term2 + term3


# ======================================================
# Implied Volatility Solver
# ======================================================

class IVComputationError(Exception):
    pass


def implied_vol(
    price, s, k, t, r, q, is_call,
    initial_guess=0.20, tol=1e-6, max_iter=50,
    min_vol=1e-4, max_vol=5.0
):
    if price <= 0 or s <= 0 or k <= 0 or t <= 0:
        return None

    intrinsic = max(0, s-k) if is_call else max(0, k-s)
    if price < intrinsic - 1e-6:
        return None

    low, high = min_vol, max_vol
    sigma = max(min(initial_guess, high), low)

    for _ in range(max_iter):
        try:
            theo = bs_price(s, k, t, r, q, sigma, is_call)
            diff = theo - price
        except ValueError:
            return None

        if abs(diff) < tol:
            return sigma

        try:
            v = bs_vega(s, k, t, r, q, sigma)
        except Exception:
            v = 0.0

        if v > 1e-8:
            sigma_new = sigma - diff / v
        else:
            sigma_new = None

        if sigma_new is None or sigma_new <= low or sigma_new >= high:
            if diff > 0:
                high = sigma
            else:
                low = sigma
            sigma = 0.5 * (low + high)
        else:
            sigma = sigma_new

    return None


# ======================================================
# DTE / time-to-expiration helpers
# ======================================================

def dte(expiration: date, as_of: Optional[date] = None) -> int:
    if as_of is None:
        as_of = date.today()
    return max((expiration - as_of).days, 0)


def tte_years(expiration: date, as_of_dt: Optional[datetime] = None) -> float:
    if as_of_dt is None:
        as_of_dt = datetime.now(timezone.utc)

    # Normalize as_of_dt to timezone-aware UTC
    if as_of_dt.tzinfo is None:
        as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
    else:
        as_of_dt = as_of_dt.astimezone(timezone.utc)

    # Make expiration datetime timezone-aware UTC
    exp_dt = datetime(expiration.year, expiration.month, expiration.day, tzinfo=timezone.utc)

    dt_days = max((exp_dt - as_of_dt).total_seconds() / 86400.0, 0.0)
    return dt_days / 365.0


# ======================================================
# Surface sampling & feature extraction
# ======================================================

def find_atm_contract(options, spot):
    best = None
    best_dist = float("inf")
    for opt in options:
        dist = abs(opt.strike - spot)
        if dist < best_dist:
            best = opt
            best_dist = dist
    return best


def sample_strikes_around_atm(options, spot, max_count=15):
    opts = list(options)
    opts.sort(key=lambda o: (abs(o.strike - spot), o.strike))
    return opts[:max_count]


def compute_contract_iv_and_delta(opt, spot, t, r, q):
    mid = opt.mid
    if mid is None:
        return None, None

    is_call = opt.option_type.upper() == "C"
    sigma = implied_vol(mid, spot, opt.strike, t, r, q, is_call)
    if sigma is None:
        return None, None

    try:
        dlt = bs_delta(spot, opt.strike, t, r, q, sigma, is_call)
    except Exception:
        dlt = None

    return sigma, dlt


def compute_surface_features_for_expiration(
    options,
    spot,
    expiration,
    as_of_dt=None,
    r=0.04,
    q=0.015,
    max_strikes=15,
):
    if as_of_dt is None:
        as_of_dt = datetime.now(timezone.utc)

    t = tte_years(expiration, as_of_dt)
    if t <= 0:
        return {  
            "atm_iv": None,
            "iv_25d_call": None,
            "iv_25d_put": None,
            "iv_10d_call": None,
            "iv_10d_put": None,
            "skew_25d": None,
            "skew_10d": None,
            "call_put_skew": None,
        }

    sampled = sample_strikes_around_atm(options, spot, max_count=max_strikes)

    ivs = {}
    deltas = {}
    calls = []
    puts = []

    for opt in sampled:
        sigma, dlt = compute_contract_iv_and_delta(opt, spot, t, r, q)
        if sigma is None or dlt is None:
            continue
        ivs[opt.symbol] = sigma
        deltas[opt.symbol] = dlt
        if opt.option_type.upper() == "C":
            calls.append((dlt, opt))
        else:
            puts.append((dlt, opt))

    if not ivs:
        return {
            "atm_iv": None,
            "iv_25d_call": None,
            "iv_25d_put": None,
            "iv_10d_call": None,
            "iv_10d_put": None,
            "skew_25d": None,
            "skew_10d": None,
            "call_put_skew": None,
        }

    def _closest(contracts, target_delta):
        best_opt = None
        best_dist = float("inf")
        for dlt, opt in contracts:
            dist = abs(dlt - target_delta)
            if dist < best_dist:
                best_dist = dist
                best_opt = opt
        return best_opt

    atm_call = find_atm_contract([opt for _, opt in calls], spot) if calls else None
    atm_put  = find_atm_contract([opt for _, opt in puts], spot) if puts else None

    atm_call_iv = ivs.get(atm_call.symbol) if atm_call else None
    atm_put_iv  = ivs.get(atm_put.symbol) if atm_put else None
    atm_iv = atm_call_iv if atm_call_iv is not None else atm_put_iv

    call_25 = _closest(calls, 0.25) if calls else None
    call_10 = _closest(calls, 0.10) if calls else None
    put_25  = _closest(puts, -0.25) if puts else None
    put_10  = _closest(puts, -0.10) if puts else None

    iv_25d_call = ivs.get(call_25.symbol) if call_25 else None
    iv_10d_call = ivs.get(call_10.symbol) if call_10 else None
    iv_25d_put  = ivs.get(put_25.symbol) if put_25 else None
    iv_10d_put  = ivs.get(put_10.symbol) if put_10 else None

    skew_25d = (
        iv_25d_put - iv_25d_call
        if iv_25d_put is not None and iv_25d_call is not None
        else None
    )
    skew_10d = (
        iv_10d_put - iv_10d_call
        if iv_10d_put is not None and iv_10d_call is not None
        else None
    )
    call_put_skew = (
        atm_call_iv - atm_put_iv
        if atm_call_iv is not None and atm_put_iv is not None
        else None
    )

    return {
        "atm_iv": atm_iv,
        "iv_25d_call": iv_25d_call,
        "iv_25d_put": iv_25d_put,
        "iv_10d_call": iv_10d_call,
        "iv_10d_put": iv_10d_put,
        "skew_25d": skew_25d,
        "skew_10d": skew_10d,
        "call_put_skew": call_put_skew,
    }


# ======================================================
# Spread theoretical pricing
# ======================================================

def compute_theoretical_spread_price(
    short_leg: OptionContract,
    long_leg: OptionContract,
    spot: float,
    as_of_dt: Optional[datetime] = None,
    r: float = 0.04,
    q: float = 0.015,
    short_iv: Optional[float] = None,
    long_iv: Optional[float] = None,
) -> Optional[float]:

    if as_of_dt is None:
        as_of_dt = datetime.now(timezone.utc)

    if short_leg.expiration != long_leg.expiration:
        log.warning("Mismatched expirations in spread calculation")
        return None

    t = tte_years(short_leg.expiration, as_of_dt)
    if t <= 0:
        return None

    short_mid = short_leg.mid
    long_mid = long_leg.mid
    if short_mid is None or long_mid is None:
        return None

    right_short = short_leg.option_type.upper()
    right_long = long_leg.option_type.upper()
    if right_short != right_long:
        log.warning("Spread mismatch: call vs put mix")
        return None

    is_call = right_short == "C"

    if short_iv is None:
        short_iv = implied_vol(short_mid, spot, short_leg.strike, t, r, q, is_call)
    if long_iv is None:
        long_iv = implied_vol(long_mid, spot, long_leg.strike, t, r, q, is_call)

    if short_iv is None or long_iv is None:
        return None

    try:
        sp = bs_price(spot, short_leg.strike, t, r, q, short_iv, is_call)
        lp = bs_price(spot, long_leg.strike, t, r, q, long_iv, is_call)
    except Exception:
        return None

    return sp - lp
