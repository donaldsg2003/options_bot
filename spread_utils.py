# helpers/spread_utils.py

from datetime import datetime, date


def parse_occ_symbol(symbol: str):
    """
    Parse an OCC option symbol into (underlying, expiration, right, strike)
    Example: SPY250117P00420000
    """
    try:
        underlying = symbol[:3]
        exp_str = symbol[3:9]   # YYMMDD
        right = symbol[9]       # P or C
        strike_raw = symbol[10:]  # strike * 1000

        strike = float(strike_raw) / 1000.0
        expiration = datetime.strptime(exp_str, "%y%m%d").date()

        return underlying, expiration, right, strike

    except Exception:
        return None, None, None, None


def extract_exp_from_symbol(symbol: str):
    """Return expiration (date) from an OCC option symbol."""
    try:
        exp_date = datetime.strptime(symbol[3:9], "%y%m%d").date()
        return exp_date
    except Exception:
        return None


def strike_from_symbol(symbol: str) -> float:
    """Extract strike price from OCC symbol (last 8 digits = strike*1000)."""
    try:
        return float(symbol[-8:]) / 1000.0
    except Exception:
        return 0.0


def calc_dte(expiration: date) -> int:
    """Return days to expiration."""
    if not isinstance(expiration, date):
        return 0
    return max((expiration - date.today()).days, 0)


def make_spread_tuple(short_sym: str, long_sym: str):
    """
    Convert two OCC symbols into:
    (expiration, short_strike, long_strike, right)
    """
    _, exp_s, right_s, strike_s = parse_occ_symbol(short_sym)
    _, exp_l, right_l, strike_l = parse_occ_symbol(long_sym)

    if exp_s != exp_l:
        # spread should always share expiration
        return None

    if right_s != right_l:
        # must be same right (P or C)
        return None

    return exp_s, strike_s, strike_l, right_s


def expiration_to_str(exp: date) -> str:
    """Convert expiration date to YYMMDD string."""
    try:
        return exp.strftime("%y%m%d")
    except Exception:
        return "000000"


def occ_symbol(underlying: str, exp: date, right: str, strike: float):
    """
    Format an OCC symbol given components.
    strike is formatted as strike*1000, 8 digits, zero-padded.
    """
    try:
        exp_str = exp.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"{underlying}{exp_str}{right}{strike_str}"
    except Exception:
        return None
