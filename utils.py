# utils.py
from datetime import datetime, timezone
import sys
import io

def utc_now_iso():
    """
    Return a clean ISO-8601 timestamp string in UTC (e.g., '2025-11-11T21:41:33Z').
    This is fully timezone-aware and backward-compatible with all Python 3 versions.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def force_utf8_output():
    """
    Ensures sys.stdout and sys.stderr use UTF-8 encoding.
    Avoids Windows CP1252 errors and ensures unicode compatibility.
    """

    encoding = sys.stdout.encoding

    # Convert to safe string before .lower()
    enc = str(encoding or "").lower()

    # Only wrap if encoding is not already UTF-8
    if enc not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

