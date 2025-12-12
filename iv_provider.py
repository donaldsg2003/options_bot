# option_bot_spreads/core/iv_provider.py

from __future__ import annotations
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


class IVSnapshot:
    """
    Canonical IV snapshot returned by any provider.
    """
    def __init__(
        self,
        atm_iv: Optional[float] = None,
        skew_25d: Optional[float] = None,
        skew_10d: Optional[float] = None,
        call_put_skew: Optional[float] = None,
        source: str = "missing",
        is_stale: bool = True,
    ):
        self.atm_iv = atm_iv
        self.skew_25d = skew_25d
        self.skew_10d = skew_10d
        self.call_put_skew = call_put_skew
        self.source = source
        self.is_stale = is_stale

    def as_db_dict(self) -> Dict:
        return {
            "atm_iv": self.atm_iv,
            "iv_skew_25d": self.skew_25d,
            "iv_skew_10d": self.skew_10d,
            "iv_call_put_skew": self.call_put_skew,
            "iv_source": self.source,
            "is_iv_stale": int(self.is_stale),
        }


class IVProvider:
    """
    Base interface for IV providers.
    """

    def get_iv_snapshot(self, timestamp: str) -> IVSnapshot:
        """
        Return IVSnapshot.
        Must NEVER throw.
        """
        try:
            return self._get_iv_snapshot_impl(timestamp)
        except Exception as e:
            log.warning("[IV_PROVIDER] failed: %s", e)
            return IVSnapshot()

    def _get_iv_snapshot_impl(self, timestamp: str) -> IVSnapshot:
        raise NotImplementedError


class MissingIVProvider(IVProvider):
    """
    Default provider when no IV source is available.
    """

    def _get_iv_snapshot_impl(self, timestamp: str) -> IVSnapshot:
        return IVSnapshot()
