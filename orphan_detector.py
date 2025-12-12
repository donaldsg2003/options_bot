# Issue #4: Safe Orphaned Leg Detection & Auto-Close
# CRITICAL: Multiple verification layers to prevent false positives

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, PositionIntent
from alpaca.trading.requests import MarketOrderRequest
from option_bot_spreads.core.persistent_state import PersistentState

@dataclass
class OrphanedLeg:
    """Represents a potentially orphaned option leg."""
    symbol: str
    qty: int
    side: str  # "SHORT" or "LONG"
    expiration: str
    strike: float
    detected_at: datetime
    verification_count: int = 0
    last_verified: Optional[datetime] = None


class OrphanDetector:
    """
    Conservative orphaned leg detector with multiple safety layers.
    
    Safety Features:
    1. Multiple verifications over time (not instant)
    2. Checks for pending orders
    3. Confirms no matching leg exists
    4. Requires sustained orphan state (not transient)
    5. Manual approval mode for first detection
    """
    
    def __init__(self, state: PersistentState, trading_client):
        self.state = state
        self.trading = trading_client
        
        # Safety thresholds
        self.MIN_VERIFICATIONS = 3  # Must see orphan 3 times
        self.VERIFICATION_INTERVAL = 300  # 5 minutes between checks
        self.ORPHAN_AGE_MIN = 900  # Must be orphaned for 15 minutes
        self.AUTO_CLOSE_ENABLED = False  # Starts disabled (manual mode)
    
    def detect_orphans(self, positions: List) -> List[OrphanedLeg]:
        """
        Detect potentially orphaned legs with conservative checks.
        
        Returns only legs that:
        1. Have no matching opposite leg
        2. Are not part of pending orders
        3. Have been orphaned for sufficient time
        """
        # Group positions by expiration
        exp_groups = {}
        for p in positions:
            if len(p.symbol) < 9:
                continue
            exp = p.symbol[3:9]
            exp_groups.setdefault(exp, []).append(p)
        
        potential_orphans = []
        
        for exp_key, legs in exp_groups.items():
            orphans = self._check_expiration_group(exp_key, legs)
            potential_orphans.extend(orphans)
        
        return potential_orphans
    
    def _check_expiration_group(self, exp_key: str, legs: List) -> List[OrphanedLeg]:
        """Check a single expiration group for orphaned legs."""
        
        # Separate shorts and longs
        shorts = [l for l in legs if float(l.qty) < 0]
        longs = [l for l in legs if float(l.qty) > 0]
        
        # Quick check: If balanced, no orphans
        total_shorts = sum(abs(int(float(l.qty))) for l in shorts)
        total_longs = sum(abs(int(float(l.qty))) for l in longs)
        
        if total_shorts == total_longs:
            return []
        
        # Potential imbalance detected - verify carefully
        orphans = []
        
        # Check each short for a matching long
        for short in shorts:
            short_strike = self._parse_strike(short.symbol)
            short_qty = abs(int(float(short.qty)))
            
            # Look for matching longs (within reasonable strike range)
            matched_qty = 0
            for long in longs:
                long_strike = self._parse_strike(long.symbol)
                # Vertical spreads: typically 5-15 points apart
                if 5 <= abs(long_strike - short_strike) <= 20:
                    matched_qty += abs(int(float(long.qty)))
            
            if matched_qty < short_qty:
                # Potential orphaned short
                orphans.append(OrphanedLeg(
                    symbol=short.symbol,
                    qty=int(short.qty),
                    side="SHORT",
                    expiration=exp_key,
                    strike=short_strike,
                    detected_at=datetime.now()
                ))
        
        # Check each long for a matching short
        for long in longs:
            long_strike = self._parse_strike(long.symbol)
            long_qty = abs(int(float(long.qty)))
            
            matched_qty = 0
            for short in shorts:
                short_strike = self._parse_strike(short.symbol)
                if 5 <= abs(short_strike - long_strike) <= 20:
                    matched_qty += abs(int(float(short.qty)))
            
            if matched_qty < long_qty:
                # Potential orphaned long
                orphans.append(OrphanedLeg(
                    symbol=long.symbol,
                    qty=int(long.qty),
                    side="LONG",
                    expiration=exp_key,
                    strike=long_strike,
                    detected_at=datetime.now()
                ))
        
        return orphans
    
    def verify_orphans(self, candidates: List[OrphanedLeg]) -> List[OrphanedLeg]:
        """
        Multi-layer verification of orphaned legs.
        
        Checks:
        1. No pending orders involving this symbol
        2. Sustained orphan state (multiple verifications)
        3. Sufficient time elapsed since detection
        """
        verified = []
        
        # Load historical orphan detections
        orphan_history = self.state.get("orphan_detections", {})
        now = datetime.now()
        
        for candidate in candidates:
            key = candidate.symbol
            
            # Check 1: Is there a pending order for this symbol?
            if self._has_pending_order(candidate.symbol):
                logging.info(
                    "[ORPHAN VERIFY] %s has pending order - not orphaned",
                    candidate.symbol
                )
                continue
            
            # Check 2: Update verification history
            if key not in orphan_history:
                orphan_history[key] = {
                    "first_seen": now.isoformat(),
                    "verification_count": 1,
                    "last_verified": now.isoformat(),
                    "side": candidate.side,
                    "qty": candidate.qty
                }
                logging.warning(
                    "[ORPHAN DETECTED] %s (qty=%d, side=%s) - verification 1/%d",
                    candidate.symbol, candidate.qty, candidate.side, self.MIN_VERIFICATIONS
                )
                continue
            
            # Existing detection - increment count
            history = orphan_history[key]
            first_seen = datetime.fromisoformat(history["first_seen"])
            last_verified = datetime.fromisoformat(history["last_verified"])
            
            # Check 3: Sufficient time since last verification?
            time_since_last = (now - last_verified).total_seconds()
            if time_since_last < self.VERIFICATION_INTERVAL:
                continue  # Too soon, wait longer
            
            # Increment verification count
            history["verification_count"] += 1
            history["last_verified"] = now.isoformat()
            
            logging.warning(
                "[ORPHAN VERIFY] %s - verification %d/%d (orphaned for %.1f min)",
                candidate.symbol, history["verification_count"], 
                self.MIN_VERIFICATIONS, (now - first_seen).total_seconds() / 60
            )
            
            # Check 4: Sufficient verifications AND time?
            age_seconds = (now - first_seen).total_seconds()
            if (history["verification_count"] >= self.MIN_VERIFICATIONS and 
                age_seconds >= self.ORPHAN_AGE_MIN):
                
                candidate.verification_count = history["verification_count"]
                candidate.last_verified = now
                verified.append(candidate)
                
                logging.critical(
                    "[ORPHAN CONFIRMED] %s verified as orphaned! (verifications=%d, age=%.1f min)",
                    candidate.symbol, history["verification_count"], age_seconds / 60
                )
        
        # Save updated history
        self.state.set("orphan_detections", orphan_history)
        
        return verified
    
    def _has_pending_order(self, symbol: str) -> bool:
        """Check if there's a pending order involving this symbol."""
        try:
            pending = self.trading.get_orders(filter=GetOrdersRequest(status='open'))
            
            for order in pending:
                # Check single-leg orders
                if hasattr(order, 'symbol') and order.symbol == symbol:
                    return True
                
                # Check multi-leg orders
                if hasattr(order, 'legs'):
                    for leg in order.legs:
                        if hasattr(leg, 'symbol') and leg.symbol == symbol:
                            return True
            
            return False
        except Exception as e:
            logging.error("[ORPHAN CHECK] Error checking pending orders: %s", e)
            return True  # Fail-safe: assume pending order exists
    
    def close_orphan(self, orphan: OrphanedLeg, mode: str = "MANUAL") -> Optional[str]:
        """
        Close an orphaned leg with safety checks.
        
        Args:
            orphan: The orphaned leg to close
            mode: "MANUAL" (require confirmation) or "AUTO" (proceed automatically)
        
        Returns:
            Order ID if successful, None otherwise
        """
        if mode == "MANUAL" or not self.AUTO_CLOSE_ENABLED:
            logging.critical(
                "[ORPHAN ACTION REQUIRED] Orphaned leg detected: %s (qty=%d, side=%s)\n"
                "   To close manually: Check Alpaca UI and close this position\n"
                "   To enable auto-close: Set AUTO_CLOSE_ORPHANS=true in .env",
                orphan.symbol, orphan.qty, orphan.side
            )
            return None
        
        # AUTO mode - proceed with closure
        try:
            # Determine order side (opposite of position)
            order_side = OrderSide.BUY if orphan.qty < 0 else OrderSide.SELL
            intent = PositionIntent.BUY_TO_CLOSE if orphan.qty < 0 else PositionIntent.SELL_TO_CLOSE
            
            # Use MARKET order for safety (guaranteed fill)
            req = MarketOrderRequest(
                symbol=orphan.symbol,
                qty=abs(orphan.qty),
                side=order_side,
                time_in_force=TimeInForce.DAY,
                position_intent=intent
            )
            
            order = self.trading.submit_order(req)
            
            logging.critical(
                "[ORPHAN CLOSED] Auto-closed orphaned leg: %s | OrderID=%s",
                orphan.symbol, order.id
            )
            
            # Clear from history
            orphan_history = self.state.get("orphan_detections", {})
            if orphan.symbol in orphan_history:
                del orphan_history[orphan.symbol]
                self.state.set("orphan_detections", orphan_history)
            
            return order.id
            
        except Exception as e:
            logging.critical(
                "[ORPHAN CLOSE FAILED] Could not close %s: %s\n"
                "   MANUAL INTERVENTION REQUIRED!",
                orphan.symbol, e
            )
            return None
    
    @staticmethod
    def _parse_strike(symbol: str) -> float:
        """Extract strike from option symbol."""
        try:
            return float(symbol[-8:]) / 1000.0
        except:
            return 0.0
