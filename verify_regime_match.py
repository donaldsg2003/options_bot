#!/usr/bin/env python3
"""
verify_regime_match.py
Verify that market data collector uses EXACT same regime logic as trading bot.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Import both regime calculators
try:
    from core.vixy_regime import get_vixy_percentile as bot_get_percentile
    from core.vixy_regime import get_adaptive_parameters
    BOT_AVAILABLE = True
except ImportError:
    BOT_AVAILABLE = False
    log.warning("Could not import bot's vixy_regime.py")

try:
    from data.market_data_collector import get_vixy_percentile as collector_get_percentile
    from data.market_data_collector import classify_regime
    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False
    log.warning("Could not import collector's regime functions")


def test_regime_matching():
    """
    Test that bot and collector produce SAME regime classification.
    """
    print("\n" + "="*80)
    print("REGIME MATCHING VERIFICATION")
    print("="*80)
    
    if not BOT_AVAILABLE:
        print("\n❌ Bot's vixy_regime.py not available")
        return False
    
    if not COLLECTOR_AVAILABLE:
        print("\n❌ Collector functions not available")
        return False
    
    print("\n[TEST 1] Bot Regime Calculation")
    print("-" * 80)
    
    try:
        # Get bot's regime
        regime, delta, width, min_credit, cooldown, bear_mode, min_dte, max_dte, delta_window = get_adaptive_parameters()
        
        # Get raw percentile from bot
        current, percentile, closes = bot_get_percentile()
        
        print(f"Bot says:")
        print(f"  VIXY Price:    ${current:.2f}")
        print(f"  Percentile:    {percentile*100:.2f}%")
        print(f"  Regime:        {regime}")
        print(f"  Target Delta:  {delta}")
        print(f"  Spread Width:  {width}")
        print(f"  DTE Range:     {min_dte}-{max_dte}")
        
        bot_regime = regime
        bot_percentile = percentile
        bot_price = current
        
    except Exception as e:
        print(f"\n❌ Bot calculation failed: {e}")
        return False
    
    print("\n[TEST 2] Collector Regime Calculation")
    print("-" * 80)
    
    try:
        # Get collector's regime
        vixy_current, vixy_pct, vixy_low, vixy_high = collector_get_percentile()
        collector_regime = classify_regime(vixy_pct)
        
        print(f"Collector says:")
        print(f"  VIXY Price:    ${vixy_current:.2f}")
        print(f"  Percentile:    {vixy_pct*100:.2f}%")
        print(f"  1Y Range:      ${vixy_low:.2f} - ${vixy_high:.2f}")
        print(f"  Regime:        {collector_regime}")
        
    except Exception as e:
        print(f"\n❌ Collector calculation failed: {e}")
        return False
    
    print("\n[TEST 3] Comparison")
    print("-" * 80)
    
    # Check if prices match (within $0.10)
    price_match = abs(bot_price - vixy_current) < 0.10
    print(f"Price match:      {price_match} (${bot_price:.2f} vs ${vixy_current:.2f})")
    
    # Check if percentiles match (within 5%)
    pct_match = abs(bot_percentile - vixy_pct) < 0.05
    print(f"Percentile match: {pct_match} ({bot_percentile*100:.2f}% vs {vixy_pct*100:.2f}%)")
    
    # Check if regimes match
    regime_match = bot_regime == collector_regime
    print(f"Regime match:     {regime_match} ({bot_regime} vs {collector_regime})")
    
    print("\n" + "="*80)
    
    if price_match and pct_match and regime_match:
        print("✅ PERFECT MATCH - Training data will be accurate!")
        print("="*80 + "\n")
        return True
    else:
        print("❌ MISMATCH DETECTED - Training data will be WRONG!")
        print("\nIssues:")
        if not price_match:
            print(f"  - Price differs by ${abs(bot_price - vixy_current):.2f}")
        if not pct_match:
            print(f"  - Percentile differs by {abs(bot_percentile - vixy_pct)*100:.2f}%")
        if not regime_match:
            print(f"  - Regime mismatch: {bot_regime} != {collector_regime}")
        print("\n⚠️  DO NOT USE THIS DATA FOR ML TRAINING!")
        print("="*80 + "\n")
        return False


def test_threshold_logic():
    """
    Verify threshold logic matches between systems.
    """
    print("\n" + "="*80)
    print("THRESHOLD LOGIC VERIFICATION")
    print("="*80)
    
    test_cases = [
        (0.05, "LOW_VOL"),   # 5th percentile
        (0.24, "LOW_VOL"),   # Just below 25%
        (0.25, "MID_VOL"),   # Exactly 25%
        (0.50, "MID_VOL"),   # Middle
        (0.75, "MID_VOL"),   # Exactly 75%
        (0.76, "HIGH_VOL"),  # Just above 75%
        (0.95, "HIGH_VOL"),  # 95th percentile
    ]
    
    print("\nTesting classification thresholds:")
    print("-" * 80)
    
    all_match = True
    for percentile, expected in test_cases:
        result = classify_regime(percentile)
        match = result == expected
        status = "✅" if match else "❌"
        
        print(f"{status} {percentile*100:5.1f}% → {result:10} (expected {expected})")
        
        if not match:
            all_match = False
    
    print("\n" + "="*80)
    
    if all_match:
        print("✅ All thresholds match bot logic!")
    else:
        print("❌ Threshold mismatch - collector needs fixing!")
    
    print("="*80 + "\n")
    return all_match


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REGIME VERIFICATION TEST SUITE")
    print("="*80)
    print("\nThis verifies that your market data collector uses")
    print("EXACTLY the same regime logic as your trading bot.")
    print("\nWhy this matters:")
    print("  - If regimes don't match, ML training will learn WRONG patterns")
    print("  - Bot might think it's in LOW_VOL, but training data says HIGH_VOL")
    print("  - ML model will make bad predictions")
    print("\n" + "="*80)
    
    # Run tests
    test1 = test_threshold_logic()
    test2 = test_regime_matching()
    
    if test1 and test2:
        print("\n🎯 SUCCESS - System is ML-ready!")
        sys.exit(0)
    else:
        print("\n⚠️  FAILURE - Fix issues before collecting training data!")
        sys.exit(1)