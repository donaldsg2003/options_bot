#!/usr/bin/env python3
"""
emergency_diagnostic.py
Check why exit logic isn't triggering for your positions.
"""

import sys
from pathlib import Path
from datetime import date, datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from option_bot_spreads.core.data_layer import DataLayer
from option_bot_spreads.core.persistent_state import PersistentState
from option_bot_spreads.config.config import SETTINGS, DTEConfig

def main():
    print("\n" + "="*70)
    print("EMERGENCY EXIT DIAGNOSTIC")
    print("="*70)
    
    # Initialize
    d = DataLayer()
    state = PersistentState()
    
    print("\n[CHECK 1] Market Status")
    print("-" * 70)
    is_open = d.is_market_open()
    print(f"Market open: {is_open}")
    if not is_open:
        print("âš ï¸ WARNING: Market closed - exit logic won't run!")
    
    print("\n[CHECK 2] Live Positions from Alpaca")
    print("-" * 70)
    try:
        positions = d.trading.get_all_positions()
        print(f"Total positions: {len(positions)}")
        
        for p in positions:
            print(f"\n  {p.symbol}")
            print(f"    Qty: {p.qty}")
            print(f"    Entry: ${float(p.avg_entry_price):.2f}")
            print(f"    Current: ${float(p.current_price):.2f}")
            print(f"    P/L: ${float(p.unrealized_pl):.2f} ({float(p.unrealized_plpc)*100:.2f}%)")
    except Exception as e:
        print(f"âŒ Error fetching positions: {e}")
        return
    
    print("\n[CHECK 3] Positions in State File")
    print("-" * 70)
    open_spreads = state.get("open_spreads", {})
    print(f"Tracked spreads: {len(open_spreads)}")
    
    if not open_spreads:
        print("âš ï¸ WARNING: No spreads in state file!")
        print("   Bot doesn't know about your positions!")
        print("   Run reconciliation to fix.")
    else:
        for skey, sdata in open_spreads.items():
            print(f"\n  {skey}")
            print(f"    Credit: ${sdata.get('credit', 0):.2f}")
            print(f"    Regime: {sdata.get('entry_regime', 'UNKNOWN')}")
            print(f"    DTE: {sdata.get('entry_dte', 'UNKNOWN')}")
    
    print("\n[CHECK 4] Group Positions (Exit Logic View)")
    print("-" * 70)
    
    pos_groups = {}
    for p in positions:
        if len(p.symbol) < 9:
            continue
        exp = p.symbol[3:9]
        pos_groups.setdefault(exp, []).append(p)
    
    print(f"Expiration groups: {len(pos_groups)}")
    
    for exp_key, legs in pos_groups.items():
        print(f"\n  Expiration: {exp_key}")
        print(f"    Legs: {len(legs)}")
        
        # Parse expiration
        try:
            exp_date = datetime.strptime(exp_key, "%y%m%d").date()
            dte = (exp_date - date.today()).days
            print(f"    DTE: {dte}")
            
            if dte <= 2:
                print(f"    âš ï¸ CRITICAL: {dte} DTE - SHOULD EMERGENCY EXIT!")
        except Exception as e:
            print(f"    Error parsing date: {e}")
        
        # Find spread pairs
        shorts = [p for p in legs if float(p.qty) < 0]
        longs = [p for p in legs if float(p.qty) > 0]
        
        print(f"    Shorts: {len(shorts)}")
        print(f"    Longs: {len(longs)}")
        
        if len(shorts) != len(longs):
            print(f"    âš ï¸ WARNING: Unbalanced spread!")
        
        # Try to pair them
        for short, long in zip(shorts, longs):
            spread_key = f"{short.symbol}|{long.symbol}|{exp_key}"
            
            # Check if in state
            in_state = spread_key in open_spreads
            print(f"\n    Spread: {short.symbol[:15]}... / {long.symbol[:15]}...")
            print(f"      In state: {in_state}")
            
            if not in_state:
                print(f"      âŒ NOT TRACKED - exit logic will skip!")
            else:
                sdata = open_spreads[spread_key]
                credit = sdata.get('credit', 0)
                regime = sdata.get('entry_regime', 'UNKNOWN')
                
                print(f"      Credit: ${credit:.2f}")
                print(f"      Regime: {regime}")
                
                # Calculate stop loss
                stop_loss = credit * SETTINGS.SL_MULT
                
                # Get current spread value
                try:
                    quotes = d.latest_quotes([short.symbol, long.symbol])
                    sb, sa = quotes.get(short.symbol, (0, 0))
                    lb, la = quotes.get(long.symbol, (0, 0))
                    
                    if sb and sa and lb and la:
                        short_mid = (sb + sa) / 2
                        long_mid = (lb + la) / 2
                        current_spread = short_mid - long_mid
                        
                        print(f"      Current spread: ${current_spread:.2f}")
                        print(f"      Stop loss: ${stop_loss:.2f}")
                        
                        if current_spread >= stop_loss:
                            print(f"STOP LOSS TRIGGERED!")
                        
                        if dte <= SETTINGS.EMERGENCY_EXIT_DTE:
                            print(f"EMERGENCY EXIT TRIGGERED!")
                except Exception as e:
                    print(f"      Error getting quotes: {e}")
    
    print("\n[CHECK 5] Exit Configuration")
    print("-" * 70)
    print(f"EMERGENCY_EXIT_DTE: {SETTINGS.EMERGENCY_EXIT_DTE}")
    print(f"SL_MULT: {SETTINGS.SL_MULT}")
    print(f"USE_MARKET_ORDER_EMERGENCY: {SETTINGS.USE_MARKET_ORDER_EMERGENCY}")
    
    print("\n[CHECK 6] Pending Close Orders")
    print("-" * 70)
    pending_keys = [k for k in state.state.keys() if k.startswith("pending_close_")]
    print(f"Pending closes: {len(pending_keys)}")
    
    for key in pending_keys:
        pending = state.get(key)
        print(f"\n  {key}")
        print(f"    Order ID: {pending.get('order_id')}")
        print(f"    Submitted: {pending.get('submitted_at')}")
        print(f"    Reasons: {pending.get('reasons')}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    print("\nâš ï¸ CRITICAL ISSUES FOUND:")
    
    issues = []
    
    if not is_open:
        issues.append("Market is closed - exit logic blocked")
    
    if not open_spreads:
        issues.append("No spreads in state file - bot doesn't know about positions")
    
    for exp_key, legs in pos_groups.items():
        try:
            exp_date = datetime.strptime(exp_key, "%y%m%d").date()
            dte = (exp_date - date.today()).days
            if dte <= 2:
                issues.append(f"Positions at {dte} DTE - emergency exit should trigger")
        except:
            pass
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No obvious issues found - check logs for errors")
    
    print("\n")

if __name__ == "__main__":
    main()