#!/usr/bin/env python3
"""
FX-Ai Risk Management Display
Standalone script to display current risk management parameters
"""

import json
import os
from pathlib import Path
from datetime import datetime

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: config.json not found!")
        return None
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON in config.json!")
        return None

def display_risk_parameters():
    """Display risk management parameters in a formatted way"""

    print("=" * 60)
    print("🎯 FX-AI RISK MANAGEMENT PARAMETERS")
    print("=" * 60)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    config = load_config()
    if not config:
        return

    # Extract risk parameters
    trading_config = config.get('trading', {})
    risk_config = config.get('risk_management', {})

    # Risk Dollar Amount Per Trade
    risk_per_trade = trading_config.get('risk_per_trade', 50.0)
    risk_type = trading_config.get('risk_type', 'fixed_dollar')

    print("💰 RISK DOLLAR AMOUNT PER TRADE")
    print("-" * 40)
    print(f"Amount:          ${risk_per_trade:.2f}")
    print(f"Risk Type:       {risk_type.replace('_', ' ').title()}")
    print(f"Risk Method:     {risk_config.get('risk_calculation_method', 'pip_based').replace('_', ' ').title()}")
    print()

    # Maximum Daily Loss
    max_daily_loss = trading_config.get('max_daily_loss', 200.0)

    print("📉 MAXIMUM DAILY LOSS")
    print("-" * 40)
    print(f"Daily Loss Limit: ${max_daily_loss:.2f}")
    print(f"Risk per Trade:   ${risk_per_trade:.2f}")
    print(f"Max Trades/Day:   {int(max_daily_loss / risk_per_trade)} (theoretical)")
    print()

    # Maximum Amount of Trades at a Time
    max_positions = trading_config.get('max_positions', 5)

    print("📊 MAXIMUM TRADES AT A TIME")
    print("-" * 40)
    print(f"Concurrent Positions: {max_positions}")
    print(f"Per Symbol Limit:    {'Yes' if trading_config.get('prevent_multiple_positions_per_symbol', True) else 'No'}")
    print()

    # Additional Risk Settings
    print("⚙️  ADDITIONAL RISK SETTINGS")
    print("-" * 40)
    print(f"Stop Loss Method:     {risk_config.get('stop_loss_method', 'atr_based').replace('_', ' ').title()}")
    print(f"ATR Multiplier:       {risk_config.get('atr_multiplier', 1.5)}x")
    print(f"Max SL Distance:      {risk_config.get('maximum_sl_pips', 50)} pips")
    print(f"Risk/Reward Ratio:    {risk_config.get('risk_reward_ratio', 3.0)}:1")
    print()

    # Position Sizing Details
    print("📏 POSITION SIZING DETAILS")
    print("-" * 40)
    print(f"Min Lot Size:         {trading_config.get('min_lot_size', 0.01)}")
    print(f"Max Lot Size:         {trading_config.get('max_lot_size', 1.0)}")
    print(f"Max Spread:           {trading_config.get('max_spread', 3.0)} pips")
    print()

    # Trading Filters
    print("🔍 TRADING FILTERS")
    print("-" * 40)
    print(f"News Filter:          {'Enabled' if trading_config.get('enable_news_filter', True) else 'Disabled'}")
    print(f"Session Filter:       {'Enabled' if trading_config.get('enable_session_filter', True) else 'Disabled'}")
    print(f"Day Trading Only:     {'Enabled' if trading_config.get('day_trading_only', True) else 'Disabled'}")
    print(f"Close Before Weekend: {'Enabled' if trading_config.get('close_before_weekend', True) else 'Disabled'}")
    print()

    # Summary
    print("📋 SUMMARY")
    print("-" * 40)
    total_risk_capacity = max_positions * risk_per_trade
    print(f"Total Risk Capacity:  ${total_risk_capacity:.2f} (at max positions)")
    print(f"Daily Risk Limit:     ${max_daily_loss:.2f}")
    print(f"Risk Utilization:     {(total_risk_capacity/max_daily_loss*100):.1f}% of daily limit")
    print()

    print("=" * 60)
    print("✅ Risk parameters loaded successfully!")
    print("💡 Remember: Always test with demo account first!")
    print("=" * 60)

def main():
    """Main function"""
    try:
        display_risk_parameters()
    except Exception as e:
        print(f"❌ Error displaying risk parameters: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())