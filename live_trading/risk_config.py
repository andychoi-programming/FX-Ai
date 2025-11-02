#!/usr/bin/env python3
"""
FX-Ai Interactive Risk Management Configuration
Standalone script to view and modify risk management parameters
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

def save_config(config):
    """Save configuration to config.json"""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path, 'w', indent=2) as f:
            json.dump(config, f, indent=2)
        print("✅ Configuration saved successfully!")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def display_current_risk_parameters(config):
    """Display current risk management parameters"""

    print("=" * 60)
    print("🎯 CURRENT FX-AI RISK MANAGEMENT PARAMETERS")
    print("=" * 60)
    print(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Extract risk parameters
    trading_config = config.get('trading', {})
    risk_config = config.get('risk_management', {})

    # Risk Dollar Amount Per Trade
    risk_per_trade = trading_config.get('risk_per_trade', 50.0)
    risk_type = trading_config.get('risk_type', 'fixed_dollar')

    print("💰 RISK DOLLAR AMOUNT PER TRADE")
    print("-" * 40)
    print(f"Current Amount:          ${risk_per_trade:.2f}")
    print(f"Risk Type:               {risk_type.replace('_', ' ').title()}")
    print(f"Risk Method:             {risk_config.get('risk_calculation_method', 'pip_based').replace('_', ' ').title()}")
    print()

    # Maximum Daily Loss
    max_daily_loss = trading_config.get('max_daily_loss', 200.0)

    print("📉 MAXIMUM DAILY LOSS")
    print("-" * 40)
    print(f"Current Daily Loss Limit: ${max_daily_loss:.2f}")
    print(f"Risk per Trade:           ${risk_per_trade:.2f}")
    print(f"Max Trades/Day:           {int(max_daily_loss / risk_per_trade)} (theoretical)")
    print()

    # Maximum Amount of Trades at a Time
    max_positions = trading_config.get('max_positions', 5)

    print("📊 MAXIMUM TRADES AT A TIME")
    print("-" * 40)
    print(f"Current Concurrent Positions: {max_positions}")
    print(f"Per Symbol Limit:            {'Yes' if trading_config.get('prevent_multiple_positions_per_symbol', True) else 'No'}")
    print()

    # Summary
    print("📋 CURRENT SUMMARY")
    print("-" * 40)
    total_risk_capacity = max_positions * risk_per_trade
    print(f"Total Risk Capacity:      ${total_risk_capacity:.2f} (at max positions)")
    print(f"Daily Risk Limit:         ${max_daily_loss:.2f}")
    print(f"Risk Utilization:         {(total_risk_capacity/max_daily_loss*100):.1f}% of daily limit")
    print()

def get_user_input():
    """Get user input for risk parameters"""
    print("=" * 60)
    print("⚙️  RISK PARAMETER CONFIGURATION")
    print("=" * 60)
    print("Enter new values (press Enter to keep current value):")
    print()

    # Risk per trade
    while True:
        try:
            risk_input = input("💰 Risk Dollar Amount per Trade (current: $50.00): $").strip()
            if risk_input == "":
                risk_per_trade = 50.0
                break
            risk_per_trade = float(risk_input)
            if risk_per_trade <= 0:
                print("❌ Risk amount must be positive!")
                continue
            if risk_per_trade > 1000:
                confirm = input(f"⚠️  Warning: ${risk_per_trade:.2f} is a high risk amount. Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    continue
            break
        except ValueError:
            print("❌ Please enter a valid number!")

    # Maximum daily loss
    while True:
        try:
            daily_loss_input = input("📉 Maximum Daily Loss (current: $500.00): $").strip()
            if daily_loss_input == "":
                max_daily_loss = 500.0
                break
            max_daily_loss = float(daily_loss_input)
            if max_daily_loss <= 0:
                print("❌ Daily loss limit must be positive!")
                continue
            if max_daily_loss < risk_per_trade:
                print(f"❌ Daily loss limit (${max_daily_loss:.2f}) cannot be less than risk per trade (${risk_per_trade:.2f})!")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number!")

    # Maximum positions
    while True:
        try:
            positions_input = input(f"📊 Maximum Trades at a Time (current: {max_positions}): ").strip()
            if positions_input == "":
                break  # Keep current value
            max_positions = int(positions_input)
            if max_positions <= 0:
                print("❌ Maximum positions must be positive!")
                continue
            if max_positions > 50:
                confirm = input(f"⚠️  Warning: {max_positions} concurrent positions is very high. Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    continue
            break
        except ValueError:
            print("❌ Please enter a valid number!")

    return {
        'risk_per_trade': risk_per_trade,
        'max_daily_loss': max_daily_loss,
        'max_positions': max_positions
    }

def update_config_with_new_values(config, new_values):
    """Update configuration with new risk values"""
    if 'trading' not in config:
        config['trading'] = {}

    config['trading']['risk_per_trade'] = new_values['risk_per_trade']
    config['trading']['max_daily_loss'] = new_values['max_daily_loss']
    config['trading']['max_positions'] = new_values['max_positions']

    return config

def display_updated_parameters(config, new_values):
    """Display the updated parameters"""
    print("\n" + "=" * 60)
    print("✅ UPDATED RISK PARAMETERS")
    print("=" * 60)

    print("💰 RISK DOLLAR AMOUNT PER TRADE")
    print("-" * 40)
    print(f"New Amount:               ${new_values['risk_per_trade']:.2f}")
    print()

    print("📉 MAXIMUM DAILY LOSS")
    print("-" * 40)
    print(f"New Daily Loss Limit:     ${new_values['max_daily_loss']:.2f}")
    print(f"Risk per Trade:           ${new_values['risk_per_trade']:.2f}")
    print(f"Max Trades/Day:           {int(new_values['max_daily_loss'] / new_values['risk_per_trade'])} (theoretical)")
    print()

    print("📊 MAXIMUM TRADES AT A TIME")
    print("-" * 40)
    print(f"New Concurrent Positions: {new_values['max_positions']}")
    print()

    # New Summary
    print("📋 UPDATED SUMMARY")
    print("-" * 40)
    total_risk_capacity = new_values['max_positions'] * new_values['risk_per_trade']
    print(f"Total Risk Capacity:      ${total_risk_capacity:.2f} (at max positions)")
    print(f"Daily Risk Limit:         ${new_values['max_daily_loss']:.2f}")
    print(f"Risk Utilization:         {(total_risk_capacity/new_values['max_daily_loss']*100):.1f}% of daily limit")
    print()

def main():
    """Main function"""
    try:
        # Load current configuration
        config = load_config()
        if not config:
            return 1

        # Display current parameters
        display_current_risk_parameters(config)

        # Ask user if they want to modify
        print("🔧 RISK MANAGEMENT CONFIGURATION")
        print("-" * 40)
        choice = input("Do you want to modify these risk parameters? (y/N): ").strip().lower()

        if choice == 'y' or choice == 'yes':
            # Get new values from user
            new_values = get_user_input()

            # Update configuration
            updated_config = update_config_with_new_values(config, new_values)

            # Display updated parameters
            display_updated_parameters(config, new_values)

            # Confirm save
            save_choice = input("💾 Save these changes to config.json? (y/N): ").strip().lower()
            if save_choice == 'y' or save_choice == 'yes':
                if save_config(updated_config):
                    print("\n🎉 Risk parameters updated successfully!")
                    print("💡 Remember to restart the trading system for changes to take effect.")
                else:
                    print("\n❌ Failed to save configuration!")
                    return 1
            else:
                print("\n📝 Changes not saved. Configuration unchanged.")
        else:
            print("\n📋 Configuration unchanged.")

        print("\n" + "=" * 60)
        print("✅ Risk management configuration complete!")
        print("💡 Always test with demo account first!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Configuration cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during configuration: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())