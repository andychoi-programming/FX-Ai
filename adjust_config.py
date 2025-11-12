"""
FX-Ai - Quick Configuration Adjuster
=====================================

This script allows you to quickly adjust key trading thresholds to test
whether signals will execute with more permissive settings.

IMPORTANT: This is for TESTING only. Do not use these settings for live trading!
"""

import json
from pathlib import Path
from datetime import datetime
import shutil

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def backup_config(config_path):
    """Create a backup of the current configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = config_path.parent / f"config_backup_{timestamp}.json"
    
    try:
        shutil.copy2(config_path, backup_path)
        print_success(f"Backup created: {backup_path}")
        return True
    except Exception as e:
        print_error(f"Failed to create backup: {e}")
        return False

def load_config(config_path):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print_success("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print_error(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in configuration: {e}")
        return None
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return None

def save_config(config_path, config):
    """Save configuration file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_success("Configuration saved successfully")
        return True
    except Exception as e:
        print_error(f"Failed to save configuration: {e}")
        return False

def show_current_settings(config):
    """Display current entry rule settings"""
    print_header("Current Entry Rule Settings")
    
    entry_rules = config.get('trading_rules', {}).get('entry_rules', {})
    
    print(f"{Colors.BOLD}Signal Thresholds:{Colors.END}")
    print(f"  min_signal_strength: {entry_rules.get('min_signal_strength', 'NOT SET')}")
    print(f"  max_spread: {entry_rules.get('max_spread', 'NOT SET')} pips")
    print(f"  min_risk_reward_ratio: {entry_rules.get('min_risk_reward_ratio', 'NOT SET')}:1")
    print(f"  require_ml_confirmation: {entry_rules.get('require_ml_confirmation', 'NOT SET')}")
    print(f"  require_technical_confirmation: {entry_rules.get('require_technical_confirmation', 'NOT SET')}")

def apply_preset(config, preset_name):
    """Apply a configuration preset"""
    
    presets = {
        'testing': {
            'name': 'Testing Mode (Very Permissive)',
            'description': 'Allows most signals to execute for testing purposes',
            'settings': {
                'min_signal_strength': 0.25,
                'max_spread': 10.0,
                'min_risk_reward_ratio': 1.0,
                'require_ml_confirmation': False,
                'require_technical_confirmation': False
            },
            'warning': 'DO NOT use for live trading! Only for testing signal execution.'
        },
        'aggressive': {
            'name': 'Aggressive Trading',
            'description': 'More trades with moderate quality threshold',
            'settings': {
                'min_signal_strength': 0.3,
                'max_spread': 5.0,
                'min_risk_reward_ratio': 1.5,
                'require_ml_confirmation': True,
                'require_technical_confirmation': True
            },
            'warning': 'Higher trade frequency, lower win rate expected'
        },
        'balanced': {
            'name': 'Balanced Trading (Default)',
            'description': 'Good balance between quality and quantity',
            'settings': {
                'min_signal_strength': 0.4,
                'max_spread': 3.0,
                'min_risk_reward_ratio': 2.0,
                'require_ml_confirmation': True,
                'require_technical_confirmation': True
            },
            'warning': None
        },
        'conservative': {
            'name': 'Conservative Trading',
            'description': 'High quality trades, fewer opportunities',
            'settings': {
                'min_signal_strength': 0.5,
                'max_spread': 2.0,
                'min_risk_reward_ratio': 2.5,
                'require_ml_confirmation': True,
                'require_technical_confirmation': True
            },
            'warning': 'May have very few trades per day'
        }
    }
    
    if preset_name not in presets:
        print_error(f"Unknown preset: {preset_name}")
        return None
    
    return presets[preset_name]

def main():
    """Main function"""
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FX-Ai Configuration Adjuster" + " " * 26 + "║")
    print("║" + " " * 68 + "║")
    print("║" + " " * 12 + "Quickly adjust entry rules for testing" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print(Colors.END)
    
    config_path = Path('config/config.json')
    
    # Load config
    config = load_config(config_path)
    if not config:
        print_error("Cannot proceed without valid configuration")
        return
    
    # Show current settings
    show_current_settings(config)
    
    # Show preset options
    print_header("Available Configuration Presets")
    print(f"{Colors.BOLD}1. Testing Mode{Colors.END} - Very permissive (for testing only)")
    print(f"   Signal: 0.25+, Spread: 10 pips, RR: 1.0, ML/Tech: Not required")
    print()
    print(f"{Colors.BOLD}2. Aggressive Trading{Colors.END} - More trades, moderate quality")
    print(f"   Signal: 0.30+, Spread: 5 pips, RR: 1.5, ML/Tech: Required")
    print()
    print(f"{Colors.BOLD}3. Balanced Trading{Colors.END} - Default settings")
    print(f"   Signal: 0.40+, Spread: 3 pips, RR: 2.0, ML/Tech: Required")
    print()
    print(f"{Colors.BOLD}4. Conservative Trading{Colors.END} - High quality, fewer trades")
    print(f"   Signal: 0.50+, Spread: 2 pips, RR: 2.5, ML/Tech: Required")
    print()
    print(f"{Colors.BOLD}5. Custom Adjustment{Colors.END} - Set individual values")
    print()
    print(f"{Colors.BOLD}0. Exit{Colors.END} - No changes")
    print()
    
    # Get user choice
    try:
        choice = input(f"{Colors.CYAN}Select option (0-5): {Colors.END}").strip()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Cancelled by user{Colors.END}")
        return
    
    if choice == '0':
        print_info("No changes made")
        return
    
    preset_map = {
        '1': 'testing',
        '2': 'aggressive',
        '3': 'balanced',
        '4': 'conservative'
    }
    
    if choice in preset_map:
        preset = apply_preset(config, preset_map[choice])
        
        if not preset:
            return
        
        print_header(f"Applying Preset: {preset['name']}")
        print(f"{preset['description']}")
        print()
        
        if preset['warning']:
            print_warning(f"WARNING: {preset['warning']}")
            print()
        
        # Show what will change
        print(f"{Colors.BOLD}New Settings:{Colors.END}")
        for key, value in preset['settings'].items():
            print(f"  {key}: {value}")
        print()
        
        # Confirm
        try:
            confirm = input(f"{Colors.YELLOW}Apply these settings? (yes/no): {Colors.END}").strip().lower()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Cancelled by user{Colors.END}")
            return
        
        if confirm != 'yes':
            print_info("Changes cancelled")
            return
        
        # Create backup
        if not backup_config(config_path):
            print_error("Backup failed - aborting changes")
            return
        
        # Apply settings
        if 'trading_rules' not in config:
            config['trading_rules'] = {}
        if 'entry_rules' not in config['trading_rules']:
            config['trading_rules']['entry_rules'] = {}
        
        config['trading_rules']['entry_rules'].update(preset['settings'])
        
        # Save
        if save_config(config_path, config):
            print_header("Configuration Updated Successfully")
            print(f"{Colors.GREEN}✅ New settings applied{Colors.END}")
            print(f"{Colors.GREEN}✅ Backup saved{Colors.END}")
            print()
            print(f"{Colors.BOLD}Next Steps:{Colors.END}")
            print("1. Restart FX-Ai system for changes to take effect")
            print("2. Monitor system logs for signal execution")
            print("3. Run diagnose_signals.py to verify changes")
            print()
            if preset['warning']:
                print_warning(f"Remember: {preset['warning']}")
    
    elif choice == '5':
        print_header("Custom Adjustment")
        print_info("Enter new values (press Enter to keep current value)")
        print()
        
        # Create backup
        if not backup_config(config_path):
            print_error("Backup failed - aborting changes")
            return
        
        entry_rules = config.get('trading_rules', {}).get('entry_rules', {})
        new_settings = {}
        
        # Signal strength
        current = entry_rules.get('min_signal_strength', 0.4)
        try:
            value = input(f"min_signal_strength (current: {current}): ").strip()
            if value:
                new_settings['min_signal_strength'] = float(value)
        except (ValueError, KeyboardInterrupt):
            pass
        
        # Max spread
        current = entry_rules.get('max_spread', 3.0)
        try:
            value = input(f"max_spread in pips (current: {current}): ").strip()
            if value:
                new_settings['max_spread'] = float(value)
        except (ValueError, KeyboardInterrupt):
            pass
        
        # Risk/reward
        current = entry_rules.get('min_risk_reward_ratio', 2.0)
        try:
            value = input(f"min_risk_reward_ratio (current: {current}): ").strip()
            if value:
                new_settings['min_risk_reward_ratio'] = float(value)
        except (ValueError, KeyboardInterrupt):
            pass
        
        if not new_settings:
            print_info("No changes made")
            return
        
        # Apply settings
        if 'trading_rules' not in config:
            config['trading_rules'] = {}
        if 'entry_rules' not in config['trading_rules']:
            config['trading_rules']['entry_rules'] = {}
        
        config['trading_rules']['entry_rules'].update(new_settings)
        
        # Save
        if save_config(config_path, config):
            print_header("Configuration Updated Successfully")
            print(f"{Colors.GREEN}✅ Custom settings applied{Colors.END}")
            print(f"{Colors.GREEN}✅ Backup saved{Colors.END}")
            print()
            print(f"{Colors.BOLD}Changes:{Colors.END}")
            for key, value in new_settings.items():
                print(f"  {key}: {value}")
    
    else:
        print_error("Invalid option")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Operation cancelled by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()