import MetaTrader5 as mt5
import json
import time
from datetime import datetime

class FillingModeDiagnostic:
    """Diagnose MT5 filling modes and generate fix code"""

    def __init__(self):
        self.symbols_to_test = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDUSD'
        ]
        self.filling_modes = {
            mt5.ORDER_FILLING_FOK: 'ORDER_FILLING_FOK',
            mt5.ORDER_FILLING_IOC: 'ORDER_FILLING_IOC',
            mt5.ORDER_FILLING_RETURN: 'ORDER_FILLING_RETURN'
        }

    def connect_mt5(self):
        """Connect to MT5"""
        print("🔌 Connecting to MT5...")
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        print("✅ MT5 connected successfully")
        return True

    def get_symbol_info(self, symbol):
        """Get symbol information"""
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"❌ Symbol {symbol} not found")
            return None
        return info

    def test_filling_mode(self, symbol, filling_mode):
        """Test a specific filling mode for a symbol"""
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False, "No tick data"

        # Create a test order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,  # Very small volume for testing
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": 10,
            "magic": 123456,
            "comment": "FILLING_MODE_TEST",
            "type_filling": filling_mode,
            "type_time": mt5.ORDER_TIME_GTC
        }

        # Check if order would be valid
        result = mt5.order_check(request)
        if result is None:
            return False, f"Order check failed: {mt5.last_error()}"

        # Return success/failure
        if result.retcode == 0:  # TRADE_RETCODE_DONE
            return True, "Valid"
        else:
            return False, f"Retcode: {result.retcode}"

    def diagnose_symbol(self, symbol):
        """Diagnose all filling modes for a symbol"""
        print(f"\n🔍 Testing {symbol}...")

        info = self.get_symbol_info(symbol)
        if not info:
            return None

        print(f"   Spread: {info.spread}, Min volume: {info.volume_min}")

        results = {}
        for mode_value, mode_name in self.filling_modes.items():
            success, message = self.test_filling_mode(symbol, mode_value)
            results[mode_name] = {
                'success': success,
                'message': message,
                'value': mode_value
            }
            status = "✅" if success else "❌"
            print(f"   {status} {mode_name}: {message}")

        return results

    def generate_fix_code(self, working_modes):
        """Generate the fix code for order_executor.py"""
        if not working_modes:
            return None

        # Find the most common working mode
        mode_counts = {}
        for symbol, modes in working_modes.items():
            for mode_name, result in modes.items():
                if result['success']:
                    mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

        if not mode_counts:
            return None

        best_mode = max(mode_counts, key=mode_counts.get)
        best_value = None

        # Get the value for the best mode
        for symbol_results in working_modes.values():
            for mode_name, result in symbol_results.items():
                if mode_name == best_mode and result['success']:
                    best_value = result['value']
                    break
            if best_value:
                break

        # Generate the fix code
        fix_code = f'''# FIX FOR MT5 FILLING MODE ISSUE
# Add this to your order request in order_executor.py

request["type_filling"] = {best_value}  # {best_mode}

# This fixes the 10030 error (invalid filling mode)
# Recommended by MT5 filling mode diagnostic on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''

        return fix_code, best_mode, best_value

    def run_diagnostic(self):
        """Run the complete diagnostic"""
        print("=" * 60)
        print("MT5 FILLING MODE DIAGNOSTIC TOOL")
        print("=" * 60)
        print(f"Testing {len(self.symbols_to_test)} symbols...")

        if not self.connect_mt5():
            return None

        working_modes = {}

        for symbol in self.symbols_to_test:
            results = self.diagnose_symbol(symbol)
            if results:
                working_modes[symbol] = results

        mt5.shutdown()

        # Generate fix
        fix_result = self.generate_fix_code(working_modes)

        print("\n" + "=" * 60)
        print("DIAGNOSTIC RESULTS")
        print("=" * 60)

        if fix_result:
            fix_code, best_mode, best_value = fix_result
            print(f"✅ RECOMMENDED FILLING MODE: {best_mode} (value: {best_value})")
            print(f"✅ Working on {len(working_modes)}/{len(self.symbols_to_test)} symbols tested")

            # Save fix code to file
            with open('order_executor_fix.py', 'w') as f:
                f.write(fix_code)

            print("\n📄 Fix code saved to: order_executor_fix.py")
            print("\n" + fix_code)

        else:
            print("❌ No working filling modes found!")
            print("This suggests a deeper MT5 configuration issue.")

        return fix_result

def main():
    diagnostic = FillingModeDiagnostic()
    diagnostic.run_diagnostic()

if __name__ == "__main__":
    main()