import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
from typing import Dict, Optional
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FillingModeTester:
    """Test the filling mode fix implementation"""

    def __init__(self):
        self.test_symbol = "EURUSD"  # Safe test symbol
        self.test_volume = 0.01  # Very small volume for testing
        self.magic_number = 123456  # Test magic number

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            logger.info("Initializing MT5...")
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False

            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.info(f"✅ Connected to: {terminal_info.name} ({terminal_info.company})")
                return True
            else:
                logger.error("Failed to get terminal info")
                return False

        except Exception as e:
            logger.error(f"MT5 initialization error: {str(e)}")
            return False

    def get_filling_mode(self, symbol: str) -> int:
        """Get the correct filling mode for a symbol (same as in order_executor.py)"""
        try:
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                logger.warning(f"No symbol info for {symbol}, using RETURN mode")
                return mt5.ORDER_FILLING_RETURN

            # Check supported modes (TIOMarkets usually uses RETURN)
            if symbol_info.filling_mode & 4:  # RETURN
                return mt5.ORDER_FILLING_RETURN
            elif symbol_info.filling_mode & 2:  # IOC
                return mt5.ORDER_FILLING_IOC
            elif symbol_info.filling_mode & 1:  # FOK
                return mt5.ORDER_FILLING_FOK
            else:
                logger.warning(f"No supported filling modes for {symbol}, using RETURN")
                return mt5.ORDER_FILLING_RETURN

        except Exception as e:
            logger.error(f"Error getting filling mode for {symbol}: {e}")
            return mt5.ORDER_FILLING_RETURN

    def test_symbol_availability(self, symbol: str) -> bool:
        """Test if symbol is available for trading"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return False

            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return False

            logger.info(f"✅ Symbol {symbol} is available for trading")
            return True

        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {e}")
            return False

    def test_order_check(self, symbol: str) -> Dict:
        """Test order validation without actually placing the order"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Cannot get tick data'}

            filling_mode = self.get_filling_mode(symbol)

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": self.test_volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 10,
                "magic": self.magic_number,
                "comment": "Filling Mode Test",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            logger.info(f"🧪 Testing order with filling mode: {filling_mode} ({self.get_mode_name(filling_mode)})")

            # Use order_check for safe testing
            result = mt5.order_check(request)

            if result:
                success = result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 0
                return {
                    'success': success,
                    'retcode': result.retcode,
                    'comment': getattr(result, 'comment', ''),
                    'filling_mode_used': filling_mode,
                    'mode_name': self.get_mode_name(filling_mode),
                    'request': request
                }
            else:
                return {'success': False, 'error': 'No result from order_check'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_actual_order(self, symbol: str, dry_run: bool = True) -> Dict:
        """Test actual order placement (dry run by default)"""
        try:
            if dry_run:
                logger.info("🔍 Performing DRY RUN (no real order will be placed)")
                return self.test_order_check(symbol)

            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Cannot get tick data'}

            filling_mode = self.get_filling_mode(symbol)

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": self.test_volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 10,
                "magic": self.magic_number,
                "comment": "Filling Mode Test - REAL ORDER",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            logger.warning("⚠️  PLACING REAL ORDER - This will execute a trade!")
            logger.info(f"📊 Order details: {symbol} {self.test_volume} lots at {tick.ask}")

            # Actually place the order
            result = mt5.order_send(request)

            if result:
                success = result.retcode == mt5.TRADE_RETCODE_DONE
                ticket = getattr(result, 'order', None)

                if success and ticket:
                    logger.info(f"✅ Order placed successfully! Ticket: {ticket}")

                    # Immediately close the test order
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": self.test_volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": ticket,
                        "price": tick.bid,
                        "deviation": 10,
                        "magic": self.magic_number,
                        "comment": "Closing test order",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling_mode,
                    }

                    time.sleep(1)  # Brief pause
                    close_result = mt5.order_send(close_request)

                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info("✅ Test order closed successfully")
                    else:
                        logger.warning("⚠️  Test order may still be open - check manually")

                return {
                    'success': success,
                    'retcode': result.retcode,
                    'comment': getattr(result, 'comment', ''),
                    'ticket': ticket,
                    'filling_mode_used': filling_mode,
                    'mode_name': self.get_mode_name(filling_mode),
                    'request': request
                }
            else:
                return {'success': False, 'error': 'No result from order_send'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_mode_name(self, mode: int) -> str:
        """Convert filling mode constant to readable name"""
        if mode == mt5.ORDER_FILLING_FOK:
            return "FOK"
        elif mode == mt5.ORDER_FILLING_IOC:
            return "IOC"
        elif mode == mt5.ORDER_FILLING_RETURN:
            return "RETURN"
        else:
            return f"UNKNOWN({mode})"

    def run_comprehensive_test(self, dry_run: bool = True):
        """Run comprehensive filling mode test"""
        print("🧪 FX-Ai Filling Mode Fix Test")
        print("=" * 50)

        if not self.initialize_mt5():
            print("❌ Failed to initialize MT5")
            return False

        try:
            # Test symbol availability
            if not self.test_symbol_availability(self.test_symbol):
                print(f"❌ Test symbol {self.test_symbol} not available")
                return False

            print(f"\n📊 Testing with symbol: {self.test_symbol}")
            print(f"📊 Test volume: {self.test_volume} lots")
            print(f"📊 Magic number: {self.magic_number}")

            # Run the test
            result = self.test_actual_order(self.test_symbol, dry_run=dry_run)

            print("\n" + "=" * 50)
            print("TEST RESULTS")
            print("=" * 50)

            if result.get('success'):
                print("✅ SUCCESS: Filling mode fix works!")
                print(f"   Mode used: {result.get('mode_name', 'Unknown')}")
                print(f"   Retcode: {result.get('retcode', 'N/A')}")
                if result.get('ticket'):
                    print(f"   Ticket: {result.get('ticket', 'N/A')}")
                print(f"   Comment: {result.get('comment', 'N/A')}")
            else:
                print("❌ FAILED: Filling mode issue persists")
                print(f"   Error: {result.get('error', 'Unknown')}")
                print(f"   Retcode: {result.get('retcode', 'N/A')}")
                print(f"   Comment: {result.get('comment', 'N/A')}")

                # Additional analysis
                if result.get('retcode') == 10030:
                    print("\n🔍 ANALYSIS: Error 10030 indicates filling mode issue")
                    print("   The dynamic filling mode detection may not be working")
                    print("   Check if the get_filling_mode method is implemented correctly")

            print("\n" + "=" * 50)

            # Recommendations
            if result.get('success'):
                print("💡 RECOMMENDATIONS:")
                print("   ✅ Your filling mode fix is working correctly")
                print("   ✅ You can proceed with live trading")
                print("   ✅ Consider running the full diagnostic for all symbols")
            else:
                print("💡 RECOMMENDATIONS:")
                print("   ❌ Run the full diagnostic: python full_system_diagnostic.py")
                print("   ❌ Check your order_executor.py get_filling_mode implementation")
                print("   ❌ Verify MT5 connection and symbol availability")

            return result.get('success', False)

        finally:
            mt5.shutdown()

def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test filling mode fix')
    parser.add_argument('--real', action='store_true',
                       help='Place a real order (default is dry run)')
    parser.add_argument('--symbol', default='EURUSD',
                       help='Symbol to test (default: EURUSD)')

    args = parser.parse_args()

    tester = FillingModeTester()
    tester.test_symbol = args.symbol

    dry_run = not args.real

    if not dry_run:
        print("⚠️  WARNING: You are about to place a REAL order!")
        confirm = input("Are you sure? (type 'yes' to continue): ")
        if confirm.lower() != 'yes':
            print("Test cancelled.")
            return

    success = tester.run_comprehensive_test(dry_run=dry_run)

    if success:
        print("\n🎉 Test passed! Your filling mode fix is working.")
    else:
        print("\n❌ Test failed. Check the recommendations above.")

if __name__ == "__main__":
    main()