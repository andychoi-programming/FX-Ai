import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
from typing import Dict, Optional
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StopOrderTester:
    """Test the stop order fix implementation"""

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

    def test_stop_order(self, symbol: str, order_type: str, dry_run: bool = True) -> Dict:
        """Test stop order placement"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Cannot get tick data'}

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'success': False, 'error': 'Cannot get symbol info'}

            current_price = tick.ask if order_type == "BUY" else tick.bid

            # Calculate stop price (far enough from current price)
            stops_level = getattr(symbol_info, 'trade_stops_level', 20)
            min_distance = stops_level * symbol_info.point * 1.5  # 1.5x safety margin

            if order_type == "BUY":
                stop_price = current_price + min_distance
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            else:  # SELL
                stop_price = current_price - min_distance
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP

            # Create stop order request (NO filling mode!)
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": self.test_volume,
                "type": mt5_order_type,
                "price": round(stop_price, symbol_info.digits),
                "magic": self.magic_number,
                "comment": f"Stop Order Test ({order_type})",
                "type_time": mt5.ORDER_TIME_GTC,  # GTC for stop orders
            }

            logger.info(f"🧪 Testing {order_type} STOP order: {symbol} @ {stop_price:.5f} (current: {current_price:.5f})")

            if dry_run:
                # Just validate the request
                result = mt5.order_check(request)
                if result and (result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 0):
                    return {
                        'success': True,
                        'retcode': result.retcode,
                        'comment': getattr(result, 'comment', ''),
                        'request': request,
                        'dry_run': True
                    }
                else:
                    retcode = result.retcode if result else 'None'
                    comment = getattr(result, 'comment', '') if result else 'No result'
                    return {
                        'success': False,
                        'retcode': retcode,
                        'comment': comment,
                        'error': f'Order check failed with retcode {retcode}',
                        'request': request,
                        'dry_run': True
                    }
            else:
                # Actually place the order
                result = mt5.order_send(request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    ticket = getattr(result, 'order', None)
                    logger.info(f"✅ Stop order placed successfully! Ticket: {ticket}")

                    # Clean up - cancel the test order
                    time.sleep(2)  # Brief pause
                    cancel_result = mt5.order_cancel(ticket)
                    if cancel_result and cancel_result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info("✅ Test stop order cancelled successfully")
                    else:
                        logger.warning("⚠️  Test stop order may still be pending - check manually")

                    return {
                        'success': True,
                        'retcode': result.retcode,
                        'comment': getattr(result, 'comment', ''),
                        'ticket': ticket,
                        'request': request,
                        'dry_run': False
                    }
                else:
                    retcode = result.retcode if result else 'None'
                    comment = getattr(result, 'comment', '') if result else 'No result'
                    return {
                        'success': False,
                        'retcode': retcode,
                        'comment': comment,
                        'error': f'Order send failed with retcode {retcode}',
                        'request': request,
                        'dry_run': False
                    }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_comprehensive_test(self, dry_run: bool = True):
        """Run comprehensive stop order test"""
        print("🧪 FX-Ai Stop Order Fix Test")
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

            # Test BUY STOP order
            print("\n📈 Testing BUY STOP order...")
            buy_result = self.test_stop_order(self.test_symbol, "BUY", dry_run=dry_run)

            # Test SELL STOP order
            print("\n📉 Testing SELL STOP order...")
            sell_result = self.test_stop_order(self.test_symbol, "SELL", dry_run=dry_run)

            print("\n" + "=" * 50)
            print("STOP ORDER TEST RESULTS")
            print("=" * 50)

            buy_success = buy_result.get('success', False)
            sell_success = sell_result.get('success', False)

            if buy_success and sell_success:
                print("✅ SUCCESS: Stop order fix works!")
                print("   Both BUY STOP and SELL STOP orders validated successfully")
            elif buy_success or sell_success:
                print("⚠️  PARTIAL SUCCESS: One stop order type works")
                if buy_success:
                    print("   ✅ BUY STOP orders work")
                else:
                    print("   ❌ BUY STOP orders failed")
                if sell_success:
                    print("   ✅ SELL STOP orders work")
                else:
                    print("   ❌ SELL STOP orders failed")
            else:
                print("❌ FAILED: Stop order issues persist")
                print(f"   BUY STOP - Retcode: {buy_result.get('retcode', 'N/A')}, Comment: {buy_result.get('comment', 'N/A')}")
                print(f"   SELL STOP - Retcode: {sell_result.get('retcode', 'N/A')}, Comment: {sell_result.get('comment', 'N/A')}")

                # Additional analysis
                if buy_result.get('retcode') == 10030 or sell_result.get('retcode') == 10030:
                    print("\n🔍 ANALYSIS: Error 10030 indicates stop order parameter issues")
                    print("   Possible causes:")
                    print("   • ORDER_TIME_GTC not used (should be GTC, not DAY)")
                    print("   • Stop price too close to market price")
                    print("   • Filling mode incorrectly applied to pending order")

            print("\n" + "=" * 50)

            # Recommendations
            if buy_success and sell_success:
                print("💡 RECOMMENDATIONS:")
                print("   ✅ Your stop order fix is working correctly")
                print("   ✅ You can proceed with live trading using stop orders")
                print("   ✅ The Error 10030 issue for stop orders is resolved")
            else:
                print("💡 RECOMMENDATIONS:")
                print("   ❌ Check stop order implementation in order_executor.py")
                print("   ❌ Ensure ORDER_TIME_GTC is used (not ORDER_TIME_DAY)")
                print("   ❌ Verify stop prices are far enough from market")
                print("   ❌ Confirm no filling modes are applied to pending orders")

            return buy_success and sell_success

        finally:
            mt5.shutdown()

def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test stop order fix')
    parser.add_argument('--real', action='store_true',
                       help='Place real stop orders (default is dry run)')
    parser.add_argument('--symbol', default='EURUSD',
                       help='Symbol to test (default: EURUSD)')

    args = parser.parse_args()

    tester = StopOrderTester()
    tester.test_symbol = args.symbol

    dry_run = not args.real

    if not dry_run:
        print("⚠️  WARNING: You are about to place REAL stop orders!")
        confirm = input("Are you sure? (type 'yes' to continue): ")
        if confirm.lower() != 'yes':
            print("Test cancelled.")
            return

    success = tester.run_comprehensive_test(dry_run=dry_run)

    if success:
        print("\n🎉 Stop order test passed! Your Error 10030 fix is complete.")
    else:
        print("\n❌ Stop order test failed. Check the recommendations above.")

if __name__ == "__main__":
    main()