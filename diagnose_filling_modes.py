import MetaTrader5 as mt5
import json
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FillingModeDiagnostic:
    """Diagnose which filling modes work for each symbol with TIOMarkets"""

    def __init__(self):
        self.results = {}
        self.symbols_to_test = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
            'XAUUSD', 'XAGUSD'  # Metals if available
        ]

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False

        logger.info("MT5 initialized successfully")
        return True

    def test_filling_mode(self, symbol: str, filling_mode: int, mode_name: str) -> Dict:
        """Test a specific filling mode for a symbol"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'supported': False, 'error': 'Cannot get tick data'}

            # Create test order request - try all modes regardless of symbol support
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": 0.01,  # Small test volume
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 10,
                "magic": 123456,
                "comment": f"Test {mode_name}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            # Use order_check instead of actual order
            result = mt5.order_check(request)

            if result and (result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 0):
                return {'supported': True, 'retcode': result.retcode}
            else:
                retcode = result.retcode if result else 'None'
                return {'supported': False, 'retcode': retcode, 'error': 'Order check failed'}

        except Exception as e:
            return {'supported': False, 'error': str(e)}

    def diagnose_symbol(self, symbol: str) -> Dict:
        """Diagnose all filling modes for a symbol"""
        logger.info(f"Diagnosing filling modes for {symbol}...")

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'symbol': symbol, 'error': 'Symbol not found'}

        # Test each filling mode
        filling_modes = [
            (mt5.ORDER_FILLING_FOK, "FOK"),
            (mt5.ORDER_FILLING_IOC, "IOC"),
            (mt5.ORDER_FILLING_RETURN, "RETURN"),
        ]

        results = {
            'symbol': symbol,
            'symbol_filling_modes': symbol_info.filling_mode,
            'modes': {},
            'recommended': None
        }

        for mode_value, mode_name in filling_modes:
            result = self.test_filling_mode(symbol, mode_value, mode_name)
            results['modes'][mode_name] = result

            if result.get('supported', False) and results['recommended'] is None:
                results['recommended'] = mode_name

        # If no mode works, default to RETURN
        if results['recommended'] is None:
            results['recommended'] = 'RETURN'

        return results

    def run_diagnosis(self) -> Dict:
        """Run diagnosis for all symbols"""
        if not self.initialize_mt5():
            return {'error': 'MT5 initialization failed'}

        logger.info("Starting filling mode diagnosis...")

        for symbol in self.symbols_to_test:
            try:
                result = self.diagnose_symbol(symbol)
                self.results[symbol] = result
                logger.info(f"✅ {symbol}: Recommended = {result.get('recommended', 'UNKNOWN')}")
            except Exception as e:
                logger.error(f"❌ Error diagnosing {symbol}: {e}")
                self.results[symbol] = {'symbol': symbol, 'error': str(e)}

        mt5.shutdown()

        # Save results
        self.save_results()

        return self.results

    def save_results(self):
        """Save diagnosis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filling_mode_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {filename}")

    def print_summary(self):
        """Print a summary of the diagnosis"""
        print("\n" + "="*60)
        print("FILLING MODE DIAGNOSIS SUMMARY")
        print("="*60)

        print(f"Tested {len(self.results)} symbols")
        print()

        for symbol, result in self.results.items():
            if 'error' in result:
                print(f"❌ {symbol}: {result['error']}")
                continue

            recommended = result.get('recommended', 'UNKNOWN')
            print(f"✅ {symbol}: Recommended = {recommended}")

            # Show working modes
            working_modes = []
            for mode_name, mode_result in result.get('modes', {}).items():
                if mode_result.get('supported', False):
                    working_modes.append(mode_name)

            if working_modes:
                print(f"   Working modes: {', '.join(working_modes)}")
            else:
                print("   No modes working - will use RETURN as fallback")

        print("\n" + "="*60)

def main():
    """Main diagnostic function"""
    diagnostic = FillingModeDiagnostic()
    results = diagnostic.run_diagnosis()

    if 'error' not in results:
        diagnostic.print_summary()
        print("\n🎯 RECOMMENDATION:")
        print("Update your order_executor.py to use dynamic filling mode detection")
        print("instead of hardcoded ORDER_FILLING_IOC")
    else:
        print(f"❌ Diagnosis failed: {results['error']}")

if __name__ == "__main__":
    main()