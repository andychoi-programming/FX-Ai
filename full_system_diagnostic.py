import MetaTrader5 as mt5
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullSystemDiagnostic:
    """Comprehensive diagnostic for MT5 filling mode issues"""

    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'mt5_connection': {},
            'symbols_tested': {},
            'filling_modes': {},
            'recommendations': [],
            'errors': []
        }

        # Test symbols commonly used in FX trading
        self.test_symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
            'XAUUSD', 'XAGUSD',  # Metals
            'BTCUSD', 'ETHUSD'   # Crypto (if available)
        ]

    def initialize_mt5(self) -> bool:
        """Initialize MT5 and test connection"""
        try:
            logger.info("Initializing MT5...")
            if not mt5.initialize():
                error = "Failed to initialize MT5"
                logger.error(error)
                self.results['errors'].append(error)
                return False

            # Test terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                self.results['mt5_connection'] = {
                    'terminal_name': terminal_info.name,
                    'company': terminal_info.company,
                    'trade_allowed': terminal_info.trade_allowed,
                    'connected': terminal_info.connected,
                    'filling_flags': getattr(terminal_info, 'trade_filling_flags', 'N/A')
                }
                logger.info(f"✅ MT5 connected to: {terminal_info.name} ({terminal_info.company})")
            else:
                error = "Failed to get terminal info"
                logger.error(error)
                self.results['errors'].append(error)
                return False

            return True

        except Exception as e:
            error = f"MT5 initialization error: {str(e)}"
            logger.error(error)
            self.results['errors'].append(error)
            return False

    def test_symbol_availability(self, symbol: str) -> Dict:
        """Test if symbol is available and get its info"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'available': False, 'error': 'Symbol not found'}

            # Try to select symbol
            if not mt5.symbol_select(symbol, True):
                return {'available': False, 'error': 'Failed to select symbol'}

            return {
                'available': True,
                'symbol_info': {
                    'name': symbol_info.name,
                    'description': symbol_info.description,
                    'filling_mode': symbol_info.filling_mode,
                    'filling_mode_binary': bin(symbol_info.filling_mode),
                    'trade_contract_size': symbol_info.trade_contract_size,
                    'volume_min': symbol_info.volume_min,
                    'volume_max': symbol_info.volume_max,
                    'volume_step': symbol_info.volume_step,
                    'point': symbol_info.point,
                    'digits': symbol_info.digits
                }
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def test_filling_mode(self, symbol: str, filling_mode: int, mode_name: str) -> Dict:
        """Test a specific filling mode for a symbol"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'tested': False, 'error': 'Cannot get tick data'}

            # Create test order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": 0.01,  # Small test volume
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 10,
                "magic": 123456,
                "comment": f"Diagnostic {mode_name}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            # Use order_check for safe testing
            result = mt5.order_check(request)

            if result and (result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 0):
                return {
                    'tested': True,
                    'works': True,
                    'retcode': result.retcode,
                    'comment': getattr(result, 'comment', 'Success')
                }
            else:
                retcode = result.retcode if result else 'None'
                comment = getattr(result, 'comment', 'Unknown') if result else 'No result'
                return {
                    'tested': True,
                    'works': False,
                    'retcode': retcode,
                    'comment': comment,
                    'error': f'Failed with retcode {retcode}'
                }

        except Exception as e:
            return {'tested': False, 'error': str(e)}

    def diagnose_symbol(self, symbol: str) -> Dict:
        """Run complete diagnosis for a symbol"""
        logger.info(f"🔍 Diagnosing {symbol}...")

        symbol_result = {
            'symbol': symbol,
            'availability': self.test_symbol_availability(symbol),
            'filling_modes': {}
        }

        # Only test filling modes if symbol is available
        if symbol_result['availability'].get('available', False):
            filling_tests = [
                (mt5.ORDER_FILLING_FOK, "FOK"),
                (mt5.ORDER_FILLING_IOC, "IOC"),
                (mt5.ORDER_FILLING_RETURN, "RETURN"),
            ]

            working_modes = []

            for mode_value, mode_name in filling_tests:
                result = self.test_filling_mode(symbol, mode_value, mode_name)
                symbol_result['filling_modes'][mode_name] = result

                if result.get('works', False):
                    working_modes.append(mode_name)

            symbol_result['working_modes'] = working_modes
            symbol_result['recommended_mode'] = working_modes[0] if working_modes else None

            if not working_modes:
                symbol_result['issue'] = "No filling modes work - this will cause 10030 errors"
                self.results['errors'].append(f"{symbol}: No working filling modes")

        return symbol_result

    def analyze_results(self):
        """Analyze diagnostic results and provide recommendations"""
        logger.info("📊 Analyzing results...")

        # Count issues
        total_symbols = len(self.results['symbols_tested'])
        available_symbols = sum(1 for s in self.results['symbols_tested'].values()
                              if s['availability'].get('available', False))
        working_symbols = sum(1 for s in self.results['symbols_tested'].values()
                            if s.get('working_modes'))

        # Generate recommendations
        if available_symbols < total_symbols:
            self.results['recommendations'].append(
                f"⚠️ Only {available_symbols}/{total_symbols} symbols are available. Check symbol subscriptions."
            )

        if working_symbols < available_symbols:
            failed_count = available_symbols - working_symbols
            self.results['recommendations'].append(
                f"🚨 {failed_count} symbols have filling mode issues that will cause 10030 errors."
            )

        # Check for common patterns
        all_recommended = [s.get('recommended_mode') for s in self.results['symbols_tested'].values()
                          if s.get('recommended_mode')]

        if all_recommended:
            most_common = max(set(all_recommended), key=all_recommended.count)
            self.results['recommendations'].append(
                f"💡 Most symbols work with {most_common} filling mode. Consider using this as default."
            )

        # Specific broker recommendations
        terminal_name = self.results['mt5_connection'].get('terminal_name', '').lower()
        if 'tiomarkets' in terminal_name or 'tio' in terminal_name:
            self.results['recommendations'].append(
                "🎯 TIOMarkets detected: Use ORDER_FILLING_RETURN (2) or dynamic detection."
            )

    def run_full_diagnostic(self) -> Dict:
        """Run complete system diagnostic"""
        logger.info("🚀 Starting Full System Diagnostic for MT5 Filling Mode Issues")
        logger.info("=" * 70)

        if not self.initialize_mt5():
            return self.results

        # Test all symbols
        for symbol in self.test_symbols:
            try:
                result = self.diagnose_symbol(symbol)
                self.results['symbols_tested'][symbol] = result

                status = "✅" if result.get('recommended_mode') else "❌"
                logger.info(f"{status} {symbol}: {result.get('recommended_mode', 'FAILED')}")

            except Exception as e:
                logger.error(f"Error diagnosing {symbol}: {e}")
                self.results['symbols_tested'][symbol] = {'error': str(e)}

        # Analyze results
        self.analyze_results()

        # Save results
        self.save_results()

        mt5.shutdown()
        return self.results

    def save_results(self):
        """Save diagnostic results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"📄 Results saved to {filename}")

    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "="*70)
        print("FULL SYSTEM DIAGNOSTIC SUMMARY")
        print("="*70)

        # Connection info
        conn = self.results['mt5_connection']
        if conn:
            print(f"🔗 MT5 Connection: {conn.get('terminal_name', 'Unknown')}")
            print(f"🏢 Broker: {conn.get('company', 'Unknown')}")
            print(f"📊 Trade Allowed: {conn.get('trade_allowed', 'Unknown')}")
        else:
            print("❌ MT5 Connection: FAILED")

        print()

        # Symbol summary
        total = len(self.results['symbols_tested'])
        available = sum(1 for s in self.results['symbols_tested'].values()
                       if s['availability'].get('available', False))
        working = sum(1 for s in self.results['symbols_tested'].values()
                     if s.get('working_modes'))

        print(f"📈 Symbol Analysis:")
        print(f"   Total tested: {total}")
        print(f"   Available: {available}")
        print(f"   Working filling modes: {working}")

        print()

        # Recommendations
        if self.results['recommendations']:
            print("💡 RECOMMENDATIONS:")
            for rec in self.results['recommendations']:
                print(f"   • {rec}")

        print()

        # Errors
        if self.results['errors']:
            print("❌ ISSUES FOUND:")
            for error in self.results['errors'][:5]:  # Show first 5
                print(f"   • {error}")
            if len(self.results['errors']) > 5:
                print(f"   • ... and {len(self.results['errors']) - 5} more")

        print("\n" + "="*70)

def main():
    """Main diagnostic function"""
    print("🔧 FX-Ai Full System Diagnostic for Error 10030")
    print("This will test MT5 connection and filling modes for all symbols")
    print()

    diagnostic = FullSystemDiagnostic()
    results = diagnostic.run_full_diagnostic()

    if results:
        diagnostic.print_summary()
    else:
        print("❌ Diagnostic failed to complete")

if __name__ == "__main__":
    main()