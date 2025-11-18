#!/usr/bin/env python3
"""
MT5 Price Issue Diagnostic Tool
Analyzes broker-specific requirements causing invalid price errors (10015)
"""

import MetaTrader5 as mt5
import json
import sys
import os
from datetime import datetime

class MT5PriceDiagnostic:
    def __init__(self):
        self.symbols = []
        self.load_config()

    def load_config(self):
        """Load trading symbols from config"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.symbols = config['trading']['symbols']
        except Exception as e:
            print(f"Error loading config: {e}")
            # Fallback to common symbols
            self.symbols = ["EURUSD", "GBPUSD", "XAUUSD", "XAGUSD"]

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        print("MT5 initialized successfully")
        return True

    def get_symbol_info(self, symbol):
        """Get detailed symbol information"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        # Debug: print available attributes
        # print(f"SymbolInfo attributes: {dir(info)}")

        return {
            'symbol': symbol,
            'point': info.point,
            'tick_size': getattr(info, 'trade_tick_size', info.point),
            'tick_value': getattr(info, 'trade_tick_value', 1.0),
            'contract_size': getattr(info, 'trade_contract_size', 100000),
            'min_lot': getattr(info, 'volume_min', 0.01),
            'max_lot': getattr(info, 'volume_max', 100.0),
            'lot_step': getattr(info, 'volume_step', 0.01),
            'spread': getattr(info, 'spread', 0),
            'stops_level': getattr(info, 'trade_stops_level', 0),
            'freeze_level': getattr(info, 'trade_freeze_level', 0),
            'margin_initial': getattr(info, 'margin_initial', 0),
            'margin_maintenance': getattr(info, 'margin_maintenance', 0),
            'digits': getattr(info, 'digits', 5),
            'description': getattr(info, 'description', symbol)
        }

    def get_current_prices(self, symbol):
        """Get current bid/ask prices"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid
        }

    def test_order_check(self, symbol, order_type, price, sl, tp, volume=0.01):
        """Test order_check with given parameters"""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 123456,
            "comment": "diagnostic test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_check(request)
        return {
            'request': request,
            'result': result,
            'error': mt5.last_error() if result is None else None
        }

    def calculate_minimum_distances(self, symbol_info, prices):
        """Calculate minimum required stop distances"""
        stops_level = symbol_info['stops_level']
        point = symbol_info['point']

        # Convert stops_level to price units
        min_stop_distance = stops_level * point

        return {
            'stops_level_points': stops_level,
            'min_stop_distance_price': min_stop_distance,
            'current_spread': prices['spread'] if prices else 0
        }

    def normalize_price(self, price, symbol_info):
        """Normalize price according to symbol's tick size"""
        tick_size = symbol_info['tick_size']
        if tick_size == 0:
            tick_size = symbol_info['point']

        # Round to nearest tick
        normalized = round(price / tick_size) * tick_size
        return normalized

    def test_filling_modes(self, symbol, order_type, price, sl, tp, volume=0.01):
        """Test different filling modes"""
        filling_modes = [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_FOK
        ]

        results = {}
        for filling in filling_modes:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 123456,
                "comment": f"test_{filling}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_check(request)
            results[filling] = {
                'result': result,
                'error': mt5.last_error() if result is None else None
            }

        return results

    def diagnose_symbol(self, symbol):
        """Complete diagnostic for a single symbol"""
        print(f"\n{'='*60}")
        print(f"DIAGNOSING: {symbol}")
        print(f"{'='*60}")

        # Get symbol info
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Cannot get symbol info for {symbol}")
            return None

        print(f"Symbol Info: {symbol_info['description']}")
        print(f"Digits: {symbol_info['digits']}, Point: {symbol_info['point']}")
        print(f"Tick Size: {symbol_info['tick_size']}, Spread: {symbol_info['spread']} points")
        print(f"Stops Level: {symbol_info['stops_level']} points")

        # Get current prices
        prices = self.get_current_prices(symbol)
        if prices is None:
            print(f"❌ Cannot get prices for {symbol}")
            return None

        print(f"Current Bid: {prices['bid']}, Ask: {prices['ask']}")
        print(f"Current Spread: {prices['spread']}")

        # Calculate minimum distances
        min_distances = self.calculate_minimum_distances(symbol_info, prices)
        print(f"Minimum Stop Distance: {min_distances['min_stop_distance_price']} price units")

        # Test BUY order
        print(f"\n--- Testing BUY Order ---")
        entry_price = prices['ask']
        normalized_entry = self.normalize_price(entry_price, symbol_info)
        sl_price = normalized_entry - (min_distances['min_stop_distance_price'] * 2)  # 2x minimum
        tp_price = normalized_entry + (min_distances['min_stop_distance_price'] * 4)  # 4x minimum for RR

        print(f"Entry (Ask): {entry_price} -> Normalized: {normalized_entry}")
        print(f"SL: {sl_price}, TP: {tp_price}")

        buy_test = self.test_order_check(symbol, mt5.ORDER_TYPE_BUY, normalized_entry, sl_price, tp_price)
        if buy_test['result'] and buy_test['result'].retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ BUY order check passed")
        else:
            print(f"❌ BUY order check failed: {buy_test['error']}")

        # Test SELL order
        print(f"\n--- Testing SELL Order ---")
        entry_price = prices['bid']
        normalized_entry = self.normalize_price(entry_price, symbol_info)
        sl_price = normalized_entry + (min_distances['min_stop_distance_price'] * 2)
        tp_price = normalized_entry - (min_distances['min_stop_distance_price'] * 4)

        print(f"Entry (Bid): {entry_price} -> Normalized: {normalized_entry}")
        print(f"SL: {sl_price}, TP: {tp_price}")

        sell_test = self.test_order_check(symbol, mt5.ORDER_TYPE_SELL, normalized_entry, sl_price, tp_price)
        if sell_test['result'] and sell_test['result'].retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ SELL order check passed")
        else:
            print(f"❌ SELL order check failed: {sell_test['error']}")

        # Test filling modes
        print(f"\n--- Testing Filling Modes ---")
        filling_results = self.test_filling_modes(symbol, mt5.ORDER_TYPE_BUY, normalized_entry, sl_price, tp_price)
        for filling, result in filling_results.items():
            filling_name = {mt5.ORDER_FILLING_IOC: 'IOC', mt5.ORDER_FILLING_RETURN: 'RETURN', mt5.ORDER_FILLING_FOK: 'FOK'}[filling]
            if result['result'] and result['result'].retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ {filling_name}: OK")
            else:
                print(f"❌ {filling_name}: {result['error']}")

        return {
            'symbol_info': symbol_info,
            'prices': prices,
            'min_distances': min_distances,
            'buy_test': buy_test,
            'sell_test': sell_test,
            'filling_tests': filling_results
        }

    def run_diagnostic(self):
        """Run complete diagnostic for all symbols"""
        if not self.initialize_mt5():
            return

        print(f"Starting MT5 Price Diagnostic for {len(self.symbols)} symbols")
        print(f"Timestamp: {datetime.now()}")

        results = {}
        for symbol in self.symbols:
            result = self.diagnose_symbol(symbol)
            if result:
                results[symbol] = result

        # Summary
        print(f"\n{'='*60}")
        print("DIAGNOSTIC SUMMARY")
        print(f"{'='*60}")

        successful_symbols = []
        failed_symbols = []

        for symbol, result in results.items():
            buy_ok = result['buy_test']['result'] and result['buy_test']['result'].retcode == mt5.TRADE_RETCODE_DONE
            sell_ok = result['sell_test']['result'] and result['sell_test']['result'].retcode == mt5.TRADE_RETCODE_DONE

            if buy_ok and sell_ok:
                successful_symbols.append(symbol)
                print(f"✅ {symbol}: OK")
            else:
                failed_symbols.append(symbol)
                print(f"❌ {symbol}: FAILED")

        print(f"\nSuccessful symbols: {len(successful_symbols)}/{len(results)}")
        print(f"Failed symbols: {len(failed_symbols)}/{len(results)}")

        if failed_symbols:
            print(f"\nFailed symbols: {', '.join(failed_symbols)}")
            print("\nCommon issues:")
            print("- Stop distances too small (check stops_level)")
            print("- Prices not normalized to tick size")
            print("- Wrong filling mode for broker")
            print("- Using bid for BUY or ask for SELL")

        mt5.shutdown()
        return results

if __name__ == "__main__":
    diagnostic = MT5PriceDiagnostic()
    results = diagnostic.run_diagnostic()

    # Save results to file
    output_file = "diagnostic_results.txt"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to {output_file}")