"""
FX-Ai Trading Engine Fix
Replaces the broken _execute_trade_safe method
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def fix_execute_trade_safe(trading_engine_instance):
    """
    Monkey patch to fix the _execute_trade_safe method
    This replaces the broken implementation with a working one
    """
    
    async def _execute_trade_safe(self, symbol: str, direction: str, volume: float, 
                                  stop_loss: Optional[float] = None, 
                                  take_profit: Optional[float] = None,
                                  comment: str = "") -> Dict:
        """Fixed implementation of trade execution"""
        
        try:
            # Initialize MT5 if needed
            if not mt5.initialize():
                return {'success': False, 'error': 'MT5 not initialized'}
                
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
                
            # Ensure symbol is selected
            if not symbol_info.select:
                if not mt5.symbol_select(symbol, True):
                    return {'success': False, 'error': f'Cannot select {symbol}'}
                    
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': f'No tick data for {symbol}'}
                
            # Determine order type and price
            if direction.upper() == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            elif direction.upper() == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                return {'success': False, 'error': f'Invalid direction: {direction}'}
                
            # CRITICAL FIX: Determine the correct filling mode for this broker
            # TIOMarkets uses FOK for most symbols
            filling_mode = mt5.ORDER_FILLING_FOK
            
            # Check if symbol actually supports this filling mode
            if not (symbol_info.filling_mode & 1):  # FOK is bit 0
                # Try IOC if FOK not supported
                if symbol_info.filling_mode & 2:  # IOC is bit 1
                    filling_mode = mt5.ORDER_FILLING_IOC
                # Try RETURN if neither FOK nor IOC
                elif symbol_info.filling_mode & 4:  # RETURN is bit 2
                    filling_mode = mt5.ORDER_FILLING_RETURN
                else:
                    # No supported filling mode!
                    return {'success': False, 'error': f'No supported filling mode for {symbol}'}
                    
            # Build the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": round(volume, 2),
                "type": order_type,
                "price": price,
                "deviation": 10,  # Allow 10 points slippage
                "magic": getattr(self, 'magic_number', 12345),
                "comment": comment or "FX-Ai",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode  # Use the correct filling mode!
            }
            
            # Add SL/TP if provided
            if stop_loss is not None:
                request["sl"] = round(stop_loss, symbol_info.digits)
            if take_profit is not None:
                request["tp"] = round(take_profit, symbol_info.digits)
                
            logger.info(f"📤 Sending order: {symbol} {direction} {volume} @ {price:.5f} (filling: {filling_mode})")
            
            # Send the order
            result = mt5.order_send(request)
            
            # Check result
            if result is None:
                logger.error(f"❌ Order send returned None for {symbol}")
                return {'success': False, 'error': 'Order send failed - no response'}
                
            # Success codes
            if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE_PARTIAL]:
                logger.info(f"✅ Order executed successfully: {symbol} ticket {result.order}")
                return {
                    'success': True,
                    'ticket': result.order,
                    'price': result.price if hasattr(result, 'price') else price,
                    'volume': result.volume if hasattr(result, 'volume') else volume,
                    'symbol': symbol,
                    'direction': direction
                }
            else:
                # Order failed
                error_msg = f"Order failed - Retcode: {result.retcode}, Comment: {getattr(result, 'comment', 'No comment')}"
                logger.error(f"❌ {error_msg}")
                
                # If Error 10030, provide specific guidance
                if result.retcode == 10030:
                    error_msg += " (Invalid filling mode - broker requires different mode)"
                    
                return {
                    'success': False,
                    'error': error_msg,
                    'symbol': symbol,
                    'direction': direction,
                    'retcode': result.retcode,
                    'comment': getattr(result, 'comment', 0)
                }
                
        except Exception as e:
            logger.error(f"Exception in _execute_trade_safe: {str(e)}")
            return {
                'success': False,
                'error': f'Exception: {str(e)}',
                'symbol': symbol,
                'direction': direction
            }
    
    # Apply the patch
    trading_engine_instance._execute_trade_safe = _execute_trade_safe.__get__(trading_engine_instance)
    print("✅ Fixed _execute_trade_safe method applied")
    return trading_engine_instance


def apply_emergency_fix():
    """
    Emergency fix to apply to the running system
    Add this to your main FX-Ai file after initializing TradingEngine
    """
    
    code = '''
# EMERGENCY FIX FOR ERROR 10030
# Add this after initializing your TradingEngine

from execute_trade_fix import fix_execute_trade_safe

# Apply the fix
trading_engine = fix_execute_trade_safe(trading_engine)
print("✅ Emergency fix applied to TradingEngine")

# Also ensure config uses market orders
if hasattr(orchestrator, 'config'):
    orchestrator.config['trading']['order_management']['default_entry_strategy'] = 'market'
    print("✅ Switched to market orders")
'''
    
    print("ADD THIS TO YOUR MAIN FILE:")
    print("=" * 60)
    print(code)
    print("=" * 60)


# Also create a config updater
def update_config_for_market_orders():
    """Update config to use market orders"""
    
    import json
    
    config_paths = [
        'config.json',
        'C:\\Users\\andyc\\python\\FX-Ai\\config.json',
    ]
    
    for path in config_paths:
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                
            # Update to market orders
            if 'trading' not in config:
                config['trading'] = {}
            if 'order_management' not in config['trading']:
                config['trading']['order_management'] = {}
                
            config['trading']['order_management']['default_entry_strategy'] = 'market'
            
            # Save
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"✅ Updated {path} to use market orders")
            return True
            
        except:
            continue
            
    print("❌ Could not find config.json to update")
    return False


if __name__ == "__main__":
    print("=" * 80)
    print("FX-Ai EMERGENCY FIX FOR ERROR 10030")
    print("=" * 80)
    
    print("\n1. Updating config to use market orders...")
    update_config_for_market_orders()
    
    print("\n2. Instructions for applying the fix:")
    apply_emergency_fix()
    
    print("\nRESTART FX-Ai after applying these fixes!")
