"""
Debug Logging Helper
Adds comprehensive logging to track signal-to-trade flow
Add this to your core/trading_engine.py or main.py
"""

import logging
from datetime import datetime

# Configure ultra-verbose logging
def setup_debug_logging():
    """Setup detailed logging for debugging"""
    
    # Create logger
    logger = logging.getLogger('FX-Ai-Debug')
    logger.setLevel(logging.DEBUG)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler('debug_trace.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize debug logger
debug_logger = setup_debug_logging()

def trace_checkpoint(checkpoint_name, data=None):
    """Log a checkpoint with optional data"""
    debug_logger.critical(f"{'='*80}")
    debug_logger.critical(f"CHECKPOINT: {checkpoint_name}")
    if data:
        for key, value in data.items():
            debug_logger.critical(f"  {key}: {value}")
    debug_logger.critical(f"{'='*80}")

# ============================================================================
# ADD THESE CHECKPOINTS TO YOUR TRADING LOOP
# ============================================================================

"""
# Example usage in your trading loop:

def trading_loop():
    trace_checkpoint("TRADING_LOOP_START")
    
    for symbol in symbols:
        trace_checkpoint("SYMBOL_PROCESSING", {
            "symbol": symbol,
            "time": datetime.now()
        })
        
        # Get market data
        data = get_market_data(symbol)
        trace_checkpoint("MARKET_DATA_RETRIEVED", {
            "symbol": symbol,
            "bid": data.bid,
            "ask": data.ask
        })
        
        # Technical analysis
        technical_score = calculate_technical(data)
        trace_checkpoint("TECHNICAL_CALCULATED", {
            "symbol": symbol,
            "score": technical_score
        })
        
        # ML prediction
        ml_prediction = get_ml_prediction(data)
        trace_checkpoint("ML_PREDICTION", {
            "symbol": symbol,
            "prediction": ml_prediction,
            "confidence": ml_prediction.confidence
        })
        
        # Generate signal
        signal = generate_signal(technical_score, ml_prediction)
        trace_checkpoint("SIGNAL_GENERATED", {
            "symbol": symbol,
            "strength": signal.strength,
            "direction": signal.direction,
            "threshold": 0.5
        })
        
        # Check threshold
        if signal.strength < 0.5:
            trace_checkpoint("SIGNAL_REJECTED_LOW_STRENGTH", {
                "symbol": symbol,
                "strength": signal.strength,
                "threshold": 0.5
            })
            continue
        
        trace_checkpoint("SIGNAL_PASSED_THRESHOLD", {
            "symbol": symbol,
            "strength": signal.strength
        })
        
        # Calculate stops
        entry = signal.entry_price
        sl = calculate_sl(data)
        tp = calculate_tp(data)
        
        trace_checkpoint("STOPS_CALCULATED", {
            "symbol": symbol,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_distance": abs(entry - sl),
            "tp_distance": abs(tp - entry)
        })
        
        # Validate risk/reward
        risk_reward_ratio = abs(tp - entry) / abs(entry - sl)
        
        trace_checkpoint("RISK_REWARD_CALCULATED", {
            "symbol": symbol,
            "ratio": f"{risk_reward_ratio:.2f}:1",
            "required": "3.0:1"
        })
        
        if risk_reward_ratio < 3.0:
            trace_checkpoint("SIGNAL_REJECTED_LOW_RR", {
                "symbol": symbol,
                "ratio": risk_reward_ratio,
                "required": 3.0
            })
            continue
        
        trace_checkpoint("SIGNAL_VALIDATED", {
            "symbol": symbol,
            "strength": signal.strength,
            "risk_reward": f"{risk_reward_ratio:.2f}:1",
            "direction": signal.direction
        })
        
        # ============================================
        # 🔴 CRITICAL: THIS IS WHERE EXECUTION SHOULD HAPPEN
        # ============================================
        
        trace_checkpoint("ABOUT_TO_EXECUTE_TRADE", {
            "symbol": symbol,
            "direction": signal.direction,
            "entry": entry,
            "sl": sl,
            "tp": tp
        })
        
        # Check if execute_trade function exists
        try:
            # OPTION 1: If you have an execute_trade method
            result = execute_trade(symbol, signal, entry, sl, tp)
            
            trace_checkpoint("EXECUTE_TRADE_CALLED", {
                "symbol": symbol,
                "result": result
            })
            
        except NameError:
            # If execute_trade doesn't exist
            trace_checkpoint("EXECUTE_TRADE_NOT_FOUND", {
                "symbol": symbol,
                "error": "execute_trade() function not found!"
            })
            
            # OPTION 2: Try direct MT5 call
            try:
                import MetaTrader5 as mt5
                
                trace_checkpoint("ATTEMPTING_DIRECT_MT5_ORDER", {
                    "symbol": symbol
                })
                
                lot_size = calculate_lot_size(symbol, risk=50.0, sl_distance=abs(entry-sl))
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": entry,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": "FX-Ai trade",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                trace_checkpoint("ORDER_REQUEST_PREPARED", {
                    "symbol": symbol,
                    "request": request
                })
                
                result = mt5.order_send(request)
                
                trace_checkpoint("MT5_ORDER_SEND_CALLED", {
                    "symbol": symbol,
                    "result": result
                })
                
                if result is None:
                    trace_checkpoint("ORDER_SEND_RETURNED_NONE", {
                        "symbol": symbol,
                        "error": mt5.last_error()
                    })
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    trace_checkpoint("ORDER_REJECTED", {
                        "symbol": symbol,
                        "retcode": result.retcode,
                        "comment": result.comment
                    })
                else:
                    trace_checkpoint("ORDER_SUCCESS", {
                        "symbol": symbol,
                        "ticket": result.order,
                        "volume": result.volume,
                        "price": result.price
                    })
                    
            except Exception as e:
                trace_checkpoint("MT5_ORDER_EXCEPTION", {
                    "symbol": symbol,
                    "error": str(e)
                })
        
        except Exception as e:
            trace_checkpoint("EXECUTE_TRADE_EXCEPTION", {
                "symbol": symbol,
                "error": str(e)
            })
    
    trace_checkpoint("TRADING_LOOP_END")
"""

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    FX-AI DEBUG LOGGING HELPER                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  This module provides comprehensive checkpoint logging to track       ║
║  the signal-to-trade execution flow.                                  ║
║                                                                       ║
║  HOW TO USE:                                                          ║
║                                                                       ║
║  1. Copy the trace_checkpoint() function to your code                 ║
║  2. Add checkpoints at key points in your trading loop                ║
║  3. Run your trading system                                           ║
║  4. Check debug_trace.log for detailed execution flow                 ║
║                                                                       ║
║  The example code above shows where to place checkpoints.             ║
║                                                                       ║
║  KEY CHECKPOINTS TO ADD:                                              ║
║  - After signal generation                                            ║
║  - After threshold check                                              ║
║  - After stops calculation                                            ║
║  - After risk/reward validation                                       ║
║  - IMMEDIATELY before trade execution call                            ║
║  - After trade execution call                                         ║
║                                                                       ║
║  OUTPUT FILES:                                                        ║
║  - debug_trace.log: Complete detailed log                             ║
║  - Console: Color-coded checkpoint messages                           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

EXAMPLE CHECKPOINT OUTPUT:

================================================================================
CHECKPOINT: SIGNAL_VALIDATED
  symbol: EURUSD
  strength: 0.627
  risk_reward: 3.0:1
  direction: SELL
================================================================================

================================================================================
CHECKPOINT: ABOUT_TO_EXECUTE_TRADE
  symbol: EURUSD
  direction: SELL
  entry: 1.15668
  sl: 1.15998
  tp: 1.14678
================================================================================

🔴 IF YOU SEE THE SECOND CHECKPOINT BUT NO "EXECUTE_TRADE_CALLED" CHECKPOINT,
   THEN THE PROBLEM IS CONFIRMED: MISSING TRADE EXECUTION CODE!

""")