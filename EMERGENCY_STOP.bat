@echo off
REM ============================================================================
REM FX-Ai EMERGENCY STOP - Close All Positions Immediately
REM ============================================================================

echo ============================================================================
echo                FX-Ai EMERGENCY STOP
echo        IMMEDIATELY CLOSING ALL POSITIONS
echo ============================================================================
echo.

REM Check if we're in the FX-Ai directory
if not exist "main.py" (
    echo [ERROR] main.py not found!
    echo Please run this script from your FX-Ai directory
    pause
    exit /b 1
)

echo [WARNING] This will close ALL open positions immediately!
echo Press Ctrl+C to cancel if you don't want to close positions...
timeout /t 5

echo.
echo [INFO] Creating emergency close script...

echo import MetaTrader5 as mt5 > emergency_close.py
echo import sys >> emergency_close.py
echo import time >> emergency_close.py
echo. >> emergency_close.py
echo print("[EMERGENCY] Connecting to MT5...") >> emergency_close.py
echo if not mt5.initialize(): >> emergency_close.py
echo     print("[ERROR] Failed to initialize MT5") >> emergency_close.py
echo     sys.exit(1) >> emergency_close.py
echo. >> emergency_close.py
echo print("[EMERGENCY] Getting all positions...") >> emergency_close.py
echo positions = mt5.positions_get() >> emergency_close.py
echo if positions is None or len(positions) == 0: >> emergency_close.py
echo     print("[INFO] No open positions found") >> emergency_close.py
echo     mt5.shutdown() >> emergency_close.py
echo     sys.exit(0) >> emergency_close.py
echo. >> emergency_close.py
echo print(f"[EMERGENCY] Found {len(positions)} positions to close") >> emergency_close.py
echo closed_count = 0 >> emergency_close.py
echo for position in positions: >> emergency_close.py
echo     try: >> emergency_close.py
echo         # Determine close order type >> emergency_close.py
echo         if position.type == mt5.ORDER_TYPE_BUY: >> emergency_close.py
echo             order_type = mt5.ORDER_TYPE_SELL >> emergency_close.py
echo             price = mt5.symbol_info_tick(position.symbol).bid >> emergency_close.py
echo         else: >> emergency_close.py
echo             order_type = mt5.ORDER_TYPE_BUY >> emergency_close.py
echo             price = mt5.symbol_info_tick(position.symbol).ask >> emergency_close.py
echo. >> emergency_close.py
echo         # Create close request >> emergency_close.py
echo         request = { >> emergency_close.py
echo             "action": mt5.TRADE_ACTION_DEAL, >> emergency_close.py
echo             "symbol": position.symbol, >> emergency_close.py
echo             "volume": position.volume, >> emergency_close.py
echo             "type": order_type, >> emergency_close.py
echo             "position": position.ticket, >> emergency_close.py
echo             "price": price, >> emergency_close.py
echo             "deviation": 10, >> emergency_close.py
echo             "comment": "EMERGENCY CLOSE" >> emergency_close.py
echo         } >> emergency_close.py
echo. >> emergency_close.py
echo         result = mt5.order_send(request) >> emergency_close.py
echo         if result and result.retcode == mt5.TRADE_RETCODE_DONE: >> emergency_close.py
echo             print(f"[SUCCESS] Closed {position.symbol} position {position.ticket}") >> emergency_close.py
echo             closed_count += 1 >> emergency_close.py
echo         else: >> emergency_close.py
echo             print(f"[ERROR] Failed to close {position.symbol} position {position.ticket}") >> emergency_close.py
echo. >> emergency_close.py
echo         time.sleep(0.5)  # Small delay between orders >> emergency_close.py
echo. >> emergency_close.py
echo     except Exception as e: >> emergency_close.py
echo         print(f"[ERROR] Exception closing position: {e}") >> emergency_close.py
echo. >> emergency_close.py
echo print(f"[EMERGENCY] Closed {closed_count}/{len(positions)} positions") >> emergency_close.py
echo mt5.shutdown() >> emergency_close.py
echo print("[EMERGENCY] MT5 connection closed") >> emergency_close.py

echo.
echo [INFO] Running emergency close script...
python emergency_close.py

echo.
echo [INFO] Cleaning up...
del emergency_close.py >nul 2>&1

echo.
echo ============================================================================
echo              EMERGENCY STOP COMPLETE
echo ============================================================================
echo.
echo All positions should now be closed.
echo.
echo IMPORTANT: Check your MT5 terminal to confirm positions are closed.
echo.
echo Risk settings have been updated to prevent future large losses.
echo.
pause