@echo off
REM *** EMERGENCY STOP *** - USE WITH EXTREME CAUTION
REM *** Closes ALL open positions AND cancels ALL pending orders ***
REM *** Affects ALL EAs and manual trades in MT5 - NOT just FX-Ai ***

echo.
echo ============================================================================
echo                *** FX-AI EMERGENCY STOP ***
echo        *** IMMEDIATELY CLOSING ALL POSITIONS AND CANCELING ALL ORDERS ***
echo        *** THIS AFFECTS ALL TRADING SYSTEMS AND MANUAL TRADES ***
echo ============================================================================
echo.

REM Change to parent directory (FX-Ai root)
cd /d "%~dp0.."

REM Check if main.py exists (verify we're in correct directory)
if not exist "main.py" (
    echo [ERROR] main.py not found!
    echo Please run this script from your FX-Ai directory
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run emergency stop script
echo [INFO] Executing emergency stop...
echo.
python live_trading\emergency_stop.py

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Emergency stop completed successfully
) else (
    echo.
    echo [WARNING] Emergency stop completed with warnings
)

echo.
pause
