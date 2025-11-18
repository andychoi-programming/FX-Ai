@echo off
REM *** FX-AI EMERGENCY STOP ***
REM *** IMMEDIATELY CLOSES ALL POSITIONS AND CANCELS ALL ORDERS ***
REM *** THIS AFFECTS ALL TRADING SYSTEMS AND MANUAL TRADES ***

echo.
echo ============================================================================
echo                *** FX-AI EMERGENCY STOP ***
echo        *** IMMEDIATELY CLOSING ALL POSITIONS AND CANCELING ALL ORDERS ***
echo        *** THIS AFFECTS ALL TRADING SYSTEMS AND MANUAL TRADES ***
echo ============================================================================
echo.

REM Change to parent directory (FX-Ai root)
cd /d "%~dp0.."

echo Current directory: %CD%
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
) else (
    echo Warning: Virtual environment not found
    echo Continuing without activation...
    echo.
)

echo Executing emergency stop...
echo.

REM Use the unified launcher for emergency stop
python fxai.py emergency-stop

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✓ Emergency stop completed successfully
) else (
    echo ✗ Emergency stop encountered an error
)

echo.
echo Press any key to exit...
pause >nul