@echo off
REM FX-Ai Risk Display Script
REM Displays current risk management parameters

echo.
echo ============================================
echo    FX-Ai Risk Management Display
echo ============================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found, using system Python...
)

REM Run the risk display script
python risk_display.py

REM Pause to keep window open
echo.
echo Press any key to exit...
pause >nul