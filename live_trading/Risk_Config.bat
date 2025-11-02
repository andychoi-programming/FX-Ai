@echo off
REM FX-Ai Interactive Risk Configuration Script
REM Allows viewing and modifying risk management parameters

echo.
echo ============================================
echo    FX-Ai Risk Management Configuration
echo ============================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found, using system Python...
)

REM Run the risk configuration script
python risk_config.py

REM Pause to keep window open
echo.
echo Press any key to exit...
pause >nul