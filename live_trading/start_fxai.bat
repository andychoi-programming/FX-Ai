@echo off
REM FX-Ai Live Trading Launcher
REM This batch file now uses the unified launcher for consistency

cd /d "%~dp0\.."

echo ================================================
echo         FX-Ai Live Trading System
echo ================================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at venv\Scripts\activate.bat
    echo Continuing without activation...
)

echo.
echo Starting FX-Ai Live Trading...
echo.

REM Use the unified launcher
python fxai.py run live

echo.
echo FX-Ai Live Trading session ended.
echo.

pause