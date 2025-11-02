2@echo off
REM FX-Ai Trading System Startup Script
REM Version 1.5.0 - November 2, 2025

REM Change to parent directory (FX-Ai root)
cd /d "%~dp0.."

echo ========================================
echo   FX-Ai Trading System Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if we're in the right directory
if not exist "main.py" (
    echo [ERROR] main.py not found in current directory
    echo Current directory: %CD%
    echo Expected to find main.py in FX-Ai root directory
    echo.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    echo Trying alternative activation method...
    call venv\Scripts\activate
    if errorlevel 1 (
        echo [ERROR] Could not activate virtual environment
        echo Please check your Python installation
        pause
        exit /b 1
    )
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip (silently)
echo [INFO] Ensuring pip is up to date...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Could not upgrade pip, continuing...
)
echo.

REM Install requirements
if exist "requirements.txt" (
    echo [INFO] Installing Python dependencies...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo Check requirements.txt and your internet connection
        echo.
        echo Try running: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [WARNING] requirements.txt not found
    echo Please ensure all dependencies are installed manually
)
echo.

REM Create necessary directories
if not exist "config" (
    echo [INFO] Creating config directory...
    mkdir config 2>nul
)

if not exist "logs" (
    echo [INFO] Creating logs directory...
    mkdir logs 2>nul
)

if not exist "models" (
    echo [INFO] Creating models directory...
    mkdir models 2>nul
)

REM Check for config file and .env (v1.5.0+)
if not exist ".env" (
    echo [WARNING] .env file not found
    echo Please copy .env.example to .env and add your MT5 credentials
    echo.
    echo Run: copy .env.example .env
    echo Then edit .env with your credentials
    echo.
)

if not exist "config\config.json" (
    echo [WARNING] config\config.json not found
    echo Please ensure config.json exists in the config directory
    echo.
)

REM Check for MT5 installation
echo [INFO] Checking MT5 installation...
python -c "import MetaTrader5 as mt5; print('[OK] MT5 module available')" 2>nul
if errorlevel 1 (
    echo [WARNING] MT5 module not properly installed
    echo Please ensure MetaTrader5 is correctly installed
)

REM Run diagnostic if requested
if "%1"=="diagnostic" (
    echo [INFO] Running MT5 diagnostic...
    python mt5_diagnostic.py
    goto :end
)

REM Run debug mode if requested
if "%1"=="debug" (
    echo [INFO] Running in debug mode...
    python trading_debug.py
    goto :end
)

REM Run the main application
echo ========================================
echo   Starting FX-Ai Trading System
echo ========================================
echo.
echo [INFO] Starting main application...
echo Press Ctrl+C to stop the system
echo.
python main.py

:end
echo.
echo ========================================
echo   FX-Ai System Stopped
echo ========================================
echo.
pause