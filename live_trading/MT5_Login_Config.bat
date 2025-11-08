@echo off
cd /d "%~dp0\.."
call venv\Scripts\activate.bat
python live_trading\mt5_config.py
pause
