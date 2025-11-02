@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python live_trading\symbol_selector.py
pause
