@echo off
REM ============================================
REM FX-Ai VS Code Performance Optimizer
REM ============================================

echo ================================
echo FX-Ai VS Code Optimizer
echo ================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Not running as administrator
    echo Some operations may fail
    echo.
)

echo Step 1: Creating backup directory...
if not exist "D:\FX-Ai-Backups" mkdir "D:\FX-Ai-Backups"
set BACKUP_DIR=D:\FX-Ai-Backups\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%
set BACKUP_DIR=%BACKUP_DIR: =0%
mkdir "%BACKUP_DIR%"
echo Backup directory: %BACKUP_DIR%
echo.

echo Step 2: Backing up current settings...
if exist "%USERPROFILE%\AppData\Roaming\Code\User\settings.json" (
    copy "%USERPROFILE%\AppData\Roaming\Code\User\settings.json" "%BACKUP_DIR%\settings_backup.json" >nul
    echo   [OK] Settings backed up
) else (
    echo   [INFO] No settings file found
)
echo.

echo Step 3: Cleaning Python cache...
if exist "C:\Users\andyc\python\FX-Ai" (
    cd /d "C:\Users\andyc\python\FX-Ai"
    
    REM Remove __pycache__ directories
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
    
    REM Remove .pyc files
    del /s /q *.pyc >nul 2>&1
    
    echo   [OK] Python cache cleaned
) else (
    echo   [WARNING] FX-Ai directory not found
)
echo.

echo Step 4: Moving log files to external storage...
if not exist "D:\FX-Ai-Data\logs" mkdir "D:\FX-Ai-Data\logs"
if exist "C:\Users\andyc\python\FX-Ai\logs\*.log" (
    move "C:\Users\andyc\python\FX-Ai\logs\*.log" "D:\FX-Ai-Data\logs\" >nul 2>&1
    echo   [OK] Log files moved
) else (
    echo   [INFO] No log files to move
)
echo.

echo Step 5: Moving database files to external storage...
if not exist "D:\FX-Ai-Data\databases" mkdir "D:\FX-Ai-Data\databases"
if exist "C:\Users\andyc\python\FX-Ai\*.db" (
    move "C:\Users\andyc\python\FX-Ai\*.db" "D:\FX-Ai-Data\databases\" >nul 2>&1
    echo   [OK] Database files moved
) else (
    echo   [INFO] No database files to move
)
echo.

echo Step 6: Moving ML models to external storage...
if not exist "D:\FX-Ai-Data\models" mkdir "D:\FX-Ai-Data\models"
if exist "C:\Users\andyc\python\FX-Ai\models\*.pkl" (
    move "C:\Users\andyc\python\FX-Ai\models\*.pkl" "D:\FX-Ai-Data\models\" >nul 2>&1
    echo   [OK] Model files moved
) else (
    echo   [INFO] No model files to move
)
echo.

echo Step 7: Clearing VS Code cache...
if exist "%APPDATA%\Code\Cache" (
    rd /s /q "%APPDATA%\Code\Cache" 2>nul
    mkdir "%APPDATA%\Code\Cache"
    echo   [OK] VS Code cache cleared
) else (
    echo   [INFO] No cache directory found
)
echo.

echo Step 8: Clearing GitHub Copilot cache...
if exist "%APPDATA%\Code\User\globalStorage\github.copilot" (
    rd /s /q "%APPDATA%\Code\User\globalStorage\github.copilot" 2>nul
    mkdir "%APPDATA%\Code\User\globalStorage\github.copilot"
    echo   [OK] Copilot cache cleared
) else (
    echo   [INFO] No Copilot cache found
)
echo.

echo Step 9: Checking workspace size...
if exist "C:\Users\andyc\python\FX-Ai" (
    cd /d "C:\Users\andyc\python\FX-Ai"
    dir /s /-c | find "File(s)"
    echo   [INFO] See above for total size
) else (
    echo   [WARNING] Cannot check workspace size
)
echo.

echo Step 10: Creating .vscode directory if not exists...
if not exist "C:\Users\andyc\python\FX-Ai\.vscode" (
    mkdir "C:\Users\andyc\python\FX-Ai\.vscode"
    echo   [OK] .vscode directory created
) else (
    echo   [INFO] .vscode directory already exists
)
echo.

echo ================================
echo Optimization Complete!
echo ================================
echo.
echo Next steps:
echo 1. Copy the optimized settings.json to:
echo    C:\Users\andyc\python\FX-Ai\.vscode\settings.json
echo.
echo 2. Copy the optimized extensions.json to:
echo    C:\Users\andyc\python\FX-Ai\.vscode\extensions.json
echo.
echo 3. Restart VS Code
echo.
echo 4. Open only the FX-Ai folder (not multiple projects)
echo.
echo Backup saved to: %BACKUP_DIR%
echo.
echo Press any key to exit...
pause >nul
