@echo off
REM VS Code Memory Optimization Script
REM Run this before starting VS Code to prevent memory issues

echo Setting Node.js memory limits...
set NODE_OPTIONS=--max-old-space-size=4096 --max-new-space-size=1024

echo Setting VS Code environment variables...
set VSCODE_NODE_OPTIONS=--max-old-space-size=4096
set ELECTRON_RUN_AS_NODE=1

echo Disabling problematic features...
set PYTHONPATH=
set PYTHONHOME=

echo Starting VS Code with optimized settings...
code --disable-extensions --disable-gpu --disable-software-rasterizer

echo VS Code started with memory optimizations.
pause
