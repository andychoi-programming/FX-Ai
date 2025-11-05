# VS Code Memory Optimization Script
# Run this before starting VS Code to prevent memory issues

Write-Host "Setting Node.js memory limits..." -ForegroundColor Green
$env:NODE_OPTIONS = "--max-old-space-size=4096 --max-new-space-size=1024"

Write-Host "Setting VS Code environment variables..." -ForegroundColor Green
$env:VSCODE_NODE_OPTIONS = "--max-old-space-size=4096"
$env:ELECTRON_RUN_AS_NODE = "1"

Write-Host "Disabling problematic features..." -ForegroundColor Green
$env:PYTHONPATH = ""
$env:PYTHONHOME = ""

Write-Host "Starting VS Code with optimized settings..." -ForegroundColor Green
& code --disable-extensions --disable-gpu --disable-software-rasterizer

Write-Host "VS Code started with memory optimizations." -ForegroundColor Green
Read-Host "Press Enter to exit"
