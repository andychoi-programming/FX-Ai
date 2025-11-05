# VS Code Cache Clear and Memory Fix Script
Write-Host "Clearing VS Code caches..." -ForegroundColor Yellow

# Clear VS Code caches
$cachePaths = @(
    "$env:APPDATA\Code\CachedData",
    "$env:APPDATA\Code\Cache",
    "$env:APPDATA\Code\GPUCache",
    "$env:APPDATA\Code\Code Cache",
    "$env:USERPROFILE\.vscode\extensions\ms-python.vscode-pylance*\node_modules",
    "$env:USERPROFILE\.vscode\extensions\github.copilot*\node_modules"
)

foreach ($path in $cachePaths) {
    if (Test-Path $path) {
        Write-Host "Clearing: $path" -ForegroundColor Gray
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Setting memory optimizations..." -ForegroundColor Green
$env:NODE_OPTIONS = "--max-old-space-size=4096 --max-new-space-size=1024"
$env:VSCODE_NODE_OPTIONS = "--max-old-space-size=4096"

Write-Host "Restarting VS Code with clean cache..." -ForegroundColor Green
Start-Process -FilePath "code" -ArgumentList "--disable-gpu --disable-software-rasterizer --disable-extensions" -Wait

Write-Host "VS Code cache cleared and restarted with memory optimizations!" -ForegroundColor Green
Read-Host "Press Enter to exit"
