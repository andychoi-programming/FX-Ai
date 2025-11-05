# Quick VS Code Refresh Script
Write-Host "Refreshing VS Code workspace..." -ForegroundColor Yellow

# Clear workspace cache
if (Test-Path ".vscode\cache") {
    Remove-Item ".vscode\cache" -Recurse -Force
    Write-Host "Cleared workspace cache" -ForegroundColor Gray
}

# Force VS Code to reload window
Write-Host "Reloading VS Code window..." -ForegroundColor Green
code --command "workbench.action.reloadWindow"

Write-Host "VS Code refreshed! Check if ghost files are gone." -ForegroundColor Green
