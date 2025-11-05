# VS Code Memory and Cache Verification Script
Write-Host "=== VS Code Memory Fix Verification ===" -ForegroundColor Cyan
Write-Host ""

# Check if documentation files are gone
Write-Host "1. Checking if documentation files are deleted..." -ForegroundColor Yellow
$deletedFiles = @("TRADING_RULES_QUICK_REF.md", "TRADING_RULES_CONFIG.md", "DAILY_TRADE_LIMIT.md")
$allGone = $true
foreach ($file in $deletedFiles) {
    if (Test-Path $file) {
        Write-Host "    $file still exists!" -ForegroundColor Red
        $allGone = $false
    } else {
        Write-Host "    $file is deleted" -ForegroundColor Green
    }
}

if ($allGone) {
    Write-Host "    All documentation files successfully removed!" -ForegroundColor Green
} else {
    Write-Host "     Some files may still appear in VS Code explorer due to caching" -ForegroundColor Yellow
}

Write-Host ""

# Check memory fix files
Write-Host "2. Checking memory fix files..." -ForegroundColor Yellow
$fixFiles = @("vscode_memory_fix.bat", "vscode_memory_fix.ps1", "VS_CODE_MEMORY_FIX_README.md", "clear_vscode_cache.ps1", "refresh_vscode.ps1")
foreach ($file in $fixFiles) {
    if (Test-Path $file) {
        Write-Host "    $file exists" -ForegroundColor Green
    } else {
        Write-Host "    $file missing" -ForegroundColor Red
    }
}

Write-Host ""

# Check VS Code settings
Write-Host "3. Checking VS Code memory settings..." -ForegroundColor Yellow
$globalSettings = "$env:APPDATA\Code\User\settings.json"
$workspaceSettings = ".vscode\settings.json"

if (Test-Path $globalSettings) {
    $memorySettings = Get-Content $globalSettings | Select-String "memory.keepLibraryAst.*false|indexing.*false" | Measure-Object
    if ($memorySettings.Count -gt 0) {
        Write-Host "    Global VS Code memory settings applied" -ForegroundColor Green
    } else {
        Write-Host "     Global memory settings may not be applied" -ForegroundColor Yellow
    }
}

if (Test-Path $workspaceSettings) {
    Write-Host "    Workspace memory settings exist" -ForegroundColor Green
} else {
    Write-Host "    Workspace settings missing" -ForegroundColor Red
}

Write-Host ""

# Git status check
Write-Host "4. Checking git status..." -ForegroundColor Yellow
$gitStatus = git status --porcelain 2>$null
if ($LASTEXITCODE -eq 0) {
    $deletedCount = ($gitStatus | Select-String "^D" | Measure-Object).Count
    if ($deletedCount -gt 0) {
        Write-Host "    $deletedCount files marked as deleted in git" -ForegroundColor Green
    } else {
        Write-Host "    Git working directory is clean" -ForegroundColor Green
    }
} else {
    Write-Host "     Git not available or not a git repository" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "If you still see ghost files in VS Code explorer:" -ForegroundColor Yellow
Write-Host "1. Press Ctrl+Shift+P (Command Palette)" -ForegroundColor White
Write-Host "2. Type 'Developer: Reload Window' and select it" -ForegroundColor White
Write-Host "3. Or run: .\clear_vscode_cache.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Memory fixes should prevent JS heap errors going forward!" -ForegroundColor Green

Read-Host "Press Enter to exit"
