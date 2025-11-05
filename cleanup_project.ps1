# Comprehensive Project Cleanup Script
# Identifies and removes conflicts, duplicates, and empty files

Write-Host "=== FX-Ai Project Cleanup Analysis ===" -ForegroundColor Cyan
Write-Host ""

# Find all empty files (excluding venv and git)
Write-Host "1. Finding empty files..." -ForegroundColor Yellow
$emptyFiles = Get-ChildItem -Path . -Recurse -File | Where-Object { 
    $_.Length -eq 0 -and 
    $_.FullName -notmatch 'venv' -and 
    $_.FullName -notmatch '\.git' -and
    $_.FullName -notmatch '__pycache__'
} | Select-Object FullName

Write-Host "Found $($emptyFiles.Count) empty files:" -ForegroundColor Red
foreach ($file in $emptyFiles) {
    Write-Host "  - $($file.FullName.Replace((Get-Location).Path + '\', ''))" -ForegroundColor Gray
}

Write-Host ""

# Find potential duplicate functionality
Write-Host "2. Analyzing potential duplicates..." -ForegroundColor Yellow

# Check for multiple health/check files
$checkFiles = Get-ChildItem -Path . -Filter "*check*" -File | Where-Object { $_.Length -gt 0 }
Write-Host "Health/Check files with content:" -ForegroundColor White
$checkFiles | ForEach-Object { Write-Host "  - $($_.Name) ($($_.Length) bytes)" -ForegroundColor Gray }

# Check for multiple risk files  
$riskFiles = Get-ChildItem -Path . -Recurse -Filter "*risk*" -File | Where-Object { $_.Length -gt 0 }
Write-Host "Risk-related files with content:" -ForegroundColor White
$riskFiles | ForEach-Object { Write-Host "  - $($_.FullName.Replace((Get-Location).Path + '\', '')) ($($_.Length) bytes)" -ForegroundColor Gray }

# Check for multiple trading system files
$tradingFiles = Get-ChildItem -Path . -Recurse -Filter "*trading*" -File | Where-Object { $_.Length -gt 0 }
Write-Host "Trading-related files with content:" -ForegroundColor White
$tradingFiles | ForEach-Object { Write-Host "  - $($_.FullName.Replace((Get-Location).Path + '\', '')) ($($_.Length) bytes)" -ForegroundColor Gray }

Write-Host ""

# Recommendations
Write-Host "3. Cleanup Recommendations:" -ForegroundColor Green
Write-Host "DELETE - Empty files ($($emptyFiles.Count) files):" -ForegroundColor Red
Write-Host "  These files contain no code and appear to be placeholders" -ForegroundColor Gray

Write-Host ""
Write-Host "REVIEW - Potential overlaps:" -ForegroundColor Yellow
Write-Host "  - Multiple check/health files: Consider consolidating" -ForegroundColor Gray
Write-Host "  - Risk management in core/ vs live_trading/: core/ seems primary" -ForegroundColor Gray
Write-Host "  - Trading systems: main.py vs live_trading/ml_trading_system.py" -ForegroundColor Gray

Write-Host ""
$confirmation = Read-Host "Do you want to DELETE all empty files? (y/N)"
if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
    Write-Host "Deleting empty files..." -ForegroundColor Yellow
    $deletedCount = 0
    foreach ($file in $emptyFiles) {
        try {
            Remove-Item $file.FullName -Force
            Write-Host "  Deleted: $($file.FullName.Replace((Get-Location).Path + '\', ''))" -ForegroundColor Green
            $deletedCount++
        } catch {
            Write-Host "  Failed to delete: $($file.FullName)" -ForegroundColor Red
        }
    }
    Write-Host "Successfully deleted $deletedCount empty files!" -ForegroundColor Green
    
    # Commit the changes
    Write-Host "Committing changes to git..." -ForegroundColor Yellow
    git add -A 2>$null; git commit -m "Clean up empty and duplicate files" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Changes committed to git!" -ForegroundColor Green
    } else {
        Write-Host "Git commit may have failed - check manually" -ForegroundColor Yellow
    }
} else {
    Write-Host "Cleanup cancelled. No files deleted." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Cleanup analysis complete!" -ForegroundColor Cyan
