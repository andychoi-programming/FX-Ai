# FX-Ai VS Code & Copilot Optimization Guide

## 🎯 Quick Fix Checklist

When Copilot becomes unresponsive, try these in order:

1. ✅ **Close unnecessary files** - Keep only 2-3 files open
2. ✅ **Reload Window** - `Ctrl+Shift+P` → "Developer: Reload Window"
3. ✅ **Clear Copilot cache** - `Ctrl+Shift+P` → "GitHub Copilot: Clear Cache"
4. ✅ **Check file size** - Large files (>1MB) can cause issues
5. ✅ **Restart VS Code** - Complete restart, not just reload

---

## 📁 File Structure Best Practices

### ✅ DO: Keep these in your project
```
FX-Ai/
├── main.py
├── trading_engine.py
├── signal_generator.py
├── risk_manager.py
├── mt5_connector.py
├── config.json
└── requirements.txt
```

### ❌ DON'T: Keep these in your project (move to external drives)
```
# Move to D:/FX-Ai-Data/
├── logs/              # All log files
├── models/            # ML model files (.h5, .pkl)
├── historical_data/   # Historical market data
├── backtest_data/     # Backtest results
└── market_data/       # Real-time data cache
```

---

## 🚀 Performance Optimization Steps

### Step 1: Clean Your Workspace
```bash
# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove old logs (keep only recent ones)
move logs\*.log D:\FX-Ai-logs\

# Remove old database files
move *.db D:\FX-Ai-Data\databases\
```

### Step 2: Limit Open Files
- Keep maximum 3-5 Python files open at once
- Close Preview files (files shown in italics)
- Use `Ctrl+W` to close files you're not actively editing

### Step 3: Disable Unnecessary Extensions
Check your installed extensions:
```
Ctrl+Shift+X → Search "Installed"
```
Disable or uninstall:
- Jupyter notebooks (if not used)
- Docker
- Any linters (Pylance handles this)
- Formatters (Black, isort) - format manually when needed

### Step 4: Monitor VS Code Performance
```
Ctrl+Shift+P → "Developer: Show Running Extensions"
```
Look for extensions using >50MB RAM or high CPU

---

## 🔧 Copilot-Specific Fixes

### Problem: Copilot suggestions not appearing
**Solution:**
1. Check status bar: Should show "GitHub Copilot" icon
2. Click icon → "Check status"
3. If offline: Sign out and sign in again
4. Clear cache: `Ctrl+Shift+P` → "GitHub Copilot: Clear Cache"

### Problem: Copilot responses are slow
**Solution:**
1. Close large files (>500 lines)
2. Reduce context: Close unrelated files
3. Simplify your question/prompt
4. Check internet connection (Copilot needs connection)

### Problem: Copilot stops working after a while
**Solution:**
1. Memory leak in VS Code - restart every 4-8 hours
2. Clear workspace cache: Delete `.vscode/.cache/`
3. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

---

## 📊 File Size Guidelines

| File Type | Recommended Size | Action if Larger |
|-----------|-----------------|------------------|
| Python (.py) | < 500 lines | Split into multiple files |
| Log files (.log) | < 1 MB | Move to external drive |
| CSV data | < 10 MB | Move to external drive |
| ML models (.pkl/.h5) | Any size | Store externally, load when needed |
| Databases (.db) | < 50 MB | Move to external drive |

---

## 🎛️ VS Code Settings Explained

### Critical Settings
```json
// Most important - prevents indexing everything
"python.analysis.indexing": false,

// Only analyzes files you have open
"python.analysis.diagnosticMode": "openFilesOnly",

// Disables type checking (faster)
"python.analysis.typeCheckingMode": "off",

// Prevents watching file changes (faster)
"files.watcherExclude": {
    "**/*.log": true,
    "**/*.db": true
}
```

### Why These Matter
- **indexing: false** - VS Code won't scan all Python files (saves 80% RAM)
- **openFilesOnly** - Only checks the file you're editing (10x faster)
- **typeCheckingMode: off** - No type errors shown (less CPU usage)

---

## 🔍 Troubleshooting Common Issues

### Issue: "Out of memory" errors
**Cause:** Too many files in workspace or large files open
**Fix:**
1. Close all files: `Ctrl+K, Ctrl+W`
2. Open only the file you need
3. Check workspace size: Should be <1GB
4. Move data files to external drive

### Issue: Copilot suggestions are wrong/irrelevant
**Cause:** Too much context from other files
**Fix:**
1. Close unrelated files
2. Add comments explaining what you want
3. Use more specific prompts
4. Check if file has syntax errors

### Issue: VS Code freezes when opening project
**Cause:** Too many files in workspace or Git repo too large
**Fix:**
1. Add `.vscodeignore` file (provided above)
2. Disable Git: `"git.enabled": false` (in settings.json)
3. Use File → "Open Folder" instead of workspace
4. Split project into multiple folders

---

## 📝 Daily Maintenance Routine

### Start of Day
1. Delete old log files: `del D:\FX-Ai-Data\logs\*.log /q`
2. Close all VS Code windows
3. Open only FX-Ai project (not multiple projects)
4. Open only main.py to start

### End of Day
1. Close all files: `Ctrl+K, Ctrl+W`
2. Commit code: `git add . && git commit -m "Daily commit"`
3. Close VS Code completely
4. (Optional) Clear temp files: `del %TEMP%\* /q`

### Weekly Maintenance
1. Update extensions: `Ctrl+Shift+P` → "Extensions: Update All Extensions"
2. Clear VS Code cache: Delete `%APPDATA%\Code\Cache\`
3. Clear Copilot cache: `Ctrl+Shift+P` → "GitHub Copilot: Clear Cache"
4. Restart computer (clears RAM)

---

## 🎯 Best Practices for FX-Ai Development

### 1. Split Your Code
Instead of one huge file:
```python
# ❌ BAD: main.py (2000 lines)
# Everything in one file

# ✅ GOOD: Split by function
main.py (100 lines)
├── trading_engine.py (300 lines)
├── signal_generator.py (400 lines)
├── risk_manager.py (200 lines)
├── ml_models.py (500 lines)
└── utils.py (150 lines)
```

### 2. Use External Storage
```python
# ✅ GOOD: Store data externally
DATA_DIR = "D:/FX-Ai-Data"
LOG_DIR = "D:/FX-Ai-Data/logs"
MODEL_DIR = "D:/FX-Ai-Data/models"

# Load only when needed
def load_model():
    return pickle.load(open(f"{MODEL_DIR}/xgboost.pkl", "rb"))
```

### 3. Logging Best Practices
```python
# ✅ GOOD: Rotate logs daily
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "D:/FX-Ai-Data/logs/trading.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

### 4. Work with Smaller Datasets
```python
# ✅ GOOD: Load data in chunks
for chunk in pd.read_csv("data.csv", chunksize=1000):
    process(chunk)

# Instead of:
# ❌ BAD: Load all at once
# df = pd.read_csv("data.csv")  # 500MB file!
```

---

## 🆘 Emergency Recovery

If VS Code is completely frozen or Copilot won't work:

### Nuclear Option 1: Reset VS Code Settings
```bash
# Backup first!
copy "%APPDATA%\Code\User\settings.json" settings_backup.json

# Delete settings (will reset to defaults)
del "%APPDATA%\Code\User\settings.json"
del "%APPDATA%\Code\User\keybindings.json"
```

### Nuclear Option 2: Reinstall VS Code
1. Uninstall VS Code
2. Delete `%APPDATA%\Code` folder
3. Delete `%USERPROFILE%\.vscode` folder
4. Reinstall VS Code
5. Install only Python + Copilot extensions

### Nuclear Option 3: Create New Workspace
```bash
# Create minimal FX-Ai workspace
mkdir FX-Ai-Minimal
cd FX-Ai-Minimal

# Copy only essential files
copy ..\FX-Ai\main.py .
copy ..\FX-Ai\config.json .
copy ..\FX-Ai\requirements.txt .

# Open in VS Code
code .
```

---

## ✅ Success Metrics

Your optimization worked if:
- VS Code opens in <5 seconds
- Copilot suggestions appear in <2 seconds
- No lag when typing
- Can work for 8+ hours without restart
- RAM usage <500MB (check Task Manager)

---

## 📞 When to Ask for Help

If after following this guide:
1. Copilot still doesn't respond after 10 seconds
2. VS Code crashes regularly (>once per day)
3. Can't open more than 1 file without freezing
4. RAM usage >2GB for VS Code

Then it might be a system issue, not configuration!
