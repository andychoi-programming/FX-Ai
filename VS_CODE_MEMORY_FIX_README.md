# VS Code Memory Fix - JS Heap Out of Memory Solution

## Problem
"Worker terminated due to reaching memory limit: JS heap out of memory"

This error occurs when VS Code's Node.js processes run out of memory, typically caused by:
- Large Python projects with many files
- Memory-intensive extensions (Pylance, Copilot, etc.)
- Insufficient system memory allocation
- Large workspace with many dependencies

## Solutions Applied

### 1. VS Code Settings Optimization
Updated `settings.json` with memory-saving configurations:
- Disabled memory-intensive features
- Limited Python analysis scope
- Disabled auto-imports and suggestions
- Optimized file watching and exclusions
- Reduced UI elements that consume memory

### 2. Memory Limit Scripts
Created two scripts to start VS Code with increased memory limits:

#### `vscode_memory_fix.bat` (Windows Batch)
- Sets Node.js heap size to 4GB
- Configures VS Code environment variables
- Disables GPU acceleration
- Clears Python environment variables

#### `vscode_memory_fix.ps1` (PowerShell)
- Same optimizations as batch file
- PowerShell-native execution
- Colored output for better feedback

## How to Use

### Option 1: Automatic Fix (Recommended)
1. Close VS Code completely
2. Run `vscode_memory_fix.bat` or `vscode_memory_fix.ps1`
3. VS Code will start with optimized memory settings

### Option 2: Manual Environment Variables
Set these environment variables before starting VS Code:
```
NODE_OPTIONS=--max-old-space-size=4096
VSCODE_NODE_OPTIONS=--max-old-space-size=4096
```

### Option 3: VS Code Command Line
Start VS Code with memory optimizations:
```
code --disable-extensions --disable-gpu --max-memory=4096
```

## Prevention Tips

1. **Keep VS Code Updated** - Latest versions have better memory management
2. **Limit Open Files** - Close unused files and editors
3. **Disable Heavy Extensions** - Temporarily disable Pylance, Copilot, or other memory-intensive extensions
4. **Use Workspace Settings** - Apply memory settings per workspace instead of globally
5. **Monitor Memory Usage** - Use Task Manager to monitor VS Code memory consumption

## Troubleshooting

### If Memory Issues Persist:
1. Increase heap size further: `--max-old-space-size=8192`
2. Disable all extensions: `code --disable-extensions`
3. Use VS Code Insiders (canary build) which may have fixes
4. Check system memory - ensure at least 8GB RAM available

### Check Memory Usage:
- Open Task Manager
- Look for "Code" processes
- Monitor memory usage over time
- Identify which extension is consuming memory

## Files Created
- `vscode_memory_fix.bat` - Windows batch script
- `vscode_memory_fix.ps1` - PowerShell script
- Updated `settings.json` - VS Code configuration

## Testing
After applying fixes:
1. Open large Python files
2. Use IntelliSense heavily
3. Monitor for memory error messages
4. Check Task Manager for stable memory usage

This solution should permanently resolve the JS heap out of memory error.
