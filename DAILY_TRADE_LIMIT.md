# One Trade Per Symbol Per Day Rule

## Overview

Implemented a strict trading rule: **Each symbol can only trade ONE time per day** based on MT5 server time.

## Purpose

- Prevents overtrading on individual symbols
- Enforces disciplined trading approach
- Allows system to skip symbols if no quality opportunity exists
- Based on MT5 server time for consistency across time zones

## Implementation Details

### RiskManager (`core/risk_manager.py`)

**New Attributes:**
- `daily_trades_per_symbol`: Dictionary tracking trades per symbol per day
  - Structure: `{symbol: {'date': 'YYYY-MM-DD', 'count': N}}`
- `max_trades_per_symbol_per_day`: Hard limit set to 1

**New Methods:**

1. **`has_traded_today(symbol)`**
   - Checks if symbol already traded today
   - Uses MT5 server time via `mt5.symbol_info_tick()`
   - Auto-resets counter if date changed
   - Returns `True` if already traded, `False` otherwise

2. **`record_trade(symbol)`**
   - Records that a trade was executed
   - Increments daily counter for the symbol
   - Uses MT5 server time for date tracking
   - Logs trade count for monitoring

3. **`can_trade(symbol)` - Enhanced**
   - Added CHECK #1: Daily trade limit per symbol
   - Returns `False` if symbol already traded today
   - Existing checks (daily loss, max positions) remain

4. **`reset_daily_stats()` - Enhanced**
   - Now also clears `daily_trades_per_symbol` dictionary

### Main Trading Loop (`main.py`)

**Trade Execution Enhancement:**
```python
if trade_result.get('success', False):
    self.session_stats['total_trades'] += 1
    
    # Record trade for daily limit tracking
    self.risk_manager.record_trade(signal['symbol'])
    self.logger.info(f"{symbol}: Trade executed and recorded - no more trades allowed today")
```

## Behavior

### Scenario 1: First Trade of the Day
```
EURUSD: Checking if can trade...
EURUSD: No trades today - ALLOWED
[TRADE EXECUTED]
EURUSD: Trade recorded for 2025-11-05 (count: 1)
EURUSD: Trade executed and recorded - no more trades allowed today
```

### Scenario 2: Second Trade Attempt (Same Day)
```
EURUSD: Checking if can trade...
EURUSD: Already traded today (2025-11-05) - BLOCKED
EURUSD: Already traded today - ONE trade per symbol per day limit
```

### Scenario 3: New Day
```
EURUSD: Checking if can trade...
[Date changed from 2025-11-05 to 2025-11-06]
EURUSD: Counter reset - ALLOWED
```

### Scenario 4: Multiple Symbols
```
Day 2025-11-05:
  EURUSD: Traded (1/1) - Blocked for rest of day
  GBPUSD: Traded (1/1) - Blocked for rest of day  
  USDJPY: No trade (0/1) - Still available
  XAUUSD: Traded (1/1) - Blocked for rest of day
```

## Check Order in `can_trade()`

1. **Daily trade limit** - New CHECK #1
2. Daily loss limit
3. Max positions
4. Multiple positions per symbol
5. Symbol cooldown

If any check fails, trading is blocked for that symbol.

## Testing

Run `test_daily_limit.py` to verify:
- First trade allowed
- Second trade blocked
- Daily reset works
- Multiple symbols independent

## Advantages

✅ Prevents revenge trading on same symbol  
✅ Forces patience and discipline  
✅ Allows skipping poor setups  
✅ Reduces transaction costs  
✅ Better risk distribution across symbols  
✅ Automatic reset at day change  
✅ Uses reliable MT5 server time  

## Configuration

No configuration needed - hard-coded to 1 trade per symbol per day.

To modify the limit (not recommended):
```python
# In RiskManager.__init__()
self.max_trades_per_symbol_per_day = 1  # Change this value
```

## Logging

All trade limit checks are logged:
- `INFO`: Trade allowed/recorded
- `WARNING`: Trade blocked (already traded today)
- `INFO`: Daily reset confirmation

## Compatibility

- Works with existing risk management systems
- Compatible with cooldown periods
- Integrates with adaptive learning
- Supports all symbols (forex, metals, etc.)
