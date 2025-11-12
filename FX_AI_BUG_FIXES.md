# 🔧 FX-AI CRITICAL BUG FIXES
## Three Bugs Preventing All Trading

**Date:** November 12, 2025  
**Issue:** System shows "already traded" for all symbols but no trades were placed

---

## 🎯 BUG #1: False "Already Traded Today" Status

### **Problem:**
System thinks all 30 symbols have been traded today, blocking ALL new trades.

### **Root Cause:**
The `daily_trades` tracking is persisting from yesterday OR was incorrectly initialized with today's date even though no trades were placed.

### **Location:**
`core/risk_manager.py` - The `has_traded_today()` or `can_trade()` method

### **FIX #1: Reset Daily Trade Tracking**

**Option A: Quick Database Reset (Immediate)**
```python
# Run this in Python console or create reset_trades.py:

import sqlite3
from datetime import datetime

# Connect to database
conn = sqlite3.connect('data/performance_history.db')
cursor = conn.cursor()

# Check current status
print("Current daily trades:")
cursor.execute("SELECT * FROM daily_trades")
for row in cursor.fetchall():
    print(row)

# Clear today's false records
today = datetime.now().date()
cursor.execute("DELETE FROM daily_trades WHERE trade_date = ?", (today,))
conn.commit()

print(f"\nCleared {cursor.rowcount} false trade records for {today}")

# Verify
cursor.execute("SELECT * FROM daily_trades WHERE trade_date = ?", (today,))
remaining = cursor.fetchall()
print(f"Remaining records for today: {len(remaining)}")

conn.close()
print("\n✅ Daily trade tracking reset!")
```

**Option B: Fix the Risk Manager Code**

Find this in `core/risk_manager.py`:
```python
def has_traded_today(self, symbol: str) -> bool:
    """Check if symbol has been traded today"""
    today = datetime.now().date()
    
    # ❌ BUG: This might be returning True when it shouldn't
    return self.daily_trades.get(symbol, {}).get('date') == today
```

**Replace with:**
```python
def has_traded_today(self, symbol: str) -> bool:
    """Check if symbol has been traded today"""
    today = datetime.now().date()
    
    # Check in-memory tracking
    if symbol in self.daily_trades:
        trade_date = self.daily_trades[symbol].get('date')
        if trade_date == today:
            # Verify trade actually exists in database
            if self._verify_trade_in_db(symbol, today):
                return True
            else:
                # False positive - remove from tracking
                logger.warning(f"Removing false 'traded today' flag for {symbol}")
                del self.daily_trades[symbol]
                return False
    
    # Also check database
    return self._check_trade_in_db(symbol, today)

def _verify_trade_in_db(self, symbol: str, date) -> bool:
    """Verify trade actually exists in database"""
    try:
        import sqlite3
        conn = sqlite3.connect('data/performance_history.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM trades 
            WHERE symbol = ? AND DATE(timestamp) = ?
        """, (symbol, date))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
        
    except Exception as e:
        logger.error(f"Error verifying trade in DB: {e}")
        return False
```

**Option C: Add Reset at Startup (Safest)**

In your `main.py` or wherever you initialize the risk manager:
```python
# After initializing risk_manager
logger.info("Resetting daily trade tracking at startup...")

# Clear in-memory tracking
risk_manager.daily_trades.clear()

# Clear false database records (only if no trades today)
from datetime import datetime
today = datetime.now().date()

import sqlite3
conn = sqlite3.connect('data/performance_history.db')
cursor = conn.cursor()

# Check if ANY actual trades exist today
cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?", (today,))
actual_trades_today = cursor.fetchone()[0]

if actual_trades_today == 0:
    # No real trades today, so clear daily_trades table
    cursor.execute("DELETE FROM daily_trades WHERE trade_date = ?", (today,))
    cleared = cursor.rowcount
    conn.commit()
    logger.info(f"Cleared {cleared} false 'already traded' flags")
else:
    logger.info(f"{actual_trades_today} real trades exist today - keeping tracking")

conn.close()
logger.info("Daily trade tracking reset complete")
```

---

## 🎯 BUG #2: MarketDataManager Missing Method

### **Problem:**
```
Error generating signal for EURGBP: 'MarketDataManager' object has no attribute 'get_market_data'
```

### **Root Cause:**
Code is calling `market_data_manager.get_market_data()` but the method doesn't exist or is named differently.

### **Location:**
- `data/market_data_manager.py` - Class definition
- Wherever signal generation calls it (probably `main.py` or `core/trading_engine.py`)

### **FIX #2A: Add Missing Method**

In `data/market_data_manager.py`:

```python
class MarketDataManager:
    """Manages market data retrieval"""
    
    def get_market_data(self, symbol: str, timeframe: str = 'M15'):
        """
        Get current market data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for data (default: M15)
            
        Returns:
            Market data dictionary or None if error
        """
        try:
            import MetaTrader5 as mt5
            
            # Get current tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Cannot get tick data for {symbol}")
                return None
            
            # Get recent bars for analysis
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
            }
            
            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M15)
            bars = mt5.copy_rates_from_pos(symbol, tf, 0, 100)
            
            if bars is None or len(bars) == 0:
                logger.error(f"Cannot get bar data for {symbol}")
                return None
            
            # Return market data
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': tick.time,
                'spread': tick.ask - tick.bid,
                'bars': bars,
                'timeframe': timeframe
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
```

### **FIX #2B: Find Correct Method Name**

The method might exist with a different name. Search for it:

```bash
cd C:\Users\andyc\python\FX-Ai
findstr /s /i "def get.*market.*data" data\*.py
findstr /s /i "def fetch.*data" data\*.py
findstr /s /i "def retrieve" data\*.py
```

If you find it's called something like `fetch_data()` or `get_data()`, update the calling code:

```python
# Change from:
# data = market_data_manager.get_market_data(symbol)

# To:
data = market_data_manager.get_data(symbol)  # Or whatever the correct name is
```

---

## 🎯 BUG #3: MT5Connector Missing Attribute

### **Problem:**
```
Error checking emergency stop conditions: 'MT5Connector' object has no attribute 'is_connected'
2025-11-12 06:18:23 - CRITICAL - Emergency stop triggered - shutting down
```

### **Root Cause:**
Emergency stop code is checking `mt5_connector.is_connected` but:
1. The attribute doesn't exist, OR
2. It's a method `is_connected()` not an attribute

### **Location:**
- `core/mt5_connector.py` - MT5Connector class
- Emergency stop check in `main.py`

### **FIX #3A: Add Missing Attribute**

In `core/mt5_connector.py`:

```python
class MT5Connector:
    def __init__(self, config):
        self.config = config
        self.connected = False  # ✅ Add this
        # ... rest of init
    
    def initialize(self):
        """Initialize MT5 connection"""
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            self.connected = False  # ✅ Set this
            return False
        
        # Login
        login = self.config['mt5']['login']
        password = self.config['mt5']['password']
        server = self.config['mt5']['server']
        
        if not mt5.login(login, password, server):
            logger.error("MT5 login failed")
            self.connected = False  # ✅ Set this
            return False
        
        logger.info("MT5 connected successfully")
        self.connected = True  # ✅ Set this
        return True
    
    @property
    def is_connected(self):
        """Check if MT5 is connected"""
        return self.connected
```

### **FIX #3B: Use Correct Method Name**

If it's a method, update the calling code:

In `main.py` or wherever emergency stop is checked:

```python
# Change from:
# if not mt5_connector.is_connected:

# To:
if not mt5_connector.is_connected():  # ✅ Add parentheses
    logger.critical("Emergency stop - MT5 disconnected")
```

### **FIX #3C: Add Proper Connection Check**

Replace the emergency stop check with a proper implementation:

```python
def check_emergency_stop():
    """Check emergency stop conditions"""
    try:
        import MetaTrader5 as mt5
        
        # Check if MT5 terminal is running
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.critical("MT5 terminal not responding")
            return True
        
        # Check if trading is allowed
        if not terminal_info.trade_allowed:
            logger.critical("Trading not allowed in MT5")
            return True
        
        # Check account connection
        account_info = mt5.account_info()
        if account_info is None:
            logger.critical("MT5 account disconnected")
            return True
        
        # All checks passed
        return False
        
    except Exception as e:
        logger.error(f"Error checking emergency stop: {e}")
        # Don't trigger stop on check error
        return False
```

---

## 🚀 QUICK FIX SCRIPT

Create `fix_trading_bugs.py`:

```python
"""
Quick fix script for FX-Ai trading bugs
Run this to reset daily tracking and verify fixes
"""

import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_daily_tracking():
    """Fix false 'already traded today' status"""
    try:
        conn = sqlite3.connect('data/performance_history.db')
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Check actual trades today
        cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?", (today,))
        actual_trades = cursor.fetchone()[0]
        
        logger.info(f"Actual trades today: {actual_trades}")
        
        # Check daily_trades table
        cursor.execute("SELECT COUNT(*) FROM daily_trades WHERE trade_date = ?", (today,))
        tracked_trades = cursor.fetchone()[0]
        
        logger.info(f"Tracked 'already traded' flags: {tracked_trades}")
        
        if tracked_trades > 0 and actual_trades == 0:
            logger.warning("Found false 'already traded' flags - clearing...")
            cursor.execute("DELETE FROM daily_trades WHERE trade_date = ?", (today,))
            conn.commit()
            logger.info(f"✅ Cleared {cursor.rowcount} false flags")
        else:
            logger.info("✅ Daily tracking looks correct")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error fixing daily tracking: {e}")
        return False

def verify_market_data_manager():
    """Verify MarketDataManager has required methods"""
    try:
        from data.market_data_manager import MarketDataManager
        
        # Check if method exists
        if hasattr(MarketDataManager, 'get_market_data'):
            logger.info("✅ MarketDataManager.get_market_data() exists")
            return True
        else:
            logger.error("❌ MarketDataManager.get_market_data() NOT FOUND")
            
            # List available methods
            methods = [m for m in dir(MarketDataManager) if not m.startswith('_')]
            logger.info(f"Available methods: {methods}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking MarketDataManager: {e}")
        return False

def verify_mt5_connector():
    """Verify MT5Connector has is_connected"""
    try:
        from core.mt5_connector import MT5Connector
        
        # Check if attribute/method exists
        if hasattr(MT5Connector, 'is_connected'):
            logger.info("✅ MT5Connector.is_connected exists")
            return True
        else:
            logger.error("❌ MT5Connector.is_connected NOT FOUND")
            
            # List available attributes
            attrs = [a for a in dir(MT5Connector) if not a.startswith('_')]
            logger.info(f"Available attributes: {attrs}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking MT5Connector: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("FX-AI BUG FIX UTILITY")
    print("="*70)
    print()
    
    # Fix 1: Daily tracking
    print("Fix #1: Resetting daily trade tracking...")
    fix_daily_tracking()
    print()
    
    # Fix 2: MarketDataManager
    print("Fix #2: Verifying MarketDataManager...")
    verify_market_data_manager()
    print()
    
    # Fix 3: MT5Connector
    print("Fix #3: Verifying MT5Connector...")
    verify_mt5_connector()
    print()
    
    print("="*70)
    print("✅ Bug fix utility complete")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Fix any ❌ errors shown above")
    print("2. Restart FX-Ai")
    print("3. Check logs for 'Already traded today' messages")
    print("4. Verify trades execute")
```

---

## 📝 STEP-BY-STEP FIX PROCEDURE

### **Step 1: Run the Fix Script (2 minutes)**
```bash
cd C:\Users\andyc\python\FX-Ai
python fix_trading_bugs.py
```

### **Step 2: Fix Identified Issues (5-10 minutes)**

Based on script output:
- If daily tracking needs clearing → Script will clear it automatically
- If `get_market_data()` missing → Add the method (see FIX #2A above)
- If `is_connected` missing → Add the property (see FIX #3A above)

### **Step 3: Add Startup Reset (2 minutes)**

In `main.py`, add this right after initializing risk_manager:

```python
# Reset daily tracking at startup (prevents false "already traded" flags)
logger.info("Verifying daily trade tracking...")
risk_manager.daily_trades.clear()

# Verify actual trades in database
from datetime import datetime
today = datetime.now().date()

import sqlite3
conn = sqlite3.connect('data/performance_history.db')
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?", (today,))
actual_trades = cursor.fetchone()[0]

if actual_trades == 0:
    cursor.execute("DELETE FROM daily_trades WHERE trade_date = ?", (today,))
    cleared = cursor.rowcount
    conn.commit()
    if cleared > 0:
        logger.warning(f"Cleared {cleared} false 'already traded' flags")

conn.close()
logger.info("Daily tracking verified")
```

### **Step 4: Restart and Test (2 minutes)**
```bash
python main.py
```

**Watch for:**
- ✅ NO "Already traded today" messages (for symbols you haven't traded)
- ✅ NO MarketDataManager errors
- ✅ NO MT5Connector errors
- ✅ NO Emergency stop triggered

### **Step 5: Verify Trading (5 minutes)**

- Check logs for signal generation
- Verify no "already traded" blocks
- Watch for trade execution
- Check MT5 for open positions

---

## ✅ SUCCESS CRITERIA

After fixes, you should see:

```
2025-11-12 06:30:00 - INFO - Starting main trading loop...
2025-11-12 06:30:01 - INFO - Processing EURUSD...
2025-11-12 06:30:01 - INFO - Signal generated for EURUSD
2025-11-12 06:30:01 - INFO - Order placed for EURUSD: Ticket #123456
```

**NOT:**
```
EURUSD: Already traded today - ONE trade per symbol per day limit
```

---

## 🆘 IF STILL HAVING ISSUES

Send me:
1. Output from `fix_trading_bugs.py`
2. The methods available in MarketDataManager (from script output)
3. The attributes available in MT5Connector (from script output)
4. New log file after fixes applied

---

## 📊 SUMMARY

**Three bugs found:**
1. ❌ False "already traded today" status blocking ALL symbols
2. ❌ Missing `get_market_data()` method in MarketDataManager
3. ❌ Missing `is_connected` in MT5Connector causing emergency shutdown

**Quick fix:**
1. Run `fix_trading_bugs.py`
2. Add missing methods/properties
3. Add startup reset to prevent future false flags
4. Restart and verify

**Time:** 15-20 minutes total

---

Good luck! These are straightforward code bugs that should be easy to fix once you add the missing methods and reset the daily tracking.
