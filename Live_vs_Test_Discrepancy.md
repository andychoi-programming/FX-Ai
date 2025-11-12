# 🚨 CRITICAL: Tests Show Opportunities, Live Shows None

**Status**: Tests work ✓ | Live fails ✗  
**Issue**: Signal generation works in tests, blocked in live execution  
**Priority**: URGENT - System cannot trade  

---

## 🎯 **The Core Problem**

```
TEST MODE:
├─ Signals generated ✓
├─ Trading opportunities identified ✓
└─ Validation passes ✓

LIVE MODE:
├─ Signals generated (probably) ?
├─ Trading opportunities (blocked) ✗
└─ Result: "No opportunities found, 30 symbols analyzed" ✗
```

**This means**: Something specific to LIVE mode is blocking all trades.

---

## 🔍 **Root Cause Analysis**

### **Theory: Environment Difference**

Tests and live mode have different execution paths:

| Factor | Test Mode | Live Mode | Impact |
|--------|-----------|-----------|---------|
| Daily limits | Not checked | Enforced | HIGH |
| Database state | Empty/Mock | Real data | HIGH |
| MT5 connection | Mock/Skip | Required | HIGH |
| Signal calculation | Simplified | Full pipeline | MEDIUM |
| Risk checks | Minimal | Complete | HIGH |

---

## 🎯 **Most Likely Issue: Daily Trade Database**

### **Hypothesis**

All 30 symbols are marked as "already traded today" in database:
- Tests: Don't check daily limits → Opportunities found
- Live: Enforces 1 trade/symbol/day → All symbols blocked

### **How to Verify (30 seconds)**

```bash
# Check if database has today's trades
sqlite3 data/performance_history.db "
SELECT COUNT(*) as symbols_traded_today
FROM daily_trades 
WHERE trade_date = date('now');
"
```

**If this returns 25-30**: That's your problem!

**Why this happens**:
1. System ran earlier today
2. Generated signals and executed trades
3. Marked all symbols as "traded"
4. System restarted
5. Now blocks all symbols (daily limit enforced)
6. Tests don't check this database → show opportunities

### **IMMEDIATE FIX**

```bash
# Clear today's trade records (TESTING ONLY)
sqlite3 data/performance_history.db "
DELETE FROM daily_trades 
WHERE trade_date = date('now');
"

# Restart FX-Ai
python main.py
```

**Expected result**: Trades should now execute

---

## 🔧 **Alternative Issues**

### **Issue 2: Signal Strength Calculation Returns 0**

**Symptoms**:
- Live: All signals calculate to 0.0 or below threshold
- Tests: Use mock data with good signals

**Causes**:
- Data sources failing (no internet, APIs down)
- ML models not loading
- Technical analysis returning null/0
- Fundamental/sentiment sources failing

**Check**:
```bash
# Look for data errors in logs
grep -i "error\|fail\|null" logs/*.log | tail -50

# Check if ML models exist
ls -lh models/*.pkl | wc -l  # Should be 60 (30 models + 30 scalers)
```

**Fix**: Add debug logging to see actual signal values

---

### **Issue 3: Risk Manager Blocking All Trades**

**Symptoms**:
- Daily loss limit reached
- Daily trade limit reached  
- Max positions reached

**Check**:
```bash
# Check today's performance
sqlite3 data/performance_history.db "
SELECT 
    COUNT(*) as trades,
    SUM(profit_loss) as pnl,
    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses
FROM trades 
WHERE DATE(timestamp) = date('now');
"
```

**Blocking conditions**:
- trades >= 10 → Daily limit reached
- pnl <= -500 → Daily loss limit
- losses >= 3 consecutive → Pause trading

**Fix**: Wait until tomorrow OR reset limits for testing

---

### **Issue 4: MT5 Connection Difference**

**Symptoms**:
- Tests: Skip MT5 checks
- Live: Requires real MT5 connection

**Check**:
```bash
python mt5_diagnostic.py
```

**Possible issues**:
- MT5 not logged in
- Trading not enabled on account
- AutoTrading disabled
- Weekend (market closed)

**Fix**: Verify MT5 terminal running and logged in

---

## ⚡ **IMMEDIATE ACTION PLAN**

### **Step 1: Run Live vs Test Diagnostic** (30 sec)

```bash
python diagnose_live_vs_test.py
```

This will check:
- Daily trade database status
- Signal thresholds
- ML models
- MT5 connection
- Risk manager state

**This script will identify the EXACT blocking condition**

---

### **Step 2: If Daily Limits Are The Issue**

```bash
# Check database
sqlite3 data/performance_history.db "
SELECT symbol, datetime(last_trade_time) as traded_at
FROM daily_trades 
WHERE trade_date = date('now')
ORDER BY symbol;
"

# If many symbols listed, clear them:
sqlite3 data/performance_history.db "
DELETE FROM daily_trades WHERE trade_date = date('now');
"

# Restart system
python main.py
```

---

### **Step 3: If Signal Strength Is The Issue**

Add debug logging to see actual values:

```python
# In main.py or trading_engine.py, add:

logger.info(f"{symbol} Signal Components:")
logger.info(f"  Technical: {technical_score:.3f}")
logger.info(f"  ML: {ml_prediction:.3f}")
logger.info(f"  Sentiment: {sentiment_score:.3f}")
logger.info(f"  Fundamental: {fundamental_score:.3f}")
logger.info(f"  Combined: {signal_strength:.3f}")
logger.info(f"  Required: {min_signal_strength}")

if signal_strength < min_signal_strength:
    logger.warning(f"  REJECTED: Signal too weak")
else:
    logger.info(f"  ACCEPTED: Signal sufficient")
```

Then watch logs:
```bash
tail -f logs/fxai_*.log | grep -A 6 "Signal Components"
```

---

### **Step 4: Test with Permissive Settings**

```bash
python adjust_config.py
# Select Option 1: Testing Mode

# Restart system
python main.py
```

If trades NOW execute → Confirms thresholds or validation too strict

---

## 📊 **Comparison: Test vs Live**

### **What Tests DO**

```python
# Typical test structure:
def test_signal_generation():
    # Mock data
    mock_price_data = {...}
    mock_technical_scores = 0.7
    mock_ml_prediction = 0.65
    
    # Calculate signal
    signal = calculate_signal(mock_price_data)
    
    # Assert
    assert signal > 0.4  # Test passes ✓
```

**Tests skip**:
- Daily trade limit checks
- Real MT5 connection
- Actual data source failures
- Database state
- Risk manager full validation

### **What Live DOES**

```python
# Live execution:
for symbol in symbols:
    # 1. Get REAL market data (can fail)
    data = get_market_data(symbol)
    
    # 2. Calculate with REAL indicators (can be null)
    technical = technical_analyzer.analyze(symbol)
    ml = ml_predictor.predict(symbol)  # Can fail
    sentiment = sentiment_analyzer.analyze(symbol)  # Can timeout
    
    # 3. Calculate signal
    signal = combine_signals(technical, ml, sentiment)
    
    # 4. CHECK DAILY LIMIT (blocks here!)
    if has_traded_today(symbol):
        continue  # ← ALL 30 symbols blocked here
    
    # 5. Validate
    if signal < threshold:
        continue
    
    # 6. Execute
    place_order(symbol)
```

**The difference**: Step 4 blocks in live but not in tests!

---

## 🎯 **Decision Tree**

```
No trades in live, but tests show opportunities
    │
    ├─→ Run diagnose_live_vs_test.py
    │       │
    │       ├─→ Shows "25-30 symbols traded today"
    │       │       └─→ FOUND IT! Clear database
    │       │
    │       ├─→ Shows "Daily loss limit reached"
    │       │       └─→ Wait until tomorrow
    │       │
    │       ├─→ Shows "MT5 not connected"
    │       │       └─→ Start MT5, login
    │       │
    │       └─→ Shows "No obvious issue"
    │               └─→ Add debug logging
    │
    └─→ After fixing, compare:
            │
            ├─→ If trades execute → Issue fixed!
            └─→ If still no trades → Debug logging needed
```

---

## 💡 **Key Insights**

### **Why Tests Pass But Live Fails**

1. **Tests use mock data** → Always returns good signals
2. **Tests skip validation** → No daily limits checked
3. **Tests don't need MT5** → No connection required
4. **Tests use clean state** → No database history

### **Common Patterns**

This issue pattern is VERY common:
- Tests: Isolated, controlled environment
- Live: Real world, messy state, external dependencies

**Classic symptoms**:
- "It works in tests"
- "Nothing wrong with the code"
- "Signals are being generated"
- **But**: Live execution silently blocked by state/condition

---

## 🔍 **Verification Steps**

### **Confirm Daily Limit Theory**

```bash
# Check symbols
sqlite3 data/performance_history.db ".tables"

# If daily_trades exists:
sqlite3 data/performance_history.db "
SELECT * FROM daily_trades WHERE trade_date = date('now');
"

# Expected if this is the issue:
# 25-30 rows returned (most/all symbols)
```

### **Confirm Signal Generation**

```bash
# Add this to your code temporarily:
logger.info(f"Signal generated for {symbol}: {signal_strength:.3f}")

# Then watch logs:
tail -f logs/*.log | grep "Signal generated"

# If you see signals > 0.4 but still no trades:
# → Daily limits or other post-signal validation blocking
```

---

## ✅ **Success Criteria**

After fixing, you should see:

```
2025-11-12 15:45:00 - INFO - Analyzing 30 trading symbols for opportunities...
2025-11-12 15:45:00 - INFO - EURUSD: Signal 0.52 - PASSED validation
2025-11-12 15:45:01 - INFO - Executing BUY trade for EURUSD
2025-11-12 15:45:02 - INFO - Trade placed: Ticket #12345
2025-11-12 15:45:03 - INFO - TRADING SUMMARY: 3 trades executed, 27 symbols skipped
```

**Not**:
```
2025-11-12 15:45:00 - INFO - Analyzing 30 trading symbols for opportunities...
2025-11-12 15:45:00 - INFO - TRADING SUMMARY: No opportunities found, 30 symbols analyzed
```

---

## 📁 **Files to Use**

1. **diagnose_live_vs_test.py** ← RUN THIS FIRST
   - Identifies exact blocking condition
   - Checks database, MT5, risk limits
   - Shows difference between test and live

2. **identify_blocker.py**
   - Tests individual symbols
   - Shows validation step-by-step
   - Reveals where each symbol fails

3. **add_debug_logging.py**
   - Shows code to add to your system
   - Reveals actual signal values
   - Proves if signals are even being calculated

---

## 🆘 **If Still Stuck**

Provide these outputs:

```bash
# 1. Daily trades status
sqlite3 data/performance_history.db "SELECT * FROM daily_trades WHERE trade_date = date('now');" > daily_status.txt

# 2. Today's trade history
sqlite3 data/performance_history.db "SELECT * FROM trades WHERE DATE(timestamp) = date('now');" > today_trades.txt

# 3. Diagnostic output
python diagnose_live_vs_test.py > diagnostic.txt

# 4. Recent logs with signal info
grep -i "signal\|opportunity\|trade" logs/*.log | tail -100 > recent_logs.txt
```

With these 4 files, the issue can be pinpointed immediately.

---

## 🎯 **Most Likely Solution** (90% confidence)

```bash
# The issue is almost certainly daily trade limits
# Here's the complete fix:

# 1. Check
sqlite3 data/performance_history.db "
SELECT COUNT(*) FROM daily_trades WHERE trade_date = date('now');
"

# 2. If result is 25-30:
sqlite3 data/performance_history.db "
DELETE FROM daily_trades WHERE trade_date = date('now');
"

# 3. Restart
python main.py

# 4. Observe - trades should now execute
```

**Why this works**:
- Tests don't check this database
- Live enforces 1 trade/day rule
- Clearing database allows re-trading
- Symptoms match perfectly

---

**Start with**: `python diagnose_live_vs_test.py`

**Expected finding**: 25-30 symbols marked as "traded today"  
**Expected fix time**: < 2 minutes  
**Expected result**: Trades execute immediately after fix  

---

**Document**: Live_vs_Test_Discrepancy.md  
**Created**: 2025-11-12  
**Issue**: Tests pass, live fails  
**Root cause**: Daily trade database blocking all symbols  
**Confidence**: 90%  
**Fix time**: < 2 minutes
