# Trading Rules Configuration

## Overview

All trading rules are now consolidated in one place: `config/config.json` under the `trading_rules` section. This makes it easy to view, modify, and manage all trading constraints from a single location.

## Configuration Location

```json
File: config/config.json
Section: "trading_rules"
```

## Complete Trading Rules

### 1. Position Limits

```json
"position_limits": {
  "max_positions": 30,                          // Maximum total open positions
  "max_positions_per_symbol": 1,               // Only 1 position per symbol at a time
  "prevent_multiple_positions_per_symbol": true,
  "max_trades_per_symbol_per_day": 1           // ⭐ ONE trade per symbol per day
}
```

**Rules:**
- ✅ Maximum 30 positions across all symbols
- ✅ Only 1 open position per symbol at any time
- ✅ **ONE trade per symbol per day** (no re-entry same day)
- ✅ If no opportunity, no trade is acceptable

### 2. Time Restrictions

```json
"time_restrictions": {
  "day_trading_only": true,        // Close all positions before day end
  "close_hour": 22,                // ⭐ Close time: 22:30 MT5 server time
  "close_minute": 30,
  "close_before_weekend": true,    // Close all before weekend
  "close_on_shutdown": true        // Close all on system shutdown
}
```

**Rules:**
- ✅ **All trades closed at 22:30** (MT5 server time)
- ✅ No overnight positions
- ✅ All positions closed before weekend
- ✅ System closes all trades on shutdown

### 3. Risk Limits

```json
"risk_limits": {
  "risk_per_trade": 50.0,          // Fixed $50 risk per trade
  "risk_type": "fixed_dollar",
  "max_daily_loss": 500.0,         // Stop trading if lose $500 in a day
  "max_consecutive_losses": 3,     // Pause after 3 losses in a row
  "pause_after_losses": true,
  "max_daily_trades": 10           // Maximum 10 trades per day total
}
```

**Rules:**
- ✅ Fixed $50 risk per trade (not percentage)
- ✅ Stop trading if daily loss reaches $500
- ✅ Pause trading after 3 consecutive losses
- ✅ Maximum 10 trades per day across all symbols

### 4. Position Sizing

```json
"position_sizing": {
  "min_lot_size": 0.01,
  "max_lot_size": 1.0,
  "risk_calculation_method": "pip_based",
  "margin_safety_factor": 0.95
}
```

**Rules:**
- ✅ Position size calculated to risk exactly $50
- ✅ Minimum 0.01 lots, maximum 1.0 lots
- ✅ Based on pip distance to stop loss
- ✅ 95% margin safety factor

### 5. Entry Rules

```json
"entry_rules": {
  "min_signal_strength": 0.4,          // Minimum 0.4 signal strength
  "max_spread": 3.0,                   // Maximum 3 pips spread
  "min_risk_reward_ratio": 2.0,        // Minimum 2:1 risk/reward
  "require_ml_confirmation": true,
  "require_technical_confirmation": true
}
```

**Rules:**
- ✅ Signal strength must be > 0.4
- ✅ Spread must be < 3 pips
- ✅ Risk/Reward ratio must be > 2:1 (TP at least 2x SL)
- ✅ ML model must confirm direction
- ✅ Technical indicators must confirm

### 6. Stop Loss Rules

```json
"stop_loss_rules": {
  "method": "atr_based",
  "sl_atr_multiplier_forex": 3.0,      // Forex: SL = ATR × 3.0
  "sl_atr_multiplier_metals": 2.5,     // Metals: SL = ATR × 2.5
  "default_sl_pips": 20,
  "max_sl_pips": 50,
  "sl_adjustment_sentiment": true,     // Adjust SL based on sentiment
  "sl_adjustment_fundamental": true    // Adjust SL based on fundamentals
}
```

**Rules:**
- ✅ SL based on ATR (dynamic, market-adaptive)
- ✅ Forex: 3.0× ATR, Metals: 2.5× ATR
- ✅ Adjusted by sentiment (0.9x-1.15x)
- ✅ Adjusted by fundamentals (1.0x-1.2x)
- ✅ Maximum 50 pips stop loss

**SL Adjustments:**
| Condition | Adjustment | Reason |
|-----------|-----------|---------|
| Strong positive sentiment (>0.7) | 0.90× (tighter) | Protect gains |
| Strong negative sentiment (<0.3) | 1.15× (wider) | Avoid premature stops |
| High-impact news (<1hr) | 1.20× (wider) | Survive volatility spike |
| Weak fundamentals (<0.3) | 1.10× (wider) | Less confident trade |

### 7. Take Profit Rules

```json
"take_profit_rules": {
  "method": "atr_based",
  "tp_atr_multiplier_forex": 6.0,      // Forex: TP = ATR × 6.0
  "tp_atr_multiplier_metals": 5.0,     // Metals: TP = ATR × 5.0
  "default_tp_pips": 40,
  "tp_adjustment_sentiment": true,     // Adjust TP based on sentiment
  "tp_adjustment_fundamental": true    // Adjust TP based on fundamentals
}
```

**Rules:**
- ✅ TP based on ATR (dynamic, market-adaptive)
- ✅ Forex: 6.0× ATR, Metals: 5.0× ATR
- ✅ Adjusted by sentiment (0.9x-1.15x)
- ✅ Adjusted by fundamentals (0.85x-1.2x)

**TP Adjustments:**
| Condition | Adjustment | Reason |
|-----------|-----------|---------|
| Strong positive sentiment (>0.7) | 1.15× (extend) | Ride the trend |
| Strong negative sentiment (<0.3) | 0.90× (tighten) | Quick profit |
| Strong fundamentals (>0.7) | 1.20× (extend) | Trend continues |
| Weak fundamentals (<0.3) | 0.85× (tighten) | Take profits early |
| High-impact news (<1hr) | 0.90× (tighten) | Lock gains before volatility |

### 8. Exit Rules

```json
"exit_rules": {
  "trailing_stop_enabled": true,
  "trailing_activation_pips": 20,      // Activate after 20 pips profit
  "trailing_distance_pips": 15,        // Trail 15 pips behind
  "breakeven_enabled": true,
  "breakeven_activation_pips": 15      // Move to breakeven at 15 pips
}
```

**Rules:**
- ✅ Trailing stop activates at 20 pips profit
- ✅ Trails 15 pips behind current price
- ✅ Breakeven activated at 15 pips profit
- ✅ Protects gains automatically

### 9. Cooldown Rules

```json
"cooldown_rules": {
  "symbol_cooldown_minutes": 5,        // 5-minute cooldown after closing
  "cooldown_after_loss": true          // Cooldown applies after losses
}
```

**Rules:**
- ✅ 5-minute cooldown per symbol after position closes
- ✅ Prevents revenge trading
- ✅ Allows system to reassess market

### 10. News Filter

```json
"news_filter": {
  "enabled": true,
  "avoid_high_impact_news": true,
  "news_buffer_minutes": 60           // Avoid trading 60 min before/after news
}
```

**Rules:**
- ✅ Avoid high-impact news events
- ✅ 60-minute buffer before/after news
- ✅ Widen SL if news within 1 hour
- ✅ Tighten TP if news within 1 hour

### 11. Session Filter

```json
"session_filter": {
  "enabled": true,
  "preferred_sessions": ["london", "newyork", "overlap"]
}
```

**Rules:**
- ✅ Trade during liquid sessions only
- ✅ London, New York, or overlap periods
- ✅ Avoid Asian session (low liquidity)

---

## Key Trading Rules Summary

### Critical Rules (Always Enforced)

1. **22:30 Close Time** - All positions closed at 22:30 MT5 server time
2. **One Trade Per Symbol Per Day** - Maximum 1 trade per symbol per day
3. **$50 Fixed Risk** - Every trade risks exactly $50
4. **2:1 Risk/Reward** - Minimum 2:1 TP:SL ratio required
5. **Max 3 Pips Spread** - No trade if spread > 3 pips

### Adaptive Rules (Market-Dependent)

1. **ATR-Based SL/TP** - Dynamic based on current volatility
2. **Sentiment Adjustments** - Tighter/wider SL & TP based on sentiment
3. **Fundamental Adjustments** - Adjust for economic strength
4. **News Protection** - Wider SL, tighter TP near high-impact news
5. **Trailing Stop** - Locks in profits automatically

---

## How to Modify Rules

### Example 1: Change close time to 21:00

```json
"time_restrictions": {
  "close_hour": 21,
  "close_minute": 0
}
```

### Example 2: Allow 2 trades per symbol per day

```json
"position_limits": {
  "max_trades_per_symbol_per_day": 2
}
```

### Example 3: Increase risk to $100 per trade

```json
"risk_limits": {
  "risk_per_trade": 100.0,
  "max_daily_loss": 1000.0
}
```

### Example 4: Tighter spread requirement (2 pips)

```json
"entry_rules": {
  "max_spread": 2.0
}
```

---

## Rule Enforcement

### RiskManager (core/risk_manager.py)
- ✅ Enforces position limits
- ✅ Tracks daily trades per symbol
- ✅ Enforces daily loss limits
- ✅ Manages cooldowns
- ✅ Validates spreads

### TradingEngine (core/trading_engine.py)
- ✅ Places orders
- ✅ Sets SL/TP levels
- ✅ Manages trailing stops
- ✅ Closes positions at 22:30

### Main Trading Loop (main.py)
- ✅ Checks time restrictions
- ✅ Validates signal strength
- ✅ Calculates SL/TP with adjustments
- ✅ Enforces risk/reward ratios
- ✅ Records trades for daily limits

---

## Testing Rules

### Test Script: `test_daily_limit.py`

Run to verify daily trade limit:

```bash
python test_daily_limit.py
```

### Expected Output:

```
Testing EURUSD:
  Has traded today? False
  Can trade? True
  [ACTION] Trade executed for EURUSD
  Has traded today (after trade)? True
  Can trade again? False
```

---

## Benefits of Consolidated Rules

1. **Easy to Find** - All rules in one place
2. **Easy to Modify** - Change one value, affects entire system
3. **Easy to Understand** - Clear structure and comments
4. **Easy to Test** - Verify rules in isolation
5. **Easy to Document** - Single source of truth

---

## Version History

- **v3.0** - Consolidated all trading rules into `trading_rules` section
- **v2.5** - Added one trade per symbol per day rule
- **v2.0** - Added sentiment/fundamental SL/TP adjustments
- **v1.5** - Added 22:30 close time configuration
- **v1.0** - Initial trading rules in separate config sections

---

## Related Files

- `config/config.json` - Main configuration (trading_rules section)
- `core/risk_manager.py` - Risk management enforcement
- `main.py` - Trading loop and rule validation
- `DAILY_TRADE_LIMIT.md` - Daily trade limit documentation
- `README.md` - Complete system documentation
