# Trading Rules Quick Reference

**Location:** `config/config.json` → `"trading_rules"` section

## Critical Rules Summary

### Position & Time Rules
| Rule | Value | Description |
|------|-------|-------------|
| **Close Time** | 22:30 | All positions closed at 22:30 MT5 server time |
| **Trades/Symbol/Day** | 1 | Only ONE trade per symbol per day |
| **Max Positions** | 30 | Maximum total open positions |
| **Day Trading Only** | Yes | No overnight positions |

### Risk Rules
| Rule | Value | Description |
|------|-------|-------------|
| **Risk Per Trade** | $50 | Fixed dollar risk (not percentage) |
| **Max Daily Loss** | $500 | Stop trading if reach $500 loss |
| **Max Consecutive Losses** | 3 | Pause after 3 losses in a row |
| **Max Daily Trades** | 10 | Maximum trades per day (all symbols) |

### Entry Rules
| Rule | Value | Description |
|------|-------|-------------|
| **Min Signal Strength** | 0.4 | Must be > 0.4 to enter |
| **Max Spread** | 3 pips | No trade if spread > 3 pips |
| **Min Risk/Reward** | 2:1 | TP must be at least 2× SL |
| **ML Confirmation** | Required | ML model must agree |

### Stop Loss Rules
| Type | Multiplier | Description |
|------|-----------|-------------|
| **Forex** | ATR × 3.0 | Dynamic based on volatility |
| **Metals** | ATR × 2.5 | Lower for metals |
| **Sentiment Adjustment** | 0.9×-1.15× | Tighter when bullish, wider when bearish |
| **News Adjustment** | 1.2× | Wider near high-impact news |

### Take Profit Rules
| Type | Multiplier | Description |
|------|-----------|-------------|
| **Forex** | ATR × 6.0 | Dynamic based on volatility |
| **Metals** | ATR × 5.0 | Lower for metals |
| **Sentiment Adjustment** | 0.9×-1.15× | Extend when bullish, tighten when bearish |
| **Fundamental Adjustment** | 0.85×-1.2× | Extend with strong fundamentals |

### Exit Rules
| Rule | Value | Description |
|------|-------|-------------|
| **Trailing Stop** | 20 pips | Activates after 20 pips profit |
| **Trail Distance** | 15 pips | Trails 15 pips behind price |
| **Breakeven** | 15 pips | Move to breakeven at 15 pips profit |

### Other Rules
| Rule | Value | Description |
|------|-------|-------------|
| **Cooldown** | 5 minutes | Per symbol after closing position |
| **News Buffer** | 60 minutes | Avoid trading 60 min before/after news |
| **Preferred Sessions** | London, NY | Trade during liquid sessions |

## Quick Modifications

### Change Close Time to 21:00
```json
"time_restrictions": {
  "close_hour": 21,
  "close_minute": 0
}
```

### Allow 2 Trades Per Symbol Per Day
```json
"position_limits": {
  "max_trades_per_symbol_per_day": 2
}
```

### Increase Risk to $100
```json
"risk_limits": {
  "risk_per_trade": 100.0,
  "max_daily_loss": 1000.0
}
```

### Tighter Spread (2 pips)
```json
"entry_rules": {
  "max_spread": 2.0
}
```

## Files to Check

- **Config:** `config/config.json` (trading_rules section)
- **Risk Manager:** `core/risk_manager.py` (rule enforcement)
- **Main Loop:** `main.py` (rule validation)
- **Full Docs:** `TRADING_RULES_CONFIG.md`

## Key Takeaways

1. ✅ All rules in ONE place (`config/config.json` → `trading_rules`)
2. ✅ Easy to modify - change one value, affects entire system
3. ✅ **22:30 close time** - No overnight risk
4. ✅ **One trade per symbol per day** - Disciplined trading
5. ✅ **Fixed $50 risk** - Consistent risk management
6. ✅ **2:1 minimum R:R** - Profitable over time
7. ✅ **Dynamic SL/TP** - Adapts to market conditions
