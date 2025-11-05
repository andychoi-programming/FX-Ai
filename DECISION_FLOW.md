# Trade Decision & SL/TP Flow

## ✅ YES - All 3 Analyzers Influence Trade Decisions

### 📊 Signal Calculation (Weighted Average)

The system combines all 3 analyzers into a **weighted signal strength**:

```python
signal_strength = (
    0.25 * technical_score     +  # Technical Analyzer (25%)
    0.30 * ml_prediction       +  # ML Predictor (30%)
    0.20 * sentiment_score     +  # Sentiment Analyzer (20%)
    0.15 * fundamental_score   +  # Fundamental Analyzer (15%)
    0.10 * sr_score               # Support/Resistance (10%)
)
```

**Trade Decision**: Only opens trade if `signal_strength > 0.4` (40% threshold)

### 🎯 Trade Direction Decision

1. **Primary**: ML Predictor determines BUY/SELL direction
2. **Override**: Reinforcement Learning agent can override ML if enabled
3. **Fallback**: If ML fails, uses Technical Analyzer only

---

## ✅ YES - SL/TP NOW ADJUSTED BY ALL 3 ANALYZERS!

### 🛑 Stop Loss Calculation

**Base Calculation** (Technical Analyzer ATR):

```python
# For FOREX:
base_sl_distance = ATR * sl_atr_multiplier (default: 3.0)

# For METALS (XAUUSD/XAGUSD):
base_sl_distance = optimized_sl_pips * pip_size
```

**Dynamic Adjustments** (Sentiment + Fundamental):

```python
sl_adjustment_factor = 1.0  # Start with base

# Sentiment adjustments:
if sentiment_score > 0.7:
    sl_adjustment_factor *= 0.90  # Tighten SL by 10% (strong positive sentiment)
elif sentiment_score < 0.3:
    sl_adjustment_factor *= 1.15  # Widen SL by 15% (strong negative sentiment)

# Fundamental adjustments:
if high_impact_news_soon:
    sl_adjustment_factor *= 1.20  # Widen SL by 20% (protect from volatility spike)
elif fundamental_score < 0.3:
    sl_adjustment_factor *= 1.10  # Widen SL by 10% (weak fundamentals)

# Final stop loss distance
stop_loss_distance = base_sl_distance * sl_adjustment_factor
```

**Sources**:
- ✅ Technical Analyzer → ATR value (base calculation)
- ✅ Sentiment Analyzer → Adjusts SL tightness based on market mood
- ✅ Fundamental Analyzer → Adjusts SL for news events and fundamental strength
- ✅ Parameter Manager → Optimized sl_pips (for metals)
- ✅ Adaptive Learning → ATR multiplier adjustments

### 🎯 Take Profit Calculation

**Base Calculation** (Technical Analyzer ATR):

```python
# For FOREX:
base_tp_distance = ATR * tp_atr_multiplier (default: 6.0)

# For METALS (XAUUSD/XAGUSD):
base_tp_distance = optimized_tp_pips * pip_size
```

**Dynamic Adjustments** (Sentiment + Fundamental):

```python
tp_adjustment_factor = 1.0  # Start with base

# Sentiment adjustments:
if sentiment_score > 0.7:
    tp_adjustment_factor *= 1.15  # Extend TP by 15% (ride the trend)
elif sentiment_score < 0.3:
    tp_adjustment_factor *= 0.90  # Tighten TP by 10% (take profits quickly)

# Fundamental adjustments:
if fundamental_score > 0.7:
    tp_adjustment_factor *= 1.20  # Extend TP by 20% (strong fundamentals)
elif fundamental_score < 0.3:
    tp_adjustment_factor *= 0.85  # Tighten TP by 15% (weak fundamentals)
elif high_impact_news_soon:
    tp_adjustment_factor *= 0.90  # Tighten TP by 10% (lock gains before news)

# Final take profit distance
take_profit_distance = base_tp_distance * tp_adjustment_factor
```

**Sources**:
- ✅ Technical Analyzer → ATR value (base calculation)
- ✅ Sentiment Analyzer → Extends/tightens TP based on market mood
- ✅ Fundamental Analyzer → Adjusts TP for news events and fundamental strength
- ✅ Parameter Manager → Optimized tp_pips (for metals)
- ✅ Adaptive Learning → ATR multiplier adjustments

---

## 📈 What Each Analyzer Actually Does

### 1. **Technical Analyzer** (`analysis/technical_analyzer.py`)
- **Purpose**: Calculate indicators (RSI, MACD, Bollinger Bands, ATR, ADX, etc.)
- **Used For**:
  - ✅ Trade decision (25% weight in signal_strength)
  - ✅ SL calculation (ATR value)
  - ✅ TP calculation (ATR value)
  - ✅ Regime detection (trending/ranging)
- **Output**: `technical_score` (0-1), ATR value, indicator values

### 2. **Sentiment Analyzer** (`analysis/sentiment_analyzer.py`)
- **Purpose**: Analyze market sentiment from news/social media
- **Used For**:
  - ✅ Trade decision (20% weight in signal_strength)
  - ❌ SL/TP calculation (NOT used)
- **Output**: `sentiment_score` (0-1)

### 3. **Fundamental Analyzer** (`analysis/fundamental_analyzer.py`)
- **Purpose**: Economic calendar, news events, correlations
- **Used For**:
  - ✅ Trade decision (15% weight in signal_strength)
  - ❌ SL/TP calculation (NOT used)
- **Output**: `fundamental_score` (0-1)

---

## 🔄 Complete Trade Flow

```
1. Collect Data
   ├─ Technical Analyzer → indicators, ATR
   ├─ Sentiment Analyzer → sentiment score
   └─ Fundamental Analyzer → fundamental score, news events

2. Calculate Signal Strength
   └─ Weighted average of all 3 + ML + S/R = signal_strength

3. Trade Decision
   └─ IF signal_strength > 0.4 → PROCEED, else SKIP

4. Determine Direction
   ├─ ML Predictor → BUY or SELL
   └─ RL Agent → Can override ML direction

5. Calculate Base SL/TP (Technical ATR)
   ├─ Base SL = ATR * 3.0 (or optimized_sl_pips for metals)
   └─ Base TP = ATR * 6.0 (or optimized_tp_pips for metals)

6. Apply Sentiment & Fundamental Adjustments
   ├─ Adjust SL based on sentiment + fundamental strength + news
   └─ Adjust TP based on sentiment + fundamental strength + news

7. Validate Risk-Reward
   └─ Must be >= 2.0:1 ratio

8. Place Order
   └─ Submit to MT5 with dynamically adjusted SL/TP
```

---

## 💡 Key Insights

### What Works Well:
✅ All 3 analyzers influence **WHETHER** to trade (decision)
✅ Technical ATR provides dynamic, market-adaptive SL/TP
✅ Sentiment + Fundamental help filter out weak signals

### New Intelligent Risk Management:
✅ **SL/TP now dynamically adjust based on market conditions**:
- ✅ Widen SL during high-impact news events (protect from volatility spikes)
- ✅ Tighten TP during negative sentiment (take profits before reversal)
- ✅ Extend TP with strong fundamentals (ride the trend longer)
- ✅ Adjust SL based on sentiment strength (confident vs defensive positioning)

### Current Behavior:
- **Trade Entry**: Multi-factor decision (all 3 analyzers)
- **Trade Exit**: Multi-factor decision (ATR + Sentiment + Fundamentals)

---

## 📊 Current Weights (Adaptive)

These weights are stored in `data/signal_weights.json` and **automatically adjusted** by adaptive learning based on performance:

```json
{
  "technical_score": 0.25,      // Can increase if technical analysis performs well
  "ml_prediction": 0.30,        // Can increase if ML predictions are accurate
  "sentiment_score": 0.20,      // Can increase if sentiment correlates with profits
  "fundamental_score": 0.15,    // Can increase if fundamentals show predictive power
  "sr_score": 0.10              // Support/Resistance levels
}
```

The system continuously learns which components are most predictive and adjusts weights accordingly!
