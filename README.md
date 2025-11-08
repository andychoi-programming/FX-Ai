# FX-Ai Trading System v3.0

## Advanced ML-Powered Forex Trading System

FX-Ai is a comprehensive machine learning-based forex trading system that combines trained ML models with advanced risk management for automated trading across multiple currency pairs and timeframes.

**Version:** 3.0  
**Date:** November 6, 2025  
**Status:** OPERATIONAL - System running normally

---

## Quick Start

### 1. Launch Main Application

```bash
python main.py
```

### 2. Launch Performance Dashboard (Real-time Monitoring)

```bash
# One-time view
python performance_dashboard.py

# Continuous monitoring (updates every 60 seconds)
python performance_dashboard.py --continuous
```

### 3. Emergency Stop (if needed)

```bash
# Using batch file (Windows)
python live_trading/emergency_stop.bat

# Or run directly
python live_trading/emergency_stop.py
```

---

## System Architecture

### Core Capabilities

- **ML Model Integration**: Trained models for 30+ currency pairs across multiple timeframes
- **Adaptive Learning**: Continuous model improvement through reinforcement learning
- **Real-time Trading**: Automated position management with advanced risk controls
- **Multi-Timeframe Support**: Optimized parameters for M1, M5, M15, H1, D1, W1, MN1
- **Risk Management**: Dynamic position sizing, daily loss limits, trade frequency controls
- **Fundamental Monitoring**: Real-time news and economic event monitoring during active trades
- **Performance Monitoring**: Real-time dashboard with system health and P&L tracking
- **Market Analysis**: Technical, fundamental, and sentiment analysis integration

### Supported Symbols (30 Pairs)

**Major FX Pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD  
**Cross Pairs:** EURGBP, EURJPY, GBPJPY, AUDJPY, EURCAD, GBPAUD, EURNZD, GBPNZD, etc.  
**Metals:** XAUUSD (Gold), XAGUSD (Silver)

---

## Configuration

### Risk Management Settings

- **Max Positions:** 30 concurrent trades
- **Risk per Trade:** $50 (fixed dollar amount)
- **Default SL/TP:** 20/40 pips
- **Max Spread:** 3.0 pips
- **Lot Size Range:** 0.01 - 1.0 lots

### Fundamental Monitoring Settings

- **Check Interval:** Every 5 minutes (configurable)
- **High Impact Exit:** Exit positions within 15 minutes of contradicting news
- **SL Tightening:** Tighten stops for upcoming high-impact events (within 30 minutes)
- **Profit Locking:** Lock in profits during volatile news events

### Trading Hours

- **Market Open:** Monday 00:00 GMT (Sunday close)
- **Market Close:** Friday 23:59 GMT
- **No Trading:** Weekend hours (Saturday 00:00 - Monday 00:00 GMT)

---

## Performance Dashboard

The `performance_dashboard.py` provides real-time monitoring:

### Features

- **System Health**: MT5 connection status, TimeManager status
- **Trading Status**: Current trading permissions, time until close
- **Account Info**: Balance, equity, margin utilization
- **Open Positions**: Position count, unrealized P&L by symbol
- **Recent Performance**: 24-hour trade statistics and win rates
- **Risk Metrics**: Drawdown percentage, margin utilization, risk level assessment

### Sample Output

```text
FX-AI PERFORMANCE DASHBOARD
SYSTEM STATUS: HEALTHY
TRADING STATUS: Trading Allowed
ACCOUNT INFO: Balance $10,000 | Equity $10,250
OPEN POSITIONS: 3 positions | Unrealized P&L +$250
RECENT PERFORMANCE: 180 trades | Win Rate 65.2%
RISK METRICS: Drawdown 0.5% | Risk Level: LOW
```

---

## Project Structure

```text
FX-Ai/
├── main.py                          # Main application entry point
├── performance_dashboard.py         # Real-time monitoring dashboard
├── requirements.txt                 # Python dependencies
├── config/
│   └── config.json                  # System configuration
├── ai/                              # Machine Learning components
│   ├── ml_predictor.py              # ML model predictions
│   ├── adaptive_learning_manager.py # Model adaptation system
│   └── reinforcement_learning_agent.py # RL optimization
├── core/                            # Core trading components
│   ├── trading_engine.py            # Main trading logic
│   ├── risk_manager.py              # Risk management system
│   └── mt5_connector.py             # MT5 integration
├── live_trading/                    # Live trading components
│   ├── fundamental_monitor.py       # Real-time fundamental monitoring
│   ├── dynamic_parameter_manager.py # Parameter optimization
│   ├── FX-Ai_Start.bat              # Windows startup script
│   └── emergency_stop.py            # Emergency shutdown
├── utils/                           # Utility modules
│   ├── logger.py                    # Logging system
│   ├── time_manager.py              # Time management
│   └── config_loader.py             # Configuration loading
├── analysis/                        # Market analysis
├── data/                            # Data management
├── models/                          # Trained ML models
├── logs/                            # Application logs
└── .env                             # Environment variables (credentials)
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- MetaTrader 5 terminal
- Valid MT5 trading account

### Installation Steps

1. **Clone Repository**

   ```bash
   git clone https://github.com/andychoi-programming/FX-Ai.git
   cd FX-Ai
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**

   ```bash
   # Windows: Copy environment template
   copy .env.example .env
   
   # Linux/Mac: Copy environment template
   cp .env.example .env

   # Edit .env with your MT5 credentials
   # MT5_LOGIN=your_login
   # MT5_PASSWORD=your_password
   # MT5_SERVER=your_server
   ```

5. **Configure System**

   ```bash
   # Edit config/config.json for your preferences
   # Adjust risk settings, symbols, etc.
   ```

6. **Test Connection**

   ```bash
   python main.py --test-connection
   ```

---

## ML Model Training

### Automated Training

The system includes automated model training for new symbols. Training data (X_train, y_train) is automatically prepared from historical market data:

```python
from ai.ml_predictor import MLPredictor
import pandas as pd

# Initialize predictor
predictor = MLPredictor(config)

# Load historical data
historical_data = pd.read_csv('data/EURUSD_H1.csv')

# Prepare features and labels (done automatically by the system)
X_train, y_train = predictor.prepare_training_data(historical_data)

# Train model for specific symbol and timeframe
predictor.train_symbol_model('EURUSD', 'H1', X_train, y_train)

# Or use the automatic training method
predictor._train_model('EURUSD', historical_data, 'H1')
```

### Model Features

- **RandomForest/GradientBoosting** classifiers
- **30+ Technical Indicators** (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- **Multi-timeframe Analysis** (M1, M5, M15, H1, D1, W1, MN1)
- **Adaptive Learning** with reinforcement learning optimization

---

## Risk Management

### Safety Features

- **Daily Loss Limits**: Automatic shutdown if daily loss exceeded
- **Max Position Limits**: Prevents over-leveraging
- **Spread Filters**: Only trade when spreads are reasonable
- **Time Filters**: Respect market hours and trading sessions
- **Symbol-specific Limits**: Max 1 position per symbol at a time

### Emergency Controls

- **Emergency Stop**: Immediate shutdown of all trading
- **Circuit Breakers**: Automatic pause during high volatility
- **Manual Override**: Administrative controls for intervention

---

## Fundamental Monitoring

### Overview

The **FundamentalMonitor** provides real-time fundamental monitoring during active trades, automatically protecting capital or locking in profits based on breaking news and economic events.

### Key Features

- **Real-time News Monitoring**: Checks for breaking news every 5 minutes
- **Economic Event Detection**: Monitors high-impact economic releases
- **Automatic Risk Management**: Takes protective actions based on event severity
- **Position Protection**: Exits positions, tightens stops, or locks profits

### Actions Taken

#### EXIT Position

- When high-impact news contradicts your position (within 15 minutes)
- Example: Long EUR/USD during negative US employment data

#### TIGHTEN Stop Loss

- When high-impact news is upcoming (within 30 minutes)
- Moves SL closer to entry (50% of current distance by default)

#### LOCK PROFITS

- When high-impact news detected and position is profitable
- Moves SL to breakeven + 20% buffer to protect gains

#### Fundamental Monitor Configuration

```json
"fundamental_monitor": {
  "enabled": true,
  "check_interval_seconds": 300,
  "high_impact_exit_threshold": 15,
  "sl_tighten_threshold": 30,
  "sl_tighten_percentage": 0.5
}
```

### Benefits

- **Disaster Prevention**: Avoids large losses from surprise announcements
- **Profit Protection**: Locks in gains before volatile events
- **24/7 Monitoring**: Automatic monitoring without manual intervention
- **Adaptive Risk Management**: Dynamic SL/TP based on market events

---

## Monitoring & Logging

### Log Files

- **Main Log**: `logs/FX-Ai_YYYY_MM_DD.log`
- **Error Log**: `logs/error_YYYY_MM_DD.log`
- **Performance Log**: `logs/performance_YYYY_MM_DD.log`

### Real-time Monitoring

- **Performance Dashboard**: Live system status
- **Health Checks**: Automatic system diagnostics
- **Alert System**: Email/SMS notifications for critical events

---

## Adaptive Learning System

### Adaptive Features

- **Model Performance Tracking**: Continuous evaluation of model accuracy
- **Automatic Retraining**: Models retrained when performance degrades
- **Parameter Optimization**: Dynamic SL/TP and entry/exit optimization
- **Market Regime Detection**: Adapts to different market conditions
- **Reinforcement Learning**: Optimizes trading strategies over time

### Learning Metrics

- **Win Rate Tracking**: Per symbol, per timeframe
- **Profit Factor**: Gross profit / gross loss ratio
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-valley decline tracking

---

## Troubleshooting

### Common Issues

#### MT5 Connection Failed

```bash
# Check MT5 terminal is running
# Verify credentials in .env file
# Check firewall/antivirus blocking
```

#### No Trading Signals

```bash
# Verify ML models are trained
# Check market hours (weekdays only)
# Review risk management settings
```

#### Fundamental Monitor Not Working

```bash
# Check fundamental_monitor.enabled = true in config.json
# Verify fundamental_monitor.py is integrated in main.py
# Check logs for monitoring activity
# Test with python test_fundamental_monitor.py
```

#### Performance Issues

```bash
# Check system resources (RAM, CPU)
# Review log files for errors
# Consider reducing max_positions setting
```

### Emergency Procedures

1. Run `emergency_stop.py` to halt all trading
2. Check logs for error details
3. Restart system after resolving issues
4. Contact support if needed

---

## Performance Expectations

### Realistic Goals

- **Monthly Return**: 5-15% (depending on account size and market conditions)
- **Win Rate**: 55-70% (with proper risk management)
- **Max Drawdown**: <10% (with circuit breakers)
- **Trades per Month**: 100-500 (depending on settings)

### Risk Warnings

- **Past Performance Does Not Guarantee Future Results**
- **Forex Trading Involves Risk of Loss**
- **Never Risk More Than You Can Afford to Lose**
- **Always Test on Demo Account First**

---

## Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Code Standards

- **Type Hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings required
- **Testing**: Unit tests for critical functions
- **Logging**: Proper error handling and logging

---

## License

This project is proprietary software. All rights reserved.

---

## Support

For technical support or questions:

- Check the troubleshooting section above
- Review log files for error details
- Test on demo account before live trading

**Disclaimer:** This software is for educational and research purposes. Use at your own risk. Always test thoroughly before deploying with real money.
