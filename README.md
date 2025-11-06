# FX-Ai Trading System v3.0

## Advanced ML-Powered Forex Trading System

FX-Ai is a comprehensive machine learning-based forex trading system that combines trained ML models with advanced risk management for automated trading across multiple currency pairs and timeframes.

**Version:** 3.0  
**Date:** November 6, 2025  
**Status:**  OPERATIONAL - System running normally

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
.\live_trading\emergency_stop.bat

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
 main.py                          # Main application entry point
 performance_dashboard.py         # Real-time monitoring dashboard
 requirements.txt                 # Python dependencies
 config/
    config.json                  # System configuration
 ai/                             # Machine Learning components
    ml_predictor.py             # ML model predictions
    adaptive_learning_manager.py # Model adaptation system
    reinforcement_learning_agent.py # RL optimization
 core/                           # Core trading components
    trading_engine.py           # Main trading logic
    risk_manager.py             # Risk management system
    mt5_connector.py            # MT5 integration
 live_trading/                   # Live trading components
    dynamic_parameter_manager.py # Parameter optimization
    FX-Ai_Start.bat            # Windows startup script
    emergency_stop.py           # Emergency shutdown
 utils/                          # Utility modules
    logger.py                   # Logging system
    time_manager.py             # Time management
    config_loader.py            # Configuration loading
 analysis/                       # Market analysis
 data/                           # Data management
 models/                         # Trained ML models
 logs/                           # Application logs
 .env                            # Environment variables (credentials)
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
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**

   ```bash
   # Copy environment template
   copy .env.example .env

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

The system includes automated model training for new symbols:

```python
from ai.ml_predictor import MLPredictor

# Initialize predictor
predictor = MLPredictor(config)

# Train model for specific symbol and timeframe
predictor.train_symbol_model('EURUSD', 'H1', X_train, y_train)
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

- **Past Performance  Future Results**
- **Forex Trading Involves Risk of Loss**
- **Never Risk More Than You Can Afford**
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
