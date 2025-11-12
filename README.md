# FX-Ai Trading System v3.0

## Advanced ML-Powered Forex Trading System

FX-Ai is a comprehensive machine learning-based forex trading system that combines trained ML models with advanced risk management for automated trading across multiple currency pairs and timeframes.

**Version:** 3.0
**Date:** November 12, 2025
**Status:** OPERATIONAL - System running normally

---

## Quick Start

### 1. Launch Main Application

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Run the main application
python main.py
```

### 2. Alternative Startup Methods

```bash
# Using the startup batch file (Windows)
live_trading\start_fxai.bat

# Direct Python execution
python -c "from app.application import FXAiApplication; import asyncio; app = FXAiApplication(); asyncio.run(app.run())"
```

### 3. Emergency Stop (if needed)

```bash
# Using batch file (Windows) - IMMEDIATE SHUTDOWN
live_trading\emergency_stop.bat

# Or run directly
python -c "from app.application import FXAiApplication; import asyncio; app = FXAiApplication(); asyncio.run(app.emergency_stop())"
```

---

## System Architecture

### Core Capabilities

- **ML Model Integration**: Trained models for 30+ currency pairs across multiple timeframes (M15, H1)
- **Adaptive Learning**: Continuous model improvement through reinforcement learning and performance tracking
- **Real-time Trading**: Automated position management with advanced risk controls
- **Multi-Timeframe Support**: Optimized parameters for M15 and H1 timeframes
- **Risk Management**: Dynamic position sizing, 3-trade daily limits per symbol, comprehensive risk controls
- **Fundamental Monitoring**: Real-time news and economic event monitoring during active trades
- **Performance Monitoring**: Real-time dashboard with system health and P&L tracking
- **Market Analysis**: Technical, fundamental, sentiment analysis, and correlation management
- **Advanced Risk Metrics**: Portfolio-level risk assessment and market regime detection

### Supported Symbols (30 Pairs)

**Major FX Pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD
**Cross Pairs:** EURGBP, EURJPY, GBPJPY, AUDJPY, EURCAD, GBPAUD, EURNZD, GBPNZD, etc.
**Metals:** XAUUSD (Gold), XAGUSD (Silver)

---

## Configuration

### Risk Management Settings

- **Max Positions:** 30 concurrent trades
- **Risk per Trade:** $50 (fixed dollar amount)
- **Daily Trade Limit:** 3 trades per symbol per day
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

The system includes comprehensive logging and monitoring. Check the `logs/` directory for detailed system status.

### Log Files

- **Main Application Log**: `logs/FX-Ai_YYYY_MM_DD.log`
- **Crash/Error Log**: `logs/crash_log.txt`
- **Performance Data**: `data/performance_history.db` (SQLite database with 494+ trades)

### Real-time Monitoring Features

- **System Health**: MT5 connection status, component initialization
- **Trading Status**: Current trading permissions, time synchronization
- **Account Info**: Balance, equity, margin utilization
- **Open Positions**: Position count, unrealized P&L by symbol
- **Recent Performance**: Historical trade statistics and win rates
- **Risk Metrics**: Drawdown percentage, correlation analysis, risk level assessment
- **AI/ML Status**: Model performance, adaptive learning progress

---

## Project Structure

```text
FX-Ai/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies (70+ packages)
├── LICENSE.txt                      # License information
├── README.md                        # This documentation
├── .env                             # Environment variables (MT5 credentials)
├── .env.example                     # Environment template
├── .flake8                          # Linting configuration
├── .vscode/                         # VS Code workspace settings
│   ├── settings.json                # Editor configuration
│   └── Extensions.json              # Recommended extensions
├── config/
│   ├── config.json                  # Main system configuration
│   └── adaptive_weights.json        # Adaptive learning weights (auto-generated)
├── ai/                              # Machine Learning components
│   ├── ml_predictor.py              # ML model predictions & training
│   ├── adaptive_learning_manager.py # Model adaptation & learning
│   ├── reinforcement_learning_agent.py # RL optimization
│   ├── correlation_manager.py       # Currency correlation analysis
│   ├── advanced_risk_metrics.py     # Portfolio risk assessment
│   ├── market_regime_detector.py    # Market condition detection
│   ├── trade_analyzer.py            # Trade performance analysis
│   ├── learning_algorithms.py       # Learning algorithm implementations
│   ├── learning_database.py         # Learning data persistence
│   └── __init__.py
├── analysis/                        # Market analysis modules
│   ├── technical_analyzer.py        # Technical indicators & analysis
│   ├── sentiment_analyzer.py        # Market sentiment analysis
│   ├── fundamental_analyzer.py      # Fundamental analysis
│   └── __init__.py
├── app/                             # Application architecture
│   ├── application.py                # Main application class
│   ├── component_initializer.py      # Component initialization
│   ├── trading_orchestrator.py       # Trading orchestration logic
│   └── __init__.py
├── core/                            # Core trading components
│   ├── trading_engine.py            # Main trading logic & execution
│   ├── risk_manager.py              # Risk management system
│   ├── mt5_connector.py             # MT5 platform integration
│   ├── position_manager.py          # Position lifecycle management
│   ├── order_executor.py            # Order execution handling
│   ├── stop_loss_manager.py         # Stop loss management
│   ├── take_profit_manager.py       # Take profit management
│   ├── position_closer.py           # Position closing logic
│   ├── clock_sync.py                # Time synchronization
│   └── __init__.py
├── data/                            # Data management
│   ├── market_data_manager.py       # Market data handling
│   └── performance_history.db       # SQLite trade database (233KB)
├── live_trading/                    # Live trading components
│   ├── fundamental_monitor.py       # Real-time fundamental monitoring
│   ├── dynamic_parameter_manager.py # Parameter optimization
│   ├── start_fxai.bat               # Windows startup script
│   ├── emergency_stop.bat           # Emergency shutdown script
│   └── __init__.py
├── logs/                            # Application logs
│   └── crash_log.txt                # Error logging
├── models/                          # ML models and optimization
│   ├── parameter_optimization/
│   │   └── optimal_parameters.json  # Optimized trading parameters (6.5MB)
│   ├── models_archive/              # Archived ML models (by date)
│   └── models_backup/               # Backup ML models (by date)
├── utils/                           # Utility modules
│   ├── logger.py                    # Comprehensive logging system
│   ├── time_manager.py              # Time management utilities
│   ├── config_loader.py             # Configuration loading
│   ├── position_monitor.py          # Position monitoring
│   ├── risk_validator.py            # Risk validation utilities
│   ├── exceptions.py                # Custom exceptions
│   └── __init__.py
└── venv/                            # Python virtual environment (excluded from git)
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- MetaTrader 5 terminal installed and running
- Valid MT5 trading account (demo or live)
- Windows/Linux/Mac OS

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

4. **Configure Environment Variables**

   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env with your MT5 credentials
   # Required fields:
   # MT5_LOGIN=your_account_number
   # MT5_PASSWORD=your_password
   # MT5_SERVER=your_broker_server
   # MT5_PATH=path_to_mt5_terminal (optional)
   ```

5. **Configure System Settings**

   ```bash
   # Edit config/config.json for your preferences
   # Key settings to review:
   # - Risk management parameters
   # - Trading symbols
   # - Fundamental monitoring settings
   ```

6. **Test System Connection**

   ```bash
   python -c "from core.mt5_connector import MT5Connector; import os; mc = MT5Connector(os.getenv('MT5_LOGIN'), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER')); print('MT5 Connection:', 'SUCCESS' if mc.initialize() else 'FAILED')"
   ```

---

## ML Model Training & Management

### Current Model Status

- **494+ Historical Trades** recorded in performance database
- **30+ Currency Pairs** with trained models (M15 and H1 timeframes)
- **Model Archives**: Automatic backup system with timestamped versions
- **Adaptive Learning**: Continuous model improvement based on trade outcomes

### Automated Training System

The system includes automated model training that analyzes historical trade performance:

```python
from ai.ml_predictor import MLPredictor
from ai.adaptive_learning_manager import AdaptiveLearningManager

# Initialize components
predictor = MLPredictor(config)
learning_manager = AdaptiveLearningManager(config)

# System automatically trains models based on:
# - Historical trade outcomes from database
# - Market conditions and volatility
# - Optimal entry/exit timing patterns
# - Risk-adjusted performance metrics

# Models are automatically retrained when:
# - Performance degrades below thresholds
# - New market conditions detected
# - Weekly optimization cycles
```

### Model Features

- **Ensemble Methods**: RandomForest and GradientBoosting classifiers
- **30+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, VWAP, etc.
- **Multi-timeframe Analysis**: M15 and H1 optimization
- **Reinforcement Learning**: Strategy optimization over time
- **Correlation Analysis**: Inter-market relationship modeling
- **Market Regime Detection**: Adapts to trending vs ranging conditions

---

## Risk Management System

### Safety Features

- **Daily Loss Limits**: Automatic shutdown if daily loss exceeded
- **Position Limits**: Max 30 concurrent positions, 3 trades per symbol daily
- **Spread Filters**: Only trade when spreads ≤ 3.0 pips
- **Time Filters**: Respect market hours and trading sessions
- **Symbol-specific Limits**: Max 1 position per symbol at a time
- **Correlation Controls**: Prevents over-concentration in correlated pairs

### Advanced Risk Metrics

- **Portfolio-level Risk**: Cross-symbol correlation analysis
- **Market Regime Detection**: Adjusts risk based on market conditions
- **Dynamic Position Sizing**: Adapts to account equity and volatility
- **Emergency Circuit Breakers**: Automatic pause during extreme volatility

### Emergency Controls

- **Emergency Stop**: Immediate shutdown of all positions and trading
- **Manual Override**: Administrative controls for intervention
- **Graceful Shutdown**: Proper cleanup and position closure

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

### Fundamental Monitor Configuration

```json
"fundamental_monitor": {
  "enabled": true,
  "check_interval_seconds": 300,
  "high_impact_exit_threshold": 15,
  "sl_tighten_threshold": 30,
  "sl_tighten_percentage": 0.5
}
```

---

## Adaptive Learning System

### Learning Capabilities

- **Performance Tracking**: Continuous evaluation of model accuracy and profitability
- **Automatic Retraining**: Models retrained when performance degrades
- **Parameter Optimization**: Dynamic SL/TP and entry/exit optimization
- **Market Regime Adaptation**: Adjusts strategies for different market conditions
- **Reinforcement Learning**: Optimizes trading strategies based on outcomes
- **Correlation Learning**: Adapts to changing inter-market relationships

### Learning Metrics Tracked

- **Win Rate**: Per symbol, per timeframe, overall portfolio
- **Profit Factor**: Gross profit / gross loss ratio
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-valley decline tracking
- **Trade Timing Optimization**: Best entry/exit times based on historical data
- **Holding Duration Analysis**: Optimal position holding periods

### Database Integration

All learning data is stored in `data/performance_history.db` with 24+ tables tracking:
- Trade outcomes and performance metrics
- Model accuracy and prediction quality
- Market conditions during trades
- Risk management effectiveness
- Adaptive parameter adjustments

---

## Monitoring & Logging

### Comprehensive Logging System

- **Main Application Log**: `logs/FX-Ai_YYYY_MM_DD.log` - All system activities
- **Crash/Error Log**: `logs/crash_log.txt` - Critical errors and crashes
- **Performance Database**: `data/performance_history.db` - Structured trade data

### Real-time Monitoring

- **System Health Checks**: Automatic diagnostics every trading cycle
- **Component Status**: MT5 connection, model loading, risk systems
- **Performance Metrics**: Live P&L, win rates, drawdown tracking
- **Risk Alerts**: Automatic notifications for risk threshold breaches
- **Market Condition Monitoring**: Volatility, spread, and liquidity checks

---

## Troubleshooting

### Common Issues & Solutions

#### MT5 Connection Failed

```bash
# Verify MT5 terminal is running
# Check credentials in .env file
# Ensure MT5 API is enabled in terminal
# Check firewall/antivirus blocking
# Try: python -c "import MetaTrader5 as mt5; print(mt5.initialize())"
```

#### No Trading Signals Generated

```bash
# Check market hours (weekdays only)
# Verify ML models are loaded: check models/ directory
# Review risk management settings in config.json
# Check logs for model prediction errors
```

#### Fundamental Monitor Not Working

```bash
# Verify fundamental_monitor.enabled = true in config.json
# Check fundamental_monitor integration in trading_orchestrator.py
# Review logs for monitoring activity
# Test: python -c "from live_trading.fundamental_monitor import FundamentalMonitor; print('OK')"
```

#### Performance Issues

```bash
# Check system resources (RAM: 8GB+, CPU: multi-core recommended)
# Review log files for errors or warnings
# Consider reducing max_positions setting (< 30)
# Check database size: data/performance_history.db should be < 1GB
```

#### Import Errors

```bash
# Ensure virtual environment is activated
# Reinstall dependencies: pip install -r requirements.txt
# Check Python version: python --version (should be 3.8+)
```

### Emergency Procedures

1. **Immediate Stop**: Run `live_trading/emergency_stop.bat`
2. **Check Logs**: Review `logs/crash_log.txt` and recent log files
3. **Database Check**: Verify `data/performance_history.db` integrity
4. **Restart System**: After resolving issues, restart with `python main.py`
5. **Demo Testing**: Always test fixes on demo account first

---

## Performance Expectations

### Realistic Goals (Based on 494+ Historical Trades)

- **Monthly Return**: 5-15% (depending on account size and market conditions)
- **Win Rate**: 55-70% (with proper risk management)
- **Max Drawdown**: <10% (with circuit breakers active)
- **Trades per Month**: 100-500 (depending on symbol activity and limits)
- **Risk per Trade**: $50 fixed amount (scalable with account size)

### Risk Warnings

⚠️ **Past Performance Does Not Guarantee Future Results**
⚠️ **Forex Trading Involves Risk of Loss**
⚠️ **Never Risk More Than You Can Afford to Lose**
⚠️ **Always Test on Demo Account First**
⚠️ **This System is Continuously Learning - Results May Vary**

---

## Development & Contributing

### Code Architecture

The system uses a modular architecture with clear separation of concerns:

- **app/**: Application lifecycle and orchestration
- **core/**: Core trading functionality
- **ai/**: Machine learning and adaptive systems
- **analysis/**: Market analysis components
- **utils/**: Shared utilities and helpers
- **live_trading/**: Live trading specific components

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly on demo account
4. Ensure all imports work: `python -c "import sys; sys.path.append('.'); from app.application import FXAiApplication"`
5. Submit pull request with detailed description

### Code Standards

- **Type Hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings required
- **Error Handling**: Proper exception handling and logging
- **Testing**: Validate changes don't break existing functionality
- **Performance**: Monitor for memory leaks and performance issues

---

## Database Schema

The system uses SQLite database (`data/performance_history.db`) with 24+ tables:

- **trades_***: Trade records by symbol/timeframe
- **analysis_***: Technical/fundamental analysis data
- **optimization_***: Parameter optimization results
- **performance_***: Performance metrics and statistics
- **learning_***: Machine learning model data

**Database Size**: ~233KB (494+ trades recorded)
**Backup**: Automatic model backups in `models_archive/` and `models_backup/`

---

## License

This project is proprietary software. All rights reserved.

---

## Support & Contact

For technical support or questions:

- **First Step**: Check the troubleshooting section above
- **Log Analysis**: Review `logs/` directory for error details
- **Database Check**: Verify trade data in `data/performance_history.db`
- **Demo Testing**: Always validate on demo account before live deployment

**Disclaimer:** This software is for educational and research purposes. Forex trading involves substantial risk of loss. Always test thoroughly on a demo account before deploying with real money. Past performance does not guarantee future results.

---

Last Updated: November 12, 2025
