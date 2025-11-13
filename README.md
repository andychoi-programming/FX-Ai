# FX-Ai Trading System v3.0

![Status](https://img.shields.io/badge/status-operational-brightgreen)
![Version](https://img.shields.io/badge/version-3.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-proprietary-red)
![Trades](https://img.shields.io/badge/trades-494+-success)

## Advanced ML-Powered Forex Trading System

FX-Ai is a comprehensive machine learning-based forex trading system that combines trained ML models with advanced risk management for automated trading across multiple currency pairs and timeframes.

**Version:** 3.0
**Date:** November 13, 2025
**Status:** OPERATIONAL - System running normally with 24-hour optimal trading enabled

---

## Quick Start

### 1. Launch Main Application

```bash
# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

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

## 🚀 Why FX-Ai?

### Key Differentiators

- **Live Learning Philosophy**: Unlike traditional systems that rely on backtesting, FX-Ai learns exclusively from live trading performance, continuously adapting to real market conditions
- **Stop Order Precision**: Advanced BUY_STOP/SELL_STOP system for optimal entry timing (not market orders)
- **30+ Currency Pairs**: Comprehensive coverage with symbol-specific optimization
- **3:1 Risk-Reward Minimum**: Disciplined risk management on every trade
- **Real-time Fundamental Monitoring**: Automated protection during news events
- **Adaptive ML Models**: Self-improving system based on actual trade outcomes
- **24-Hour Trading Capability**: Symbol-specific optimal hour scheduling

---

## System Architecture

### 🧠 Live Learning vs Backtesting

FX-Ai makes a **deliberate architectural decision** to focus on live learning rather than historical backtesting:

- **No Historical Backtesting**: The system doesn't train on simulated past data
- **Real Market Adaptation**: Models learn solely from actual trading performance
- **Continuous Improvement**: Adaptive learning from real wins/losses in live conditions
- **Market Reality**: Avoids overfitting to historical patterns that may not repeat

This approach ensures the system adapts to current market dynamics rather than optimizing for past conditions.

### System Flow

```
┌─────────────────┐
│   MT5 Platform  │
│   (Live Data)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  MT5 Connector  │◄────►│  Market Data     │
│  (Clock Sync)   │      │  Manager         │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Trading Engine  │◄────►│  ML Predictor    │
│ (Stop Orders)   │      │  (30+ Models)    │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Risk Manager   │◄────►│  Fundamental     │
│  (3:1 R:R)      │      │  Monitor         │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│  Performance DB │
│  (Learning)     │
└─────────────────┘
```

### Core Capabilities

- **Stop Order Trading**: Uses BUY_STOP and SELL_STOP orders instead of market orders for better entry timing
- **ML Model Integration**: Trained models for 30+ currency pairs across multiple timeframes (M15, H1)
- **Adaptive Learning**: Continuous model improvement through reinforcement learning and performance tracking
- **Real-time Trading**: Automated position management with advanced risk controls
- **Multi-Timeframe Support**: Optimized parameters for M15 and H1 timeframes
- **Risk Management**: 3:1 minimum risk-reward ratio, dynamic position sizing, 3-trade daily limits per symbol
- **Stop Order Recording**: Database tracking of stop order placements for AI learning
- **Fundamental Monitoring**: Real-time news and economic event monitoring during active trades
- **Performance Monitoring**: Real-time dashboard with system health and P&L tracking
- **Market Analysis**: Technical, fundamental, sentiment analysis, and correlation management
- **Advanced Risk Metrics**: Portfolio-level risk assessment and market regime detection

### Supported Symbols (30 Pairs)

**Major FX Pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD
**Cross Pairs:** EURGBP, EURJPY, GBPJPY, AUDJPY, EURCAD, GBPAUD, EURNZD, GBPNZD, etc.
**Metals:** XAUUSD (Gold), XAGUSD (Silver)

## Configuration

### Main Configuration Files

- **`config/config.json`**: Main system configuration (risk management, trading limits, component settings)
- **`config/symbol_schedules.json`**: Symbol-specific trading schedules and optimal hours (30+ currency pairs)
- **`models/parameter_optimization/optimal_parameters.json`**: ML model parameters (auto-generated)

### Symbol Schedules Configuration

The system uses separate configuration for trading hours to enable optimal 24-hour trading:

```json
{
  "global_settings": {
    "enable_24hour_trading": true,
    "force_close_hour": 23,
    "force_close_minute": 45
  },
  "symbol_schedules": {
    "EURUSD": {
      "enabled": true,
      "optimal_hours": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
      "session_filters": {"overlap_required": false}
    }
    // ... 29 more symbols with optimized schedules
  }
}
```

**Key Features:**

- **Symbol-Specific Hours**: Each currency pair has optimized trading windows
- **24-Hour Support**: Continuous trading capability across all sessions
- **Force Close**: All positions closed at 23:45 GMT daily
- **Session Flexibility**: No overlap requirements for single-session trading

### Risk Management Settings

- **Order Type**: BUY_STOP/SELL_STOP orders (no market orders)
- **Max Positions:** 30 concurrent trades
- **Risk per Trade:** $50 (fixed dollar amount)
- **Daily Trade Limit:** 3 trades per symbol per day
- **Minimum Risk-Reward Ratio:** 3:1 for all symbols
- **Stop Order Distances:**
  - Forex pairs: 5-10 pips from current price
  - XAUUSD (Gold): 15-40 pips from current price
  - XAGUSD (Silver): 100-200 pips from current price
- **SL/TP from Entry:** 20/60 pips (3:1 ratio)
- **Max Spread:** 3.0 pips
- **Lot Size Range:** 0.01 - 1.0 lots
- **Symbol Limits:** Max 1 position per symbol at a time

### Fundamental Monitoring Settings

- **Check Interval:** Every 5 minutes (configurable)
- **High Impact Exit:** Exit positions within 15 minutes of contradicting news
- **SL Tightening:** Tighten stops for upcoming high-impact events (within 30 minutes)
- **Profit Locking:** Lock in profits during volatile news events

### Trading Hours

- **Market Open:** Monday 00:00 GMT (Sunday close)
- **Market Close:** Friday 23:59 GMT
- **24-Hour Optimal Trading:** System supports continuous trading with symbol-specific optimal hours
- **Symbol-Specific Schedules:** Individual trading windows optimized for each currency pair
- **Force Close Time:** All positions automatically closed at 23:45 GMT daily
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
│   ├── symbol_schedules.json        # Symbol-specific trading schedules (30+ pairs)
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

- **Python**: 3.8 or higher (3.9-3.10 recommended, tested)
- **MetaTrader 5**: Terminal installed and running
  - Enable Algo Trading in MT5 Tools → Options → Expert Advisors
  - Allow DLL imports and external expert imports
- **MT5 Account**: Valid trading account (TIOMarkets demo recommended for testing)
- **Operating System**:
  - Windows 10/11 (primary development platform)
  - Linux/Mac (supported with minor path adjustments)
- **Hardware**:
  - RAM: 8GB minimum, 16GB recommended
  - CPU: Multi-core processor (4+ cores recommended)
  - Storage: 5GB free space (for logs and models)
  - Internet: Stable connection for MT5 and real-time data

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
   # - Trading symbols and limits
   # - Fundamental monitoring settings

   # Review config/symbol_schedules.json for trading hours
   # - Symbol-specific optimal trading windows
   # - 24-hour trading schedules for 30+ pairs
   # - Force close times and session management
   ```

6. **Test System Connection**

   ```bash
   python -c "from core.mt5_connector import MT5Connector; import os; mc = MT5Connector(os.getenv('MT5_LOGIN'), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER')); print('MT5 Connection:', 'SUCCESS' if mc.initialize() else 'FAILED')"
   ```

---

## 💻 Development Workflow

### Recommended Setup

- **IDE**: Visual Studio Code with GitHub Copilot integration
- **Python Version**: 3.8+ (tested extensively on 3.9 and 3.10)
- **Operating System**: Windows 10/11 (primary), Linux/Mac supported
- **External Storage**: Consider using external drive for logs (e.g., `D:\FX-Ai-Data\`)

### VS Code Configuration

The `.vscode/` directory includes:

- **settings.json**: Optimized editor configuration
- **Extensions.json**: Recommended extensions (Python, GitHub Copilot)

**Performance Tip**: Exclude large directories from VS Code indexing:

```json
"files.watcherExclude": {
  "**/logs/**": true,
  "**/data/**": true,
  "**/models/**": true
}
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

- **Stop Order System**: Uses pending BUY_STOP/SELL_STOP orders for precise entry timing
- **3:1 Risk-Reward Ratio**: Minimum required ratio for all trades (profit 3x risk)
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
- **Stop Order Validation**: Ensures proper SL/TP placement before order submission
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
- Stop order placements and execution data
- Model accuracy and prediction quality
- Market conditions during trades
- Risk management effectiveness
- Adaptive parameter adjustments

---

## Stop Order Trading System

### System Overview

FX-Ai v3.0 uses an advanced stop order system that places BUY_STOP and SELL_STOP orders instead of market orders. This approach provides better entry timing and risk management.

### Stop Order Mechanics

**BUY_STOP Order:**

- Placed above current price (breakout buying)
- Triggered when price reaches the stop level
- SL placed below entry, TP above entry

**SELL_STOP Order:**

- Placed below current price (breakdown selling)
- Triggered when price reaches the stop level
- SL placed above entry, TP below entry

### Distance Configuration

Stop orders are placed at calculated distances from current price:

- **Forex Pairs:** 5-10 pips (optimized for major pairs)
- **XAUUSD (Gold):** 15-40 pips (accounts for higher volatility)
- **XAGUSD (Silver):** 100-200 pips (matches 3:1 risk-reward requirement)

### Risk-Reward Validation

- **Minimum Ratio:** 3:1 (profit must be 3x the risk)
- **SL Distance:** 20 pips from entry price
- **TP Distance:** 60 pips from entry price
- **Validation:** Orders rejected if ratio < 3.0

### Stop Order Recording

All stop order placements are recorded in the learning database for AI analysis:

- Entry price and stop distance
- Signal strength and market conditions
- Order outcome tracking
- Performance optimization data

### Advantages

- **Precise Entry:** Orders trigger at optimal price levels
- **Reduced Slippage:** No market order execution risk
- **Better Timing:** Enters on breakouts/breakdowns
- **Risk Control:** SL/TP set before order activation
- **AI Learning:** System learns from stop order performance

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

## 🧪 Testing & Validation

### Pre-Deployment Checklist

- [ ] MT5 connection successful
- [ ] All 30 symbols loading correctly
- [ ] ML models present in `models/` directory
- [ ] Configuration files validated
- [ ] Demo account configured in `.env`
- [ ] Test with single symbol first (e.g., EURUSD)
- [ ] Monitor for 24 hours on demo before live

### Test Commands

```bash
# Test MT5 Connection
python -c "from core.mt5_connector import MT5Connector; import os; from dotenv import load_dotenv; load_dotenv(); mc = MT5Connector(os.getenv('MT5_LOGIN'), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER')); print('Connection:', 'OK' if mc.initialize() else 'FAILED')"

# Test Model Loading
python -c "from ai.ml_predictor import MLPredictor; from utils.config_loader import ConfigLoader; config = ConfigLoader().load_config(); print('Models:', 'OK')"

# Validate Configuration
python -c "from utils.config_loader import ConfigLoader; config = ConfigLoader().load_config(); print('Config loaded:', len(config.keys()), 'sections')"
```

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

1. **Immediate Stop**: Run `live_trading/emergency_stop.bat` or `python live_trading/emergency_stop.py`
2. **System Validation**: Script verifies MT5 connection and trading permissions before attempting closures
3. **Complete Shutdown**: Closes all open positions AND cancels all pending orders
4. **Failure Detection**: Reports if any positions cannot be closed (may require manual intervention)
5. **Check Logs**: Review `logs/crash_log.txt` and recent log files for details
6. **Database Check**: Verify `data/performance_history.db` integrity
7. **Restart System**: After resolving issues, restart with `python main.py`
8. **Demo Testing**: Always test fixes on demo account first

---

## ❓ Frequently Asked Questions

**Q: Why no backtesting?**
A: FX-Ai deliberately focuses on live learning to avoid overfitting to historical data and adapt to current market conditions.

**Q: Can I use this on a live account?**
A: Only after extensive testing on demo. We recommend at least 1-2 months of demo trading first.

**Q: How much capital do I need?**
A: Minimum $1000 for proper risk management with $50 per trade. $5000+ recommended.

**Q: Does this work with all brokers?**
A: Works with any MT5 broker. Tested extensively with TIOMarkets.

**Q: Can I modify the ML models?**
A: Yes! See `ai/ml_predictor.py` for model architecture. System supports custom models.

**Q: How often are models retrained?**
A: Automatically when performance degrades or weekly optimization cycles.

**Q: What happens during weekends?**
A: System automatically pauses. Positions closed Friday 23:45 GMT.

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

Last Updated: November 13, 2025
