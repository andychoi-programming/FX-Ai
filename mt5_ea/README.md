# FX-Ai MT5 Expert Advisor Setup Guide

## Overview

The FX-Ai MT5 Expert Advisor (EA) enables automated trading execution directly in MetaTrader 5, receiving signals from the Python-based FX-Ai trading system.

## Files Created

- `mt5_ea/FX-Ai_Connector.mq5` - The MQL5 Expert Advisor
- `utils/mt5_ea_communicator.py` - Python communication module

## Installation Steps

### 1. Install the EA in MetaTrader 5

1. Open MetaTrader 5
2. Go to **File → Open Data Folder**
3. Navigate to `MQL5/Experts/`
4. Copy `FX-Ai_Connector.mq5` to this folder
5. Restart MetaTrader 5 or refresh the Navigator (Ctrl+N)

### 2. Configure the EA

1. In MetaTrader 5, go to **Navigator → Expert Advisors**
2. Right-click on **FX-Ai_Connector** and select **Properties**
3. Configure the following parameters:

#### Trading Settings

**Note**: LotSize, StopLoss, and TakeProfit parameters are controlled by the Python system and ignored by the EA.

- **LotSize**: 0.01 (IGNORED - controlled by Python risk management)
- **MaxSpread**: 30 (maximum allowed spread in points)
- **Slippage**: 30 (maximum slippage in points)
- **UseStopLoss**: true (enables stop loss from Python signals)
- **UseTakeProfit**: true (enables take profit from Python signals)
- **StopLoss**: 500 (IGNORED - stop loss comes from Python signals)
- **TakeProfit**: 1000 (IGNORED - take profit comes from Python signals)

#### Signal Settings

- **SignalFile**: fxai_signals.txt (must match Python communicator)
- **SignalTimeout**: 300 (5 minutes)
- **AutoTrading**: false (set to true for automated trading)

#### Risk Management

- **MaxDailyLoss**: 100.0 (maximum daily loss in $)
- **MaxTrades**: 5 (maximum open positions)
- **MagicNumber**: 20241029 (unique identifier)

### 3. Enable Automated Trading

1. In MetaTrader 5, go to **Tools → Options → Expert Advisors**
2. Check **Allow automated trading**
3. Check **Allow DLL imports** (if needed)
4. Check **Allow automated trading when minimized**

### 4. Attach EA to Chart

1. Open a chart for your trading symbol (e.g., EURUSD)
2. Drag the **FX-Ai_Connector** from Navigator to the chart
3. Configure parameters in the dialog
4. Click **OK** to attach

## Usage

### Python Integration

```python
from utils.mt5_ea_communicator import MT5EACommunicator

# Initialize communicator
ea_comm = MT5EACommunicator()

# Send a buy signal
ea_comm.send_signal(
    symbol="EURUSD",
    direction="BUY",
    entry_price=0.0,  # Market order
    stop_loss=500,    # 50 pips
    take_profit=1000  # 100 pips
)

# Send a sell signal
ea_comm.send_signal(
    symbol="GBPUSD",
    direction="SELL",
    entry_price=0.0,
    stop_loss=300,
    take_profit=600
)
```

### Signal Format

Signals are written to `fxai_signals.txt` in the format:

```text
SYMBOL,DIRECTION,PRICE,STOPLOSS,TAKEPROFIT,LOTSIZE,TIMESTAMP
EURUSD,BUY,0.00000,500,1000,0.02000,2025.10.28 15:30:00
XAUUSD,SELL,1950.00000,1000,2000,0.00500,2025.10.28 15:30:00
```

## Features

### Risk Management Features

- **Daily Loss Limit**: Automatically closes all positions when daily loss exceeds limit
- **Maximum Trades**: Limits the number of concurrent open positions
- **Spread Control**: Rejects trades when spread is too high
- **Stop Loss/Take Profit**: Configurable SL/TP levels

### Signal Processing

- **File-based Communication**: Reliable signal transfer between Python and MT5
- **Signal Timeout**: Prevents stale signals from being executed
- **Manual Override**: Can disable auto-trading for manual review

### Logging

- **Comprehensive Logs**: All actions logged to MetaTrader 5 Experts tab
- **Trade Tracking**: Monitors position profits and losses
- **Error Handling**: Detailed error messages for troubleshooting

## Troubleshooting

### Common Issues

1. **"Trading not allowed" error**
   - Enable automated trading in MT5 options
   - Check that EA is attached to chart
   - Verify account has trading permissions

2. **Signals not being read**
   - Check signal file path in EA properties
   - Ensure Python and MT5 are writing to same file location
   - Verify file permissions

3. **Spread too high**
   - Increase MaxSpread parameter
   - Trade during active market hours
   - Consider ECN account for tighter spreads

4. **No signals received**
   - Check Python logs for communication errors
   - Verify MT5 data path in communicator
   - Test with manual signal file creation

### Debug Mode

Set `AutoTrading = false` to see signals in logs without executing trades.

## Security Notes

- Test with small lot sizes initially
- Use stop loss on all trades
- Monitor account balance regularly
- Keep backup of EA settings
- Use demo account for testing

## Support

For issues or questions:

1. Check MetaTrader 5 logs (Experts tab)
2. Review Python console output
3. Verify signal file contents
4. Test with manual signal creation

---
**FX-Ai MT5 EA - Version 1.0.0**
**Last updated: October 29, 2025**
