//+------------------------------------------------------------------+
//|                                              FX-Ai_Connector.mq5|
//|                        FX-Ai Automated Trading System            |
//|                        MetaTrader 5 Expert Advisor               |
//+------------------------------------------------------------------+
#property copyright     "FX-Ai Development Team"
#property link          "https://github.com/andychoi-programming/FX-EURJPY"
#property version       "1.00"
#property description   "FX-Ai Connector for automated forex trading"

//--- Input parameters
input group "=== TRADING SETTINGS ==="
input double   LotSize         = 0.01;        // Lot size per trade
input int      MaxSpread       = 30;          // Maximum allowed spread (points)
input int      Slippage        = 30;          // Maximum slippage (points)
input bool     UseStopLoss     = true;        // Enable stop loss
input bool     UseTakeProfit   = true;        // Enable take profit
input int      StopLoss        = 500;         // Stop loss in points
input int      TakeProfit      = 1000;        // Take profit in points

input group "=== SIGNAL SETTINGS ==="
input string   SignalFile      = "fxai_signals.txt"; // Signal file name
input int      SignalTimeout   = 300;         // Signal timeout (seconds)
input bool     AutoTrading     = false;       // Enable automated trading

input group "=== RISK MANAGEMENT ==="
input double   MaxDailyLoss    = 100.0;       // Maximum daily loss ($)
input int      MaxTrades       = 5;           // Maximum open trades
input int      MagicNumber     = 20241028;    // Magic number for trades

//--- Global variables
string signalFilePath;
datetime lastSignalTime = 0;
double dailyLoss = 0.0;
datetime lastResetDate = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Check trading permissions
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
    {
        Print("❌ Trading not allowed in terminal settings");
        return(INIT_FAILED);
    }

    if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
    {
        Print("❌ Automated trading not allowed");
        return(INIT_FAILED);
    }

    //--- Set up signal file path
    signalFilePath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + SignalFile;

    //--- Create signal file if it doesn't exist
    if(!FileIsExist(SignalFile))
    {
        int handle = FileOpen(SignalFile, FILE_WRITE|FILE_TXT|FILE_COMMON);
        if(handle != INVALID_HANDLE)
        {
            FileWrite(handle, "# FX-Ai Signal File");
            FileWrite(handle, "# Format: SYMBOL,DIRECTION,PRICE,STOPLOSS,TAKEPROFIT,LOTSIZE,TIMESTAMP");
            FileClose(handle);
            Print("✓ Signal file created: ", SignalFile);
        }
        else
        {
            Print("❌ Failed to create signal file: ", SignalFile);
            return(INIT_FAILED);
        }
    }

    //--- Initialize daily loss tracking
    ResetDailyLoss();

    Print("✓ FX-Ai EA initialized successfully");
    Print("  Signal file: ", SignalFile);
    Print("  Auto trading: ", AutoTrading ? "ENABLED" : "DISABLED");
    Print("  Magic number: ", MagicNumber);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("FX-Ai EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check for new signals
    CheckSignals();

    //--- Update daily loss tracking
    UpdateDailyLoss();

    //--- Check risk limits
    if(dailyLoss >= MaxDailyLoss)
    {
        Print("⚠️ Daily loss limit reached: $", DoubleToString(dailyLoss, 2));
        CloseAllTrades();
        return;
    }

    //--- Check maximum trades limit
    if(PositionsTotal() >= MaxTrades)
    {
        return;
    }
}//+------------------------------------------------------------------+
//| Check for new trading signals                                    |
//+------------------------------------------------------------------+
void CheckSignals()
{
    //--- Read signal file
    int handle = FileOpen(SignalFile, FILE_READ|FILE_TXT|FILE_COMMON);
    if(handle == INVALID_HANDLE)
    {
        return;
    }

    string line;
    bool hasNewSignal = false;

    while(!FileIsEnding(handle))
    {
        line = FileReadString(handle);
        StringTrimLeft(line);
        StringTrimRight(line);

        // Skip comments and empty lines
        if(StringLen(line) == 0 || StringSubstr(line, 0, 1) == "#")
            continue;

        // Parse signal
        string parts[];
        if(StringSplit(line, ',', parts) == 7)
        {
            string symbol = parts[0];
            string direction = parts[1];
            double price = StringToDouble(parts[2]);
            double sl = StringToDouble(parts[3]);
            double tp = StringToDouble(parts[4]);
            double lotSize = StringToDouble(parts[5]);
            datetime timestamp = StringToTime(parts[6]);

            // Check if signal is new
            if(timestamp > lastSignalTime)
            {
                lastSignalTime = timestamp;
                hasNewSignal = true;

                if(AutoTrading)
                {
                    ExecuteSignal(symbol, direction, price, sl, tp, lotSize);
                }
                else
                {
                    Print("📊 Signal received: ", symbol, " ", direction, " Lot: ", DoubleToString(lotSize, 5));
                }
            }
        }
    }

    FileClose(handle);

    if(hasNewSignal && !AutoTrading)
    {
        Print("⚠️ New signals detected but auto-trading is disabled");
    }
}

//+------------------------------------------------------------------+
//| Execute trading signal                                           |
//+------------------------------------------------------------------+
bool ExecuteSignal(string symbol, string direction, double price, double sl, double tp, double lotSize)
{
    //--- Check if symbol is available
    if(!SymbolSelect(symbol, true))
    {
        Print("❌ Symbol not available: ", symbol);
        return false;
    }

    //--- Check spread
    long spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
    if(spread > MaxSpread)
    {
        Print("❌ Spread too high: ", symbol, " (", spread, " > ", MaxSpread, ")");
        return false;
    }

    //--- Prepare trade request
    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = lotSize;
    request.magic = MagicNumber;
    request.deviation = (ulong)Slippage;
    request.type_filling = ORDER_FILLING_FOK;

    //--- Set order type
    if(direction == "BUY")
    {
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
    }
    else if(direction == "SELL")
    {
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
    }
    else
    {
        Print("❌ Invalid direction: ", direction);
        return false;
    }

    //--- Set stop loss and take profit
    if(UseStopLoss && sl > 0)
    {
        if(direction == "BUY")
            request.sl = request.price - sl * Point();
        else
            request.sl = request.price + sl * Point();
    }

    if(UseTakeProfit && tp > 0)
    {
        if(direction == "BUY")
            request.tp = request.price + tp * Point();
        else
            request.tp = request.price - tp * Point();
    }

    //--- Send order
    if(OrderSend(request, result))
    {
        Print("✅ Order executed: ", symbol, " ", direction, " Lot: ", DoubleToString(lotSize, 5));
        return true;
    }
    else
    {
        Print("❌ Order failed: ", symbol, " ", direction, " Error: ", result.retcode);
        return false;
    }
}

//+------------------------------------------------------------------+
//| Close all open positions                                         |
//+------------------------------------------------------------------+
void CloseAllTrades()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0)
        {
            PositionSelectByTicket(ticket);
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                MqlTradeRequest request = {};
                MqlTradeResult result = {};

                request.action = TRADE_ACTION_DEAL;
                request.position = ticket;
                request.symbol = PositionGetString(POSITION_SYMBOL);
                request.volume = PositionGetDouble(POSITION_VOLUME);
                request.magic = MagicNumber;
                request.deviation = Slippage;

                if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                    request.type = ORDER_TYPE_SELL;
                    request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
                }
                else
                {
                    request.type = ORDER_TYPE_BUY;
                    request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
                }

                if(!OrderSend(request, result))
                {
                    Print("❌ Failed to close position: ", ticket, " Error: ", result.retcode);
                }
            }
        }
    }

    Print("✓ All positions closed due to risk management");
}

//+------------------------------------------------------------------+
//| Update daily loss tracking                                       |
//+------------------------------------------------------------------+
void UpdateDailyLoss()
{
    //--- Reset daily loss on new day
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);

    if(dt.day != lastResetDate)
    {
        ResetDailyLoss();
    }

    //--- Calculate current daily loss
    dailyLoss = 0.0;
    for(int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0)
        {
            PositionSelectByTicket(ticket);
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                double profit = PositionGetDouble(POSITION_PROFIT);
                if(profit < 0)
                {
                    dailyLoss += MathAbs(profit);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Reset daily loss tracking                                        |
//+------------------------------------------------------------------+
void ResetDailyLoss()
{
    dailyLoss = 0.0;
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    lastResetDate = dt.day;
    Print("✓ Daily loss reset to $0.00");
}

//+------------------------------------------------------------------+
//| Expert trade transaction function                                |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
    //--- Log trade transactions
    if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
    {
        Print("📈 Trade executed - Ticket: ", trans.deal, " Profit: ", DoubleToString(trans.price, 2));
    }
}

//+------------------------------------------------------------------+
//| Expert trade function                                            |
//+------------------------------------------------------------------+
void OnTrade()
{
    //--- Update position information
    int total = PositionsTotal();
    if(total > 0)
    {
        Print("📊 Active positions: ", total);
    }
}
//+------------------------------------------------------------------+
