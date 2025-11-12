"""
Test trading opportunities check
"""

import json
import asyncio
import MetaTrader5 as mt5
from core.mt5_connector import MT5Connector
from utils.time_manager import get_time_manager
from analysis.technical_analyzer import TechnicalAnalyzer
from data.market_data_manager import MarketDataManager
from core.risk_manager import RiskManager

async def test_trading_opportunities():
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)

    print('Testing trading opportunities check...')

    # Initialize components
    mt5_conn = MT5Connector()
    mt5_conn.connect()
    time_manager = get_time_manager(mt5_conn)
    market_data = MarketDataManager(mt5_conn, config)
    technical_analyzer = TechnicalAnalyzer(config)
    risk_manager = RiskManager(config, mt5_conn)

    # Get current session and threshold
    current_session = time_manager.get_current_session()
    threshold = time_manager.get_session_signal_threshold(config)
    print(f'Current session: {current_session}, Threshold: {threshold}')

    symbols = config.get('trading', {}).get('symbols', [])[:5]  # Test first 5 symbols
    opportunities_found = 0

    print(f'Checking {len(symbols)} symbols...')

    for symbol in symbols:
        try:
            # Check if we can trade this symbol
            can_trade, reason = risk_manager.can_trade(symbol)
            if not can_trade:
                print(f'[{symbol}] Cannot trade: {reason}')
                continue

            # Get market data (H1 bars for technical analysis)
            h1_data = market_data.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
            if h1_data is None or len(h1_data) < 50:
                print(f'[{symbol}] No market data')
                continue

            # Format data as expected by analyzers
            market_data_dict = {'H1': h1_data}

            # Get current price
            tick_data = market_data.get_market_data(symbol)
            if tick_data:
                current_price = tick_data.get('last', tick_data.get('bid', 0))
            else:
                current_price = h1_data['close'].iloc[-1] if len(h1_data) > 0 else 0

            # Get analysis scores
            technical_score = technical_analyzer.analyze(symbol, market_data_dict)
            fundamental_score = 0.5  # Mock
            sentiment_score = 0.5    # Mock

            # Combine signals
            signal_strength = (technical_score * 0.6 + fundamental_score * 0.25 + sentiment_score * 0.15)

            # Check minimum signal strength
            if signal_strength < threshold:
                print(f'[{symbol}] Signal strength {signal_strength:.3f} < threshold {threshold:.3f}')
                continue

            # Determine trade direction
            direction = 'BUY' if technical_score > 0.5 else 'SELL'

            opportunities_found += 1
            print(f'[{symbol}] ✅ OPPORTUNITY: {direction} (Strength: {signal_strength:.3f})')

        except Exception as e:
            print(f'[{symbol}] Error: {e}')

    print(f'\\nTotal opportunities found: {opportunities_found}')

if __name__ == '__main__':
    asyncio.run(test_trading_opportunities())