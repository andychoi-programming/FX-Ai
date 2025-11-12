"""
Quick signal generation test
"""

import json
import MetaTrader5 as mt5
from core.mt5_connector import MT5Connector
from utils.time_manager import get_time_manager
from ai.ml_predictor import MLPredictor
from analysis.technical_analyzer import TechnicalAnalyzer
from data.market_data_manager import MarketDataManager

def test_signal_generation():
    # Load config
    with open('config/config.json', 'r') as f:
        config = json.load(f)

    print('Testing signal generation with session filtering disabled...')

    # Initialize components
    mt5_conn = MT5Connector()
    mt5_conn.connect()
    time_manager = get_time_manager(mt5_conn)
    ml_predictor = MLPredictor(config['ml'])
    market_data = MarketDataManager(mt5_conn, config)

    # Get current session and threshold
    current_session = time_manager.get_current_session()
    threshold = time_manager.get_session_signal_threshold(config)
    trading_allowed = time_manager.is_trading_allowed()

    print(f'Current session: {current_session}')
    print(f'Signal threshold: {threshold}')
    print(f'Trading allowed: {trading_allowed}')

    # Test EURUSD signal generation
    symbol = 'EURUSD'
    print(f'\nTesting {symbol} signal generation...')

    # Get market data
    data = market_data.get_bars(symbol, mt5.TIMEFRAME_H1, 100)
    if data is None or len(data) == 0:
        print('No market data available')
        return

    print(f'Got {len(data)} bars of data')

    # Create technical signals
    technical_signals = {
        'rsi': 50.0,
        'macd_signal': 0.0,
        'bb_position': 0.0,
        'stoch_k': 50.0,
        'stoch_d': 50.0,
        'trend_strength': 0.5,
        'momentum': 0.0
    }

    # Generate ML signal
    ml_signal = ml_predictor.predict_signal(symbol, data, technical_signals, 'H1')
    signal_strength = float(ml_signal.get('signal_strength', 0))
    direction = int(ml_signal.get('direction', 0))

    print(f'ML signal: direction={direction}, strength={signal_strength:.3f}')
    print(f'Signal passes threshold: {signal_strength >= threshold}')

    # Check if this would result in a trade
    if signal_strength >= threshold and trading_allowed[0]:
        print('✅ TRADE SIGNAL GENERATED!')
        print(f'Direction: {"BUY" if direction == 1 else "SELL"} {symbol}')
        print(f'Signal strength: {signal_strength:.3f}')
    else:
        print('❌ No trade signal')
        if not trading_allowed[0]:
            print(f'Reason: {trading_allowed[1]}')
        else:
            print(f'Reason: Signal strength {signal_strength:.3f} < threshold {threshold}')

if __name__ == '__main__':
    test_signal_generation()