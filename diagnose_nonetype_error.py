"""
FX-Ai NoneType Error Diagnostic Script
This script helps identify where the 'NoneType' object has no attribute 'get' error is occurring
"""

import sys
import os

# Add project root to path
sys.path.insert(0, r'C:\Users\andyc\python\FX-Ai')

from core.mt5_connector import MT5Connector
from analysis.technical_analyzer import TechnicalAnalyzer
from data.market_data_manager import MarketDataManager
from utils.config_loader import ConfigLoader
import MetaTrader5 as mt5

def test_technical_data_retrieval():
    """Test if technical analyzer returns None for some symbols"""
    
    print("="*80)
    print("TESTING TECHNICAL DATA RETRIEVAL")
    print("="*80)
    
    # Initialize components
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    mt5_config = config.get('mt5', {})
    mt5_connector = MT5Connector(
        mt5_config.get('login'),
        mt5_config.get('password'), 
        mt5_config.get('server')
    )
    market_data = MarketDataManager(mt5_connector)
    technical_analyzer = TechnicalAnalyzer(config)
    
    # Test symbols that failed
    failed_symbols = ['USDJPY', 'AUDUSD', 'NZDUSD', 'AUDCAD', 'AUDCHF', 
                     'AUDJPY', 'AUDNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
                     'CADJPY', 'CHFJPY', 'EURJPY', 'GBPJPY']
    
    for symbol in failed_symbols:
        print(f"\n{'='*80}")
        print(f"Testing: {symbol}")
        print(f"{'='*80}")
        
        try:
            # Get market data
            h1_data = market_data.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
            print(f"✓ H1 Data: {len(h1_data) if h1_data is not None else 'None'} bars")
            
            if h1_data is None or len(h1_data) == 0:
                print(f"✗ ERROR: No H1 data available for {symbol}")
                continue
            
            # Analyze technical data
            technical_data = technical_analyzer.analyze(symbol, h1_data)
            print(f"✓ Technical Data Type: {type(technical_data)}")
            
            if technical_data is None:
                print(f"✗ ERROR: technical_analyzer.analyze() returned None for {symbol}")
                continue
            
            # Check if it's a dict
            if not isinstance(technical_data, dict):
                print(f"✗ ERROR: Technical data is not a dict, it's {type(technical_data)}")
                continue
            
            # Try to access ATR
            print(f"✓ Technical Data Keys: {list(technical_data.keys())}")
            
            atr = technical_data.get('atr')
            print(f"✓ ATR Value: {atr}")
            
            if atr is None:
                print(f"✗ WARNING: ATR is None for {symbol}")
            
            # Check for other critical fields
            for field in ['ema_fast', 'ema_slow', 'rsi', 'vwap', 'support', 'resistance']:
                value = technical_data.get(field)
                print(f"  - {field}: {value}")
                if value is None:
                    print(f"    ✗ WARNING: {field} is None")
            
        except Exception as e:
            print(f"✗ EXCEPTION: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

def test_atr_calculation():
    """Test ATR calculation specifically"""
    
    print("\n" + "="*80)
    print("TESTING ATR CALCULATION")
    print("="*80)
    
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    mt5_config = config.get('mt5', {})
    mt5_connector = MT5Connector(
        mt5_config.get('login'),
        mt5_config.get('password'), 
        mt5_config.get('server')
    )
    market_data = MarketDataManager(mt5_connector)
    
    symbol = 'USDJPY'
    print(f"\nTesting ATR for {symbol}")
    
    try:
        # Get data
        h1_data = market_data.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
        
        if h1_data is None or len(h1_data) < 14:
            print(f"✗ Insufficient data: {len(h1_data) if h1_data else 0} bars")
            return
        
        # Manual ATR calculation
        import pandas as pd
        df = pd.DataFrame(h1_data)
        
        print(f"✓ Data columns: {list(df.columns)}")
        print(f"✓ Data shape: {df.shape}")
        print(f"✓ Last 5 rows:")
        print(df[['time', 'open', 'high', 'low', 'close']].tail())
        
        # Calculate True Range
        df['h_l'] = df['high'] - df['low']
        df['h_pc'] = abs(df['high'] - df['close'].shift(1))
        df['l_pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        
        # Calculate ATR(14)
        atr_14 = df['tr'].rolling(window=14).mean().iloc[-1]
        print(f"\n✓ Manual ATR(14) calculation: {atr_14}")
        
        # Now test technical analyzer
        technical_analyzer = TechnicalAnalyzer(config)
        tech_data = technical_analyzer.analyze(symbol, h1_data)
        
        print(f"\n✓ Technical Analyzer ATR: {tech_data.get('atr') if tech_data else 'None'}")
        
    except Exception as e:
        print(f"✗ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_position_sizing():
    """Test the position sizing calculation where error might occur"""
    
    print("\n" + "="*80)
    print("TESTING POSITION SIZING")
    print("="*80)
    
    from core.risk_manager import RiskManager
    
    config = ConfigLoader()
    mt5_connector = MT5Connector(config)
    risk_manager = RiskManager(config, mt5_connector)
    
    symbol = 'USDJPY'
    direction = 'BUY'
    entry_price = 154.60
    
    print(f"\nCalculating position size for {symbol}")
    print(f"  Direction: {direction}")
    print(f"  Entry: {entry_price}")
    
    try:
        # This is where the error likely occurs
        position_params = risk_manager.calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            signal_data={'score': 0.5, 'confidence': 0.6}
        )
        
        print(f"\n✓ Position Parameters:")
        if position_params:
            for key, value in position_params.items():
                print(f"  - {key}: {value}")
        else:
            print(f"✗ position_params is None!")
        
    except AttributeError as e:
        print(f"\n✗ AttributeError Found!")
        print(f"  Error: {e}")
        print(f"  This is likely the source of 'NoneType' object has no attribute 'get'")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"✗ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FX-Ai NoneType Error Diagnostic")
    print("="*80)
    print("\nThis script will help identify the source of the NoneType error")
    print("that's preventing trades from executing.\n")
    
    # Run all tests
    test_technical_data_retrieval()
    test_atr_calculation()
    test_position_sizing()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nReview the output above to identify where None values appear.")
    print("The error occurs when code tries to call .get() on a None object.")
