"""
Retrain XAGUSD and XAUUSD models with corrected 3-month data
This ensures models are trained with recent data and proper metal characteristics
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from ai.ml_predictor import MLPredictor

def setup_logging():
    """Setup logging for training process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('metal_training_log.txt'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from config file"""
    config_path = Path("config/config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)

def initialize_mt5(config):
    """Initialize MT5 connection"""
    logger = logging.getLogger(__name__)

    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False

    # Login to MT5
    mt5_config = config.get('mt5', {})
    login_result = mt5.login(
        login=int(mt5_config.get('login', 0)),
        password=mt5_config.get('password', ''),
        server=mt5_config.get('server', ''),
        timeout=int(mt5_config.get('timeout', 60000))
    )

    if not login_result:
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        return False

    logger.info("MT5 connection established successfully")
    return True

def get_historical_data(symbol, timeframe, start_date, end_date):
    """Get historical data for a symbol within date range"""
    logger = logging.getLogger(__name__)

    # Convert timeframe string to MT5 constant
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'D1': mt5.TIMEFRAME_D1
    }

    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

    logger.info(f"Fetching {timeframe} data for {symbol} from {start_date} to {end_date}")

    # Get data from MT5
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)

    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get data for {symbol}: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Rename columns to match expected format
    df = df.rename(columns={
        'tick_volume': 'volume'
    })

    logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
    return df

def prepare_training_data(df):
    """Prepare data for training with features"""
    logger = logging.getLogger(__name__)
    
    if df is None or len(df) < 50:
        logger.error("Insufficient data for training")
        return None, None
    
    # Calculate technical indicators
    df['returns'] = df['close'].pct_change()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Target: 1 if next close is higher, 0 if lower
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    # Prepare features
    feature_columns = ['returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 
                       'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
                       'atr', 'volatility']
    
    X = df[feature_columns].values
    y = df['target'].values
    
    logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
    return X, y

def train_metal_models():
    """Train models for XAGUSD and XAUUSD with 3-month data"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting Metal Model Training Process")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize MT5
    if not initialize_mt5(config):
        logger.error("Failed to initialize MT5, exiting...")
        return
    
    # Metal symbols
    symbols = ['XAGUSD', 'XAUUSD']
    timeframes = ['M1', 'M5', 'M15', 'H1']
    
    # Use 3 months of data (matching the optimized parameters)
    end_date = datetime(2025, 10, 31)
    start_date = end_date - timedelta(days=90)
    
    # Initialize ML Predictor
    ml_predictor = MLPredictor(config)
    
    for symbol in symbols:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training models for {symbol}")
        logger.info(f"{'=' * 60}")
        
        for timeframe in timeframes:
            try:
                logger.info(f"\nProcessing {symbol} {timeframe}...")
                
                # Get historical data
                df = get_historical_data(symbol, timeframe, start_date, end_date)
                
                if df is None or len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}, skipping...")
                    continue
                
                # Prepare training data
                X, y = prepare_training_data(df)
                
                if X is None or len(X) < 100:
                    logger.warning(f"Insufficient training samples for {symbol} {timeframe}, skipping...")
                    continue
                
                # Train the model
                logger.info(f"Training model for {symbol} {timeframe}...")
                ml_predictor.train_symbol_model(symbol, timeframe, X, y)
                
                logger.info(f"[OK] Successfully trained {symbol} {timeframe} model")
                
            except Exception as e:
                logger.error(f"Error training {symbol} {timeframe}: {e}")
                continue
        
        logger.info(f"\n[OK] Completed training all timeframes for {symbol}")
    
    # Shutdown MT5
    mt5.shutdown()
    
    logger.info("\n" + "=" * 60)
    logger.info("Metal Model Training Complete!")
    logger.info("=" * 60)
    logger.info("\nTrained models:")
    for symbol in symbols:
        for timeframe in timeframes:
            model_path = Path(f"models/{symbol}_{timeframe}_model.pkl")
            if model_path.exists():
                logger.info(f"  [OK] {symbol}_{timeframe}_model.pkl")

if __name__ == "__main__":
    train_metal_models()
