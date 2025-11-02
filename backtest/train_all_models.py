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
            logging.FileHandler('training_log.txt'),
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

def get_historical_data(symbol, timeframe='H1', bars=1000):
    """Get historical data for a symbol"""
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

    logger.info(f"Fetching {bars} bars of {timeframe} ({mt5_timeframe}) data for {symbol}")

    # Get data from MT5
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        logger.error(f"Failed to get data for {symbol}: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Rename columns to match expected format
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    })

    logger.info(f"Retrieved {len(df)} bars for {symbol}")
    return df

def train_all_models():
    """Train ML models for all configured symbols on multiple timeframes"""
    logger = setup_logging()
    logger.info("Starting training of all 30 ML models on multiple timeframes")

    try:
        # Load configuration
        config = load_config()
        symbols = config['trading']['symbols']

        logger.info(f"Found {len(symbols)} symbols to train: {symbols}")

        # Initialize MT5
        if not initialize_mt5(config):
            logger.error("Failed to initialize MT5")
            return False

        # Define timeframes to train on
        timeframes = ['M1', 'M5', 'M15', 'H1']  # Skip D1 for now to save time
        logger.info(f"Training on timeframes: {timeframes}")

        # Initialize ML Predictor
        ml_predictor = MLPredictor(config)

        # Train models for each symbol and timeframe
        trained_count = 0
        failed_count = 0

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    logger.info(f"Training model for {symbol} on {timeframe} timeframe")

                    # Get historical data for this timeframe
                    data = get_historical_data(symbol, timeframe=timeframe, bars=5000)  # More data for intraday
                    if data is None or len(data) < 100:
                        logger.warning(f"Insufficient data for {symbol} {timeframe}")
                        continue

                    # Train the model for this timeframe
                    ml_predictor._train_model(symbol, data, timeframe)

                    # Check if model was created
                    model_key = f"{symbol}_{timeframe}"
                    if model_key in ml_predictor.models:
                        trained_count += 1
                        logger.info(f"Successfully trained model for {symbol} on {timeframe}")
                    else:
                        logger.warning(f"Model training failed for {symbol} on {timeframe}")
                        failed_count += 1

                except Exception as e:
                    logger.error(f"Error training model for {symbol} on {timeframe}: {e}")
                    failed_count += 1

        # Shutdown MT5
        mt5.shutdown()

        logger.info(f"Training completed: {trained_count} successful, {failed_count} failed")

        if trained_count == len(symbols) * len(timeframes):
            logger.info("All models trained successfully!")
            return True
        else:
            logger.warning(f"Some models failed to train: {failed_count}/{len(symbols) * len(timeframes)}")
            return False

    except Exception as e:
        logger.error(f"Training process failed: {e}")
        return False

if __name__ == "__main__":
    success = train_all_models()
    sys.exit(0 if success else 1)