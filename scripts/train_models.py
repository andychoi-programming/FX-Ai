#!/usr/bin/env python3
"""
Unified ML Model Training for FX-Ai
Trains ML models for all supported trading symbols
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai.ml_predictor import MLPredictor
from data.market_data_manager import MarketDataManager
from core.mt5_connector import MT5Connector
from core.config_manager import ConfigManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_models_for_symbols(symbols: list, config: ConfigManager, logger: logging.Logger):
    """
    Train ML models for specified symbols

    Args:
        symbols: List of symbols to train
        config: Configuration manager
        logger: Logger instance
    """
    print("\n" + "="*60)
    print("FX-AI ML MODEL TRAINING")
    print("="*60 + "\n")

    # Initialize components
    try:
        mt5 = MT5Connector()
        if not mt5.connect():
            logger.error("Failed to connect to MT5")
            return False

        data_manager = MarketDataManager(mt5_connector=mt5)
        ml_predictor = MLPredictor(config.config)

        trained_count = 0
        failed_count = 0

        for symbol in symbols:
            try:
                logger.info(f"Training model for {symbol}...")

                # Get historical data (3 years for comprehensive training)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1095)  # 3 years

                # Get H1 data for training
                bars = data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe='H1',
                    start_date=start_date,
                    end_date=end_date
                )

                if bars is None or len(bars) < 1000:
                    logger.warning(f"Insufficient data for {symbol}: {len(bars) if bars else 0} bars")
                    failed_count += 1
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(bars)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # Train model
                success = ml_predictor.train_symbol_model(symbol, df)

                if success:
                    logger.info(f"✅ Successfully trained model for {symbol}")
                    trained_count += 1
                else:
                    logger.error(f"❌ Failed to train model for {symbol}")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
                failed_count += 1

        # Summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total symbols: {len(symbols)}")
        print(f"Successfully trained: {trained_count}")
        print(f"Failed: {failed_count}")
        print(".1f")
        print("="*60)

        return trained_count > 0

    except Exception as e:
        logger.error(f"Fatal error during training: {e}")
        return False

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ML models for FX-Ai')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train (default: all)')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live',
                       help='Training mode (default: live)')
    parser.add_argument('--config', default='config/config.json',
                       help='Configuration file path')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    try:
        # Load configuration
        config = ConfigManager(mode=args.mode)

        # Get symbols to train
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = config.get_symbols()

        logger.info(f"Starting training for {len(symbols)} symbols in {args.mode} mode")

        # Train models
        success = train_models_for_symbols(symbols, config, logger)

        if success:
            logger.info("Training completed successfully")
            sys.exit(0)
        else:
            logger.error("Training failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()