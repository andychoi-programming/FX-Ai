#!/usr/bin/env python3
"""
Train All ML Models for FX-Ai
Trains ML models for all supported trading symbols
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import MetaTrader5 as mt5

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ml_predictor import MLPredictor
from data.market_data_manager import MarketDataManager
from core.mt5_connector import MT5Connector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_all_models():
    """Train ML models for all supported symbols"""
    print("\n" + "="*60)
    print("FX-AI ML MODEL TRAINING")
    print("="*60 + "\n")

    # Initialize components
    try:
        mt5 = MT5Connector()
        if not mt5.connect():
            logger.error("Failed to connect to MT5")
            return

        data_manager = MarketDataManager(mt5_connector=mt5)
        
        # Load config
        config_path = 'config/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        ml_predictor = MLPredictor(config)

        # Symbols to train
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']

        print(f"Training models for {len(symbols)} symbols...")
        print("This may take several minutes...\n")

        trained_count = 0

        for symbol in symbols:
            try:
                print(f"[UP] Training model for {symbol}...")

                # Get historical data (last 1000 bars for training)
                data = data_manager.get_bars(symbol, 16385, 1000)  # 16385 = H1 timeframe

                if data is None or len(data) < 100:
                    logger.warning(f"Insufficient data for {symbol} ({len(data) if data else 0} bars)")
                    continue

                print(f"   Got {len(data)} bars of data")

                # Generate technical signals (simplified)
                technical_signals = {
                    'rsi': 50.0,
                    'macd_signal': 0.0,
                    'bb_position': 0.0,
                    'stoch_k': 50.0,
                    'stoch_d': 50.0,
                    'trend_strength': 0.5,
                    'momentum': 0.0
                }

                # Train the model
                ml_predictor._load_or_train_model(symbol, data, 'H1')

                print(f"   [PASS] Model trained for {symbol}")
                trained_count += 1

            except Exception as e:
                logger.error(f"Failed to train model for {symbol}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"[PASS] Successfully trained {trained_count}/{len(symbols)} models")
        print(f"[EMOJI] Models saved to: {ml_predictor.model_dir}")

        if trained_count > 0:
            print("\n[TRADE] Ready for live trading!")
            print("   Run: python main.py")
        else:
            print("\n[FAIL] No models were trained - check logs for errors")        # Cleanup
        mt5.disconnect()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

    return trained_count > 0

if __name__ == "__main__":
    success = train_all_models()
    sys.exit(0 if success else 1)