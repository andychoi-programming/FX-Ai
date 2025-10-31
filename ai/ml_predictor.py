"""
ML Predictor Module
Machine learning models for price prediction and signal generation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class MLPredictor:
    """Machine learning predictor for trading signals"""

    def __init__(self, config: Dict):
        """
        Initialize ML predictor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model storage
        self.models = {}
        self.scalers = {}

        # Model parameters
        self.model_type = config.get('model_type', 'random_forest')
        self.confidence_threshold = config.get('confidence_threshold', 0.6)

        # Feature engineering
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.technical_indicators = [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'ema_21', 'vwap'
        ]

        # Model paths
        self.model_dir = config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def predict_signal(self, symbol: str, data: pd.DataFrame,
                      technical_signals: Dict) -> Dict:
        """
        Generate ML-based trading signal

        Args:
            symbol: Trading symbol
            data: Historical price data
            technical_signals: Technical analysis signals

        Returns:
            dict: ML prediction results
        """
        try:
            # Check if model exists
            if symbol not in self.models:
                self._load_or_train_model(symbol, data)

            if symbol not in self.models:
                return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0}

            # Prepare features
            features = self._prepare_features(data, technical_signals)

            if features is None:
                return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0}

            # Make prediction
            model = self.models[symbol]
            scaler = self.scalers.get(symbol)

            if scaler:
                features_scaled = scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)

            # Get prediction probabilities
            probabilities = model.predict_proba(features_scaled)[0]

            # Determine direction and confidence
            if probabilities[1] > probabilities[0]:  # Bullish
                direction = 'bullish'
                confidence = probabilities[1]
            else:  # Bearish
                direction = 'bearish'
                confidence = probabilities[0]

            # Calculate signal strength
            signal_strength = (confidence - 0.5) * 2  # Scale to -1 to 1

            # Convert direction to numeric for compatibility
            direction_numeric = 1 if direction == 'bullish' else 0

            return {
                'direction': direction_numeric,  # 1 for bullish, 0 for bearish
                'probability': confidence,       # Use confidence as probability
                'signal_strength': signal_strength,
                'probabilities': {
                    'bearish': probabilities[0],
                    'bullish': probabilities[1]
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating ML prediction for {symbol}: {e}")
            return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0}

    async def predict(self, symbol: str, data: Union[pd.DataFrame, Dict], technical_signals: Dict) -> Dict:
        """
        Async wrapper for predict_signal method

        Args:
            symbol: Trading symbol
            data: Historical price data (DataFrame or dict of DataFrames by timeframe)
            technical_signals: Technical analysis signals

        Returns:
            dict: ML prediction results
        """
        # Handle both DataFrame and dict formats
        if isinstance(data, dict):
            # Extract H1 data from timeframe dictionary
            data = data.get('H1')
            if data is None:
                return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0, 'probability': 0}
        
        return self.predict_signal(symbol, data, technical_signals)

    def _prepare_features(self, data: pd.DataFrame, technical_signals: Dict) -> Optional[np.ndarray]:
        """
        Prepare features for ML model

        Args:
            data: Historical price data
            technical_signals: Technical analysis signals

        Returns:
            np.ndarray: Feature array or None
        """
        try:
            if len(data) < 50:
                return None

            features = []

            # Price-based features
            close_prices = data['close'].values[-50:]
            high_prices = data['high'].values[-50:]
            low_prices = data['low'].values[-50:]
            volume = data.get('volume', data.get('tick_volume', np.ones(50)))  # Handle both volume column names
            volume = volume[-50:] if len(volume) >= 50 else volume  # Ensure volume matches price data length
            
            # print(f"DEBUG: Close prices length: {len(close_prices)}")
            # print(f"DEBUG: Volume length: {len(volume)}")

            # Returns
            returns_1d = np.diff(close_prices[-2:])[0] / close_prices[-2] if len(close_prices) >= 2 else 0.0
            returns_5d = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) >= 6 else 0.0
            
            # Ensure returns are scalars
            returns_1d = float(returns_1d) if np.isscalar(returns_1d) or returns_1d.size == 1 else 0
            returns_5d = float(returns_5d) if np.isscalar(returns_5d) else 0

            features.extend([returns_1d, returns_5d])
            # print(f"DEBUG: After returns, features length: {len(features)}")

            # Volatility features
            volatility_5d = float(np.std(np.diff(close_prices[-6:]))) if len(close_prices) >= 6 else 0.0
            volatility_20d = float(np.std(np.diff(close_prices[-21:]))) if len(close_prices) >= 21 else 0.0
            
            # Ensure volatilities are scalars
            volatility_5d = float(volatility_5d)
            volatility_20d = float(volatility_20d)

            features.extend([volatility_5d, volatility_20d])

            # Volume features
            avg_volume = float(np.mean(volume))
            volume_ratio = float(volume.iloc[-1] / avg_volume) if avg_volume > 0 else 1.0
            
            # Ensure volume features are scalars
            avg_volume = float(avg_volume)
            volume_ratio = float(volume_ratio)

            features.append(volume_ratio)

            # Technical indicators from signals
            if technical_signals:
                # RSI
                rsi = float(technical_signals.get('rsi', {}).get('value', 50))
                features.append(rsi / 100.0)  # Normalize

                # VWAP position
                vwap_pos = 1.0 if technical_signals.get('vwap', {}).get('position') == 'above' else 0.0
                features.append(vwap_pos)

                # EMA trend
                ema_trend = technical_signals.get('ema', {}).get('trend', 'neutral')
                ema_score = float({'bullish': 1, 'bearish': -1, 'bullish_crossover': 1.5, 'bearish_crossover': -1.5}.get(ema_trend, 0))
                features.append(ema_score)

            # Fill missing features with zeros
            while len(features) < 10:  # Ensure minimum feature count
                features.append(0.0)
            
            # Ensure all features are floats
            features = [float(f) for f in features]

            return np.array(features[:10], dtype=np.float32)  # Limit to 10 features

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None

    def _load_or_train_model(self, symbol: str, data: pd.DataFrame):
        """
        Load existing model or train new one

        Args:
            symbol: Trading symbol
            data: Historical data for training
        """
        try:
            model_path = os.path.join(self.model_dir, f'{symbol}_model.pkl')
            scaler_path = os.path.join(self.model_dir, f'{symbol}_scaler.pkl')

            # Try to load existing model
            if os.path.exists(model_path):
                self.models[symbol] = joblib.load(model_path)
                if os.path.exists(scaler_path):
                    self.scalers[symbol] = joblib.load(scaler_path)
                self.logger.info(f"Loaded existing model for {symbol}")
                return

            # Train new model
            self.logger.info(f"Training new model for {symbol}")
            self._train_model(symbol, data)

        except Exception as e:
            self.logger.error(f"Error loading/training model for {symbol}: {e}")

    def _train_model(self, symbol: str, data: pd.DataFrame):
        """
        Train ML model for symbol

        Args:
            symbol: Trading symbol
            data: Historical data
        """
        try:
            if len(data) < 80:  # Reduced from 100 for demo purposes
                # Only log once per symbol to avoid spam
                if symbol not in getattr(self, '_insufficient_data_logged', set()):
                    if not hasattr(self, '_insufficient_data_logged'):
                        self._insufficient_data_logged = set()
                    self._insufficient_data_logged.add(symbol)
                    self.logger.info(f"Waiting for more data to train {symbol} model ({len(data)}/80 bars)")
                return

            # Create target variable (future returns)
            close_prices = data['close'].values
            future_returns = np.diff(close_prices[20:]) / close_prices[20:-1]  # 20-period future returns

            # Create binary target (1 for positive return, 0 for negative)
            target = (future_returns > 0).astype(int)

            # Prepare features for all available data points
            features_list = []
            valid_indices = []

            for i in range(50, len(data) - 20):  # Ensure enough history and future data
                window_data = data.iloc[i-50:i+1]
                # Mock technical signals for training
                mock_signals = {}
                features = self._prepare_features(window_data, mock_signals)

                if features is not None:
                    features_list.append(features)
                    valid_indices.append(i - 50)  # Adjust for the 20-period shift

            if len(features_list) < 10:  # Reduced from 50 for demo purposes
                self.logger.warning(f"Insufficient valid features for training {symbol} ({len(features_list)}/10)")
                return

            # Align features with targets
            X = np.array(features_list)
            y = target[valid_indices[:len(X)]]  # Align with valid feature indices

            if len(X) != len(y):
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            if self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )

            model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)

            self.logger.info(f"Model for {symbol} - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler

            # Save model
            model_path = os.path.join(self.model_dir, f'{symbol}_model.pkl')
            scaler_path = os.path.join(self.model_dir, f'{symbol}_scaler.pkl')

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)

            self.logger.info(f"Trained and saved model for {symbol}")

        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")

    def get_model_performance(self, symbol: str) -> Dict:
        """
        Get model performance metrics

        Args:
            symbol: Trading symbol

        Returns:
            dict: Performance metrics
        """
        # This would track actual performance vs predictions
        return {
            'accuracy': 0.55,  # Mock value
            'precision': 0.52,
            'recall': 0.58,
            'total_predictions': 0
        }

    def get_model_version(self, symbol: str) -> Optional[str]:
        """
        Return a lightweight model version identifier for a given symbol.

        Currently implemented as the model file modification timestamp (ISO string)
        when a saved model exists, otherwise returns None.
        """
        try:
            model_path = os.path.join(self.model_dir, f'{symbol}_model.pkl')
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                return datetime.fromtimestamp(mtime).isoformat()
            return None
        except Exception as e:
            self.logger.debug(f"Error getting model version for {symbol}: {e}")
            return None

    def get_model_version(self, symbol: str) -> str:
        """
        Get model version for symbol

        Args:
            symbol: Trading symbol

        Returns:
            str: Model version string
        """
        if symbol in self.models:
            return "1.0.0"  # Current model version
        return "none"  # No model available