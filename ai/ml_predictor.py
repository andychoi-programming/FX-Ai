"""
ML Predictor Module
Machine learning models for price prediction and signal generation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Union
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

        # Support multiple timeframes for training
        self.supported_timeframes = ['M1', 'M5', 'M15', 'H1', 'D1']
        self.primary_timeframe = config.get('primary_timeframe', 'H1')

        # Feature engineering
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        self.technical_indicators = [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_9', 'ema_21', 'vwap'
        ]

        # Model paths
        self.model_dir = config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        # Feature consistency tracking
        self.expected_features = [
            'returns_1d', 'returns_5d', 'volatility_5d', 'volatility_20d', 'volume_ratio',
            'rsi_norm', 'vwap_position', 'bb_position', 'macd_signal', 'trend_strength',
            'momentum', 'support_resistance', 'regime_score'
        ]

    def validate_feature_consistency(self, features: Union[pd.DataFrame, Dict]) -> bool:
        """
        Ensure live features match training structure

        Args:
            features: Features to validate (DataFrame or dict)

        Returns:
            bool: True if features are consistent
        """
        try:
            if isinstance(features, pd.DataFrame):
                actual_features = list(features.columns)
            elif isinstance(features, dict):
                actual_features = list(features.keys())
            else:
                self.logger.error("Features must be DataFrame or dict")
                return False

            missing = set(self.expected_features) - set(actual_features)
            extra = set(actual_features) - set(self.expected_features)

            if missing or extra:
                self.logger.warning(f"Feature mismatch! Missing: {missing}, Extra: {extra}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating feature consistency: {e}")
            return False

    def predict_signal(self, symbol: str, data: pd.DataFrame,
                      technical_signals: Dict, timeframe: str = 'H1') -> Dict:
        """
        Generate ML-based trading signal

        Args:
            symbol: Trading symbol
            data: Historical price data
            technical_signals: Technical analysis signals
            timeframe: Timeframe for prediction (M1, M5, M15, H1, D1)

        Returns:
            dict: ML prediction results
        """
        try:
            # Check if model exists for this timeframe
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.models:
                # Try to load existing model without blocking
                model_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_model.pkl')
                if os.path.exists(model_path):
                    try:
                        self.logger.info(f"{symbol}: Loading model from {model_path}...")
                        self.models[model_key] = joblib.load(model_path)
                        scaler_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_scaler.pkl')
                        if os.path.exists(scaler_path):
                            self.scalers[model_key] = joblib.load(scaler_path)
                        self.logger.info(f"{symbol}: Model loaded successfully")
                    except Exception as e:
                        self.logger.error(f"{symbol}: Failed to load model: {e}")
                        return {'direction': 0, 'confidence': 0.5, 'signal_strength': 0, 'probability': 0.5}
                else:
                    # No model exists - return neutral
                    self.logger.warning(f"No pre-trained model for {symbol} {timeframe}")
                    return {'direction': 0, 'confidence': 0.5, 'signal_strength': 0, 'probability': 0.5}

            if model_key not in self.models:
                return {'direction': 0, 'confidence': 0.5, 'signal_strength': 0, 'probability': 0.5}

            # Prepare features
            self.logger.info(f"{symbol}: Preparing features...")
            features = self._prepare_features(data, technical_signals)
            self.logger.info(f"{symbol}: Features prepared successfully")

            if features is None:
                self.logger.warning(f"{symbol}: Feature preparation returned None")
                return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0}

            # Make prediction
            model = self.models[model_key]
            scaler = self.scalers.get(model_key)

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
                'confidence': confidence,        # Use confidence as the main confidence score
                'probability': confidence,       # Keep probability for backward compatibility
                'signal_strength': signal_strength,
                'probabilities': {
                    'bearish': probabilities[0],
                    'bullish': probabilities[1]
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating ML prediction for {symbol}: {e}")
            return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0}

    async def predict(self, symbol: str, data: Union[pd.DataFrame, Dict], technical_signals: Dict, timeframe: str = 'H1') -> Dict:
        """
        Async wrapper for predict_signal method

        Args:
            symbol: Trading symbol
            data: Historical price data (DataFrame or dict of DataFrames by timeframe)
            technical_signals: Technical analysis signals
            timeframe: Timeframe for prediction

        Returns:
            dict: ML prediction results
        """
        # Handle both DataFrame and dict formats
        if isinstance(data, dict):
            # Extract data for the specified timeframe
            data = data.get(timeframe)  # type: ignore
            if data is None:
                return {'direction': 'neutral', 'confidence': 0, 'signal_strength': 0, 'probability': 0}
        
        return self.predict_signal(symbol, data, technical_signals, timeframe)  # type: ignore

    def prepare_features(self, symbol: str, data: Union[pd.DataFrame, Dict], technical_signals: Dict) -> Optional[pd.DataFrame]:
        """
        Public method to prepare features for ML models using advanced feature engineering

        Args:
            symbol: Trading symbol
            data: Historical price data (DataFrame or dict of DataFrames by timeframe)
            technical_signals: Technical analysis signals

        Returns:
            pd.DataFrame: Prepared features or None if preparation fails
        """
        try:
            # Handle both DataFrame and dict formats
            if isinstance(data, dict):
                # Extract H1 data from timeframe dictionary
                data = data.get('H1')  # type: ignore
                if data is None:
                    return None

            # Use basic feature preparation (advanced feature engineering removed)
            features_array = self._prepare_features(data, technical_signals)  # type: ignore
            if features_array is None:
                return None

            # Convert to DataFrame for compatibility
            feature_names = [
                'returns_1d', 'returns_5d', 'volatility_5d', 'volatility_20d', 'volume_ratio',
                'rsi_norm', 'vwap_position', 'bb_position', 'macd_signal', 'trend_strength',
                'momentum', 'support_resistance', 'regime_score'
            ]

            # Ensure we have the right number of features
            if len(features_array) != len(feature_names):
                self.logger.warning(f"Feature count mismatch: got {len(features_array)}, expected {len(feature_names)}")
                # Pad or truncate to match expected length
                if len(features_array) < len(feature_names):
                    features_array = np.pad(features_array, (0, len(feature_names) - len(features_array)), 'constant')
                else:
                    features_array = features_array[:len(feature_names)]

            features_df = pd.DataFrame([features_array], columns=feature_names)

            return features_df

        except Exception as e:
            self.logger.error(f"Error preparing features for {symbol}: {e}")
            return None

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
            volume_data = data.get('volume', data.get('tick_volume', np.ones(len(close_prices))))
            
            # Ensure volume has the same length as price data
            if len(volume_data) > len(close_prices):
                volume = volume_data[-len(close_prices):]
            elif len(volume_data) < len(close_prices):
                # Pad volume with ones if shorter
                volume = np.ones(len(close_prices))
                volume[-len(volume_data):] = volume_data
            else:
                volume = volume_data
            
            # Convert to numpy array if it's a pandas Series
            if hasattr(volume, 'values'):
                volume = volume.values  # type: ignore
            elif not isinstance(volume, np.ndarray):
                volume = np.array(volume)

            # Returns
            returns_1d = np.diff(close_prices[-2:])[0] / close_prices[-2] if len(close_prices) >= 2 else 0.0  # type: ignore
            returns_5d = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) >= 6 else 0.0
            
            # Ensure returns are scalars
            returns_1d = float(returns_1d) if np.isscalar(returns_1d) or returns_1d.size == 1 else 0  # type: ignore
            returns_5d = float(returns_5d) if np.isscalar(returns_5d) else 0  # type: ignore

            features.extend([returns_1d, returns_5d])

            # Volatility features
            volatility_5d = float(np.std(np.diff(close_prices[-6:]))) if len(close_prices) >= 6 else 0.0  # type: ignore
            volatility_20d = float(np.std(np.diff(close_prices[-21:]))) if len(close_prices) >= 21 else 0.0  # type: ignore
            
            # Ensure volatilities are scalars
            volatility_5d = float(volatility_5d)
            volatility_20d = float(volatility_20d)

            features.extend([volatility_5d, volatility_20d])

            # Volume features
            avg_volume = float(np.mean(volume))  # type: ignore
            volume_ratio = float(volume[-1] / avg_volume) if avg_volume > 0 else 1.0
            
            # Ensure volume features are scalars
            avg_volume = float(avg_volume)
            volume_ratio = float(volume_ratio)

            features.append(volume_ratio)

            # Technical indicators from signals
            if technical_signals:
                # Handle both nested dict format (from feature engineer) and flat format (from backtester)
                if isinstance(technical_signals.get('rsi'), dict):
                    # Nested format: {'rsi': {'value': 50}}
                    rsi = float(technical_signals.get('rsi', {}).get('value', 50))
                else:
                    # Flat format: {'rsi': 50}
                    rsi = float(technical_signals.get('rsi', 50))
                features.append(rsi / 100.0)  # Normalize

                # VWAP position
                if isinstance(technical_signals.get('vwap'), dict):
                    vwap_pos = 1.0 if technical_signals.get('vwap', {}).get('position') == 'above' else 0.0
                else:
                    # For flat format, assume price > vwap means above
                    current_price = close_prices[-1] if len(close_prices) > 0 else 0
                    vwap_value = technical_signals.get('vwap', current_price)
                    vwap_pos = 1.0 if current_price > vwap_value else 0.0
                features.append(vwap_pos)

                # EMA trend
                if isinstance(technical_signals.get('ema'), dict):
                    ema_trend = technical_signals.get('ema', {}).get('trend', 'neutral')
                    ema_score = float({'bullish': 1, 'bearish': -1, 'bullish_crossover': 1.5, 'bearish_crossover': -1.5}.get(ema_trend, 0))
                else:
                    # For flat format, compare EMA values
                    ema9 = technical_signals.get('ema_9', current_price)
                    ema21 = technical_signals.get('ema_21', current_price)
                    if ema9 > ema21:
                        ema_score = 1.0  # Bullish
                    elif ema9 < ema21:
                        ema_score = -1.0  # Bearish
                    else:
                        ema_score = 0.0  # Neutral
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

    def prepare_ensemble_features(self, symbol: str, data: Union[pd.DataFrame, Dict], technical_signals: Dict) -> Optional[pd.DataFrame]:
        """
        Prepare features for ensemble ML models

        Args:
            symbol: Trading symbol
            data: Historical price data
            technical_signals: Technical analysis signals

        Returns:
            pd.DataFrame: Feature DataFrame or None
        """
        try:
            # Convert data to DataFrame if it's a dict
            if isinstance(data, dict):
                # Assume it's bars data with H1 timeframe
                if 'H1' in data and data['H1'] is not None and not data['H1'].empty:
                    df = pd.DataFrame(data['H1'])
                    if len(df) == 0:
                        return None
                    # Use only the most recent data point for prediction
                    data = df.tail(1)
                else:
                    return None

            if not isinstance(data, pd.DataFrame) or data.empty:
                return None

            # Prepare features using existing method
            features_array = self._prepare_features(data, technical_signals)
            if features_array is None:
                return None

            # Convert to DataFrame with feature names
            feature_names = [
                'rsi', 'macd_signal', 'bb_position', 'ema_trend',
                'volume_ratio', 'price_change', 'vwap_position',
                'ema_score', 'feature_8', 'feature_9'
            ]

            # Ensure we have the right number of features
            if len(features_array) < len(feature_names):
                # Pad with zeros if needed
                features_array = np.pad(features_array, (0, len(feature_names) - len(features_array)), 'constant')
            elif len(features_array) > len(feature_names):
                # Truncate if too many
                features_array = features_array[:len(feature_names)]

            features_df = pd.DataFrame([features_array], columns=feature_names[:len(features_array)])

            return features_df

        except Exception as e:
            self.logger.error(f"Error preparing features DataFrame: {e}")
            return None

    def _load_or_train_model(self, symbol: str, data: pd.DataFrame, timeframe: str = 'H1'):
        """
        Load existing model or train new one

        Args:
            symbol: Trading symbol
            data: Historical data for training
            timeframe: Timeframe for the model
        """
        try:
            model_key = f"{symbol}_{timeframe}"
            model_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_model.pkl')
            scaler_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_scaler.pkl')

            # Try to load existing model
            if os.path.exists(model_path):
                self.models[model_key] = joblib.load(model_path)
                if os.path.exists(scaler_path):
                    self.scalers[model_key] = joblib.load(scaler_path)
                self.logger.info(f"Loaded existing model for {symbol} {timeframe}")
                return

            # Train new model
            self.logger.info(f"Training new model for {symbol} {timeframe}")
            self._train_model(symbol, data, timeframe)

        except Exception as e:
            self.logger.error(f"Error loading/training model for {symbol}: {e}")

    def _train_model(self, symbol: str, data: pd.DataFrame, timeframe: str = 'H1'):
        """
        Train ML model for symbol and timeframe

        Args:
            symbol: Trading symbol
            data: Historical data
            timeframe: Timeframe for the model (M1, M5, M15, H1, D1)
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
            future_returns = np.diff(close_prices[20:]) / close_prices[20:-1]  # 20-period future returns  # type: ignore

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

            # Store model and scaler with timeframe key
            model_key = f"{symbol}_{timeframe}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler

            # Save model with timeframe
            model_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_model.pkl')
            scaler_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_scaler.pkl')

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)

            self.logger.info(f"Trained and saved model for {symbol}")

        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")

    def get_feature_importance_feedback(self, symbol: str, timeframe: str = 'H1') -> Dict:
        """
        Analyze which features are most predictive for the ML model

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis

        Returns:
            dict: Feature importance analysis
        """
        try:
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.models:
                self.logger.warning(f"No trained model found for {symbol} {timeframe}")
                return {}

            model = self.models[model_key]

            # Get feature importance if the model supports it
            if hasattr(model, 'feature_importances_'):
                # For tree-based models (Random Forest, Gradient Boosting)
                feature_importances = model.feature_importances_

                # Map feature indices to names (simplified mapping)
                feature_names = self._get_feature_names()

                # Create importance dictionary
                importance_dict = {}
                for i, importance in enumerate(feature_importances):
                    if i < len(feature_names):
                        feature_name = feature_names[i]
                        importance_dict[feature_name] = float(importance)

                # Categorize by analyzer type
                technical_features = {}
                fundamental_features = {}
                sentiment_features = {}

                for feature, importance in importance_dict.items():
                    if any(tech in feature.lower() for tech in ['rsi', 'macd', 'bb', 'ema', 'vwap', 'sma', 'stoch']):
                        technical_features[feature] = importance
                    elif any(fund in feature.lower() for fund in ['rate', 'gdp', 'cpi', 'employment']):
                        fundamental_features[feature] = importance
                    elif any(sent in feature.lower() for sent in ['sentiment', 'news', 'social']):
                        sentiment_features[feature] = importance

                return {
                    'technical_features': technical_features,
                    'fundamental_features': fundamental_features,
                    'sentiment_features': sentiment_features,
                    'overall_importance': importance_dict,
                    'top_features': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                }

            elif hasattr(model, 'coef_'):
                # For linear models
                coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                feature_names = self._get_feature_names()

                importance_dict = {}
                for i, coef in enumerate(coefficients):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = abs(float(coef))

                return {
                    'feature_coefficients': importance_dict,
                    'top_features': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                }

            else:
                self.logger.warning(f"Model type {type(model)} doesn't support feature importance analysis")
                return {}

        except Exception as e:
            self.logger.error(f"Error getting feature importance for {symbol}: {e}")
            return {}

    def suggest_technical_parameters(self, importance_data: Dict, timeframe: str = 'H1') -> Dict:
        """
        Suggest optimal technical indicator parameters based on ML model performance

        Args:
            importance_data: Feature importance data from get_feature_importance_feedback
            timeframe: Timeframe for analysis

        Returns:
            dict: Suggested parameter adjustments
        """
        try:
            if not importance_data or 'technical_features' not in importance_data:
                return {}

            technical_importance = importance_data['technical_features']

            suggestions = {}

            # Analyze RSI importance
            rsi_features = {k: v for k, v in technical_importance.items() if 'rsi' in k.lower()}
            if rsi_features:
                avg_rsi_importance = sum(rsi_features.values()) / len(rsi_features)
                if avg_rsi_importance > 0.1:  # High importance
                    suggestions['rsi_period'] = 12  # Shorter period for more responsive RSI
                    suggestions['rsi_overbought'] = 72  # Adjust thresholds
                    suggestions['rsi_oversold'] = 28
                elif avg_rsi_importance < 0.05:  # Low importance
                    suggestions['rsi_weight'] = 0.3  # Reduce weight in signal combination

            # Analyze EMA importance
            ema_features = {k: v for k, v in technical_importance.items() if 'ema' in k.lower()}
            if ema_features:
                avg_ema_importance = sum(ema_features.values()) / len(ema_features)
                if avg_ema_importance > 0.08:
                    suggestions['ema_fast_period'] = 9   # Shorter for trending markets
                    suggestions['ema_slow_period'] = 21
                else:
                    suggestions['ema_weight'] = 0.4  # Moderate weight

            # Analyze MACD importance
            macd_features = {k: v for k, v in technical_importance.items() if 'macd' in k.lower()}
            if macd_features:
                avg_macd_importance = sum(macd_features.values()) / len(macd_features)
                if avg_macd_importance > 0.08:
                    suggestions['macd_fast'] = 12
                    suggestions['macd_slow'] = 26
                    suggestions['macd_signal'] = 9
                else:
                    suggestions['macd_weight'] = 0.4

            # Analyze Bollinger Bands importance
            bb_features = {k: v for k, v in technical_importance.items() if 'bb' in k.lower()}
            if bb_features:
                avg_bb_importance = sum(bb_features.values()) / len(bb_features)
                if avg_bb_importance > 0.07:
                    suggestions['bb_period'] = 20
                    suggestions['bb_std_dev'] = 2.0
                else:
                    suggestions['bb_weight'] = 0.3

            return suggestions

        except Exception as e:
            self.logger.error(f"Error suggesting technical parameters: {e}")
            return {}

    def _get_feature_names(self) -> list:
        """
        Get the list of feature names used by the model

        Returns:
            list: Feature names
        """
        # This should match the feature engineering in _prepare_features
        base_features = []

        # Price-based features (50 periods)
        for i in range(50):
            base_features.extend([
                f'close_{i}',
                f'high_{i}',
                f'low_{i}',
                f'volume_{i}'
            ])

        # Technical indicators
        for indicator in self.technical_indicators:
            for period in self.lookback_periods:
                base_features.append(f'{indicator}_{period}')

        # Add fundamental and sentiment features if available
        base_features.extend([
            'interest_rate_diff',
            'gdp_growth',
            'news_sentiment',
            'social_sentiment'
        ])

        return base_features

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

    def get_model_version(self, symbol: str, timeframe: str = 'H1') -> Optional[str]:
        """
        Return a lightweight model version identifier for a given symbol and timeframe.

        Currently implemented as the model file modification timestamp (ISO string)
        when a saved model exists, otherwise returns None.
        """
        try:
            model_path = os.path.join(self.model_dir, f'{symbol}_{timeframe}_model.pkl')
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                return datetime.fromtimestamp(mtime).isoformat()
            return None
        except Exception as e:
            self.logger.debug(f"Error getting model version for {symbol} {timeframe}: {e}")
            return None