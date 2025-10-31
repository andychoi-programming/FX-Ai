"""
Advanced Feature Engineering Module for FX-Ai Trading System

This module provides sophisticated feature creation techniques for enhanced ML model performance:
- Technical indicators and oscillators
- Statistical and mathematical transformations
- Time-series features and patterns
- Domain-specific trading features
- Feature interactions and combinations
- Automated feature selection and engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, time
import logging
from scipy import stats
from scipy.signal import find_peaks
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for trading signals with comprehensive feature creation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced feature engineer

        Args:
            config: Configuration dictionary with feature engineering settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Feature engineering settings
        fe_config = config.get('advanced_feature_engineering', {})
        self.enabled = fe_config.get('enabled', True)
        self.feature_categories = fe_config.get('categories', [
            'technical', 'statistical', 'time_series', 'domain_specific', 'interactions'
        ])

        # Technical indicator parameters
        self.technical_params = fe_config.get('technical_params', {
            'rsi_periods': [7, 14, 21],
            'macd_params': [(12, 26, 9), (8, 21, 5)],
            'bb_periods': [20, 30],
            'bb_std': [2.0, 2.5],
            'stoch_params': [(14, 3, 3), (21, 5, 5)],
            'williams_r_periods': [14, 21],
            'cci_periods': [14, 20],
            'mfi_periods': [14, 21],
            'adx_periods': [14, 21],
            'atr_periods': [14, 21],
            'ema_periods': [9, 21, 50, 200],
            'sma_periods': [10, 20, 50, 100]
        })

        # Statistical feature parameters
        self.statistical_params = fe_config.get('statistical_params', {
            'rolling_windows': [5, 10, 20, 50],
            'percentile_periods': [10, 20],
            'zscore_windows': [20, 50],
            'skewness_windows': [20, 50],
            'kurtosis_windows': [20, 50]
        })

        # Domain-specific parameters
        self.domain_params = fe_config.get('domain_params', {
            'session_weights': {
                'london': 1.2, 'new_york': 1.1, 'tokyo': 1.0, 'sydney': 0.9
            },
            'currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
            'volatility_thresholds': [0.5, 1.0, 1.5, 2.0]
        })

        # Feature selection parameters
        self.selection_params = fe_config.get('selection_params', {
            'method': 'mutual_info',  # 'mutual_info', 'pca', 'correlation'
            'k_features': 50,
            'correlation_threshold': 0.95
        })

        # Scalers for feature normalization
        self.scalers = {}
        self.feature_names = []

        self.logger.info("Advanced Feature Engineer initialized")

    def create_features(self, symbol: str, data: pd.DataFrame,
                       technical_signals: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models

        Args:
            symbol: Trading symbol
            data: Historical price data (OHLCV)
            technical_signals: Pre-computed technical signals

        Returns:
            pd.DataFrame: Engineered features
        """
        if not self.enabled:
            return pd.DataFrame()

        try:
            self.logger.debug(f"Creating advanced features for {symbol}")

            # Start with basic price features
            features_df = self._create_basic_price_features(data)

            # Add technical indicator features
            if 'technical' in self.feature_categories:
                tech_features = self._create_technical_features(data)
                features_df = pd.concat([features_df, tech_features], axis=1)

            # Add statistical features
            if 'statistical' in self.feature_categories:
                stat_features = self._create_statistical_features(data)
                features_df = pd.concat([features_df, stat_features], axis=1)

            # Add time-series features
            if 'time_series' in self.feature_categories:
                ts_features = self._create_time_series_features(data)
                features_df = pd.concat([features_df, ts_features], axis=1)

            # Add domain-specific features
            if 'domain_specific' in self.feature_categories:
                domain_features = self._create_domain_features(symbol, data)
                features_df = pd.concat([features_df, domain_features], axis=1)

            # Add interaction features
            if 'interactions' in self.feature_categories:
                interaction_features = self._create_interaction_features(features_df)
                features_df = pd.concat([features_df, interaction_features], axis=1)

            # Clean and normalize features
            features_df = self._clean_and_normalize_features(features_df)

            # Feature selection
            features_df = self._select_features(features_df, data)

            self.logger.debug(f"Created {len(features_df.columns)} features for {symbol}")
            return features_df

        except Exception as e:
            self.logger.error(f"Error creating features for {symbol}: {e}")
            return pd.DataFrame()

    def _create_basic_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic price-based features"""
        features = pd.DataFrame(index=data.index)

        # Price returns
        features['returns_1d'] = data['close'].pct_change(1)
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_10d'] = data['close'].pct_change(10)
        features['returns_20d'] = data['close'].pct_change(20)

        # Price momentum
        features['momentum_1d'] = data['close'] - data['close'].shift(1)
        features['momentum_5d'] = data['close'] - data['close'].shift(5)
        features['momentum_10d'] = data['close'] - data['close'].shift(10)

        # Price ratios
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['body_size'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
        features['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
        features['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])

        # Volume features (if available)
        if 'volume' in data.columns or 'tick_volume' in data.columns:
            volume_col = 'volume' if 'volume' in data.columns else 'tick_volume'
            features['volume_change'] = data[volume_col].pct_change(1)
            features['volume_ma_ratio'] = data[volume_col] / data[volume_col].rolling(20).mean()

        return features

    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicator features"""
        features = pd.DataFrame(index=data.index)

        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            open_price = data['open'].values

            # RSI features
            for period in self.technical_params['rsi_periods']:
                if len(close) > period:
                    rsi = talib.RSI(close, timeperiod=period)
                    features[f'rsi_{period}'] = rsi
                    features[f'rsi_{period}_slope'] = pd.Series(rsi).diff(3)

            # MACD features
            for fast, slow, signal in self.technical_params['macd_params']:
                if len(close) > slow:
                    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fast,
                                                          slowperiod=slow, signalperiod=signal)
                    features[f'macd_{fast}_{slow}_{signal}'] = macd
                    features[f'macd_signal_{fast}_{slow}_{signal}'] = macdsignal
                    features[f'macd_hist_{fast}_{slow}_{signal}'] = macdhist
                    features[f'macd_crossover_{fast}_{slow}_{signal}'] = (macd > macdsignal).astype(int)

            # Bollinger Bands
            for period in self.technical_params['bb_periods']:
                for nbdev in self.technical_params['bb_std']:
                    if len(close) > period:
                        upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
                        features[f'bb_upper_{period}_{nbdev}'] = upper
                        features[f'bb_middle_{period}_{nbdev}'] = middle
                        features[f'bb_lower_{period}_{nbdev}'] = lower
                        features[f'bb_position_{period}_{nbdev}'] = (close - lower) / (upper - lower)
                        features[f'bb_width_{period}_{nbdev}'] = (upper - lower) / middle

            # Stochastic Oscillator
            for k_period, slow_k, slow_d in self.technical_params['stoch_params']:
                if len(data) > k_period:
                    slowk, slowd = talib.STOCH(high, low, close,
                                             fastk_period=k_period, slowk_period=slow_k,
                                             slowd_period=slow_d)
                    features[f'stoch_k_{k_period}_{slow_k}_{slow_d}'] = slowk
                    features[f'stoch_d_{k_period}_{slow_k}_{slow_d}'] = slowd

            # Williams %R
            for period in self.technical_params['williams_r_periods']:
                if len(data) > period:
                    willr = talib.WILLR(high, low, close, timeperiod=period)
                    features[f'williams_r_{period}'] = willr

            # Commodity Channel Index
            for period in self.technical_params['cci_periods']:
                if len(data) > period:
                    cci = talib.CCI(high, low, close, timeperiod=period)
                    features[f'cci_{period}'] = cci

            # Money Flow Index
            for period in self.technical_params['mfi_periods']:
                if len(data) > period and ('volume' in data.columns or 'tick_volume' in data.columns):
                    volume_col = 'volume' if 'volume' in data.columns else 'tick_volume'
                    mfi = talib.MFI(high, low, close, data[volume_col].values, timeperiod=period)
                    features[f'mfi_{period}'] = mfi

            # ADX (Average Directional Movement Index)
            for period in self.technical_params['adx_periods']:
                if len(data) > period:
                    adx = talib.ADX(high, low, close, timeperiod=period)
                    features[f'adx_{period}'] = adx

            # ATR (Average True Range)
            for period in self.technical_params['atr_periods']:
                if len(data) > period:
                    atr = talib.ATR(high, low, close, timeperiod=period)
                    features[f'atr_{period}'] = atr
                    features[f'atr_ratio_{period}'] = atr / data['close']

            # Moving Averages
            for period in self.technical_params['ema_periods']:
                if len(close) > period:
                    ema = talib.EMA(close, timeperiod=period)
                    features[f'ema_{period}'] = ema
                    features[f'ema_ratio_{period}'] = close / ema

            for period in self.technical_params['sma_periods']:
                if len(close) > period:
                    sma = talib.SMA(close, timeperiod=period)
                    features[f'sma_{period}'] = sma
                    features[f'sma_ratio_{period}'] = close / sma

        except Exception as e:
            self.logger.warning(f"Error creating technical features: {e}")

        return features

    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        features = pd.DataFrame(index=data.index)

        try:
            close = data['close']

            # Rolling statistics
            for window in self.statistical_params['rolling_windows']:
                if len(close) > window:
                    # Basic statistics
                    features[f'rolling_mean_{window}'] = close.rolling(window).mean()
                    features[f'rolling_std_{window}'] = close.rolling(window).std()
                    features[f'rolling_skew_{window}'] = close.rolling(window).skew()
                    features[f'rolling_kurt_{window}'] = close.rolling(window).kurt()

                    # Advanced statistics
                    features[f'rolling_zscore_{window}'] = (close - close.rolling(window).mean()) / close.rolling(window).std()
                    features[f'rolling_percentile_{window}'] = close.rolling(window).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))

                    # Volatility measures
                    returns = close.pct_change()
                    features[f'rolling_volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)  # Annualized

            # Percentile features
            for period in self.statistical_params['percentile_periods']:
                if len(close) > period:
                    features[f'percentile_{period}d'] = close.rolling(period).apply(
                        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
                    )

            # Distribution features
            for window in self.statistical_params['skewness_windows']:
                if len(close) > window:
                    features[f'skewness_{window}'] = close.rolling(window).skew()

            for window in self.statistical_params['kurtosis_windows']:
                if len(close) > window:
                    features[f'kurtosis_{window}'] = close.rolling(window).kurt()

        except Exception as e:
            self.logger.warning(f"Error creating statistical features: {e}")

        return features

    def _create_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-series specific features"""
        features = pd.DataFrame(index=data.index)

        try:
            close = data['close']

            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = close.shift(lag)
                features[f'return_lag_{lag}'] = close.pct_change(lag)

            # Trend features
            features['trend_5d'] = (close > close.shift(5)).astype(int)
            features['trend_10d'] = (close > close.shift(10)).astype(int)
            features['trend_20d'] = (close > close.shift(20)).astype(int)

            # Acceleration (second derivative)
            features['acceleration_1d'] = close.pct_change().diff()
            features['acceleration_5d'] = close.pct_change(5).diff(5)

            # Peak/valley detection
            peaks, _ = find_peaks(close.values, distance=10)
            valleys, _ = find_peaks(-close.values, distance=10)

            features['near_peak'] = 0
            features['near_valley'] = 0

            for idx in peaks:
                if idx < len(features):
                    features.iloc[idx, features.columns.get_loc('near_peak')] = 1

            for idx in valleys:
                if idx < len(features):
                    features.iloc[idx, features.columns.get_loc('near_valley')] = 1

            # Seasonal decomposition (simple)
            if len(close) >= 24:  # At least 24 periods for daily seasonality
                # Simple seasonal features based on hour of day
                if 'timestamp' in data.columns:
                    hour = pd.to_datetime(data['timestamp']).dt.hour
                    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        except Exception as e:
            self.logger.warning(f"Error creating time-series features: {e}")

        return features

    def _create_domain_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for forex trading"""
        features = pd.DataFrame(index=data.index)

        try:
            # Currency pair characteristics
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]

            # Currency strength indicators (simplified)
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
            features['is_major_pair'] = int(base_currency in major_currencies and quote_currency in major_currencies)
            features['is_usd_base'] = int(base_currency == 'USD')
            features['is_usd_quote'] = int(quote_currency == 'USD')
            features['is_jpy_pair'] = int('JPY' in symbol)

            # Volatility regime features
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()

            for threshold in self.domain_params['volatility_thresholds']:
                features[f'volatility_regime_{threshold}'] = (volatility > threshold).astype(int)

            # Session-based features (simplified)
            if 'timestamp' in data.columns:
                dt = pd.to_datetime(data['timestamp'])
                hour = dt.dt.hour

                # Trading session weights
                features['london_session'] = ((hour >= 8) & (hour <= 16)).astype(int) * self.domain_params['session_weights']['london']
                features['new_york_session'] = ((hour >= 13) & (hour <= 21)).astype(int) * self.domain_params['session_weights']['new_york']
                features['tokyo_session'] = ((hour >= 0) & (hour <= 8)).astype(int) * self.domain_params['session_weights']['tokyo']
                features['sydney_session'] = ((hour >= 22) & (hour <= 6)).astype(int) * self.domain_params['session_weights']['sydney']

            # Carry trade indicators (simplified)
            # Positive for high-yield currencies vs low-yield
            interest_rate_diff = 0.0  # This would need actual interest rate data
            features['carry_opportunity'] = interest_rate_diff

        except Exception as e:
            self.logger.warning(f"Error creating domain features for {symbol}: {e}")

        return features

    def _create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different indicators"""
        interaction_features = pd.DataFrame(index=features_df.index)

        try:
            # RSI and Momentum interactions
            rsi_cols = [col for col in features_df.columns if 'rsi' in col and not 'slope' in col]
            momentum_cols = [col for col in features_df.columns if 'momentum' in col]

            for rsi_col in rsi_cols[:2]:  # Limit to avoid too many features
                for momentum_col in momentum_cols[:2]:
                    if rsi_col in features_df.columns and momentum_col in features_df.columns:
                        interaction_features[f'{rsi_col}_{momentum_col}_interact'] = (
                            features_df[rsi_col] * features_df[momentum_col]
                        )

            # MACD and Trend interactions
            macd_cols = [col for col in features_df.columns if 'macd' in col and 'hist' in col]
            trend_cols = [col for col in features_df.columns if 'trend' in col]

            for macd_col in macd_cols[:1]:
                for trend_col in trend_cols[:2]:
                    if macd_col in features_df.columns and trend_col in features_df.columns:
                        interaction_features[f'{macd_col}_{trend_col}_interact'] = (
                            features_df[macd_col] * features_df[trend_col]
                        )

            # Volatility and Volume interactions
            vol_cols = [col for col in features_df.columns if 'volatility' in col]
            volume_cols = [col for col in features_df.columns if 'volume' in col]

            for vol_col in vol_cols[:1]:
                for volume_col in volume_cols[:1]:
                    if vol_col in features_df.columns and volume_col in features_df.columns:
                        interaction_features[f'{vol_col}_{volume_col}_interact'] = (
                            features_df[vol_col] * features_df[volume_col]
                        )

        except Exception as e:
            self.logger.warning(f"Error creating interaction features: {e}")

        return interaction_features

    def _clean_and_normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features"""
        try:
            # Remove features with too many NaN values
            nan_threshold = 0.5  # Remove features with >50% NaN
            features_df = features_df.dropna(thresh=len(features_df) * (1 - nan_threshold), axis=1)

            # Forward fill remaining NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')

            # Remove features with zero variance
            features_df = features_df.loc[:, features_df.std() > 1e-8]

            # Normalize features (optional - can be disabled for tree-based models)
            if self.config.get('normalize_features', False):
                for col in features_df.columns:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    features_df[col] = self.scalers[col].fit_transform(features_df[col].values.reshape(-1, 1)).flatten()

            return features_df

        except Exception as e:
            self.logger.warning(f"Error cleaning features: {e}")
            return features_df

    def _select_features(self, features_df: pd.DataFrame, target_data: pd.DataFrame) -> pd.DataFrame:
        """Select most important features"""
        try:
            if len(features_df.columns) <= self.selection_params['k_features']:
                return features_df

            method = self.selection_params['method']

            if method == 'mutual_info':
                # Create target variable (future returns)
                target = target_data['close'].pct_change(5).shift(-5).fillna(0)
                target = target[len(target) - len(features_df):]  # Align lengths

                if len(target) == len(features_df):
                    selector = SelectKBest(score_func=mutual_info_regression,
                                          k=self.selection_params['k_features'])
                    X_selected = selector.fit_transform(features_df, target)

                    # Get selected feature names
                    selected_mask = selector.get_support()
                    selected_features = features_df.columns[selected_mask]

                    return features_df[selected_features]

            elif method == 'correlation':
                # Remove highly correlated features
                corr_matrix = features_df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                to_drop = []
                for column in upper.columns:
                    if any(upper[column] > self.selection_params['correlation_threshold']):
                        to_drop.append(column)

                features_df = features_df.drop(columns=to_drop)

                # Limit to k features if still too many
                if len(features_df.columns) > self.selection_params['k_features']:
                    # Keep features with highest variance
                    variances = features_df.var().sort_values(ascending=False)
                    selected_features = variances.head(self.selection_params['k_features']).index
                    features_df = features_df[selected_features]

            elif method == 'pca':
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=self.selection_params['k_features'])
                X_pca = pca.fit_transform(features_df)

                # Create new feature names
                feature_names = [f'pca_{i}' for i in range(self.selection_params['k_features'])]
                features_df = pd.DataFrame(X_pca, columns=feature_names, index=features_df.index)

            return features_df

        except Exception as e:
            self.logger.warning(f"Error in feature selection: {e}")
            return features_df

    def get_feature_importance(self, features_df: pd.DataFrame,
                             target: pd.Series) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            importance_scores = {}

            # Mutual information scores
            mi_scores = mutual_info_regression(features_df, target)
            for i, col in enumerate(features_df.columns):
                importance_scores[col] = mi_scores[i]

            return importance_scores

        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")
            return {}

    def save_feature_engineering_state(self, filepath: str) -> None:
        """Save feature engineering state for reproducibility"""
        try:
            state = {
                'feature_names': self.feature_names,
                'scaler_params': {name: scaler.get_params() for name, scaler in self.scalers.items()},
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.info(f"Feature engineering state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving feature engineering state: {e}")

    def load_feature_engineering_state(self, filepath: str) -> None:
        """Load feature engineering state"""
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.feature_names = state.get('feature_names', [])
            scaler_params = state.get('scaler_params', {})

            # Recreate scalers
            for name, params in scaler_params.items():
                scaler = StandardScaler()
                scaler.set_params(**params)
                self.scalers[name] = scaler

            self.logger.info(f"Feature engineering state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading feature engineering state: {e}")