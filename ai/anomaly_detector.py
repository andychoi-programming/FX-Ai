"""
Anomaly Detection Module for FX-Ai
Detects unusual market conditions using statistical and ML methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Anomaly detection for unusual market conditions"""

    def __init__(self, config: Dict):
        """
        Initialize anomaly detector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Anomaly detection parameters
        self.anomaly_config = config.get('anomaly_detection', {})
        self.enabled = self.anomaly_config.get('enabled', True)

        # Statistical thresholds
        self.z_score_threshold = self.anomaly_config.get('z_score_threshold', 3.0)
        self.iqr_multiplier = self.anomaly_config.get('iqr_multiplier', 1.5)
        self.mad_threshold = self.anomaly_config.get('mad_threshold', 3.5)

        # ML-based detection
        self.isolation_forest_contamination = self.anomaly_config.get('isolation_forest_contamination', 0.1)
        self.ocsvm_nu = self.anomaly_config.get('ocsvm_nu', 0.1)

        # Time windows for analysis
        self.short_window = self.anomaly_config.get('short_window', 20)
        self.medium_window = self.anomaly_config.get('medium_window', 50)
        self.long_window = self.anomaly_config.get('long_window', 200)

        # Feature scaling
        self.scalers = {}

        # ML models
        self.isolation_forests = {}
        self.ocsvm_models = {}

        # Historical data for training
        self.historical_data = {}
        self.anomaly_history = {}

        self.logger.info("Anomaly Detector initialized")

    def detect_statistical_anomalies(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods

        Args:
            data: Price data DataFrame
            symbol: Trading symbol

        Returns:
            dict: Anomaly detection results
        """
        try:
            results = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'anomalies_detected': False,
                'anomaly_types': [],
                'severity_score': 0.0,
                'details': {}
            }

            if len(data) < self.long_window:
                return results

            # Extract price data
            prices = data['close'].values
            volumes = data.get('volume', np.ones(len(prices))).values
            returns = np.diff(prices) / prices[:-1]

            # 1. Z-Score based anomaly detection
            z_score_anomalies = self._detect_zscore_anomalies(returns)

            # 2. IQR based anomaly detection
            iqr_anomalies = self._detect_iqr_anomalies(returns)

            # 3. MAD (Median Absolute Deviation) based detection
            mad_anomalies = self._detect_mad_anomalies(returns)

            # 4. Volume spike detection
            volume_anomalies = self._detect_volume_anomalies(volumes)

            # 5. Price gap detection
            gap_anomalies = self._detect_price_gaps(prices)

            # 6. Volatility spike detection
            volatility_anomalies = self._detect_volatility_anomalies(returns)

            # Combine results
            all_anomalies = {
                'z_score': z_score_anomalies,
                'iqr': iqr_anomalies,
                'mad': mad_anomalies,
                'volume': volume_anomalies,
                'price_gap': gap_anomalies,
                'volatility': volatility_anomalies
            }

            # Calculate severity score
            severity_score = 0.0
            anomaly_types = []

            for anomaly_type, anomaly_data in all_anomalies.items():
                if anomaly_data['detected']:
                    anomaly_types.append(anomaly_type)
                    severity_score += anomaly_data['severity']
                    results['details'][anomaly_type] = anomaly_data

            results['anomalies_detected'] = len(anomaly_types) > 0
            results['anomaly_types'] = anomaly_types
            results['severity_score'] = min(severity_score, 10.0)  # Cap at 10

            return results

        except Exception as e:
            self.logger.warning(f"Error in statistical anomaly detection for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'anomalies_detected': False,
                'anomaly_types': [],
                'severity_score': 0.0,
                'details': {'error': str(e)}
            }

    def _detect_zscore_anomalies(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using Z-score method"""
        try:
            if len(returns) < 30:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Calculate rolling Z-scores
            rolling_mean = pd.Series(returns).rolling(window=self.medium_window).mean()
            rolling_std = pd.Series(returns).rolling(window=self.medium_window).std()

            z_scores = (returns - rolling_mean) / rolling_std
            z_scores = z_scores.dropna()

            if len(z_scores) == 0:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Check for extreme Z-scores
            extreme_z = np.abs(z_scores) > self.z_score_threshold
            anomaly_count = extreme_z.sum()

            severity = min(anomaly_count * 0.5, 3.0) if anomaly_count > 0 else 0.0

            return {
                'detected': anomaly_count > 0,
                'severity': severity,
                'details': {
                    'anomaly_count': int(anomaly_count),
                    'max_z_score': float(np.abs(z_scores).max()),
                    'threshold': self.z_score_threshold
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_iqr_anomalies(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using IQR method"""
        try:
            if len(returns) < 30:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Calculate IQR
            q1 = np.percentile(returns, 25)
            q3 = np.percentile(returns, 75)
            iqr = q3 - q1

            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)

            # Find outliers
            outliers = (returns < lower_bound) | (returns > upper_bound)
            outlier_count = outliers.sum()

            severity = min(outlier_count * 0.3, 2.0) if outlier_count > 0 else 0.0

            return {
                'detected': outlier_count > 0,
                'severity': severity,
                'details': {
                    'outlier_count': int(outlier_count),
                    'iqr_range': [float(lower_bound), float(upper_bound)],
                    'multiplier': self.iqr_multiplier
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_mad_anomalies(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using Median Absolute Deviation"""
        try:
            if len(returns) < 30:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Calculate MAD
            median = np.median(returns)
            mad = np.median(np.abs(returns - median))

            if mad == 0:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Modified Z-score
            modified_z_scores = 0.6745 * (returns - median) / mad
            anomalies = np.abs(modified_z_scores) > self.mad_threshold
            anomaly_count = anomalies.sum()

            severity = min(anomaly_count * 0.4, 2.5) if anomaly_count > 0 else 0.0

            return {
                'detected': anomaly_count > 0,
                'severity': severity,
                'details': {
                    'anomaly_count': int(anomaly_count),
                    'max_modified_z': float(np.abs(modified_z_scores).max()),
                    'threshold': self.mad_threshold
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_volume_anomalies(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Detect unusual volume spikes"""
        try:
            if len(volumes) < 30:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Calculate volume moving average
            volume_ma = pd.Series(volumes).rolling(window=self.medium_window).mean()

            # Volume ratio (current vs average)
            volume_ratios = volumes / volume_ma
            volume_ratios = volume_ratios.dropna()

            if len(volume_ratios) == 0:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Check for extreme volume spikes
            extreme_volume = volume_ratios > 3.0  # 3x average volume
            spike_count = extreme_volume.sum()

            severity = min(spike_count * 0.8, 4.0) if spike_count > 0 else 0.0

            return {
                'detected': spike_count > 0,
                'severity': severity,
                'details': {
                    'spike_count': int(spike_count),
                    'max_volume_ratio': float(volume_ratios.max()),
                    'avg_volume_ratio': float(volume_ratios.mean())
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_price_gaps(self, prices: np.ndarray) -> Dict[str, Any]:
        """Detect significant price gaps"""
        try:
            if len(prices) < 2:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Calculate price changes
            price_changes = np.abs(np.diff(prices))
            avg_price = np.mean(prices)

            # Gap threshold (percentage of average price)
            gap_threshold = avg_price * 0.02  # 2% gap

            gaps = price_changes > gap_threshold
            gap_count = gaps.sum()

            severity = min(gap_count * 1.0, 5.0) if gap_count > 0 else 0.0

            return {
                'detected': gap_count > 0,
                'severity': severity,
                'details': {
                    'gap_count': int(gap_count),
                    'max_gap_pct': float((price_changes.max() / avg_price) * 100),
                    'threshold_pct': 2.0
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_volatility_anomalies(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect volatility spikes"""
        try:
            if len(returns) < 30:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            # Calculate rolling volatility
            rolling_volatility = pd.Series(returns).rolling(window=self.medium_window).std()

            # Current volatility vs historical average
            avg_volatility = rolling_volatility.mean()
            current_volatility = rolling_volatility.iloc[-1]

            if avg_volatility == 0:
                return {'detected': False, 'severity': 0.0, 'details': {}}

            volatility_ratio = current_volatility / avg_volatility

            # Check for extreme volatility
            extreme_volatility = volatility_ratio > 2.0  # 2x normal volatility

            severity = min((volatility_ratio - 1) * 2.0, 3.0) if extreme_volatility else 0.0

            return {
                'detected': extreme_volatility,
                'severity': severity,
                'details': {
                    'volatility_ratio': float(volatility_ratio),
                    'current_volatility': float(current_volatility),
                    'avg_volatility': float(avg_volatility)
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def detect_ml_anomalies(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Detect anomalies using machine learning methods

        Args:
            data: Price data DataFrame
            symbol: Trading symbol

        Returns:
            dict: ML-based anomaly detection results
        """
        try:
            results = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'anomalies_detected': False,
                'anomaly_types': [],
                'severity_score': 0.0,
                'details': {}
            }

            if len(data) < self.long_window:
                return results

            # Prepare features for ML
            features = self._prepare_ml_features(data)

            if features is None or len(features) == 0:
                return results

            # 1. Isolation Forest detection
            if_anomalies = self._detect_isolation_forest_anomalies(features, symbol)

            # 2. One-Class SVM detection
            svm_anomalies = self._detect_ocsvm_anomalies(features, symbol)

            # 3. Clustering-based detection
            cluster_anomalies = self._detect_clustering_anomalies(features)

            # Combine ML results
            ml_anomalies = {
                'isolation_forest': if_anomalies,
                'one_class_svm': svm_anomalies,
                'clustering': cluster_anomalies
            }

            # Calculate severity score
            severity_score = 0.0
            anomaly_types = []

            for anomaly_type, anomaly_data in ml_anomalies.items():
                if anomaly_data['detected']:
                    anomaly_types.append(anomaly_type)
                    severity_score += anomaly_data['severity']
                    results['details'][anomaly_type] = anomaly_data

            results['anomalies_detected'] = len(anomaly_types) > 0
            results['anomaly_types'] = anomaly_types
            results['severity_score'] = min(severity_score, 10.0)

            return results

        except Exception as e:
            self.logger.warning(f"Error in ML anomaly detection for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'anomalies_detected': False,
                'anomaly_types': [],
                'severity_score': 0.0,
                'details': {'error': str(e)}
            }

    def _prepare_ml_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML-based anomaly detection"""
        try:
            # Extract basic features
            prices = data['close'].values
            volumes = data.get('volume', np.ones(len(prices))).values

            if len(prices) < 50:
                return None

            # Calculate returns (this will be 1 element shorter than prices)
            returns = np.diff(prices) / prices[:-1]

            # Ensure we have enough data after differencing
            if len(returns) < 49:
                return None

            # Create aligned feature arrays (all should be length 49 to match returns)
            features = []

            # Price-based features (align with returns length)
            features.append(prices[-49:])  # Last 49 prices to match returns
            features.append(returns[-49:])  # Last 49 returns

            # Technical indicators (ensure they match the 49-element window)
            sma_20 = pd.Series(prices).rolling(20).mean().values[-49:]
            sma_50 = pd.Series(prices).rolling(50).mean().values[-49:]
            rsi = self._calculate_rsi(prices)[-49:]

            # Volatility (calculated on returns, already 49 elements)
            volatility = pd.Series(returns).rolling(20).std().values[-49:]

            features.extend([sma_20, sma_50, rsi, volatility])

            # Volume features (align with the 49-element window)
            volume_ma = pd.Series(volumes).rolling(20).mean().values[-49:]
            volume_ratio = volumes[-49:] / np.maximum(volume_ma, 1e-8)  # Avoid division by zero

            features.extend([volume_ma, volume_ratio])

            # Combine all features
            feature_matrix = np.column_stack(features)

            # Remove any rows with NaN values
            feature_matrix = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]

            if len(feature_matrix) == 0:
                return None

            return feature_matrix

        except Exception as e:
            self.logger.warning(f"Error preparing ML features: {e}")
            return None

    def _detect_isolation_forest_anomalies(self, features: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        try:
            # Initialize or get existing model
            if symbol not in self.isolation_forests:
                self.isolation_forests[symbol] = IsolationForest(
                    contamination=self.isolation_forest_contamination,
                    random_state=42
                )

                # Fit on historical data if available
                if symbol in self.historical_data and len(self.historical_data[symbol]) >= 100:
                    hist_features = self._prepare_ml_features(self.historical_data[symbol])
                    if hist_features is not None and len(hist_features) >= 50:
                        self.isolation_forests[symbol].fit(hist_features)

            model = self.isolation_forests[symbol]

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()

            scaled_features = self.scalers[symbol].fit_transform(features)

            # Predict anomalies
            predictions = model.fit_predict(scaled_features)
            anomaly_scores = model.decision_function(scaled_features)

            # Anomalies are marked as -1
            anomalies = predictions == -1
            anomaly_count = anomalies.sum()

            severity = min(anomaly_count * 0.5, 3.0) if anomaly_count > 0 else 0.0

            return {
                'detected': anomaly_count > 0,
                'severity': severity,
                'details': {
                    'anomaly_count': int(anomaly_count),
                    'anomaly_score': float(np.mean(anomaly_scores[anomalies])) if anomaly_count > 0 else 0.0,
                    'contamination': self.isolation_forest_contamination
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_ocsvm_anomalies(self, features: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Detect anomalies using One-Class SVM"""
        try:
            # Initialize or get existing model
            if symbol not in self.ocsvm_models:
                self.ocsvm_models[symbol] = OneClassSVM(
                    nu=self.ocsvm_nu,
                    kernel='rbf',
                    gamma='scale'
                )

                # Fit on historical data if available
                if symbol in self.historical_data and len(self.historical_data[symbol]) >= 100:
                    hist_features = self._prepare_ml_features(self.historical_data[symbol])
                    if hist_features is not None and len(hist_features) >= 50:
                        self.ocsvm_models[symbol].fit(hist_features)

            model = self.ocsvm_models[symbol]

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()

            scaled_features = self.scalers[symbol].fit_transform(features)

            # Predict anomalies
            predictions = model.predict(scaled_features)
            anomaly_scores = model.decision_function(scaled_features)

            # Anomalies are marked as -1
            anomalies = predictions == -1
            anomaly_count = anomalies.sum()

            severity = min(anomaly_count * 0.6, 3.5) if anomaly_count > 0 else 0.0

            return {
                'detected': anomaly_count > 0,
                'severity': severity,
                'details': {
                    'anomaly_count': int(anomaly_count),
                    'anomaly_score': float(np.mean(anomaly_scores[anomalies])) if anomaly_count > 0 else 0.0,
                    'nu': self.ocsvm_nu
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _detect_clustering_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using clustering (DBSCAN)"""
        try:
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(scaled_features)

            # Points labeled as -1 are anomalies
            anomalies = cluster_labels == -1
            anomaly_count = anomalies.sum()

            # Calculate distance to nearest cluster
            distances = dbscan.core_sample_indices_  # This is not quite right for distance

            severity = min(anomaly_count * 0.4, 2.5) if anomaly_count > 0 else 0.0

            return {
                'detected': anomaly_count > 0,
                'severity': severity,
                'details': {
                    'anomaly_count': int(anomaly_count),
                    'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    'noise_ratio': float(anomaly_count / len(features))
                }
            }

        except Exception as e:
            return {'detected': False, 'severity': 0.0, 'details': {'error': str(e)}}

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return np.full(len(prices), 50.0)

            gains = np.diff(prices)
            gains = np.where(gains > 0, gains, 0)
            losses = np.where(gains == 0, -gains, 0)

            avg_gain = pd.Series(gains).rolling(window=period).mean()
            avg_loss = pd.Series(losses).rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.fillna(50).values

        except Exception:
            return np.full(len(prices), 50.0)

    def get_comprehensive_anomaly_report(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive anomaly detection report

        Args:
            symbol: Trading symbol
            data: Price data DataFrame

        Returns:
            dict: Comprehensive anomaly report
        """
        try:
            # Statistical anomalies
            statistical_report = self.detect_statistical_anomalies(data, symbol)

            # ML-based anomalies
            ml_report = self.detect_ml_anomalies(data, symbol)

            # Combine reports
            comprehensive_report = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'statistical_anomalies': statistical_report,
                'ml_anomalies': ml_report,
                'overall_anomalies_detected': statistical_report['anomalies_detected'] or ml_report['anomalies_detected'],
                'overall_severity_score': statistical_report['severity_score'] + ml_report['severity_score'],
                'recommendation': self._generate_trading_recommendation(statistical_report, ml_report)
            }

            # Store in history
            if symbol not in self.anomaly_history:
                self.anomaly_history[symbol] = []
            self.anomaly_history[symbol].append(comprehensive_report)

            # Keep only last 1000 entries
            if len(self.anomaly_history[symbol]) > 1000:
                self.anomaly_history[symbol] = self.anomaly_history[symbol][-1000:]

            return comprehensive_report

        except Exception as e:
            self.logger.error(f"Error generating comprehensive anomaly report for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'overall_anomalies_detected': False,
                'overall_severity_score': 0.0,
                'recommendation': 'normal_trading',
                'error': str(e)
            }

    def _generate_trading_recommendation(self, statistical_report: Dict, ml_report: Dict) -> str:
        """Generate trading recommendation based on anomaly detection"""
        try:
            total_severity = statistical_report['severity_score'] + ml_report['severity_score']

            if total_severity >= 8.0:
                return 'halt_trading'
            elif total_severity >= 5.0:
                return 'reduce_position_size'
            elif total_severity >= 3.0:
                return 'increase_stops'
            elif total_severity >= 1.0:
                return 'monitor_closely'
            else:
                return 'normal_trading'

        except Exception:
            return 'normal_trading'

    def update_historical_data(self, symbol: str, data: pd.DataFrame):
        """Update historical data for model training"""
        self.historical_data[symbol] = data.copy()

    def get_anomaly_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get anomaly detection history for a symbol"""
        if symbol not in self.anomaly_history:
            return []
        return self.anomaly_history[symbol][-limit:]