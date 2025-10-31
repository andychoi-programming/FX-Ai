"""
Concept Drift Detection Module for FX-Ai Trading System

This module implements sophisticated concept drift detection to identify
changes in market conditions and trigger model adaptations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DriftEvent:
    """Represents a detected concept drift event"""
    timestamp: datetime
    drift_type: str
    severity: float
    confidence: float
    affected_features: List[str]
    market_regime_before: str
    market_regime_after: str
    detection_method: str


@dataclass
class DriftAnalysis:
    """Comprehensive drift analysis results"""
    total_events: int
    events: List[DriftEvent]
    current_regime: str
    regime_stability: float
    drift_frequency: float
    risk_assessment: str


class ConceptDriftDetector:
    """
    Advanced concept drift detection for market condition changes
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize concept drift detector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Drift detection configuration
        drift_config = config.get('concept_drift', {})
        self.enabled = drift_config.get('enabled', True)

        # Detection parameters
        self.drift_window = drift_config.get('drift_window', 200)
        self.reference_window = drift_config.get('reference_window', 500)
        self.significance_level = drift_config.get('significance_level', 0.05)
        self.min_drift_severity = drift_config.get('min_drift_severity', 0.1)

        # Statistical test parameters
        self.ks_test_threshold = drift_config.get('ks_test_threshold', 0.05)
        self.js_divergence_threshold = drift_config.get('js_divergence_threshold', 0.1)
        self.correlation_change_threshold = drift_config.get('correlation_change_threshold', 0.2)

        # ML-based detection parameters
        self.isolation_forest_contamination = drift_config.get('isolation_forest_contamination', 0.1)
        self.pca_variance_threshold = drift_config.get('pca_variance_threshold', 0.8)

        # Market regime detection
        self.regime_config = drift_config.get('regime_detection', {
            'volatility_clusters': 3,
            'trend_clusters': 3,
            'volume_clusters': 3
        })

        # Detection methods
        self.detection_methods = drift_config.get('detection_methods', [
            'statistical_tests',
            'distribution_comparison',
            'correlation_analysis',
            'ml_based_detection'
        ])

        # Data storage
        self.feature_history = {}
        self.drift_events: List[DriftEvent] = []
        self.regime_history = []

        # ML components
        self.isolation_forest = None
        self.feature_scaler = StandardScaler()
        self.pca_model = PCA(n_components=self.pca_variance_threshold)

        # Initialize components
        self._initialize_detection_components()

        self.logger.info("Concept Drift Detector initialized")

    def _initialize_detection_components(self):
        """Initialize detection components"""
        try:
            # Initialize isolation forest for anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=self.isolation_forest_contamination,
                random_state=42,
                n_estimators=100
            )

            self.logger.info("Detection components initialized")

        except Exception as e:
            self.logger.error(f"Error initializing detection components: {e}")

    def detect_drift(self, symbol: str, current_data: pd.DataFrame,
                    feature_columns: List[str]) -> Optional[DriftEvent]:
        """
        Detect concept drift in current market data

        Args:
            symbol: Trading symbol
            current_data: Current market data
            feature_columns: Feature columns to monitor

        Returns:
            DriftEvent if drift detected, None otherwise
        """
        try:
            if not self.enabled or len(current_data) < self.drift_window:
                return None

            # Extract features
            features = current_data[feature_columns].values

            # Store feature history
            if symbol not in self.feature_history:
                self.feature_history[symbol] = []

            self.feature_history[symbol].append({
                'timestamp': current_data.index[-1] if hasattr(current_data.index, '__getitem__') else datetime.now(),
                'features': features[-1].copy(),
                'data': current_data.copy()
            })

            # Keep only recent history
            max_history = self.reference_window + self.drift_window
            if len(self.feature_history[symbol]) > max_history:
                self.feature_history[symbol] = self.feature_history[symbol][-max_history:]

            # Check if we have enough data for drift detection
            if len(self.feature_history[symbol]) < self.reference_window + self.drift_window:
                return None

            # Perform drift detection using multiple methods
            drift_results = []

            for method in self.detection_methods:
                try:
                    if method == 'statistical_tests':
                        drift = self._statistical_drift_detection(symbol, features, feature_columns)
                    elif method == 'distribution_comparison':
                        drift = self._distribution_drift_detection(symbol, features, feature_columns)
                    elif method == 'correlation_analysis':
                        drift = self._correlation_drift_detection(symbol, features, feature_columns)
                    elif method == 'ml_based_detection':
                        drift = self._ml_based_drift_detection(symbol, features, feature_columns)
                    else:
                        continue

                    if drift:
                        drift_results.append(drift)

                except Exception as e:
                    self.logger.warning(f"Error in {method} drift detection: {e}")
                    continue

            # Aggregate drift results
            if drift_results:
                aggregated_drift = self._aggregate_drift_results(drift_results, symbol)

                if aggregated_drift and aggregated_drift.severity >= self.min_drift_severity:
                    # Store drift event
                    self.drift_events.append(aggregated_drift)

                    # Update regime history
                    self.regime_history.append({
                        'timestamp': aggregated_drift.timestamp,
                        'regime': aggregated_drift.market_regime_after,
                        'drift_type': aggregated_drift.drift_type
                    })

                    self.logger.info(f"Concept drift detected for {symbol}: {aggregated_drift.drift_type} (severity: {aggregated_drift.severity:.3f})")
                    return aggregated_drift

            return None

        except Exception as e:
            self.logger.error(f"Error detecting drift for {symbol}: {e}")
            return None

    def _statistical_drift_detection(self, symbol: str, features: np.ndarray,
                                   feature_columns: List[str]) -> Optional[DriftEvent]:
        """
        Statistical drift detection using Kolmogorov-Smirnov test

        Args:
            symbol: Trading symbol
            features: Feature matrix
            feature_columns: Feature column names

        Returns:
            DriftEvent if drift detected
        """
        try:
            history = self.feature_history[symbol]

            # Get reference and current windows
            reference_data = np.array([h['features'] for h in history[-self.reference_window:]])
            current_data = features[-self.drift_window:]

            if len(reference_data) < 50 or len(current_data) < 50:
                return None

            # Perform KS test on each feature
            drift_features = []
            max_statistic = 0.0

            for i, feature_name in enumerate(feature_columns):
                try:
                    ref_feature = reference_data[:, i]
                    curr_feature = current_data[:, i]

                    # Kolmogorov-Smirnov test
                    statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)

                    if p_value < self.ks_test_threshold:
                        drift_features.append(feature_name)
                        max_statistic = max(max_statistic, statistic)

                except Exception:
                    continue

            if drift_features and max_statistic > 0.1:
                # Determine market regimes
                regime_before = self._classify_market_regime(reference_data)
                regime_after = self._classify_market_regime(current_data)

                return DriftEvent(
                    timestamp=datetime.now(),
                    drift_type="statistical_distribution_change",
                    severity=min(max_statistic * 2, 1.0),  # Scale severity
                    confidence=1 - self.ks_test_threshold,
                    affected_features=drift_features,
                    market_regime_before=regime_before,
                    market_regime_after=regime_after,
                    detection_method="statistical_tests"
                )

            return None

        except Exception as e:
            self.logger.warning(f"Error in statistical drift detection: {e}")
            return None

    def _distribution_drift_detection(self, symbol: str, features: np.ndarray,
                                    feature_columns: List[str]) -> Optional[DriftEvent]:
        """
        Distribution drift detection using Jensen-Shannon divergence

        Args:
            symbol: Trading symbol
            features: Feature matrix
            feature_columns: Feature column names

        Returns:
            DriftEvent if drift detected
        """
        try:
            history = self.feature_history[symbol]

            # Get reference and current windows
            reference_data = np.array([h['features'] for h in history[-self.reference_window:]])
            current_data = features[-self.drift_window:]

            if len(reference_data) < 50 or len(current_data) < 50:
                return None

            # Calculate Jensen-Shannon divergence for each feature
            max_divergence = 0.0
            drift_features = []

            for i, feature_name in enumerate(feature_columns):
                try:
                    ref_feature = reference_data[:, i]
                    curr_feature = current_data[:, i]

                    # Create histograms
                    hist_range = (min(np.min(ref_feature), np.min(curr_feature)),
                                max(np.max(ref_feature), np.max(curr_feature)))

                    ref_hist, _ = np.histogram(ref_feature, bins=20, range=hist_range, density=True)
                    curr_hist, _ = np.histogram(curr_feature, bins=20, range=hist_range, density=True)

                    # Add small value to avoid zero probabilities
                    ref_hist = ref_hist + 1e-8
                    curr_hist = curr_hist + 1e-8

                    # Normalize
                    ref_hist = ref_hist / np.sum(ref_hist)
                    curr_hist = curr_hist / np.sum(curr_hist)

                    # Jensen-Shannon divergence
                    js_divergence = jensenshannon(ref_hist, curr_hist)

                    if js_divergence > self.js_divergence_threshold:
                        drift_features.append(feature_name)
                        max_divergence = max(max_divergence, js_divergence)

                except Exception:
                    continue

            if drift_features and max_divergence > self.js_divergence_threshold:
                # Determine market regimes
                regime_before = self._classify_market_regime(reference_data)
                regime_after = self._classify_market_regime(current_data)

                return DriftEvent(
                    timestamp=datetime.now(),
                    drift_type="distribution_shift",
                    severity=min(max_divergence, 1.0),
                    confidence=0.8,
                    affected_features=drift_features,
                    market_regime_before=regime_before,
                    market_regime_after=regime_after,
                    detection_method="distribution_comparison"
                )

            return None

        except Exception as e:
            self.logger.warning(f"Error in distribution drift detection: {e}")
            return None

    def _correlation_drift_detection(self, symbol: str, features: np.ndarray,
                                   feature_columns: List[str]) -> Optional[DriftEvent]:
        """
        Correlation drift detection

        Args:
            symbol: Trading symbol
            features: Feature matrix
            feature_columns: Feature column names

        Returns:
            DriftEvent if drift detected
        """
        try:
            history = self.feature_history[symbol]

            # Get reference and current windows
            reference_data = np.array([h['features'] for h in history[-self.reference_window:]])
            current_data = features[-self.drift_window:]

            if len(reference_data) < 50 or len(current_data) < 50:
                return None

            # Calculate correlation matrices
            ref_corr = np.corrcoef(reference_data.T)
            curr_corr = np.corrcoef(current_data.T)

            # Calculate correlation change
            corr_diff = np.abs(ref_corr - curr_corr)
            max_corr_change = np.max(corr_diff)

            # Find affected feature pairs
            affected_features = []
            corr_changes = {}

            for i in range(len(feature_columns)):
                for j in range(i+1, len(feature_columns)):
                    change = abs(ref_corr[i, j] - curr_corr[i, j])
                    if change > self.correlation_change_threshold:
                        feature_pair = f"{feature_columns[i]}_{feature_columns[j]}"
                        affected_features.append(feature_pair)
                        corr_changes[feature_pair] = change

            if affected_features and max_corr_change > self.correlation_change_threshold:
                # Determine market regimes
                regime_before = self._classify_market_regime(reference_data)
                regime_after = self._classify_market_regime(current_data)

                return DriftEvent(
                    timestamp=datetime.now(),
                    drift_type="correlation_structure_change",
                    severity=min(max_corr_change, 1.0),
                    confidence=0.7,
                    affected_features=affected_features,
                    market_regime_before=regime_before,
                    market_regime_after=regime_after,
                    detection_method="correlation_analysis"
                )

            return None

        except Exception as e:
            self.logger.warning(f"Error in correlation drift detection: {e}")
            return None

    def _ml_based_drift_detection(self, symbol: str, features: np.ndarray,
                                feature_columns: List[str]) -> Optional[DriftEvent]:
        """
        ML-based drift detection using isolation forest

        Args:
            symbol: Trading symbol
            features: Feature matrix
            feature_columns: Feature column names

        Returns:
            DriftEvent if drift detected
        """
        try:
            history = self.feature_history[symbol]

            # Get recent data for training and testing
            recent_data = np.array([h['features'] for h in history[-self.reference_window:]])
            current_data = features[-self.drift_window:]

            if len(recent_data) < 100 or len(current_data) < 50:
                return None

            # Scale features
            all_data = np.vstack([recent_data, current_data])
            scaled_data = self.feature_scaler.fit_transform(all_data)

            # Split back into reference and current
            ref_scaled = scaled_data[:len(recent_data)]
            curr_scaled = scaled_data[len(recent_data):]

            # Train isolation forest on reference data
            self.isolation_forest.fit(ref_scaled)

            # Score current data
            anomaly_scores = self.isolation_forest.decision_function(curr_scaled)
            anomaly_ratio = np.mean(anomaly_scores < 0)  # Proportion of anomalies

            # Also check PCA reconstruction error
            pca_reconstructed = self.pca_model.inverse_transform(
                self.pca_model.fit_transform(ref_scaled)
            )
            pca_error_ref = np.mean((ref_scaled - pca_reconstructed) ** 2)

            pca_reconstructed_curr = self.pca_model.inverse_transform(
                self.pca_model.transform(curr_scaled)
            )
            pca_error_curr = np.mean((curr_scaled - pca_reconstructed_curr) ** 2)

            pca_error_change = pca_error_curr - pca_error_ref

            # Determine if drift detected
            drift_detected = False
            severity = 0.0

            if anomaly_ratio > 0.2:  # More than 20% anomalies
                drift_detected = True
                severity = min(anomaly_ratio, 1.0)
            elif pca_error_change > 0.1:  # Significant PCA error increase
                drift_detected = True
                severity = min(pca_error_change, 1.0)

            if drift_detected:
                # Determine market regimes
                regime_before = self._classify_market_regime(recent_data)
                regime_after = self._classify_market_regime(current_data)

                return DriftEvent(
                    timestamp=datetime.now(),
                    drift_type="ml_based_anomaly",
                    severity=severity,
                    confidence=0.9,
                    affected_features=feature_columns,  # All features potentially affected
                    market_regime_before=regime_before,
                    market_regime_after=regime_after,
                    detection_method="ml_based_detection"
                )

            return None

        except Exception as e:
            self.logger.warning(f"Error in ML-based drift detection: {e}")
            return None

    def _classify_market_regime(self, features: np.ndarray) -> str:
        """
        Classify market regime based on features

        Args:
            features: Feature matrix

        Returns:
            Market regime classification
        """
        try:
            if features.shape[1] < 3:
                return "unknown"

            # Simple regime classification based on feature statistics
            volatility = np.std(features[:, 0])  # Assume first feature is price-related
            trend = np.mean(features[:, 1]) if features.shape[1] > 1 else 0  # Assume second feature is trend-related
            volume = np.mean(features[:, 2]) if features.shape[1] > 2 else 1  # Assume third feature is volume-related

            if volatility > np.percentile(features[:, 0], 75):
                vol_regime = "high_volatility"
            elif volatility < np.percentile(features[:, 0], 25):
                vol_regime = "low_volatility"
            else:
                vol_regime = "normal_volatility"

            if trend > 0.001:
                trend_regime = "bull_trend"
            elif trend < -0.001:
                trend_regime = "bear_trend"
            else:
                trend_regime = "sideways"

            return f"{vol_regime}_{trend_regime}"

        except Exception:
            return "unknown"

    def _aggregate_drift_results(self, drift_results: List[DriftEvent], symbol: str) -> Optional[DriftEvent]:
        """
        Aggregate multiple drift detection results

        Args:
            drift_results: List of detected drift events
            symbol: Trading symbol

        Returns:
            Aggregated drift event
        """
        try:
            if not drift_results:
                return None

            # Find the most severe drift
            most_severe = max(drift_results, key=lambda x: x.severity * x.confidence)

            # Aggregate affected features
            all_features = set()
            for drift in drift_results:
                all_features.update(drift.affected_features)

            # Determine consensus regime change
            regime_after = most_severe.market_regime_after

            return DriftEvent(
                timestamp=datetime.now(),
                drift_type="aggregated_drift",
                severity=most_severe.severity,
                confidence=np.mean([d.confidence for d in drift_results]),
                affected_features=list(all_features),
                market_regime_before=most_severe.market_regime_before,
                market_regime_after=regime_after,
                detection_method="aggregated"
            )

        except Exception as e:
            self.logger.error(f"Error aggregating drift results: {e}")
            return None

    def analyze_drift_history(self, symbol: Optional[str] = None,
                            days_back: int = 30) -> DriftAnalysis:
        """
        Analyze drift detection history

        Args:
            symbol: Optional symbol filter
            days_back: Days of history to analyze

        Returns:
            Comprehensive drift analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Filter events
            relevant_events = [event for event in self.drift_events
                             if event.timestamp >= cutoff_date]

            if symbol:
                # In a real implementation, events would be tagged by symbol
                pass

            # Calculate metrics
            total_events = len(relevant_events)

            if total_events == 0:
                return DriftAnalysis(
                    total_events=0,
                    events=[],
                    current_regime="stable",
                    regime_stability=1.0,
                    drift_frequency=0.0,
                    risk_assessment="low_risk"
                )

            # Calculate drift frequency (events per day)
            days_covered = (datetime.now() - cutoff_date).total_seconds() / (24 * 3600)
            drift_frequency = total_events / days_covered

            # Assess regime stability
            recent_regimes = [event.market_regime_after for event in relevant_events[-10:]]
            if recent_regimes:
                most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
                regime_stability = recent_regimes.count(most_common_regime) / len(recent_regimes)
            else:
                regime_stability = 1.0
                most_common_regime = "unknown"

            # Risk assessment
            avg_severity = np.mean([event.severity for event in relevant_events])

            if drift_frequency > 1.0 or avg_severity > 0.7:
                risk_assessment = "high_risk"
            elif drift_frequency > 0.3 or avg_severity > 0.4:
                risk_assessment = "moderate_risk"
            else:
                risk_assessment = "low_risk"

            return DriftAnalysis(
                total_events=total_events,
                events=relevant_events,
                current_regime=most_common_regime,
                regime_stability=regime_stability,
                drift_frequency=drift_frequency,
                risk_assessment=risk_assessment
            )

        except Exception as e:
            self.logger.error(f"Error analyzing drift history: {e}")
            return DriftAnalysis(
                total_events=0,
                events=[],
                current_regime="error",
                regime_stability=0.0,
                drift_frequency=0.0,
                risk_assessment="unknown"
            )

    def get_drift_alerts(self, symbol: str, severity_threshold: float = 0.5) -> List[DriftEvent]:
        """
        Get recent drift alerts above severity threshold

        Args:
            symbol: Trading symbol
            severity_threshold: Minimum severity for alerts

        Returns:
            List of significant drift events
        """
        try:
            recent_events = [event for event in self.drift_events[-20:]  # Last 20 events
                           if event.severity >= severity_threshold]

            # In a real implementation, filter by symbol
            return recent_events

        except Exception as e:
            self.logger.error(f"Error getting drift alerts: {e}")
            return []

    def create_drift_report(self) -> Dict[str, Any]:
        """
        Create comprehensive drift detection report

        Returns:
            Drift report dictionary
        """
        try:
            analysis = self.analyze_drift_history()

            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_drift_events': analysis.total_events,
                    'current_regime': analysis.current_regime,
                    'regime_stability': analysis.regime_stability,
                    'drift_frequency_per_day': analysis.drift_frequency,
                    'risk_assessment': analysis.risk_assessment
                },
                'drift_types': {},
                'severity_distribution': {},
                'regime_transitions': {},
                'recommendations': []
            }

            # Analyze drift types
            drift_types = {}
            severities = []

            for event in analysis.events:
                drift_types[event.drift_type] = drift_types.get(event.drift_type, 0) + 1
                severities.append(event.severity)

            report['drift_types'] = drift_types

            # Severity distribution
            if severities:
                report['severity_distribution'] = {
                    'mean': np.mean(severities),
                    'median': np.median(severities),
                    'max': np.max(severities),
                    'min': np.min(severities)
                }

            # Regime transitions
            transitions = {}
            for i in range(1, len(analysis.events)):
                transition = f"{analysis.events[i-1].market_regime_after}_to_{analysis.events[i].market_regime_after}"
                transitions[transition] = transitions.get(transition, 0) + 1

            report['regime_transitions'] = transitions

            # Generate recommendations
            report['recommendations'] = self._generate_drift_recommendations(analysis)

            return report

        except Exception as e:
            self.logger.error(f"Error creating drift report: {e}")
            return {}

    def _generate_drift_recommendations(self, analysis: DriftAnalysis) -> List[str]:
        """Generate drift-based recommendations"""
        try:
            recommendations = []

            if analysis.drift_frequency > 1.0:
                recommendations.append("frequent_drift_increase_model_adaptation")
            elif analysis.drift_frequency < 0.1:
                recommendations.append("stable_market_conditions_maintain_current_models")

            if analysis.regime_stability < 0.5:
                recommendations.append("low_regime_stability_implement_regime_aware_models")
            elif analysis.regime_stability > 0.8:
                recommendations.append("high_regime_stability_focus_on_optimization")

            if analysis.risk_assessment == "high_risk":
                recommendations.append("high_drift_risk_implement_conservative_strategies")
            elif analysis.risk_assessment == "low_risk":
                recommendations.append("low_drift_risk_opportunity_for_aggressive_strategies")

            if not recommendations:
                recommendations.append("drift_detection_operating_normally")

            return recommendations

        except Exception:
            return ["review_drift_detection_system"]

    def save_drift_state(self, filepath: str) -> None:
        """Save drift detection state"""
        try:
            state = {
                'drift_events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'drift_type': event.drift_type,
                        'severity': event.severity,
                        'confidence': event.confidence,
                        'affected_features': event.affected_features,
                        'market_regime_before': event.market_regime_before,
                        'market_regime_after': event.market_regime_after,
                        'detection_method': event.detection_method
                    } for event in self.drift_events[-100:]  # Last 100 events
                ],
                'regime_history': self.regime_history[-100:] if self.regime_history else [],
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.info(f"Drift detection state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving drift state: {e}")

    def load_drift_state(self, filepath: str) -> None:
        """Load drift detection state"""
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Load drift events
            self.drift_events = [
                DriftEvent(
                    timestamp=datetime.fromisoformat(event['timestamp']),
                    drift_type=event['drift_type'],
                    severity=event['severity'],
                    confidence=event['confidence'],
                    affected_features=event['affected_features'],
                    market_regime_before=event['market_regime_before'],
                    market_regime_after=event['market_regime_after'],
                    detection_method=event['detection_method']
                ) for event in state.get('drift_events', [])
            ]

            # Load regime history
            self.regime_history = state.get('regime_history', [])

            self.logger.info(f"Drift detection state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading drift state: {e}")