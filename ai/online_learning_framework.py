"""
Online Learning Framework for FX-Ai Trading System

This module implements incremental learning capabilities for continuous model
improvement and adaptation to changing market conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OnlineLearningState:
    """Current state of online learning models"""
    model_id: str
    last_updated: datetime
    training_samples: int
    performance_metrics: Dict[str, float]
    model_parameters: Dict[str, Any]
    drift_detected: bool


@dataclass
class LearningUpdate:
    """Result of an online learning update"""
    timestamp: datetime
    model_id: str
    new_samples: int
    performance_change: float
    drift_detected: bool
    adaptation_performed: bool


class OnlineLearningFramework:
    """
    Online learning framework for continuous model improvement
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize online learning framework

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Online learning configuration
        ol_config = config.get('online_learning', {})
        self.enabled = ol_config.get('enabled', True)

        # Learning parameters
        self.learning_rate = ol_config.get('learning_rate', 0.01)
        self.batch_size = ol_config.get('batch_size', 100)
        self.update_frequency = ol_config.get('update_frequency', 24)  # hours
        self.min_samples_for_update = ol_config.get('min_samples_for_update', 50)

        # Drift detection parameters
        self.drift_threshold = ol_config.get('drift_threshold', 0.1)
        self.drift_window = ol_config.get('drift_window', 200)
        self.stability_period = ol_config.get('stability_period', 168)  # hours

        # Model types
        self.model_types = ol_config.get('model_types', [
            'prediction_model',
            'risk_model',
            'signal_model'
        ])

        # Online learning models
        self.models = {}
        self.scalers = {}
        self.model_states = {}

        # Learning history
        self.learning_history: List[LearningUpdate] = []
        self.data_buffer = {}

        # Initialize models
        self._initialize_models()

        self.logger.info("Online Learning Framework initialized")

    def _initialize_models(self):
        """Initialize online learning models"""
        try:
            for model_type in self.model_types:
                # Initialize different model types
                if 'prediction' in model_type:
                    # Regression model for price prediction
                    model = SGDRegressor(
                        loss='squared_loss',
                        penalty='l2',
                        alpha=0.01,
                        learning_rate='adaptive',
                        eta0=self.learning_rate,
                        random_state=42
                    )
                elif 'risk' in model_type:
                    # Classification model for risk assessment
                    model = SGDClassifier(
                        loss='log_loss',
                        penalty='l2',
                        alpha=0.01,
                        learning_rate='adaptive',
                        eta0=self.learning_rate,
                        random_state=42
                    )
                elif 'signal' in model_type:
                    # Regression model for signal strength
                    model = AdaBoostRegressor(
                        n_estimators=50,
                        learning_rate=self.learning_rate,
                        random_state=42
                    )
                else:
                    # Default regression model
                    model = SGDRegressor(
                        loss='squared_loss',
                        penalty='l2',
                        alpha=0.01,
                        random_state=42
                    )

                # Store model and scaler
                self.models[model_type] = model
                self.scalers[model_type] = StandardScaler()

                # Initialize model state
                self.model_states[model_type] = OnlineLearningState(
                    model_id=model_type,
                    last_updated=datetime.now(),
                    training_samples=0,
                    performance_metrics={},
                    model_parameters={},
                    drift_detected=False
                )

                # Initialize data buffer
                self.data_buffer[model_type] = []

            self.logger.info(f"Initialized {len(self.model_types)} online learning models")

        except Exception as e:
            self.logger.error(f"Error initializing online learning models: {e}")

    def update_model(self, model_type: str, features: np.ndarray, targets: np.ndarray,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[LearningUpdate]:
        """
        Update online learning model with new data

        Args:
            model_type: Type of model to update
            features: Feature matrix
            targets: Target values
            metadata: Additional metadata

        Returns:
            LearningUpdate if update performed, None otherwise
        """
        try:
            if model_type not in self.models:
                self.logger.warning(f"Unknown model type: {model_type}")
                return None

            # Add data to buffer
            for i in range(len(features)):
                self.data_buffer[model_type].append({
                    'features': features[i],
                    'target': targets[i],
                    'timestamp': datetime.now(),
                    'metadata': metadata or {}
                })

            # Check if we have enough data for an update
            if len(self.data_buffer[model_type]) < self.min_samples_for_update:
                return None

            # Check update frequency
            state = self.model_states[model_type]
            time_since_update = (datetime.now() - state.last_updated).total_seconds() / 3600

            if time_since_update < self.update_frequency:
                return None

            # Perform model update
            return self._perform_model_update(model_type)

        except Exception as e:
            self.logger.error(f"Error updating model {model_type}: {e}")
            return None

    def _perform_model_update(self, model_type: str) -> LearningUpdate:
        """
        Perform actual model update

        Args:
            model_type: Model type to update

        Returns:
            LearningUpdate with results
        """
        try:
            # Get recent data
            recent_data = self.data_buffer[model_type][-self.batch_size:]
            features = np.array([d['features'] for d in recent_data])
            targets = np.array([d['target'] for d in recent_data])

            # Scale features
            scaler = self.scalers[model_type]
            features_scaled = scaler.fit_transform(features)

            # Get current model state
            state = self.model_states[model_type]
            old_performance = state.performance_metrics.copy()

            # Update model
            model = self.models[model_type]

            if hasattr(model, 'partial_fit'):
                # Online learning capable model
                model.partial_fit(features_scaled, targets)
            else:
                # Fallback to fit (less efficient for online learning)
                model.fit(features_scaled, targets)

            # Evaluate new performance
            predictions = model.predict(features_scaled)
            new_performance = self._evaluate_model_performance(targets, predictions, model_type)

            # Check for concept drift
            drift_detected = self._detect_concept_drift(model_type, old_performance, new_performance)

            # Calculate performance change
            performance_change = self._calculate_performance_change(old_performance, new_performance)

            # Update model state
            state.last_updated = datetime.now()
            state.training_samples += len(recent_data)
            state.performance_metrics = new_performance
            state.drift_detected = drift_detected

            # Create learning update
            update = LearningUpdate(
                timestamp=datetime.now(),
                model_id=model_type,
                new_samples=len(recent_data),
                performance_change=performance_change,
                drift_detected=drift_detected,
                adaptation_performed=drift_detected  # Adapt if drift detected
            )

            # Store update
            self.learning_history.append(update)

            # Clear processed data from buffer (keep some for context)
            self.data_buffer[model_type] = self.data_buffer[model_type][-self.drift_window:]

            self.logger.info(f"Updated model {model_type}: performance_change={performance_change:.4f}, drift={drift_detected}")
            return update

        except Exception as e:
            self.logger.error(f"Error performing model update for {model_type}: {e}")
            return LearningUpdate(
                timestamp=datetime.now(),
                model_id=model_type,
                new_samples=0,
                performance_change=0.0,
                drift_detected=False,
                adaptation_performed=False
            )

    def _evaluate_model_performance(self, targets: np.ndarray, predictions: np.ndarray,
                                  model_type: str) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            targets: True target values
            predictions: Model predictions
            model_type: Type of model

        Returns:
            Performance metrics dictionary
        """
        try:
            metrics = {}

            if 'classification' in model_type or 'risk' in model_type:
                # Classification metrics
                predictions_binary = (predictions > 0.5).astype(int)
                accuracy = accuracy_score(targets, predictions_binary)
                metrics['accuracy'] = accuracy

                # Additional classification metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                try:
                    precision = precision_score(targets, predictions_binary, average='weighted')
                    recall = recall_score(targets, predictions_binary, average='weighted')
                    f1 = f1_score(targets, predictions_binary, average='weighted')

                    metrics['precision'] = precision
                    metrics['recall'] = recall
                    metrics['f1_score'] = f1
                except:
                    pass

            else:
                # Regression metrics
                mse = mean_squared_error(targets, predictions)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(targets - predictions))

                # R-squared
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                metrics['mse'] = mse
                metrics['rmse'] = rmse
                metrics['mae'] = mae
                metrics['r_squared'] = r_squared

                # Additional regression metrics
                mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
                metrics['mape'] = mape

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {e}")
            return {'error': 'evaluation_failed'}

    def _detect_concept_drift(self, model_type: str, old_performance: Dict[str, float],
                            new_performance: Dict[str, float]) -> bool:
        """
        Detect concept drift in model performance

        Args:
            model_type: Model type
            old_performance: Previous performance metrics
            new_performance: New performance metrics

        Returns:
            True if drift detected
        """
        try:
            # Check for significant performance degradation
            primary_metric = self._get_primary_metric(model_type)

            if primary_metric in old_performance and primary_metric in new_performance:
                old_value = old_performance[primary_metric]
                new_value = new_performance[primary_metric]

                # For accuracy/classification metrics, lower is worse
                # For regression metrics, higher error is worse
                if 'accuracy' in primary_metric or 'r_squared' in primary_metric:
                    change = old_value - new_value
                else:
                    change = new_value - old_value

                # Check if change exceeds threshold
                if abs(change) > self.drift_threshold:
                    self.logger.info(f"Concept drift detected in {model_type}: {primary_metric} changed by {change:.4f}")
                    return True

            # Check for sudden changes in recent predictions
            if len(self.data_buffer[model_type]) >= self.drift_window:
                recent_errors = []
                for i in range(-min(50, len(self.data_buffer[model_type])), 0):
                    data_point = self.data_buffer[model_type][i]
                    prediction = self.models[model_type].predict(
                        self.scalers[model_type].transform([data_point['features']])
                    )[0]

                    if 'classification' in model_type or 'risk' in model_type:
                        error = abs(data_point['target'] - prediction)
                    else:
                        error = (data_point['target'] - prediction) ** 2

                    recent_errors.append(error)

                # Check for sudden increase in errors
                if len(recent_errors) >= 10:
                    recent_avg_error = np.mean(recent_errors[-10:])
                    older_avg_error = np.mean(recent_errors[:-10]) if len(recent_errors) > 10 else recent_avg_error

                    if older_avg_error > 0 and (recent_avg_error / older_avg_error) > (1 + self.drift_threshold):
                        self.logger.info(f"Sudden error increase detected in {model_type}")
                        return True

            return False

        except Exception as e:
            self.logger.warning(f"Error detecting concept drift: {e}")
            return False

    def _get_primary_metric(self, model_type: str) -> str:
        """Get primary performance metric for model type"""
        if 'classification' in model_type or 'risk' in model_type:
            return 'accuracy'
        else:
            return 'rmse'

    def _calculate_performance_change(self, old_performance: Dict[str, float],
                                    new_performance: Dict[str, float]) -> float:
        """
        Calculate performance change

        Args:
            old_performance: Old performance metrics
            new_performance: New performance metrics

        Returns:
            Performance change (positive = improvement)
        """
        try:
            primary_metric = self._get_primary_metric("default")  # Simplified

            if primary_metric in old_performance and primary_metric in new_performance:
                old_value = old_performance[primary_metric]
                new_value = new_performance[primary_metric]

                # For accuracy/r_squared, higher is better
                # For error metrics, lower is better
                if 'accuracy' in primary_metric or 'r_squared' in primary_metric:
                    return new_value - old_value
                else:
                    return old_value - new_value

            return 0.0

        except Exception:
            return 0.0

    def predict_online(self, model_type: str, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Make online predictions

        Args:
            model_type: Model type to use
            features: Feature matrix

        Returns:
            Predictions or None if model not available
        """
        try:
            if model_type not in self.models:
                return None

            # Scale features
            scaler = self.scalers[model_type]
            features_scaled = scaler.transform(features)

            # Make predictions
            model = self.models[model_type]
            predictions = model.predict(features_scaled)

            return predictions

        except Exception as e:
            self.logger.error(f"Error making online predictions for {model_type}: {e}")
            return None

    def get_model_state(self, model_type: str) -> Optional[OnlineLearningState]:
        """
        Get current state of a model

        Args:
            model_type: Model type

        Returns:
            Model state or None
        """
        try:
            return self.model_states.get(model_type)
        except Exception:
            return None

    def get_learning_history(self, model_type: Optional[str] = None,
                           hours_back: int = 168) -> List[LearningUpdate]:
        """
        Get learning history

        Args:
            model_type: Optional model type filter
            hours_back: Hours of history to retrieve

        Returns:
            List of learning updates
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            history = [update for update in self.learning_history
                      if update.timestamp >= cutoff_time]

            if model_type:
                history = [update for update in history if update.model_id == model_type]

            return history

        except Exception as e:
            self.logger.error(f"Error getting learning history: {e}")
            return []

    def force_model_update(self, model_type: str) -> Optional[LearningUpdate]:
        """
        Force an immediate model update

        Args:
            model_type: Model type to update

        Returns:
            LearningUpdate if successful
        """
        try:
            if model_type not in self.models:
                return None

            if len(self.data_buffer[model_type]) < self.min_samples_for_update:
                self.logger.warning(f"Insufficient data for {model_type} update")
                return None

            return self._perform_model_update(model_type)

        except Exception as e:
            self.logger.error(f"Error forcing model update for {model_type}: {e}")
            return None

    def reset_model(self, model_type: str) -> bool:
        """
        Reset a model to initial state

        Args:
            model_type: Model type to reset

        Returns:
            True if successful
        """
        try:
            if model_type not in self.models:
                return False

            # Reinitialize model
            if 'prediction' in model_type:
                self.models[model_type] = SGDRegressor(
                    loss='squared_loss',
                    penalty='l2',
                    alpha=0.01,
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42
                )
            elif 'risk' in model_type:
                self.models[model_type] = SGDClassifier(
                    loss='log_loss',
                    penalty='l2',
                    alpha=0.01,
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42
                )
            elif 'signal' in model_type:
                self.models[model_type] = AdaBoostRegressor(
                    n_estimators=50,
                    learning_rate=self.learning_rate,
                    random_state=42
                )

            # Reset scaler
            self.scalers[model_type] = StandardScaler()

            # Reset model state
            self.model_states[model_type] = OnlineLearningState(
                model_id=model_type,
                last_updated=datetime.now(),
                training_samples=0,
                performance_metrics={},
                model_parameters={},
                drift_detected=False
            )

            # Clear data buffer
            self.data_buffer[model_type] = []

            self.logger.info(f"Reset model {model_type}")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting model {model_type}: {e}")
            return False

    def create_learning_report(self) -> Dict[str, Any]:
        """
        Create comprehensive online learning report

        Returns:
            Learning report dictionary
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_models': len(self.models),
                    'active_models': len([m for m in self.model_states.values() if m.training_samples > 0]),
                    'total_updates': len(self.learning_history),
                    'drift_events': len([u for u in self.learning_history if u.drift_detected])
                },
                'model_states': {},
                'performance_trends': {},
                'recommendations': []
            }

            # Model states
            for model_type, state in self.model_states.items():
                report['model_states'][model_type] = {
                    'training_samples': state.training_samples,
                    'last_updated': state.last_updated.isoformat(),
                    'drift_detected': state.drift_detected,
                    'performance': state.performance_metrics
                }

            # Performance trends
            for model_type in self.models.keys():
                history = self.get_learning_history(model_type, hours_back=168)
                if history:
                    performance_changes = [u.performance_change for u in history]
                    report['performance_trends'][model_type] = {
                        'avg_performance_change': np.mean(performance_changes),
                        'performance_volatility': np.std(performance_changes),
                        'drift_frequency': np.mean([u.drift_detected for u in history]),
                        'updates_count': len(history)
                    }

            # Generate recommendations
            report['recommendations'] = self._generate_learning_recommendations(report)

            return report

        except Exception as e:
            self.logger.error(f"Error creating learning report: {e}")
            return {}

    def _generate_learning_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on report"""
        try:
            recommendations = []

            # Check model activity
            active_models = report['summary']['active_models']
            total_models = report['summary']['total_models']

            if active_models < total_models * 0.5:
                recommendations.append("increase_data_collection")

            # Check drift frequency
            drift_events = report['summary']['drift_events']
            total_updates = report['summary']['total_updates']

            if total_updates > 0:
                drift_rate = drift_events / total_updates
                if drift_rate > 0.3:
                    recommendations.append("high_concept_drift_investigate_market_changes")
                elif drift_rate < 0.05:
                    recommendations.append("low_concept_drift_consider_model_stability")

            # Check performance trends
            for model_type, trends in report['performance_trends'].items():
                if trends['avg_performance_change'] < -0.1:
                    recommendations.append(f"degrading_performance_{model_type}_consider_reset")
                elif trends['performance_volatility'] > 0.2:
                    recommendations.append(f"unstable_performance_{model_type}_increase_regularization")

            if not recommendations:
                recommendations.append("learning_system_operating_normally")

            return recommendations

        except Exception:
            return ["review_learning_system"]

    def save_learning_state(self, filepath: str) -> None:
        """Save online learning state"""
        try:
            # Note: In a real implementation, you'd save model weights
            # For now, just save metadata
            state = {
                'model_states': {
                    k: {
                        'model_id': v.model_id,
                        'last_updated': v.last_updated.isoformat(),
                        'training_samples': v.training_samples,
                        'performance_metrics': v.performance_metrics,
                        'drift_detected': v.drift_detected
                    } for k, v in self.model_states.items()
                },
                'learning_history': [
                    {
                        'timestamp': u.timestamp.isoformat(),
                        'model_id': u.model_id,
                        'new_samples': u.new_samples,
                        'performance_change': u.performance_change,
                        'drift_detected': u.drift_detected,
                        'adaptation_performed': u.adaptation_performed
                    } for u in self.learning_history[-100:]  # Last 100 updates
                ],
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.info(f"Online learning state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving learning state: {e}")

    def load_learning_state(self, filepath: str) -> None:
        """Load online learning state"""
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Load model states
            for k, v in state.get('model_states', {}).items():
                if k in self.model_states:
                    self.model_states[k] = OnlineLearningState(
                        model_id=v['model_id'],
                        last_updated=datetime.fromisoformat(v['last_updated']),
                        training_samples=v['training_samples'],
                        performance_metrics=v['performance_metrics'],
                        model_parameters={},
                        drift_detected=v['drift_detected']
                    )

            # Load learning history
            self.learning_history = [
                LearningUpdate(
                    timestamp=datetime.fromisoformat(u['timestamp']),
                    model_id=u['model_id'],
                    new_samples=u['new_samples'],
                    performance_change=u['performance_change'],
                    drift_detected=u['drift_detected'],
                    adaptation_performed=u['adaptation_performed']
                ) for u in state.get('learning_history', [])
            ]

            self.logger.info(f"Online learning state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading learning state: {e}")