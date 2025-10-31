#!/usr/bin/env python3
"""
Ensemble ML Predictor
Combines multiple ML models for improved prediction accuracy
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble ML predictor combining multiple models"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Ensemble configuration
        ensemble_config = self.config.get('ml', {}).get('ensemble', {})
        self.enabled = ensemble_config.get('enabled', True)
        self.voting_method = ensemble_config.get('voting_method', 'soft')  # 'hard' or 'soft'
        self.models_to_use = ensemble_config.get('models', [
            'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'mlp'
        ])

        # Model hyperparameters
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'mlp': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'random_state': 42
            }
        }

        # Initialize models
        self.models = {}
        self.ensemble_model = None
        self.model_weights = {}
        self.feature_importance = {}

        # Performance tracking
        self.model_performance = {}
        self.ensemble_performance = {}

        logger.info("Ensemble Predictor initialized")

    def initialize_models(self):
        """Initialize individual ML models"""
        if not self.enabled:
            logger.info("Ensemble predictor disabled")
            return

        try:
            estimators = []

            # Random Forest
            if 'random_forest' in self.models_to_use:
                rf = RandomForestClassifier(**self.model_params['random_forest'])
                self.models['random_forest'] = rf
                estimators.append(('rf', rf))

            # Gradient Boosting
            if 'gradient_boosting' in self.models_to_use:
                gb = GradientBoostingClassifier(**self.model_params['gradient_boosting'])
                self.models['gradient_boosting'] = gb
                estimators.append(('gb', gb))

            # XGBoost
            if 'xgboost' in self.models_to_use:
                xgb_model = xgb.XGBClassifier(**self.model_params['xgboost'])
                self.models['xgboost'] = xgb_model
                estimators.append(('xgb', xgb_model))

            # LightGBM
            if 'lightgbm' in self.models_to_use:
                lgb_model = lgb.LGBMClassifier(**self.model_params['lightgbm'])
                self.models['lightgbm'] = lgb_model
                estimators.append(('lgb', lgb_model))

            # MLP
            if 'mlp' in self.models_to_use:
                mlp = MLPClassifier(**self.model_params['mlp'])
                self.models['mlp'] = mlp
                estimators.append(('mlp', mlp))

            # Create ensemble voting classifier
            if estimators:
                self.ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting=self.voting_method
                )

            logger.info(f"Initialized {len(self.models)} models for ensemble")

        except Exception as e:
            logger.error(f"Error initializing ensemble models: {e}")

    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None) -> Dict:
        """Train the ensemble model"""
        if not self.enabled or not self.ensemble_model:
            return {}

        try:
            logger.info("Training ensemble models...")

            # Train individual models and track performance
            individual_predictions = {}
            individual_scores = {}

            for name, model in self.models.items():
                logger.debug(f"Training {name}...")

                # Train model
                model.fit(X_train, y_train)

                # Get predictions and performance
                train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)

                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_accuracy = accuracy_score(y_val, val_pred)
                    individual_predictions[name] = val_pred
                    individual_scores[name] = val_accuracy
                    logger.debug(f"{name} - Train: {train_accuracy:.3f}, Val: {val_accuracy:.3f}")
                else:
                    individual_scores[name] = train_accuracy
                    logger.debug(f"{name} - Train: {train_accuracy:.3f}")

                # Store performance
                self.model_performance[name] = {
                    'train_accuracy': train_accuracy,
                    'val_accuracy': individual_scores.get(name, train_accuracy),
                    'last_trained': datetime.now()
                }

            # Calculate model weights based on performance
            self._calculate_model_weights(individual_scores)

            # Train ensemble model
            logger.debug("Training ensemble model...")
            self.ensemble_model.fit(X_train, y_train)

            # Evaluate ensemble
            ensemble_train_pred = self.ensemble_model.predict(X_train)
            ensemble_train_accuracy = accuracy_score(y_train, ensemble_train_pred)

            ensemble_metrics = {
                'train_accuracy': ensemble_train_accuracy,
                'individual_scores': individual_scores,
                'model_weights': self.model_weights.copy(),
                'num_models': len(self.models)
            }

            if X_val is not None and y_val is not None:
                ensemble_val_pred = self.ensemble_model.predict(X_val)
                ensemble_val_accuracy = accuracy_score(y_val, ensemble_val_pred)
                ensemble_metrics['val_accuracy'] = ensemble_val_accuracy

                # Store ensemble performance
                self.ensemble_performance = {
                    'train_accuracy': ensemble_train_accuracy,
                    'val_accuracy': ensemble_val_accuracy,
                    'last_evaluated': datetime.now()
                }

            logger.info(f"Ensemble training completed - Train: {ensemble_train_accuracy:.3f}")

            return ensemble_metrics

        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {}

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Make ensemble predictions"""
        if not self.enabled or not self.ensemble_model:
            # Fallback to random prediction if ensemble not available
            return np.random.choice([0, 1], size=len(X)), {}

        try:
            # Get ensemble prediction
            ensemble_pred = self.ensemble_model.predict(X)

            # Get individual model predictions for analysis
            individual_preds = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    individual_preds[name] = pred
                except Exception as e:
                    logger.warning(f"Error getting prediction from {name}: {e}")
                    individual_preds[name] = np.zeros(len(X))

            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(individual_preds)

            # Get prediction probabilities if available
            try:
                ensemble_proba = self.ensemble_model.predict_proba(X)
                probabilities = ensemble_proba[:, 1]  # Probability of positive class
            except:
                probabilities = np.full(len(X), 0.5)

            prediction_info = {
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': individual_preds,
                'confidence': confidence,
                'probabilities': probabilities,
                'model_weights': self.model_weights.copy()
            }

            return ensemble_pred, prediction_info

        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            # Fallback prediction
            return np.random.choice([0, 1], size=len(X)), {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.enabled or not self.ensemble_model:
            return np.full((len(X), 2), 0.5)

        try:
            return self.ensemble_model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            return np.full((len(X), 2), 0.5)

    def _calculate_model_weights(self, individual_scores: Dict[str, float]):
        """Calculate weights for each model based on performance"""
        if not individual_scores:
            # Equal weights if no scores available
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            return

        # Weight models by their validation accuracy
        total_score = sum(individual_scores.values())

        if total_score > 0:
            self.model_weights = {
                name: score / total_score for name, score in individual_scores.items()
            }
        else:
            # Equal weights fallback
            self.model_weights = {name: 1.0 / len(individual_scores) for name in individual_scores.keys()}

        logger.debug(f"Model weights: {self.model_weights}")

    def _calculate_prediction_confidence(self, individual_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate confidence score for each prediction"""
        if not individual_preds:
            return np.full(len(next(iter(individual_preds.values()))), 0.5)

        # Calculate agreement between models
        predictions_array = np.array(list(individual_preds.values()))

        # For each sample, calculate fraction of models that agree
        confidence_scores = []
        for i in range(predictions_array.shape[1]):
            sample_preds = predictions_array[:, i]
            # Agreement is the fraction of models predicting the majority class
            majority_vote = np.bincount(sample_preds.astype(int)).argmax()
            agreement = np.mean(sample_preds == majority_vote)
            confidence_scores.append(agreement)

        return np.array(confidence_scores)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models"""
        importance_dict = {}

        try:
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    importance_dict[name] = np.abs(model.coef_[0])

            # Average importance across models
            if importance_dict:
                all_importances = list(importance_dict.values())
                avg_importance = np.mean(all_importances, axis=0)
                importance_dict['ensemble_average'] = avg_importance

        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")

        return importance_dict

    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            model_data = {
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'model_weights': self.model_weights,
                'model_performance': self.model_performance,
                'ensemble_performance': self.ensemble_performance,
                'config': self.config
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Ensemble models saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self, filepath: str) -> bool:
        """Load trained models from disk"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.models = model_data.get('models', {})
            self.ensemble_model = model_data.get('ensemble_model')
            self.model_weights = model_data.get('model_weights', {})
            self.model_performance = model_data.get('model_performance', {})
            self.ensemble_performance = model_data.get('ensemble_performance', {})

            logger.info(f"Ensemble models loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def get_model_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            'ensemble_performance': self.ensemble_performance,
            'individual_performance': self.model_performance,
            'model_weights': self.model_weights,
            'num_models': len(self.models),
            'enabled': self.enabled
        }