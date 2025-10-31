"""
Strategy Adaptation Module for FX-Ai Trading System

This module enables dynamic strategy adjustment based on validation results,
market conditions, and performance feedback for optimal trading performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .performance_metrics_calculator import PerformanceMetricsCalculator
from .rolling_window_validator import RollingWindowValidator


class AdaptationStrategy(Enum):
    """Strategy adaptation approaches"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    DYNAMIC = "dynamic"
    MARKET_REGIME = "market_regime"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class AdaptationParameters:
    """Parameters for strategy adaptation"""
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    stop_loss_pips: int
    take_profit_pips: int
    position_size_multiplier: float
    risk_multiplier: float


@dataclass
class MarketCondition:
    """Current market condition assessment"""
    volatility_regime: str  # "low", "medium", "high"
    trend_strength: str     # "weak", "moderate", "strong"
    volume_regime: str      # "low", "normal", "high"
    market_phase: str       # "bull", "bear", "sideways"
    confidence_score: float


@dataclass
class AdaptationDecision:
    """Strategy adaptation decision"""
    timestamp: datetime
    reason: str
    old_parameters: AdaptationParameters
    new_parameters: AdaptationParameters
    expected_improvement: float
    confidence_level: float
    market_condition: MarketCondition


class StrategyAdapter:
    """
    Dynamic strategy adaptation based on performance and market conditions
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy adapter

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Adaptation configuration
        adapt_config = config.get('strategy_adaptation', {})
        self.enabled = adapt_config.get('enabled', True)

        # Adaptation strategy
        self.adaptation_strategy = AdaptationStrategy(
            adapt_config.get('adaptation_strategy', 'dynamic')
        )

        # Adaptation thresholds
        self.performance_threshold = adapt_config.get('performance_threshold', 0.1)  # 10% improvement
        self.confidence_threshold = adapt_config.get('confidence_threshold', 0.7)    # 70% confidence
        self.min_adaptation_interval = adapt_config.get('min_adaptation_interval', 24)  # hours

        # Parameter bounds
        self.parameter_bounds = adapt_config.get('parameter_bounds', {
            'rsi_period': {'min': 7, 'max': 21, 'step': 2},
            'macd_fast': {'min': 8, 'max': 16, 'step': 2},
            'macd_slow': {'min': 21, 'max': 31, 'step': 2},
            'macd_signal': {'min': 5, 'max': 11, 'step': 1},
            'bb_period': {'min': 15, 'max': 25, 'step': 2},
            'bb_std': {'min': 1.8, 'max': 2.5, 'step': 0.1},
            'stop_loss_pips': {'min': 15, 'max': 35, 'step': 5},
            'take_profit_pips': {'min': 30, 'max': 70, 'step': 10},
            'position_size_multiplier': {'min': 0.5, 'max': 2.0, 'step': 0.1},
            'risk_multiplier': {'min': 0.5, 'max': 1.5, 'step': 0.1}
        })

        # Learning components
        self.performance_predictor = None
        self.parameter_scaler = StandardScaler()
        self.market_condition_classifier = None

        # Adaptation history
        self.adaptation_history: List[AdaptationDecision] = []
        self.last_adaptation_time = None

        # Performance calculator
        self.metrics_calculator = PerformanceMetricsCalculator(config)

        # Rolling validator for market assessment
        self.validator = RollingWindowValidator(config)

        # Initialize ML components
        self._initialize_ml_components()

        self.logger.info("Strategy Adapter initialized")

    def _initialize_ml_components(self):
        """Initialize machine learning components for adaptation"""
        try:
            # Performance predictor (predicts parameter performance)
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            # Market condition classifier would be initialized here
            # For now, we'll use rule-based classification

            self.logger.info("ML components initialized for strategy adaptation")

        except Exception as e:
            self.logger.warning(f"Failed to initialize ML components: {e}")

    def should_adapt_strategy(self, symbol: str, current_performance: Dict[str, float],
                            market_data: pd.DataFrame, current_params: AdaptationParameters) -> bool:
        """
        Determine if strategy adaptation is needed

        Args:
            symbol: Trading symbol
            current_performance: Current performance metrics
            market_data: Recent market data
            current_params: Current strategy parameters

        Returns:
            True if adaptation is recommended
        """
        try:
            # Check minimum adaptation interval
            if self.last_adaptation_time is not None:
                time_since_last = (datetime.now() - self.last_adaptation_time).total_seconds() / 3600
                if time_since_last < self.min_adaptation_interval:
                    return False

            # Assess market conditions
            market_condition = self._assess_market_condition(market_data)

            # Evaluate current performance
            performance_score = self._evaluate_performance_score(current_performance)

            # Check adaptation criteria based on strategy
            if self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
                return self._conservative_adaptation_check(performance_score, market_condition)

            elif self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
                return self._aggressive_adaptation_check(performance_score, market_condition)

            elif self.adaptation_strategy == AdaptationStrategy.DYNAMIC:
                return self._dynamic_adaptation_check(performance_score, market_condition)

            elif self.adaptation_strategy == AdaptationStrategy.MARKET_REGIME:
                return self._market_regime_adaptation_check(market_condition)

            elif self.adaptation_strategy == AdaptationStrategy.PERFORMANCE_BASED:
                return self._performance_based_adaptation_check(performance_score)

            else:
                return False

        except Exception as e:
            self.logger.error(f"Error checking adaptation need: {e}")
            return False

    def adapt_strategy(self, symbol: str, current_performance: Dict[str, float],
                      market_data: pd.DataFrame, current_params: AdaptationParameters) -> Optional[AdaptationDecision]:
        """
        Perform strategy adaptation

        Args:
            symbol: Trading symbol
            current_performance: Current performance metrics
            market_data: Recent market data
            current_params: Current strategy parameters

        Returns:
            AdaptationDecision if adaptation performed, None otherwise
        """
        try:
            if not self.should_adapt_strategy(symbol, current_performance, market_data, current_params):
                return None

            self.logger.info(f"Adapting strategy for {symbol}")

            # Assess market conditions
            market_condition = self._assess_market_condition(market_data)

            # Generate adaptation candidates
            candidates = self._generate_adaptation_candidates(current_params, market_condition)

            # Evaluate candidates
            best_candidate, expected_improvement, confidence = self._evaluate_candidates(
                candidates, market_data, current_performance
            )

            if best_candidate is None or expected_improvement < self.performance_threshold:
                self.logger.info("No beneficial adaptation found")
                return None

            # Create adaptation decision
            decision = AdaptationDecision(
                timestamp=datetime.now(),
                reason=self._determine_adaptation_reason(market_condition, current_performance),
                old_parameters=current_params,
                new_parameters=best_candidate,
                expected_improvement=expected_improvement,
                confidence_level=confidence,
                market_condition=market_condition
            )

            # Record adaptation
            self.adaptation_history.append(decision)
            self.last_adaptation_time = datetime.now()

            # Update learning model
            self._update_learning_model(decision, market_data)

            self.logger.info(f"Strategy adapted for {symbol}: expected improvement {expected_improvement:.2%}")
            return decision

        except Exception as e:
            self.logger.error(f"Error adapting strategy: {e}")
            return None

    def _assess_market_condition(self, market_data: pd.DataFrame) -> MarketCondition:
        """
        Assess current market conditions

        Args:
            market_data: Recent market data

        Returns:
            MarketCondition assessment
        """
        try:
            if len(market_data) < 20:
                return MarketCondition("unknown", "unknown", "unknown", "unknown", 0.0)

            # Calculate returns
            returns = market_data['close'].pct_change().dropna()

            # Volatility assessment
            volatility = returns.std()
            if volatility > 0.02:
                volatility_regime = "high"
            elif volatility > 0.01:
                volatility_regime = "medium"
            else:
                volatility_regime = "low"

            # Trend strength assessment
            if len(market_data) >= 50:
                ma_short = market_data['close'].rolling(20).mean()
                ma_long = market_data['close'].rolling(50).mean()
                trend_diff = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]

                if abs(trend_diff) > 0.02:
                    trend_strength = "strong"
                elif abs(trend_diff) > 0.01:
                    trend_strength = "moderate"
                else:
                    trend_strength = "weak"
            else:
                trend_strength = "unknown"

            # Volume assessment (if available)
            if 'volume' in market_data.columns:
                avg_volume = market_data['volume'].mean()
                recent_volume = market_data['volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

                if volume_ratio > 1.5:
                    volume_regime = "high"
                elif volume_ratio > 0.7:
                    volume_regime = "normal"
                else:
                    volume_regime = "low"
            else:
                volume_regime = "unknown"

            # Market phase assessment
            recent_trend = market_data['close'].iloc[-1] / market_data['close'].iloc[-20]
            if recent_trend > 1.02:
                market_phase = "bull"
            elif recent_trend < 0.98:
                market_phase = "bear"
            else:
                market_phase = "sideways"

            # Confidence score based on data quality
            confidence_score = min(1.0, len(market_data) / 100)

            return MarketCondition(
                volatility_regime=volatility_regime,
                trend_strength=trend_strength,
                volume_regime=volume_regime,
                market_phase=market_phase,
                confidence_score=confidence_score
            )

        except Exception as e:
            self.logger.warning(f"Error assessing market condition: {e}")
            return MarketCondition("unknown", "unknown", "unknown", "unknown", 0.0)

    def _evaluate_performance_score(self, performance: Dict[str, float]) -> float:
        """
        Evaluate overall performance score

        Args:
            performance: Performance metrics

        Returns:
            Performance score (0-1)
        """
        try:
            score = 0.0
            weights = 0.0

            # Sharpe ratio (30% weight)
            sharpe = performance.get('sharpe_ratio', 0)
            if sharpe > 1.0:
                score += 0.3
            elif sharpe > 0:
                score += 0.15
            weights += 0.3

            # Win rate (25% weight)
            win_rate = performance.get('win_rate', 0)
            score += win_rate * 0.25
            weights += 0.25

            # Profit factor (25% weight)
            profit_factor = performance.get('profit_factor', 1)
            if profit_factor > 1.5:
                score += 0.25
            elif profit_factor > 1.0:
                score += 0.125
            weights += 0.25

            # Drawdown penalty (20% weight)
            max_dd = abs(performance.get('max_drawdown', 0))
            drawdown_penalty = min(1.0, max_dd * 5)  # 20% DD = full penalty
            score += (1.0 - drawdown_penalty) * 0.2
            weights += 0.2

            return score / weights if weights > 0 else 0.0

        except Exception:
            return 0.0

    def _conservative_adaptation_check(self, performance_score: float,
                                     market_condition: MarketCondition) -> bool:
        """Conservative adaptation strategy - only adapt when clearly needed"""
        try:
            # Only adapt if performance is poor AND market conditions suggest change
            poor_performance = performance_score < 0.4
            significant_market_change = market_condition.confidence_score > 0.8

            return poor_performance and significant_market_change

        except Exception:
            return False

    def _aggressive_adaptation_check(self, performance_score: float,
                                   market_condition: MarketCondition) -> bool:
        """Aggressive adaptation strategy - adapt frequently"""
        try:
            # Adapt if performance could be better or market conditions changed
            suboptimal_performance = performance_score < 0.7
            market_change = market_condition.confidence_score > 0.6

            return suboptimal_performance or market_change

        except Exception:
            return False

    def _dynamic_adaptation_check(self, performance_score: float,
                                market_condition: MarketCondition) -> bool:
        """Dynamic adaptation based on multiple factors"""
        try:
            # Adapt based on performance degradation or market regime change
            performance_decline = performance_score < 0.5
            regime_change = market_condition.volatility_regime in ["high", "low"]
            trend_change = market_condition.trend_strength == "strong"

            return performance_decline or regime_change or trend_change

        except Exception:
            return False

    def _market_regime_adaptation_check(self, market_condition: MarketCondition) -> bool:
        """Adapt based on market regime changes"""
        try:
            # Always adapt on significant regime changes
            return market_condition.confidence_score > 0.8

        except Exception:
            return False

    def _performance_based_adaptation_check(self, performance_score: float) -> bool:
        """Adapt based on performance thresholds"""
        try:
            return performance_score < 0.6  # Adapt if performance below 60%

        except Exception:
            return False

    def _generate_adaptation_candidates(self, current_params: AdaptationParameters,
                                      market_condition: MarketCondition) -> List[AdaptationParameters]:
        """
        Generate candidate parameter sets for adaptation

        Args:
            current_params: Current parameters
            market_condition: Current market condition

        Returns:
            List of candidate parameter sets
        """
        try:
            candidates = []

            # Generate candidates based on market condition
            if market_condition.volatility_regime == "high":
                # More conservative parameters for high volatility
                candidates.extend(self._generate_conservative_candidates(current_params))
            elif market_condition.volatility_regime == "low":
                # More aggressive parameters for low volatility
                candidates.extend(self._generate_aggressive_candidates(current_params))
            else:
                # Balanced parameters for normal volatility
                candidates.extend(self._generate_balanced_candidates(current_params))

            # Add some random variations
            candidates.extend(self._generate_random_candidates(current_params))

            # Ensure we have at least the current parameters as baseline
            candidates.append(current_params)

            return candidates[:20]  # Limit to 20 candidates

        except Exception as e:
            self.logger.warning(f"Error generating adaptation candidates: {e}")
            return [current_params]

    def _generate_conservative_candidates(self, base_params: AdaptationParameters) -> List[AdaptationParameters]:
        """Generate conservative parameter candidates"""
        candidates = []

        # Wider stops, longer periods, smaller position sizes
        for sl_mult in [1.2, 1.5]:
            for period_mult in [1.2, 1.5]:
                candidate = AdaptationParameters(
                    rsi_period=min(int(base_params.rsi_period * period_mult), self.parameter_bounds['rsi_period']['max']),
                    macd_fast=base_params.macd_fast,
                    macd_slow=min(int(base_params.macd_slow * period_mult), self.parameter_bounds['macd_slow']['max']),
                    macd_signal=base_params.macd_signal,
                    bb_period=min(int(base_params.bb_period * period_mult), self.parameter_bounds['bb_period']['max']),
                    bb_std=max(base_params.bb_std * 0.8, self.parameter_bounds['bb_std']['min']),
                    stop_loss_pips=min(int(base_params.stop_loss_pips * sl_mult), self.parameter_bounds['stop_loss_pips']['max']),
                    take_profit_pips=base_params.take_profit_pips,
                    position_size_multiplier=max(base_params.position_size_multiplier * 0.8, self.parameter_bounds['position_size_multiplier']['min']),
                    risk_multiplier=max(base_params.risk_multiplier * 0.9, self.parameter_bounds['risk_multiplier']['min'])
                )
                candidates.append(candidate)

        return candidates

    def _generate_aggressive_candidates(self, base_params: AdaptationParameters) -> List[AdaptationParameters]:
        """Generate aggressive parameter candidates"""
        candidates = []

        # Tighter stops, shorter periods, larger position sizes
        for sl_mult in [0.8, 0.6]:
            for period_mult in [0.8, 0.6]:
                candidate = AdaptationParameters(
                    rsi_period=max(int(base_params.rsi_period * period_mult), self.parameter_bounds['rsi_period']['min']),
                    macd_fast=base_params.macd_fast,
                    macd_slow=max(int(base_params.macd_slow * period_mult), self.parameter_bounds['macd_slow']['min']),
                    macd_signal=base_params.macd_signal,
                    bb_period=max(int(base_params.bb_period * period_mult), self.parameter_bounds['bb_period']['min']),
                    bb_std=min(base_params.bb_std * 1.2, self.parameter_bounds['bb_std']['max']),
                    stop_loss_pips=max(int(base_params.stop_loss_pips * sl_mult), self.parameter_bounds['stop_loss_pips']['min']),
                    take_profit_pips=base_params.take_profit_pips,
                    position_size_multiplier=min(base_params.position_size_multiplier * 1.2, self.parameter_bounds['position_size_multiplier']['max']),
                    risk_multiplier=min(base_params.risk_multiplier * 1.1, self.parameter_bounds['risk_multiplier']['max'])
                )
                candidates.append(candidate)

        return candidates

    def _generate_balanced_candidates(self, base_params: AdaptationParameters) -> List[AdaptationParameters]:
        """Generate balanced parameter candidates"""
        candidates = []

        # Small variations around current parameters
        variations = [-0.1, 0.1, -0.2, 0.2]

        for var in variations:
            candidate = AdaptationParameters(
                rsi_period=max(min(int(base_params.rsi_period * (1 + var)), self.parameter_bounds['rsi_period']['max']), self.parameter_bounds['rsi_period']['min']),
                macd_fast=base_params.macd_fast,
                macd_slow=max(min(int(base_params.macd_slow * (1 + var)), self.parameter_bounds['macd_slow']['max']), self.parameter_bounds['macd_slow']['min']),
                macd_signal=base_params.macd_signal,
                bb_period=max(min(int(base_params.bb_period * (1 + var)), self.parameter_bounds['bb_period']['max']), self.parameter_bounds['bb_period']['min']),
                bb_std=max(min(base_params.bb_std * (1 + var), self.parameter_bounds['bb_std']['max']), self.parameter_bounds['bb_std']['min']),
                stop_loss_pips=max(min(int(base_params.stop_loss_pips * (1 + var)), self.parameter_bounds['stop_loss_pips']['max']), self.parameter_bounds['stop_loss_pips']['min']),
                take_profit_pips=base_params.take_profit_pips,
                position_size_multiplier=max(min(base_params.position_size_multiplier * (1 + var), self.parameter_bounds['position_size_multiplier']['max']), self.parameter_bounds['position_size_multiplier']['min']),
                risk_multiplier=max(min(base_params.risk_multiplier * (1 + var), self.parameter_bounds['risk_multiplier']['max']), self.parameter_bounds['risk_multiplier']['min'])
            )
            candidates.append(candidate)

        return candidates

    def _generate_random_candidates(self, base_params: AdaptationParameters) -> List[AdaptationParameters]:
        """Generate random parameter candidates"""
        candidates = []

        for _ in range(5):
            candidate = AdaptationParameters(
                rsi_period=np.random.randint(self.parameter_bounds['rsi_period']['min'],
                                           self.parameter_bounds['rsi_period']['max'] + 1),
                macd_fast=np.random.randint(self.parameter_bounds['macd_fast']['min'],
                                          self.parameter_bounds['macd_fast']['max'] + 1),
                macd_slow=np.random.randint(self.parameter_bounds['macd_slow']['min'],
                                          self.parameter_bounds['macd_slow']['max'] + 1),
                macd_signal=np.random.randint(self.parameter_bounds['macd_signal']['min'],
                                            self.parameter_bounds['macd_signal']['max'] + 1),
                bb_period=np.random.randint(self.parameter_bounds['bb_period']['min'],
                                          self.parameter_bounds['bb_period']['max'] + 1),
                bb_std=np.random.uniform(self.parameter_bounds['bb_std']['min'],
                                       self.parameter_bounds['bb_std']['max']),
                stop_loss_pips=np.random.randint(self.parameter_bounds['stop_loss_pips']['min'],
                                               self.parameter_bounds['stop_loss_pips']['max'] + 1),
                take_profit_pips=np.random.randint(self.parameter_bounds['take_profit_pips']['min'],
                                                 self.parameter_bounds['take_profit_pips']['max'] + 1),
                position_size_multiplier=np.random.uniform(self.parameter_bounds['position_size_multiplier']['min'],
                                                        self.parameter_bounds['position_size_multiplier']['max']),
                risk_multiplier=np.random.uniform(self.parameter_bounds['risk_multiplier']['min'],
                                                self.parameter_bounds['risk_multiplier']['max'])
            )
            candidates.append(candidate)

        return candidates

    def _evaluate_candidates(self, candidates: List[AdaptationParameters], market_data: pd.DataFrame,
                           current_performance: Dict[str, float]) -> Tuple[Optional[AdaptationParameters], float, float]:
        """
        Evaluate adaptation candidates

        Args:
            candidates: Candidate parameter sets
            market_data: Market data for evaluation
            current_performance: Current performance metrics

        Returns:
            Best candidate, expected improvement, confidence level
        """
        try:
            best_candidate = None
            best_improvement = 0.0
            best_confidence = 0.0

            current_score = self._evaluate_performance_score(current_performance)

            for candidate in candidates:
                # Simulate performance with candidate parameters
                simulated_performance = self._simulate_strategy_performance(candidate, market_data)

                if simulated_performance:
                    candidate_score = self._evaluate_performance_score(simulated_performance)
                    improvement = candidate_score - current_score

                    # Calculate confidence based on simulation stability
                    confidence = self._calculate_simulation_confidence(simulated_performance)

                    if improvement > best_improvement and confidence > self.confidence_threshold:
                        best_candidate = candidate
                        best_improvement = improvement
                        best_confidence = confidence

            return best_candidate, best_improvement, best_confidence

        except Exception as e:
            self.logger.error(f"Error evaluating candidates: {e}")
            return None, 0.0, 0.0

    def _simulate_strategy_performance(self, params: AdaptationParameters,
                                     market_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Simulate strategy performance with given parameters

        Args:
            params: Strategy parameters
            market_data: Market data

        Returns:
            Simulated performance metrics
        """
        try:
            # Simple simulation based on parameter characteristics
            # In a real implementation, this would run the actual strategy

            returns = market_data['close'].pct_change().dropna()

            # Adjust returns based on parameter risk profile
            risk_adjustment = params.risk_multiplier * (1 - params.stop_loss_pips / 50)  # Rough risk adjustment
            adjusted_returns = returns * risk_adjustment

            # Calculate simulated metrics
            total_return = np.prod(1 + adjusted_returns) - 1
            volatility = adjusted_returns.std() * np.sqrt(252)
            sharpe_ratio = np.mean(adjusted_returns) / volatility * np.sqrt(252) if volatility > 0 else 0
            max_drawdown = (np.cumprod(1 + adjusted_returns) - np.maximum.accumulate(np.cumprod(1 + adjusted_returns))) / np.maximum.accumulate(np.cumprod(1 + adjusted_returns))
            max_drawdown = np.min(max_drawdown)
            win_rate = np.mean(adjusted_returns > 0)

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'profit_factor': 1.5 if win_rate > 0.5 else 1.0  # Simplified
            }

        except Exception as e:
            self.logger.warning(f"Error simulating strategy performance: {e}")
            return None

    def _calculate_simulation_confidence(self, performance: Dict[str, float]) -> float:
        """Calculate confidence in simulation results"""
        try:
            # Simple confidence calculation based on metric consistency
            sharpe = performance.get('sharpe_ratio', 0)
            win_rate = performance.get('win_rate', 0)
            profit_factor = performance.get('profit_factor', 1)

            confidence = 0.0

            if sharpe > 0.5:
                confidence += 0.4
            elif sharpe > 0:
                confidence += 0.2

            if win_rate > 0.55:
                confidence += 0.3
            elif win_rate > 0.5:
                confidence += 0.15

            if profit_factor > 1.2:
                confidence += 0.3
            elif profit_factor > 1.0:
                confidence += 0.15

            return min(confidence, 1.0)

        except Exception:
            return 0.0

    def _determine_adaptation_reason(self, market_condition: MarketCondition,
                                   performance: Dict[str, float]) -> str:
        """Determine the reason for adaptation"""
        try:
            if market_condition.volatility_regime == "high":
                return "high_volatility_adaptation"
            elif market_condition.volatility_regime == "low":
                return "low_volatility_opportunity"
            elif performance.get('sharpe_ratio', 0) < 0.3:
                return "poor_performance_adaptation"
            elif market_condition.trend_strength == "strong":
                return "strong_trend_adaptation"
            else:
                return "general_optimization"

        except Exception:
            return "unknown_reason"

    def _update_learning_model(self, decision: AdaptationDecision, market_data: pd.DataFrame):
        """Update the learning model with adaptation results"""
        try:
            # This would update the ML model with the adaptation outcome
            # For now, just log the adaptation
            self.logger.debug(f"Learning model updated with adaptation: {decision.reason}")

        except Exception as e:
            self.logger.warning(f"Error updating learning model: {e}")

    def get_adaptation_history(self, symbol: Optional[str] = None) -> List[AdaptationDecision]:
        """
        Get adaptation history

        Args:
            symbol: Optional symbol filter

        Returns:
            List of adaptation decisions
        """
        try:
            if symbol is None:
                return self.adaptation_history.copy()
            else:
                # In a real implementation, filter by symbol
                return self.adaptation_history.copy()

        except Exception as e:
            self.logger.error(f"Error getting adaptation history: {e}")
            return []

    def create_adaptation_report(self) -> Dict[str, Any]:
        """
        Create comprehensive adaptation report

        Returns:
            Adaptation report dictionary
        """
        try:
            total_adaptations = len(self.adaptation_history)
            recent_adaptations = [d for d in self.adaptation_history
                                if (datetime.now() - d.timestamp).days <= 30]

            success_rate = np.mean([d.expected_improvement > 0 for d in self.adaptation_history]) if self.adaptation_history else 0

            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_adaptations': total_adaptations,
                    'recent_adaptations': len(recent_adaptations),
                    'success_rate': success_rate,
                    'adaptation_strategy': self.adaptation_strategy.value
                },
                'performance': {
                    'avg_expected_improvement': np.mean([d.expected_improvement for d in self.adaptation_history]) if self.adaptation_history else 0,
                    'avg_confidence': np.mean([d.confidence_level for d in self.adaptation_history]) if self.adaptation_history else 0
                },
                'reasons': self._analyze_adaptation_reasons(),
                'recommendations': self._generate_adaptation_recommendations()
            }

            return report

        except Exception as e:
            self.logger.error(f"Error creating adaptation report: {e}")
            return {}

    def _analyze_adaptation_reasons(self) -> Dict[str, int]:
        """Analyze frequency of adaptation reasons"""
        try:
            reasons = {}
            for decision in self.adaptation_history:
                reason = decision.reason
                reasons[reason] = reasons.get(reason, 0) + 1

            return reasons

        except Exception:
            return {}

    def _generate_adaptation_recommendations(self) -> List[str]:
        """Generate adaptation recommendations"""
        try:
            recommendations = []

            if len(self.adaptation_history) < 5:
                recommendations.append("increase_adaptation_frequency")
            elif len(self.adaptation_history) > 20:
                recommendations.append("reduce_adaptation_frequency")

            success_rate = np.mean([d.expected_improvement > 0 for d in self.adaptation_history]) if self.adaptation_history else 0

            if success_rate < 0.6:
                recommendations.append("review_adaptation_criteria")
            elif success_rate > 0.8:
                recommendations.append("current_adaptation_effective")

            return recommendations

        except Exception:
            return ["review_adaptation_system"]