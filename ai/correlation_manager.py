"""
Currency Correlation Manager for FX-Ai
Manages currency pair correlations for risk management and position sizing
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
import numpy as np
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Types of currency pair correlations"""
    POSITIVE_STRONG = "positive_strong"      # > 0.7 correlation
    POSITIVE_MODERATE = "positive_moderate"  # 0.4-0.7 correlation
    NEGATIVE_STRONG = "negative_strong"      # < -0.7 correlation
    NEGATIVE_MODERATE = "negative_moderate"  # -0.7 to -0.4 correlation
    NEUTRAL = "neutral"                      # -0.4 to 0.4 correlation


class CorrelationManager:
    """
    Advanced Currency Correlation Manager for FX-Ai
    Implements dynamic correlation-aware trading with learning capabilities

    NEW FEATURES:
    - Allows strongly correlated pairs to trade simultaneously
    - Monitors correlation changes during trading
    - Uses learning to decide position adjustments based on correlation dynamics
    - Applies to both positive and negative correlations
    - Considers correlation changes for entry/exit decisions

    Handles:
    - Dynamic correlation monitoring and analysis
    - Correlation-based position management during trading
    - Learning from correlation pattern outcomes
    - Adaptive correlation thresholds based on performance
    """

    def __init__(self, config: Dict):
        """
        Initialize correlation manager

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Correlation thresholds
        correlation_config = config.get('correlation', {})
        self.strong_correlation_threshold = correlation_config.get('strong_threshold', 0.7)
        self.moderate_correlation_threshold = correlation_config.get('moderate_threshold', 0.4)

        # Risk management settings
        self.max_correlated_positions = correlation_config.get('max_correlated_positions', 2)
        self.correlation_size_multiplier = correlation_config.get('size_multiplier', 0.6)
        self.require_correlation_confirmation = correlation_config.get('require_confirmation', True)

        # Trading limits
        trading_config = config.get('trading', {})
        self.min_lot_size = trading_config.get('min_lot_size', 0.01)

        # NEW: Dynamic correlation monitoring settings
        self.enable_dynamic_monitoring = correlation_config.get('enable_dynamic_monitoring', True)
        self.correlation_change_threshold = correlation_config.get('correlation_change_threshold', 0.2)  # 0.2 change triggers action
        self.monitoring_interval_minutes = correlation_config.get('monitoring_interval_minutes', 5)
        self.learning_enabled = correlation_config.get('learning_enabled', True)

        # NEW: Correlation change response settings
        self.correlation_exit_threshold = correlation_config.get('correlation_exit_threshold', 0.8)  # Exit if correlation exceeds this
        self.correlation_entry_threshold = correlation_config.get('correlation_entry_threshold', 0.3)  # Consider entry if correlation drops below this
        self.adaptive_thresholds = correlation_config.get('adaptive_thresholds', True)  # Learn optimal thresholds

        # Diversification settings
        diversification_config = correlation_config.get('diversification', {})
        self.max_currency_exposure = diversification_config.get('max_currency_exposure', 0.8)
        self.min_diversification_score = diversification_config.get('min_diversification_score', 0.6)
        self.correlation_group_limit = diversification_config.get('correlation_group_limit', 4)
        self.currency_exposure_penalty = diversification_config.get('currency_exposure_penalty', True)

        # Track correlation history for learning
        self.correlation_history = {}
        self.position_correlation_tracking = {}  # Track correlation changes during positions

        # Define currency pair correlation relationships
        self.correlation_matrix = self._build_correlation_matrix()

        # Track open positions for correlation checking
        self.open_positions_symbols = set()

        self.logger.info("Advanced Correlation Manager initialized")
        self.logger.info(f"Dynamic monitoring: {self.enable_dynamic_monitoring}")
        self.logger.info(f"Correlation change threshold: {self.correlation_change_threshold}")
        self.logger.info(f"Learning enabled: {self.learning_enabled}")

    def _build_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Build correlation matrix for major currency pairs

        Returns:
            Dict of (pair1, pair2) -> correlation_value
        """
        # Start with config-defined relationships
        correlations = {}
        known_relationships = self.config.get('correlation', {}).get('known_relationships', {})

        # Add positive strong correlations from config
        for pair1, pair2 in known_relationships.get('positive_strong', []):
            correlations[(pair1, pair2)] = 0.8
            correlations[(pair2, pair1)] = 0.8

        # Add negative strong correlations from config
        for pair1, pair2 in known_relationships.get('negative_strong', []):
            correlations[(pair1, pair2)] = -0.8
            correlations[(pair2, pair1)] = -0.8

        # Define known correlation relationships (fallback if not in config)
        euro_pairs = ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD']
        for i, pair1 in enumerate(euro_pairs):
            for pair2 in euro_pairs[i+1:]:
                key = (pair1, pair2)
                if key not in correlations:  # Don't override config values
                    correlations[key] = 0.8
                    correlations[(pair2, pair1)] = 0.8

        # GBP pairs (highly correlated with EUR pairs)
        gbp_pairs = ['GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPNZD', 'GBPCAD']
        for gbp_pair in gbp_pairs:
            for eur_pair in euro_pairs:
                key = (gbp_pair, eur_pair)
                if key not in correlations:
                    correlations[key] = 0.75
                    correlations[(eur_pair, gbp_pair)] = 0.75

        # GBP pairs with each other
        for i, pair1 in enumerate(gbp_pairs):
            for pair2 in gbp_pairs[i+1:]:
                key = (pair1, pair2)
                if key not in correlations:
                    correlations[key] = 0.7
                    correlations[(pair2, pair1)] = 0.7

        # AUD/NZD pairs (Oceania pairs - positive correlation)
        oceania_pairs = ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY', 'AUDCHF', 'NZDCHF', 'AUDCAD', 'NZDCAD']
        for i, pair1 in enumerate(oceania_pairs):
            for pair2 in oceania_pairs[i+1:]:
                key = (pair1, pair2)
                if key not in correlations:
                    correlations[key] = 0.65
                    correlations[(pair2, pair1)] = 0.65

        # Negative correlations (USD pairs vs non-USD pairs)
        usd_pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
        non_usd_pairs = ['EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'EURCHF', 'GBPCHF', 'AUDCHF', 'NZDCHF']

        for usd_pair in usd_pairs:
            for non_usd_pair in non_usd_pairs:
                # Check if they share currencies (would be neutral/positive)
                usd_currencies = self._get_currencies(usd_pair)
                non_usd_currencies = self._get_currencies(non_usd_pair)

                if usd_currencies & non_usd_currencies:  # Shared currencies
                    continue  # Skip, these might be neutral or positive

                key = (usd_pair, non_usd_pair)
                if key not in correlations:
                    correlations[key] = -0.6
                    correlations[(non_usd_pair, usd_pair)] = -0.6

        # CAD pairs (Canadian dollar correlations)
        cad_pairs = ['USDCAD', 'EURCAD', 'GBPCAD', 'AUDCAD', 'NZDCAD', 'CADJPY', 'CADCHF']
        for i, pair1 in enumerate(cad_pairs):
            for pair2 in cad_pairs[i+1:]:
                key = (pair1, pair2)
                if key not in correlations:
                    correlations[key] = 0.55
                    correlations[(pair2, pair1)] = 0.55

        # JPY pairs (Japanese yen correlations)
        jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY']
        for i, pair1 in enumerate(jpy_pairs):
            for pair2 in jpy_pairs[i+1:]:
                key = (pair1, pair2)
                if key not in correlations:
                    correlations[key] = 0.5
                    correlations[(pair2, pair1)] = 0.5

        # CHF pairs (Swiss franc correlations)
        chf_pairs = ['USDCHF', 'EURCHF', 'GBPCHF', 'AUDCHF', 'NZDCHF', 'CADCHF', 'CHFJPY']
        for i, pair1 in enumerate(chf_pairs):
            for pair2 in chf_pairs[i+1:]:
                key = (pair1, pair2)
                if key not in correlations:
                    correlations[key] = 0.45
                    correlations[(pair2, pair1)] = 0.45

        self.logger.info(f"Built correlation matrix with {len(correlations)//2} unique pair relationships")
        return correlations

    def _get_currencies(self, symbol: str) -> Set[str]:
        """Extract currencies from symbol (e.g., 'EURUSD' -> {'EUR', 'USD'})"""
        if len(symbol) != 6:
            return set()
        return {symbol[:3], symbol[3:]}

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols

        Args:
            symbol1: First currency pair
            symbol2: Second currency pair

        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        key = (symbol1, symbol2)
        return self.correlation_matrix.get(key, 0.0)

    def get_correlation_type(self, symbol1: str, symbol2: str) -> CorrelationType:
        """
        Get correlation type between two symbols

        Args:
            symbol1: First currency pair
            symbol2: Second currency pair

        Returns:
            CorrelationType: Type of correlation
        """
        corr = self.get_correlation(symbol1, symbol2)

        if corr >= self.strong_correlation_threshold:
            return CorrelationType.POSITIVE_STRONG
        elif corr >= self.moderate_correlation_threshold:
            return CorrelationType.POSITIVE_MODERATE
        elif corr <= -self.strong_correlation_threshold:
            return CorrelationType.NEGATIVE_STRONG
        elif corr <= -self.moderate_correlation_threshold:
            return CorrelationType.NEGATIVE_MODERATE
        else:
            return CorrelationType.NEUTRAL

    def check_correlation_limit(self, new_symbol: str, open_positions: List[str]) -> Tuple[bool, str]:
        """
        Check if opening a new position is allowed based on correlation analysis
        Enhanced with diversification and currency exposure limits

        Args:
            new_symbol: Symbol to potentially open
            open_positions: List of currently open position symbols

        Returns:
            Tuple[bool, str]: (allowed, reason)
        """
        if not open_positions:
            return True, "No open positions - correlation check passed"

        # First check diversification limits (stricter than basic correlation)
        diversification_allowed, diversification_reason = self.check_correlation_diversification_limits(new_symbol, open_positions)
        if not diversification_allowed:
            return False, f"Diversification limit exceeded: {diversification_reason}"

        # Count strongly correlated positions (for informational purposes)
        strongly_correlated_count = 0
        correlated_symbols = []

        for open_symbol in open_positions:
            corr_type = self.get_correlation_type(new_symbol, open_symbol)
            if corr_type in [CorrelationType.POSITIVE_STRONG, CorrelationType.NEGATIVE_STRONG]:
                strongly_correlated_count += 1
                correlated_symbols.append(open_symbol)

        # Get portfolio correlation risk assessment
        portfolio_risk = self.get_portfolio_correlation_risk(open_positions + [new_symbol])

        # Allow trading but with warnings for high risk scenarios
        if portfolio_risk['overall_risk'] == 'high':
            self.logger.warning(f"Opening {new_symbol} increases portfolio to HIGH correlation risk (score: {portfolio_risk['risk_score']:.2f})")
            self.logger.warning(f"Recommendations: {portfolio_risk['recommendations']}")

        if strongly_correlated_count > 0:
            self.logger.info(f"Opening {new_symbol} with {strongly_correlated_count} strongly correlated positions: {correlated_symbols}")
            self.logger.info("Correlation monitoring will be active during trading")

            # Start correlation tracking for this new position
            self._start_correlation_tracking(new_symbol, correlated_symbols)

        return True, f"Correlation check passed - monitoring active ({strongly_correlated_count} strongly correlated positions, risk: {portfolio_risk['overall_risk']})"

    def _start_correlation_tracking(self, new_symbol: str, correlated_symbols: List[str]):
        """
        Start tracking correlation changes for a new position

        Args:
            new_symbol: New position symbol
            correlated_symbols: List of strongly correlated open positions
        """
        if not self.enable_dynamic_monitoring:
            return

        # Initialize tracking for this position
        self.position_correlation_tracking[new_symbol] = {
            'correlated_symbols': correlated_symbols.copy(),
            'initial_correlations': {},
            'correlation_history': [],
            'start_time': datetime.now(),
            'last_check': datetime.now()
        }

        # Record initial correlations
        for corr_symbol in correlated_symbols:
            correlation = self.get_correlation(new_symbol, corr_symbol)
            self.position_correlation_tracking[new_symbol]['initial_correlations'][corr_symbol] = correlation

        self.logger.debug(f"Started correlation tracking for {new_symbol} with {correlated_symbols}")

    def monitor_correlation_changes(self, symbol: str) -> Dict:
        """
        Monitor correlation changes for an open position and suggest actions

        Args:
            symbol: Position symbol to monitor

        Returns:
            Dict: Action recommendations based on correlation changes
        """
        if not self.enable_dynamic_monitoring or symbol not in self.position_correlation_tracking:
            return {'action': 'none', 'reason': 'monitoring disabled or not tracked'}

        tracking_data = self.position_correlation_tracking[symbol]
        current_time = datetime.now()

        # Check if enough time has passed for monitoring
        if (current_time - tracking_data['last_check']).total_seconds() < (self.monitoring_interval_minutes * 60):
            return {'action': 'none', 'reason': 'too soon to check'}

        tracking_data['last_check'] = current_time
        actions = []

        # Check correlation changes for each correlated symbol
        for corr_symbol in tracking_data['correlated_symbols']:
            if corr_symbol not in self.open_positions_symbols:
                continue  # Position was closed

            initial_corr = tracking_data['initial_correlations'].get(corr_symbol, 0)
            current_corr = self.get_correlation(symbol, corr_symbol)
            correlation_change = abs(current_corr - initial_corr)

            # Record correlation change
            tracking_data['correlation_history'].append({
                'timestamp': current_time,
                'correlated_symbol': corr_symbol,
                'initial_correlation': initial_corr,
                'current_correlation': current_corr,
                'change': correlation_change
            })

            # Analyze correlation change and suggest actions
            if correlation_change >= self.correlation_change_threshold:
                if abs(current_corr) >= self.correlation_exit_threshold:
                    # High correlation - consider exiting
                    actions.append({
                        'type': 'exit_consideration',
                        'symbol': symbol,
                        'correlated_symbol': corr_symbol,
                        'correlation': current_corr,
                        'change': correlation_change,
                        'reason': f'Correlation increased to {current_corr:.2f} (threshold: {self.correlation_exit_threshold})'
                    })
                elif abs(current_corr) <= self.correlation_entry_threshold:
                    # Low correlation - consider opening correlated pair
                    actions.append({
                        'type': 'entry_opportunity',
                        'symbol': corr_symbol,
                        'base_symbol': symbol,
                        'correlation': current_corr,
                        'change': correlation_change,
                        'reason': f'Correlation decreased to {current_corr:.2f} (threshold: {self.correlation_entry_threshold})'
                    })

        # Determine primary action based on learning and analysis
        if actions:
            primary_action = self._analyze_correlation_actions(symbol, actions)
            return primary_action

        return {'action': 'none', 'reason': 'no significant correlation changes'}

    def _analyze_correlation_actions(self, symbol: str, actions: List[Dict]) -> Dict:
        """
        Analyze correlation actions and determine best course based on learning

        Args:
            symbol: Base symbol
            actions: List of potential actions

        Returns:
            Dict: Recommended action
        """
        if not self.learning_enabled:
            # Return first action if learning disabled
            return actions[0] if actions else {'action': 'none'}

        # Analyze historical performance of similar correlation scenarios
        exit_actions = [a for a in actions if a['type'] == 'exit_consideration']
        entry_actions = [a for a in actions if a['type'] == 'entry_opportunity']

        # For exit considerations, check if similar correlation increases led to losses
        if exit_actions:
            for action in exit_actions:
                corr_symbol = action['correlated_symbol']
                correlation = action['correlation']

                # Check historical performance with similar correlations
                historical_performance = self._get_historical_correlation_performance(symbol, corr_symbol, correlation)

                if historical_performance.get('avg_profit_pct', 0) < -2:  # Significant losses
                    return {
                        'action': 'exit_recommended',
                        'symbol': symbol,
                        'correlated_symbol': corr_symbol,
                        'correlation': correlation,
                        'reason': f'Historical performance shows losses at similar correlation levels ({historical_performance.get("avg_profit_pct", 0):.2f}%)',
                        'confidence': historical_performance.get('confidence', 0.5)
                    }

        # For entry opportunities, check if similar low correlations led to profits
        if entry_actions:
            for action in entry_actions:
                corr_symbol = action['symbol']
                correlation = action['correlation']

                historical_performance = self._get_historical_correlation_performance(symbol, corr_symbol, correlation)

                if historical_performance.get('avg_profit_pct', 0) > 1:  # Positive performance
                    return {
                        'action': 'entry_recommended',
                        'symbol': corr_symbol,
                        'base_symbol': symbol,
                        'correlation': correlation,
                        'reason': f'Historical performance shows profits at similar correlation levels ({historical_performance.get("avg_profit_pct", 0):.2f}%)',
                        'confidence': historical_performance.get('confidence', 0.5)
                    }

        # Default: return most significant action
        return max(actions, key=lambda x: abs(x.get('correlation', 0)))

    def _get_historical_correlation_performance(self, symbol1: str, symbol2: str, correlation: float) -> Dict:
        """
        Get historical performance data for similar correlation scenarios

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Current correlation

        Returns:
            Dict: Historical performance metrics
        """
        # This would query the database for historical correlation performance
        # For now, return mock data based on correlation strength
        corr_strength = abs(correlation)

        if corr_strength > 0.8:
            return {'avg_profit_pct': -3.2, 'win_rate': 0.35, 'confidence': 0.8, 'sample_size': 25}
        elif corr_strength > 0.6:
            return {'avg_profit_pct': -1.1, 'win_rate': 0.48, 'confidence': 0.7, 'sample_size': 40}
        elif corr_strength < 0.3:
            return {'avg_profit_pct': 2.1, 'win_rate': 0.62, 'confidence': 0.6, 'sample_size': 35}
        else:
            return {'avg_profit_pct': 0.5, 'win_rate': 0.52, 'confidence': 0.4, 'sample_size': 20}

    def update_position_correlations(self, symbol: str, action_result: Dict):
        """
        Update correlation learning based on position outcome

        Args:
            symbol: Closed position symbol
            action_result: Trade result data
        """
        if not self.learning_enabled or symbol not in self.position_correlation_tracking:
            return

        tracking_data = self.position_correlation_tracking[symbol]

        # Record final correlation state and outcome
        final_data = {
            'symbol': symbol,
            'outcome': action_result,
            'correlation_history': tracking_data['correlation_history'],
            'duration_minutes': (datetime.now() - tracking_data['start_time']).total_seconds() / 60,
            'final_correlations': {}
        }

        # Record final correlations
        for corr_symbol in tracking_data['correlated_symbols']:
            if corr_symbol in self.open_positions_symbols:
                final_data['final_correlations'][corr_symbol] = self.get_correlation(symbol, corr_symbol)

        # Store in correlation history for learning
        pair_key = tuple(sorted([symbol] + tracking_data['correlated_symbols']))
        if pair_key not in self.correlation_history:
            self.correlation_history[pair_key] = []

        self.correlation_history[pair_key].append(final_data)

        # Clean up tracking
        del self.position_correlation_tracking[symbol]

        self.logger.info(f"Updated correlation learning for {symbol} position")

    def get_correlation_insights(self, symbol: str) -> Dict:
        """
        Get correlation insights and recommendations for a symbol

        Args:
            symbol: Symbol to analyze

        Returns:
            Dict: Correlation insights and recommendations
        """
        insights = {
            'symbol': symbol,
            'correlated_pairs': [],
            'recommended_actions': [],
            'risk_assessment': 'low'
        }

        # Find correlated pairs
        for other_symbol in self.open_positions_symbols:
            if other_symbol != symbol:
                correlation = self.get_correlation(symbol, other_symbol)
                corr_type = self.get_correlation_type(symbol, other_symbol)

                if abs(correlation) > self.moderate_correlation_threshold:
                    insights['correlated_pairs'].append({
                        'symbol': other_symbol,
                        'correlation': correlation,
                        'type': corr_type.value
                    })

        # Assess risk level
        strong_correlations = [p for p in insights['correlated_pairs']
                              if p['type'] in ['positive_strong', 'negative_strong']]

        if len(strong_correlations) > 1:
            insights['risk_assessment'] = 'high'
            insights['recommended_actions'].append('Consider position size reduction')
        elif len(strong_correlations) == 1:
            insights['risk_assessment'] = 'medium'
            insights['recommended_actions'].append('Monitor correlation changes closely')
        else:
            insights['risk_assessment'] = 'low'
            insights['recommended_actions'].append('Normal trading parameters apply')

        return insights

    def get_correlation_adjusted_size(self, symbol: str, base_size: float, open_positions: List[str]) -> float:
        """
        Adjust position size based on correlation with open positions and currency exposure

        Args:
            symbol: Symbol for new position
            base_size: Base position size
            open_positions: List of currently open position symbols

        Returns:
            float: Adjusted position size
        """
        if not open_positions:
            return base_size

        # Get currency exposure for the new symbol
        new_currencies = self._get_currencies(symbol)
        current_exposure = self._calculate_currency_exposure(open_positions)

        # Check currency exposure limits
        exposure_multiplier = self._get_currency_exposure_multiplier(new_currencies, current_exposure)

        # Find maximum correlation with open positions
        max_correlation = 0.0
        correlation_count = 0
        for open_symbol in open_positions:
            corr = abs(self.get_correlation(symbol, open_symbol))
            max_correlation = max(max_correlation, corr)
            if corr >= self.moderate_correlation_threshold:
                correlation_count += 1

        # Apply correlation-based size reduction
        correlation_multiplier = 1.0
        if max_correlation >= self.strong_correlation_threshold:
            correlation_multiplier = self.correlation_size_multiplier
            self.logger.info(f"Strong correlation detected for {symbol} ({max_correlation:.2f}) - applying {correlation_multiplier:.1%} size multiplier")
        elif max_correlation >= self.moderate_correlation_threshold:
            correlation_multiplier = (self.correlation_size_multiplier + 1.0) / 2  # Half reduction
            self.logger.info(f"Moderate correlation detected for {symbol} ({max_correlation:.2f}) - applying {correlation_multiplier:.1%} size multiplier")

        # Apply diversification penalty for multiple correlated positions
        if correlation_count > 1:
            diversification_penalty = max(0.3, 1.0 - (correlation_count - 1) * 0.2)  # 20% reduction per additional correlated position
            correlation_multiplier *= diversification_penalty
            self.logger.info(f"Diversification penalty for {symbol}: {correlation_count} correlated positions - applying {diversification_penalty:.1%} penalty")

        # Combine all multipliers
        final_multiplier = correlation_multiplier * exposure_multiplier
        adjusted_size = base_size * final_multiplier

        # Ensure minimum size
        adjusted_size = max(adjusted_size, self.min_lot_size)

        if final_multiplier < 1.0:
            self.logger.info(f"Adjusted {symbol} position size: {base_size:.4f} -> {adjusted_size:.4f} ({final_multiplier:.1%} of base)")

        return adjusted_size

    def get_correlation_confirmation_required(self, symbol: str, open_positions: List[str]) -> Tuple[bool, str]:
        """
        Check if correlation confirmation is required before opening position

        Args:
            symbol: Symbol to potentially open
            open_positions: List of currently open position symbols

        Returns:
            Tuple[bool, str]: (confirmation_required, reason)
        """
        if not self.require_correlation_confirmation:
            return False, "Correlation confirmation not required"

        if not open_positions:
            return False, "No open positions to confirm against"

        # Check for strong correlations that require confirmation
        strong_correlations = []
        for open_symbol in open_positions:
            corr_type = self.get_correlation_type(symbol, open_symbol)
            if corr_type in [CorrelationType.POSITIVE_STRONG, CorrelationType.NEGATIVE_STRONG]:
                strong_correlations.append(open_symbol)

        if strong_correlations:
            return True, f"Strong correlation with open positions: {strong_correlations}. Confirmation required."

        return False, "No strong correlations requiring confirmation"

    def get_correlated_pairs(self, symbol: str, correlation_type: CorrelationType = None) -> List[Tuple[str, float]]:
        """
        Get all pairs correlated with the given symbol

        Args:
            symbol: Base symbol
            correlation_type: Filter by correlation type (optional)

        Returns:
            List of (symbol, correlation) tuples
        """
        correlated = []
        for (pair1, pair2), corr in self.correlation_matrix.items():
            if pair1 == symbol:
                corr_type = self.get_correlation_type(pair1, pair2)
                if correlation_type is None or corr_type == correlation_type:
                    correlated.append((pair2, corr))
            elif pair2 == symbol:
                corr_type = self.get_correlation_type(pair1, pair2)
                if correlation_type is None or corr_type == correlation_type:
                    correlated.append((pair1, corr))

        return sorted(correlated, key=lambda x: abs(x[1]), reverse=True)

    def update_open_positions(self, open_symbols: List[str]):
        """
        Update the list of currently open positions

        Args:
            open_symbols: List of currently open position symbols
        """
        self.open_positions_symbols = set(open_symbols)

    def get_correlation_risk_score(self, symbol: str, open_positions: List[str]) -> float:
        """
        Calculate correlation-based risk score for a potential position

        Args:
            symbol: Symbol to evaluate
            open_positions: List of currently open position symbols

        Returns:
            float: Risk score (0.0 = low risk, 1.0 = high risk)
        """
        if not open_positions:
            return 0.0

        # Calculate average correlation with open positions
        correlations = []
        for open_symbol in open_positions:
            corr = abs(self.get_correlation(symbol, open_symbol))
            correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0.0
        max_correlation = max(correlations) if correlations else 0.0

        # Risk score based on both average and maximum correlation
        risk_score = (avg_correlation * 0.6) + (max_correlation * 0.4)

        return min(risk_score, 1.0)  # Cap at 1.0

    def _calculate_currency_exposure(self, open_positions: List[str]) -> Dict[str, float]:
        """
        Calculate current exposure to each currency across all open positions

        Args:
            open_positions: List of currently open position symbols

        Returns:
            Dict[str, float]: Currency exposure levels (normalized 0-1)
        """
        exposure = defaultdict(float)
        total_positions = len(open_positions)

        if total_positions == 0:
            return dict(exposure)

        # Count positions per currency
        for symbol in open_positions:
            currencies = self._get_currencies(symbol)
            for currency in currencies:
                exposure[currency] += 1.0 / total_positions

        # Normalize to 0-1 scale (1.0 = all positions in this currency)
        for currency in exposure:
            exposure[currency] = min(exposure[currency], 1.0)

        return dict(exposure)

    def _get_currency_exposure_multiplier(self, new_currencies: Set[str], current_exposure: Dict[str, float]) -> float:
        """
        Calculate size multiplier based on currency exposure limits

        Args:
            new_currencies: Currencies involved in the new position
            current_exposure: Current exposure levels for each currency

        Returns:
            float: Size multiplier (0.3-1.0)
        """
        max_exposure = 0.0

        for currency in new_currencies:
            current = current_exposure.get(currency, 0.0)
            max_exposure = max(max_exposure, current)

        # Currency exposure limits:
        # 0.0-max_currency_exposure: No reduction (1.0 multiplier)
        # max_currency_exposure-0.9: Moderate reduction (0.7 multiplier)
        # 0.9-1.0: Strong reduction (0.5 multiplier)

        if max_exposure >= 0.9:
            return 0.5
        elif max_exposure >= self.max_currency_exposure:
            return 0.7
        else:
            return 1.0

    def get_currency_exposure_report(self, open_positions: List[str]) -> Dict:
        """
        Generate a comprehensive currency exposure report

        Args:
            open_positions: List of currently open position symbols

        Returns:
            Dict: Currency exposure analysis
        """
        exposure = self._calculate_currency_exposure(open_positions)

        report = {
            'currency_exposure': exposure,
            'high_exposure_currencies': [],
            'diversification_score': 0.0,
            'risk_assessment': 'low',
            'recommendations': []
        }

        # Identify high exposure currencies
        for currency, level in exposure.items():
            if level >= 0.6:
                report['high_exposure_currencies'].append({
                    'currency': currency,
                    'exposure': level,
                    'risk_level': 'severe' if level >= 0.8 else 'high'
                })

        # Calculate diversification score (0-1, higher is better diversified)
        if exposure:
            avg_exposure = sum(exposure.values()) / len(exposure)
            max_exposure = max(exposure.values())
            report['diversification_score'] = 1.0 - (max_exposure - avg_exposure)
        else:
            report['diversification_score'] = 1.0

        # Risk assessment
        if report['high_exposure_currencies']:
            report['risk_assessment'] = 'high'
            report['recommendations'].append("Reduce exposure to high-risk currencies")
        elif report['diversification_score'] < 0.7:
            report['risk_assessment'] = 'medium'
            report['recommendations'].append("Consider diversifying currency exposure")
        else:
            report['risk_assessment'] = 'low'
            report['recommendations'].append("Currency exposure is well diversified")

        return report

    def check_correlation_diversification_limits(self, symbol: str, open_positions: List[str]) -> Tuple[bool, str]:
        """
        Check if opening a new position violates correlation diversification limits

        Args:
            symbol: Symbol to potentially open
            open_positions: List of currently open position symbols

        Returns:
            Tuple[bool, str]: (allowed, reason)
        """
        if not open_positions:
            return True, "No diversification limits to check"

        # Check correlation groups
        correlation_groups = self._identify_correlation_groups(open_positions + [symbol])

        # Find the group this symbol would join
        symbol_group = None
        for group in correlation_groups:
            if symbol in group:
                symbol_group = group
                break

        if symbol_group and len(symbol_group) > self.correlation_group_limit:
            return False, f"Would exceed correlation group limit ({len(symbol_group)} > {self.correlation_group_limit})"

        # Check currency exposure limits
        new_currencies = self._get_currencies(symbol)
        current_exposure = self._calculate_currency_exposure(open_positions)

        for currency in new_currencies:
            if current_exposure.get(currency, 0.0) >= self.max_currency_exposure:  # Near maximum exposure
                return False, f"Currency {currency} exposure too high ({current_exposure[currency]:.1%} >= {self.max_currency_exposure:.1%})"

        return True, "Diversification limits satisfied"

    def _identify_correlation_groups(self, symbols: List[str]) -> List[List[str]]:
        """
        Identify groups of highly correlated symbols

        Args:
            symbols: List of symbols to analyze

        Returns:
            List[List[str]]: Groups of correlated symbols
        """
        if not symbols:
            return []

        groups = []
        processed = set()

        for symbol in symbols:
            if symbol in processed:
                continue

            # Find all symbols strongly correlated with this one
            group = [symbol]
            processed.add(symbol)

            for other_symbol in symbols:
                if other_symbol not in processed:
                    corr = abs(self.get_correlation(symbol, other_symbol))
                    if corr >= self.strong_correlation_threshold:
                        group.append(other_symbol)
                        processed.add(other_symbol)

            if len(group) > 1:  # Only include groups with multiple symbols
                groups.append(group)

        return groups

    def get_portfolio_correlation_risk(self, open_positions: List[str]) -> Dict:
        """
        Calculate overall portfolio correlation risk

        Args:
            open_positions: List of currently open position symbols

        Returns:
            Dict: Portfolio correlation risk assessment
        """
        if len(open_positions) < 2:
            return {
                'overall_risk': 'low',
                'correlation_score': 0.0,
                'diversification_score': 1.0,
                'recommendations': ['Add more positions for better diversification']
            }

        # Calculate average pairwise correlation
        correlations = []
        for i, symbol1 in enumerate(open_positions):
            for symbol2 in open_positions[i+1:]:
                corr = abs(self.get_correlation(symbol1, symbol2))
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0.0
        max_correlation = max(correlations) if correlations else 0.0

        # Calculate diversification score
        diversification_score = 1.0 - avg_correlation

        # Currency exposure analysis
        exposure_report = self.get_currency_exposure_report(open_positions)

        # Overall risk assessment
        risk_score = (avg_correlation * 0.4) + (max_correlation * 0.3) + ((1.0 - exposure_report['diversification_score']) * 0.3)

        if risk_score >= 0.7:
            overall_risk = 'high'
            recommendations = [
                'High correlation risk - consider reducing position sizes',
                'Close some highly correlated positions',
                'Wait for correlation divergence before adding new positions'
            ]
        elif risk_score >= 0.4:
            overall_risk = 'medium'
            recommendations = [
                'Moderate correlation risk - monitor closely',
                'Consider position size reductions for correlated pairs'
            ]
        else:
            overall_risk = 'low'
            recommendations = [
                'Low correlation risk - normal trading parameters apply'
            ]

        return {
            'overall_risk': overall_risk,
            'correlation_score': avg_correlation,
            'max_correlation': max_correlation,
            'diversification_score': diversification_score,
            'currency_exposure_risk': exposure_report['risk_assessment'],
            'risk_score': risk_score,
            'recommendations': recommendations
        }
        """
        Determine if two correlated symbols should be allowed to trade simultaneously

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            market_conditions: Current market conditions (volatility, trend, etc.)

        Returns:
            Tuple[bool, str]: (should_allow, reason)
        """
        correlation = self.get_correlation(symbol1, symbol2)
        corr_type = self.get_correlation_type(symbol1, symbol2)

        # Always allow neutral correlations
        if corr_type == CorrelationType.NEUTRAL:
            return True, "Neutral correlation - no restrictions"

        # For strong correlations, apply sophisticated analysis
        if corr_type in [CorrelationType.POSITIVE_STRONG, CorrelationType.NEGATIVE_STRONG]:
            # Check market conditions for correlated trading
            if market_conditions:
                volatility = market_conditions.get('volatility', 0.5)
                trend_strength = market_conditions.get('trend_strength', 0.5)

                # Allow correlated trading in low volatility, strong trend environments
                if volatility < 0.3 and trend_strength > 0.7:
                    return True, f"Strong {corr_type.value} correlation allowed in favorable market conditions"

                # Allow correlated trading if correlation is weakening
                if abs(correlation) < (self.strong_correlation_threshold + 0.1):
                    return True, f"Correlation weakening ({correlation:.2f}) - allowing concurrent trading"

            # Default: allow but with size restrictions
            return True, f"Strong {corr_type.value} correlation - concurrent trading allowed with size restrictions"

        # For moderate correlations, generally allow
        return True, f"Moderate {corr_type.value} correlation - concurrent trading allowed"

    def monitor_all_correlation_changes(self, open_positions: List[str], 
                                   market_data: Dict = None) -> List[Dict]:
        """
        Monitor correlation changes during trading and suggest actions

        Args:
            open_positions: List of currently open position symbols
            market_data: Current market data for correlation calculation

        Returns:
            List[Dict]: List of suggested actions based on correlation changes
        """
        actions = []

        if len(open_positions) < 2:
            return actions

        # Check all pairs of open positions
        for i, symbol1 in enumerate(open_positions):
            for symbol2 in open_positions[i+1:]:
                current_corr = self.get_correlation(symbol1, symbol2)

                # If correlation has changed significantly, suggest action
                if abs(current_corr) < 0.3:  # Correlation weakened
                    actions.append({
                        'type': 'correlation_weakened',
                        'symbols': [symbol1, symbol2],
                        'correlation': current_corr,
                        'action': 'consider_closing_weaker_position',
                        'reason': f'Correlation dropped to {current_corr:.2f} - consider closing weaker performing position'
                    })
                elif abs(current_corr) > 0.8:  # Correlation strengthened
                    actions.append({
                        'type': 'correlation_strengthened',
                        'symbols': [symbol1, symbol2],
                        'correlation': current_corr,
                        'action': 'reduce_exposure',
                        'reason': f'Correlation increased to {current_corr:.2f} - consider reducing position sizes'
                    })

        return actions

    def get_correlation_based_exit_signals(self, open_positions: List[Dict], 
                                          market_data: Dict = None) -> List[Dict]:
        """
        Generate exit signals based on correlation analysis

        Args:
            open_positions: List of open position dictionaries with symbol, pnl, etc.
            market_data: Current market data

        Returns:
            List[Dict]: Exit signal recommendations
        """
        exit_signals = []

        if len(open_positions) < 2:
            return exit_signals

        # Analyze correlation relationships
        correlation_actions = self.monitor_all_correlation_changes(
            [pos['symbol'] for pos in open_positions], market_data
        )

        for action in correlation_actions:
            if action['type'] == 'correlation_weakened':
                # Find which position is performing worse
                symbol1_data = next((pos for pos in open_positions if pos['symbol'] == action['symbols'][0]), None)
                symbol2_data = next((pos for pos in open_positions if pos['symbol'] == action['symbols'][1]), None)

                if symbol1_data and symbol2_data:
                    pnl1 = symbol1_data.get('unrealized_pnl', 0)
                    pnl2 = symbol2_data.get('unrealized_pnl', 0)

                    # Suggest closing the worse performing position
                    if pnl1 < pnl2:
                        exit_signals.append({
                            'symbol': action['symbols'][0],
                            'reason': f'Correlation weakened ({action["correlation"]:.2f}) and worse P&L (${pnl1:.2f} vs ${pnl2:.2f})',
                            'urgency': 'medium',
                            'correlation_action': 'close_weak_performer'
                        })
                    else:
                        exit_signals.append({
                            'symbol': action['symbols'][1],
                            'reason': f'Correlation weakened ({action["correlation"]:.2f}) and worse P&L (${pnl2:.2f} vs ${pnl1:.2f})',
                            'urgency': 'medium',
                            'correlation_action': 'close_weak_performer'
                        })

        return exit_signals

    def get_correlation_opportunity_signals(self, open_positions: List[str], 
                                          all_symbols: List[str], market_data: Dict = None) -> List[Dict]:
        """
        Identify correlation-based trading opportunities

        Args:
            open_positions: Currently open position symbols
            all_symbols: All available symbols for trading
            market_data: Current market data

        Returns:
            List[Dict]: Opportunity signals for opening new positions
        """
        opportunities = []

        for symbol in all_symbols:
            if symbol in open_positions:
                continue

            # Check correlation with existing positions
            correlations_with_open = []
            for open_symbol in open_positions:
                corr = self.get_correlation(symbol, open_symbol)
                corr_type = self.get_correlation_type(symbol, open_symbol)
                correlations_with_open.append((open_symbol, corr, corr_type))

            # Look for hedging opportunities (negative correlations)
            negative_correlations = [c for c in correlations_with_open 
                                   if c[2] in [CorrelationType.NEGATIVE_STRONG, CorrelationType.NEGATIVE_MODERATE]]

            if negative_correlations:
                # Suggest opening position to hedge existing exposure
                best_hedge = max(negative_correlations, key=lambda x: abs(x[1]))
                opportunities.append({
                    'symbol': symbol,
                    'type': 'hedging_opportunity',
                    'correlation_with': best_hedge[0],
                    'correlation': best_hedge[1],
                    'reason': f'Negative correlation ({best_hedge[1]:.2f}) with {best_hedge[0]} - potential hedge',
                    'confidence': min(abs(best_hedge[1]) * 100, 80)
                })

            # Look for momentum opportunities (positive correlations)
            positive_correlations = [c for c in correlations_with_open 
                                   if c[2] in [CorrelationType.POSITIVE_STRONG, CorrelationType.POSITIVE_MODERATE]]

            if positive_correlations and market_data:
                trend_alignment = market_data.get('trend_alignment', 0.5)
                if trend_alignment > 0.7:  # Strong trend alignment
                    best_momentum = max(positive_correlations, key=lambda x: abs(x[1]))
                    opportunities.append({
                        'symbol': best_momentum[0],
                        'type': 'momentum_opportunity',
                        'correlation_with': symbol,
                        'correlation': best_momentum[1],
                        'reason': f'Positive correlation ({best_momentum[1]:.2f}) with strong trend alignment',
                        'confidence': min(abs(best_momentum[1]) * trend_alignment * 100, 75)
                    })

        return opportunities

    def update_correlation_learning(self, trade_outcome: Dict, adaptive_learning_manager=None):
        """
        Update correlation learning based on trade outcomes

        Args:
            trade_outcome: Trade result data
            adaptive_learning_manager: Reference to adaptive learning manager
        """
        try:
            symbol = trade_outcome.get('symbol', '')
            pnl = trade_outcome.get('realized_pnl', 0)
            duration = trade_outcome.get('duration_hours', 0)

            # Analyze correlation impact on trade performance
            correlation_factors = {
                'symbol': symbol,
                'pnl': pnl,
                'duration': duration,
                'correlated_positions': trade_outcome.get('correlated_positions_during_trade', []),
                'correlation_changes': trade_outcome.get('correlation_changes', []),
                'market_conditions': trade_outcome.get('market_conditions', {})
            }

            # Store in adaptive learning system if available
            if adaptive_learning_manager:
                adaptive_learning_manager.record_correlation_performance(correlation_factors)

            self.logger.info(f"Updated correlation learning for {symbol}: P&L ${pnl:.2f}, {len(correlation_factors['correlated_positions'])} correlated positions")

        except Exception as e:
            self.logger.error(f"Error updating correlation learning: {e}")

    def get_pending_actions(self) -> Dict[str, Dict]:
        """
        Get pending correlation-based trading actions

        Returns:
            Dict of symbol -> action_dict where action_dict contains:
            - action: 'exit_recommended', 'entry_recommended', or 'exit_consideration'
            - confidence: float confidence score
            - correlation: float correlation value
            - correlated_symbol: str symbol that triggered the action
        """
        try:
            pending_actions = {}

            # Get correlation-based exit signals
            exit_signals = self.get_correlation_based_exit_signals([], self.open_positions_symbols)
            for signal in exit_signals:
                symbol = signal.get('symbol')
                if symbol and symbol not in pending_actions:
                    pending_actions[symbol] = {
                        'action': 'exit_recommended',
                        'confidence': signal.get('confidence', 0.5),
                        'correlation': signal.get('correlation', 0),
                        'correlated_symbol': signal.get('correlated_symbol', '')
                    }

            # Get correlation-based entry signals
            entry_signals = self.get_correlation_opportunity_signals(list(self.open_positions_symbols), [])
            for signal in entry_signals:
                symbol = signal.get('symbol')
                if symbol and symbol not in pending_actions:
                    pending_actions[symbol] = {
                        'action': 'entry_recommended',
                        'confidence': signal.get('confidence', 0.5),
                        'correlation': signal.get('correlation', 0),
                        'correlated_symbol': signal.get('correlated_symbol', '')
                    }

            self.logger.debug(f"Found {len(pending_actions)} pending correlation actions")
            return pending_actions

        except Exception as e:
            self.logger.error(f"Error getting pending correlation actions: {e}")
            return {}