"""
Reinforcement Learning Module for FX-Ai
Implements Q-learning for optimal trade execution timing
"""

import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
import json
import os
import pickle
from enum import Enum

class Action(Enum):
    """Trading actions for RL agent"""
    HOLD = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    EXIT = 3

class MarketState:
    """Represents the current market state for RL"""

    def __init__(self, regime: str, rsi: float, macd_signal: float,
                 bb_position: float, trend_strength: float, volatility: float,
                 position_status: int):
        """
        Initialize market state

        Args:
            regime: Current market regime ('bull', 'bear', 'ranging', 'trending')
            rsi: RSI indicator value
            macd_signal: MACD signal line
            bb_position: Bollinger Band position (-1 to 1)
            trend_strength: ADX or trend strength indicator
            volatility: ATR or volatility measure
            position_status: -1 (short), 0 (no position), 1 (long)
        """
        self.regime = regime
        self.rsi = rsi
        self.macd_signal = macd_signal
        self.bb_position = bb_position
        self.trend_strength = trend_strength
        self.volatility = volatility
        self.position_status = position_status

    def to_tuple(self) -> Tuple:
        """Convert state to hashable tuple for Q-table"""
        return (
            self.regime,
            round(self.rsi, 1),
            round(self.macd_signal, 3),
            round(self.bb_position, 2),
            round(self.trend_strength, 1),
            round(self.volatility, 4),
            self.position_status
        )

    def discretize_value(self, value: float, bins: int, min_val: float, max_val: float) -> int:
        """Discretize continuous value into bins"""
        if value <= min_val:
            return 0
        elif value >= max_val:
            return bins - 1
        else:
            return int((value - min_val) / (max_val - min_val) * bins)

class RLAgent:
    """Reinforcement Learning Agent for Trading"""

    def __init__(self, config: Dict):
        """
        Initialize RL agent

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # RL parameters
        rl_config = config.get('reinforcement_learning', {})
        self.enabled = rl_config.get('enabled', True)
        self.learning_rate = rl_config.get('learning_rate', 0.1)
        self.discount_factor = rl_config.get('discount_factor', 0.95)
        self.epsilon = rl_config.get('epsilon', 0.1)  # Exploration rate
        self.epsilon_decay = rl_config.get('epsilon_decay', 0.995)
        self.min_epsilon = rl_config.get('min_epsilon', 0.01)

        # State discretization
        self.state_bins = rl_config.get('state_bins', {
            'rsi': 10,  # 0-100 in 10 bins
            'macd': 20,  # -2 to 2 in 20 bins
            'bb_position': 10,  # -1 to 1 in 10 bins
            'trend_strength': 10,  # 0-100 in 10 bins
            'volatility': 10  # 0-0.01 in 10 bins
        })

        # Q-table: state -> action -> q_value
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Experience replay
        self.max_experiences = rl_config.get('max_experiences', 10000)
        self.experiences = []

        # Performance tracking
        self.episode_rewards = []
        self.episode_count = 0

        # Model persistence
        self.model_dir = config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'rl_agent.pkl')

        # Load existing model if available
        self.load_model()

        # Current state tracking
        self.current_state = None
        self.last_action = None
        self.last_reward = 0

        # Pending experiences for learning when trades close
        self.pending_experiences = {}

    def get_state_from_data(self, symbol: str, market_data: pd.DataFrame,
                           technical_signals: Dict, position_status: int,
                           regime: str) -> MarketState:
        """
        Extract market state from current data

        Args:
            symbol: Trading symbol
            market_data: Historical price data
            technical_signals: Technical analysis signals
            position_status: Current position (-1, 0, 1)
            regime: Current market regime

        Returns:
            MarketState: Current market state
        """
        try:
            # Extract technical indicators
            rsi = technical_signals.get('rsi', 50.0)
            macd_signal = technical_signals.get('macd_signal', 0.0)
            bb_position = technical_signals.get('bb_position', 0.0)
            trend_strength = technical_signals.get('adx', 25.0)

            # Calculate volatility (ATR)
            if len(market_data) >= 14:
                high_low = market_data['high'] - market_data['low']
                high_close = (market_data['high'] - market_data['close'].shift(1)).abs()
                low_close = (market_data['low'] - market_data['close'].shift(1)).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                volatility = tr.rolling(14).mean().iloc[-1] / market_data['close'].iloc[-1]
            else:
                volatility = 0.001  # Default volatility

            return MarketState(
                regime=regime,
                rsi=rsi,
                macd_signal=macd_signal,
                bb_position=bb_position,
                trend_strength=trend_strength,
                volatility=volatility,
                position_status=position_status
            )

        except Exception as e:
            self.logger.error(f"Error creating market state: {e}")
            # Return default state
            return MarketState('ranging', 50.0, 0.0, 0.0, 25.0, 0.001, 0)

    def discretize_state(self, state: MarketState) -> Tuple:
        """
        Discretize continuous state values for Q-table

        Args:
            state: MarketState object

        Returns:
            tuple: Discretized state tuple
        """
        regime_map = {'bull': 0, 'bear': 1, 'ranging': 2, 'trending': 3}
        regime_idx = regime_map.get(state.regime, 2)

        rsi_bin = self.discretize_value(state.rsi, self.state_bins['rsi'], 0, 100)
        macd_bin = self.discretize_value(state.macd_signal, self.state_bins['macd'], -2, 2)
        bb_bin = self.discretize_value(state.bb_position, self.state_bins['bb_position'], -1, 1)
        trend_bin = self.discretize_value(state.trend_strength, self.state_bins['trend_strength'], 0, 100)
        vol_bin = self.discretize_value(state.volatility, self.state_bins['volatility'], 0, 0.01)

        return (regime_idx, rsi_bin, macd_bin, bb_bin, trend_bin, vol_bin, state.position_status)

    def discretize_value(self, value: float, bins: int, min_val: float, max_val: float) -> int:
        """Discretize continuous value into bins"""
        if value <= min_val:
            return 0
        elif value >= max_val:
            return bins - 1
        else:
            return int((value - min_val) / (max_val - min_val) * bins)

    def choose_action(self, state: MarketState, training: bool = True) -> Action:
        """
        Choose action using epsilon-greedy policy

        Args:
            state: Current market state
            training: Whether in training mode

        Returns:
            Action: Chosen action
        """
        state_key = self.discretize_state(state)

        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return random.choice(list(Action))
        else:
            # Exploit: best action
            q_values = {action: self.q_table[state_key][action.value] for action in Action}
            best_action_value = max(q_values, key=lambda x: q_values[x])
            return Action(best_action_value)

    def choose_action_from_dict(self, state_dict: Dict, training: bool = True) -> str:
        """
        Choose action from simplified state dictionary

        Args:
            state_dict: State dictionary with simplified features
            training: Whether in training mode

        Returns:
            str: Action as string ('hold', 'buy', 'sell', 'close_position')
        """
        # Discretize the state for Q-table lookup
        state_key = self.discretize_state_dict(state_dict)

        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(['hold', 'buy', 'sell', 'close_position'])
        else:
            # Exploit: best action based on Q-values
            q_values = {}
            for action in ['hold', 'buy', 'sell', 'close_position']:
                action_idx = self.action_to_index(action)
                q_values[action] = self.q_table[state_key][action_idx]
            
            action = max(q_values, key=lambda x: q_values[x])

        return action

    def discretize_state_dict(self, state_dict: Dict) -> Tuple:
        """
        Discretize simplified state dictionary for Q-table

        Args:
            state_dict: Simplified state dictionary

        Returns:
            tuple: Discretized state tuple
        """
        # Map regime to index
        regime_map = {'bull_market': 0, 'bear_market': 1, 'ranging': 2, 'trending_up': 3, 'trending_down': 4}
        regime_idx = regime_map.get(state_dict.get('market_regime', 'ranging'), 2)

        # Get state bins from config
        state_bins = self.config.get('reinforcement_learning', {}).get('state_bins', {
            'rsi_bins': 10,
            'adx_bins': 8,
            'volatility_bins': 6,
            'trend_bins': 5,
            'regime_bins': 4
        })

        # Discretize each feature
        rsi_bin = self.discretize_value(state_dict.get('rsi', 50), state_bins['rsi_bins'], 0, 100)
        adx_bin = self.discretize_value(state_dict.get('adx', 25), state_bins['adx_bins'], 0, 100)
        vol_bin = self.discretize_value(state_dict.get('volatility_ratio', 0.001), state_bins['volatility_bins'], 0, 0.01)
        trend_bin = self.discretize_value(state_dict.get('trend_strength', 25), state_bins['trend_bins'], 0, 100)
        regime_bin = min(regime_idx, state_bins['regime_bins'] - 1)

        return (regime_bin, rsi_bin, adx_bin, vol_bin, trend_bin, state_dict.get('position_status', 0))

    def action_to_index(self, action: str) -> int:
        """Convert action string to index"""
        action_map = {'hold': 0, 'buy': 1, 'sell': 2, 'close_position': 3}
        return action_map.get(action, 0)

    def calculate_reward(self, entry_price: float, exit_price: float, direction: str, duration_minutes: int) -> float:
        """
        Calculate reward for a completed trade

        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: Trade direction ('BUY' or 'SELL')
            duration_minutes: Trade duration in minutes

        Returns:
            float: Reward value
        """
        # Calculate P&L
        if direction == 'BUY':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        # Base reward from P&L (normalized)
        reward = pnl / entry_price * 100  # Percentage return

        # Duration penalty (prefer shorter trades if profitable, longer if losing)
        if reward > 0:
            # Penalize very short profitable trades (may be noise)
            if duration_minutes < 30:
                reward *= 0.8
        else:
            # Penalize very long losing trades
            if duration_minutes > 480:  # 8 hours
                reward *= 0.9

        # Cap rewards
        reward = max(min(reward, 5.0), -5.0)

        return reward

    def update_q_table(self, state_dict: Dict, action: str, reward: float, next_state_dict: Dict):
        """
        Update Q-table using simplified state dictionaries

        Args:
            state_dict: Current state dictionary
            action: Action taken as string
            reward: Reward received
            next_state_dict: Next state dictionary
        """
        state_key = self.discretize_state_dict(state_dict)
        next_state_key = self.discretize_state_dict(next_state_dict)
        action_idx = self.action_to_index(action)

        # Current Q-value
        current_q = self.q_table[state_key][action_idx]

        # Max Q-value for next state
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0

        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_action_recommendation(self, symbol: str, market_data: pd.DataFrame,
                                 technical_signals: Dict, position_status: int,
                                 regime: str) -> Dict:
        """
        Get RL-based action recommendation

        Args:
            symbol: Trading symbol
            market_data: Historical price data
            technical_signals: Technical signals
            position_status: Current position status
            regime: Current market regime

        Returns:
            dict: Action recommendation with confidence
        """
        try:
            state = self.get_state_from_data(symbol, market_data, technical_signals,
                                           position_status, regime)

            action = self.choose_action(state, training=False)

            # Get Q-values for confidence calculation
            state_key = self.discretize_state(state)
            q_values = [self.q_table[state_key][a.value] for a in Action]
            max_q = max(q_values) if q_values else 0
            min_q = min(q_values) if q_values else 0

            # Confidence based on Q-value spread
            if max_q - min_q > 0:
                confidence = (max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-6)
            else:
                confidence = 0.5  # Default confidence

            return {
                'action': action,
                'confidence': min(confidence, 1.0),
                'q_values': dict(zip([a.name for a in Action], q_values)),
                'state': state.to_tuple()
            }

        except Exception as e:
            self.logger.error(f"Error getting RL recommendation: {e}")
            return {
                'action': Action.HOLD,
                'confidence': 0.0,
                'q_values': {},
                'state': None
            }

    def save_model(self):
        """Save RL model to disk"""
        try:
            model_data = {
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'episode_count': self.episode_count,
                'experiences': self.experiences[-1000:],  # Save last 1000 experiences
                'config': self.config.get('reinforcement_learning', {})
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"RL model saved to {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error saving RL model: {e}")

    def load_model(self):
        """Load RL model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.q_table = defaultdict(lambda: defaultdict(float), model_data.get('q_table', {}))
                self.epsilon = model_data.get('epsilon', self.epsilon)
                self.episode_count = model_data.get('episode_count', 0)
                self.experiences = model_data.get('experiences', [])

                self.logger.info(f"RL model loaded from {self.model_path}")
                return True
            else:
                self.logger.info("No existing RL model found, starting fresh")
                return False

        except Exception as e:
            self.logger.error(f"Error loading RL model: {e}")
            return False

    def get_performance_stats(self) -> Dict:
        """Get RL agent performance statistics"""
        return {
            'episodes_trained': self.episode_count,
            'q_table_size': len(self.q_table),
            'experiences_stored': len(self.experiences),
            'current_epsilon': self.epsilon,
            'average_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }