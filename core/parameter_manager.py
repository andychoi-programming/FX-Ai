"""
Unified Parameter Manager for FX-Ai
Centralizes all parameter optimization and management
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
from datetime import datetime

class ParameterManager:
    """Unified parameter manager for all trading systems"""

    def __init__(self, config_manager):
        """
        Initialize parameter manager

        Args:
            config_manager: ConfigManager instance
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)

        # Parameter storage paths
        self.models_dir = Path('models')
        self.param_dir = self.models_dir / 'parameter_optimization'
        self.param_file = self.param_dir / 'optimal_parameters.json'

        # Ensure directories exist
        self.param_dir.mkdir(parents=True, exist_ok=True)

        # Load parameters
        self.parameters = self._load_parameters()

    def _load_parameters(self) -> Dict:
        """Load optimized parameters from file"""
        if self.param_file.exists():
            try:
                with open(self.param_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load parameters: {e}")
                return {}
        else:
            self.logger.warning("Optimized parameters file not found")
            return {}

    def save_parameters(self):
        """Save current parameters to file"""
        try:
            with open(self.param_file, 'w') as f:
                json.dump(self.parameters, f, indent=2)
            self.logger.info(f"Saved parameters for {len(self.parameters)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}")

    def get_optimal_parameters(self, symbol: str, timeframe: str = 'H1') -> Dict:
        """
        Get optimal parameters for a symbol and timeframe

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe ('H1', 'D1', 'W1', 'MN1')

        Returns:
            Dict with optimal parameters
        """
        # Try to get symbol-specific parameters
        if symbol in self.parameters and timeframe in self.parameters[symbol]:
            params = self.parameters[symbol][timeframe]['optimal_params']
            self.logger.debug(f"Using optimized params for {symbol} {timeframe}: SL={params.get('sl_pips', 20)}, TP={params.get('tp_pips', 40)}")
            return params

        # Fallback to symbol-specific H1 parameters
        if symbol in self.parameters and 'H1' in self.parameters[symbol]:
            params = self.parameters[symbol]['H1']['optimal_params']
            self.logger.debug(f"Using H1 fallback params for {symbol} {timeframe}: SL={params.get('sl_pips', 20)}, TP={params.get('tp_pips', 40)}")
            return params

        # Use defaults from config
        default_sl = self.config.get('trading.default_sl_pips', 20)
        default_tp = self.config.get('trading.default_tp_pips', 60)

        self.logger.debug(f"Using default params for {symbol} {timeframe}: SL={default_sl}, TP={default_tp}")
        return {
            'sl_pips': default_sl,
            'tp_pips': default_tp,
            'risk_reward_ratio': self.config.get('trading.min_risk_reward_ratio', 3.0)
        }

    def update_parameters(self, symbol: str, timeframe: str, params: Dict, performance_metrics: Dict = None):
        """
        Update optimal parameters for a symbol

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            params: New optimal parameters
            performance_metrics: Optional performance data
        """
        if symbol not in self.parameters:
            self.parameters[symbol] = {}

        if timeframe not in self.parameters[symbol]:
            self.parameters[symbol][timeframe] = {}

        self.parameters[symbol][timeframe]['optimal_params'] = params
        self.parameters[symbol][timeframe]['last_updated'] = datetime.now().isoformat()

        if performance_metrics:
            self.parameters[symbol][timeframe]['performance'] = performance_metrics

        self.logger.info(f"Updated parameters for {symbol} {timeframe}")

    def get_all_symbols_with_params(self) -> List[str]:
        """Get list of all symbols that have optimized parameters"""
        return list(self.parameters.keys())

    def get_parameter_summary(self) -> Dict:
        """Get summary of parameter optimization status"""
        summary = {
            'total_symbols': len(self.parameters),
            'symbols_by_timeframe': {},
            'last_updates': []
        }

        for symbol, timeframes in self.parameters.items():
            for timeframe, data in timeframes.items():
                if timeframe not in summary['symbols_by_timeframe']:
                    summary['symbols_by_timeframe'][timeframe] = []
                summary['symbols_by_timeframe'][timeframe].append(symbol)

                if 'last_updated' in data:
                    summary['last_updates'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'last_updated': data['last_updated']
                    })

        return summary

    def validate_parameters(self, symbol: str, timeframe: str = 'H1') -> bool:
        """
        Validate that parameters exist and are reasonable

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            True if parameters are valid
        """
        params = self.get_optimal_parameters(symbol, timeframe)

        # Check required parameters exist
        required = ['sl_pips', 'tp_pips']
        for param in required:
            if param not in params:
                self.logger.warning(f"Missing required parameter {param} for {symbol} {timeframe}")
                return False

        # Check parameter ranges
        if params['sl_pips'] <= 0 or params['tp_pips'] <= 0:
            self.logger.warning(f"Invalid parameter values for {symbol} {timeframe}: SL={params['sl_pips']}, TP={params['tp_pips']}")
            return False

        # Check risk-reward ratio
        if 'risk_reward_ratio' in params:
            actual_rr = params['tp_pips'] / params['sl_pips']
            if actual_rr < 1.0:
                self.logger.warning(f"Poor risk-reward ratio for {symbol} {timeframe}: {actual_rr:.2f}")
                return False

        return True

    def reset_symbol_parameters(self, symbol: str):
        """
        Reset parameters for a symbol to defaults

        Args:
            symbol: Trading symbol to reset
        """
        if symbol in self.parameters:
            del self.parameters[symbol]
            self.logger.info(f"Reset parameters for {symbol}")

    def backup_parameters(self):
        """Create a backup of current parameters"""
        backup_file = self.param_dir / f'optimal_parameters_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        try:
            with open(backup_file, 'w') as f:
                json.dump(self.parameters, f, indent=2)
            self.logger.info(f"Created parameter backup: {backup_file}")
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")