"""
Config Loader Module
Handles loading and validation of configuration files
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration loader with validation and defaults"""

    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initialize config loader

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self.defaults = {
            'mt5': {
                'server': 'MetaQuotes-Demo',
                'login': 123456,
                'password': 'password',
                'timeout': 60000
            },
            'trading': {
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'timeframe': 'H1',
                'max_positions': 5,
                'risk_per_trade': 50.0  # $50 per trade instead of 2%
            },
            'risk_management': {
                'max_risk_per_trade': 50.0,  # $50 per trade instead of 2%
                'max_daily_risk': 200.0,     # $200 daily loss instead of 5%
                'max_open_positions': 5,
                'max_correlation': 0.7
            },
            'technical_analysis': {
                'indicators': ['rsi', 'macd', 'ema', 'vwap'],
                'rsi_period': 14,
                'ema_fast': 9,
                'ema_slow': 21
            },
            'ml_model': {
                'model_type': 'random_forest',
                'confidence_threshold': 0.6,
                'lookback_periods': [5, 10, 20, 50]
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/FX-Ai',
                'rotation_type': 'time',
                'max_size': 10485760,  # 10MB
                'backup_count': 5
            },
            'data': {
                'cache_timeout': 60,
                'max_historical_bars': 10000,
                'update_interval': 60
            }
        }

        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)

                # Merge with defaults
                self.config = self._merge_configs(self.defaults, user_config)
                self.logger.info(f"Configuration loaded from {self.config_path}, trading.max_positions={self.config.get('trading', {}).get('max_positions', 'NOT_FOUND')}")
            else:
                # Use defaults and create config file
                self.config = self.defaults.copy()
                self.save_config()
                self.logger.info(f"Default configuration created at {self.config_path}")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.config = self.defaults.copy()

    def save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure directory exists
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)

            self.logger.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")

    def _merge_configs(self, defaults: Dict, user_config: Dict) -> Dict:
        """
        Recursively merge user config with defaults

        Args:
            defaults: Default configuration
            user_config: User configuration

        Returns:
            dict: Merged configuration
        """
        merged = defaults.copy()

        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value

        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def validate_config(self) -> bool:
        """
        Validate configuration values

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate MT5 settings
            mt5_config = self.get('mt5', {})
            if not isinstance(mt5_config.get('login'), int):
                self.logger.error("MT5 login must be an integer")
                return False

            if not mt5_config.get('password'):
                self.logger.error("MT5 password is required")
                return False

            # Validate trading settings
            trading_config = self.get('trading', {})
            if not trading_config.get('symbols'):
                self.logger.error("At least one trading symbol is required")
                return False

            # Validate risk settings
            risk_config = self.get('risk_management', {})
            if risk_config.get('max_risk_per_trade', 0) > 0.1:  # 10% max
                self.logger.warning("Risk per trade seems very high (>10%)")

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    def create_default_config(self):
        """Create default configuration file"""
        self.config = self.defaults.copy()
        self.save_config()

    def get_section(self, section: str) -> Dict:
        """
        Get entire configuration section

        Args:
            section: Section name

        Returns:
            dict: Configuration section
        """
        return self.config.get(section, {})

    def update_from_dict(self, updates: Dict):
        """
        Update configuration from dictionary

        Args:
            updates: Dictionary of updates
        """
        def update_nested_dict(base_dict, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    update_nested_dict(base_dict[key], value)
                else:
                    base_dict[key] = value

        update_nested_dict(self.config, updates)
        self.save_config()

    def __str__(self) -> str:
        """String representation of config"""
        return json.dumps(self.config, indent=2)