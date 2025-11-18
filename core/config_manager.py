"""
Unified Configuration Manager for FX-Ai
Handles configuration for both live trading and backtesting modes
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """Unified configuration manager for FX-Ai"""

    def __init__(self, mode: str = 'live'):
        """
        Initialize configuration manager

        Args:
            mode: 'live' or 'backtest'
        """
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        # Configuration file paths
        self.config_dir = Path('config')
        self.main_config_file = self.config_dir / 'config.json'
        self.env_file = Path('.env')

        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configuration based on mode"""
        if not self.main_config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.main_config_file}")

        # Load main configuration
        with open(self.main_config_file, 'r') as f:
            config = json.load(f)

        # Load environment variables for sensitive data
        env_config = self._load_env_config()

        # Merge environment config
        self._merge_env_config(config, env_config)

        # Apply mode-specific overrides
        if self.mode == 'backtest':
            self._apply_backtest_overrides(config)
        elif self.mode == 'live':
            self._apply_live_overrides(config)

        return config

    def _load_env_config(self) -> Dict[str, str]:
        """Load configuration from .env file"""
        env_config = {}

        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_config[key.strip()] = value.strip()

        return env_config

    def _merge_env_config(self, config: Dict[str, Any], env_config: Dict[str, str]):
        """Merge environment variables into configuration"""
        # MT5 credentials
        if 'MT5_LOGIN' in env_config:
            config['mt5']['login'] = int(env_config['MT5_LOGIN'])
        if 'MT5_PASSWORD' in env_config:
            config['mt5']['password'] = env_config['MT5_PASSWORD']
        if 'MT5_SERVER' in env_config:
            config['mt5']['server'] = env_config['MT5_SERVER']
        if 'MT5_PATH' in env_config:
            config['mt5']['path'] = env_config['MT5_PATH']

    def _apply_backtest_overrides(self, config: Dict[str, Any]):
        """Apply backtest-specific configuration overrides"""
        # Disable live trading features
        config['trading']['dry_run'] = True

        # Adjust risk settings for backtesting
        config['trading']['risk_per_trade'] = 100.0  # Higher risk for faster testing

        # Disable real-time features
        config['trading']['enable_news_filter'] = False

        # Set backtest-specific database
        if 'database' not in config:
            config['database'] = {}
        config['database']['performance_db'] = 'backtest_results.db'

    def _apply_live_overrides(self, config: Dict[str, Any]):
        """Apply live trading-specific configuration overrides"""
        # Ensure live trading features are enabled
        config['trading']['dry_run'] = False

        # Set live database
        if 'database' not in config:
            config['database'] = {}
        config['database']['performance_db'] = 'performance_history.db'

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'trading.risk_per_trade')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'trading.risk_per_trade')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to the parent of the final key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the final value
        config[keys[-1]] = value

    def save(self, file_path: Optional[Path] = None):
        """
        Save current configuration to file

        Args:
            file_path: Optional file path, defaults to main config file
        """
        if file_path is None:
            file_path = self.main_config_file

        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_symbols(self) -> list:
        """Get list of trading symbols"""
        return self.get('trading.symbols', [])

    def get_risk_settings(self) -> Dict[str, Any]:
        """Get risk management settings"""
        return {
            'risk_per_trade': self.get('trading.risk_per_trade', 50.0),
            'risk_type': self.get('trading.risk_type', 'fixed_dollar'),
            'max_daily_loss': self.get('trading.max_daily_loss', 500.0),
            'max_positions': self.get('trading.max_positions', 30),
        }

    def get_mt5_config(self) -> Dict[str, Any]:
        """Get MT5 connection settings"""
        return self.config.get('mt5', {})

    def is_backtest_mode(self) -> bool:
        """Check if running in backtest mode"""
        return self.mode == 'backtest'

    def is_live_mode(self) -> bool:
        """Check if running in live mode"""
        return self.mode == 'live'