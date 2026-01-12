"""
Configuration loader utility.
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML and environment variables."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file and environment variables."""
        # Load environment variables from .env file
        load_dotenv()

        # Load YAML configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Replace environment variable placeholders
        self._replace_env_vars(self._config)

    def _replace_env_vars(self, config: Any) -> None:
        """
        Recursively replace environment variable references in config.

        Args:
            config: Configuration dictionary or value
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, value)
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                    env_var = item[2:-1]
                    config[i] = os.getenv(env_var, item)
                elif isinstance(item, (dict, list)):
                    self._replace_env_vars(item)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path (e.g., "model.training.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated key path.

        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config
