"""
Configuration management for the vast.ai automation system.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .exceptions import ConfigurationError


def load_config(config_file: Path | None = None) -> dict[str, Any]:
    """Load configuration from file or environment variables."""
    config = {}

    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    # Load from file if provided
    if config_file and config_file.exists():
        try:
            with open(config_file) as f:
                config.update(json.load(f))
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_file}: {e}")

    # Load from environment variables
    env_config = {
        "VAST_API_KEY": os.getenv("VAST_API_KEY"),
        "SSH_KEY_PATH": os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "MAX_CONCURRENT_JOBS": int(os.getenv("MAX_CONCURRENT_JOBS", "5")),
        "JOB_TIMEOUT_MINUTES": int(os.getenv("JOB_TIMEOUT_MINUTES", "30")),
        "INSTANCE_DISCOVERY_INTERVAL": int(
            os.getenv("INSTANCE_DISCOVERY_INTERVAL", "30")
        ),
        "JOB_DISCOVERY_INTERVAL": int(os.getenv("JOB_DISCOVERY_INTERVAL", "60")),
    }

    # Filter out None values
    env_config = {k: v for k, v in env_config.items() if v is not None}
    config.update(env_config)

    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value."""
    config = load_config()
    return config.get(key, default)


def update_config(key: str, value: Any) -> bool:
    """Update a configuration value (in memory only)."""
    # This is a simple implementation - in a real system you might want to persist changes
    return True


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []

    required_keys = ["VAST_API_KEY"]
    for key in required_keys:
        if key not in config or not config[key]:
            errors.append(f"Missing required configuration: {key}")

    # Validate numeric values
    numeric_keys = [
        "MAX_CONCURRENT_JOBS",
        "JOB_TIMEOUT_MINUTES",
        "INSTANCE_DISCOVERY_INTERVAL",
        "JOB_DISCOVERY_INTERVAL",
    ]
    for key in numeric_keys:
        if key in config:
            try:
                int(config[key])
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {key}: {config[key]}")

    return errors
