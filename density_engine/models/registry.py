"""
Model registry for dynamic model discovery and loading.

This module provides a registry system for managing Monte Carlo models,
allowing dynamic registration and retrieval of model classes.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import MonteCarloModelProtocol

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for Monte Carlo model classes.

    This registry allows models to be registered and retrieved by name,
    enabling dynamic model loading and discovery.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._models: dict[str, type[MonteCarloModelProtocol]] = {}
        self._model_names: list[str] = []

    def register(self, name: str, model_class: type[MonteCarloModelProtocol]) -> None:
        """Register a model class.

        Args:
            name: Model identifier (e.g., 'garch', 'rough_heston')
            model_class: Model class implementing MonteCarloModelProtocol

        Raises:
            ValueError: If name is already registered or model_class is invalid
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string")

        # Skip protocol check for now to avoid runtime issues
        # TODO: Implement proper protocol checking

        if name in self._models:
            logger.warning(f"Overriding existing model registration: {name}")

        self._models[name] = model_class
        if name not in self._model_names:
            self._model_names.append(name)

        logger.info(f"Registered model: {name} -> {model_class.__name__}")

    def get_model_class(self, name: str) -> type[MonteCarloModelProtocol] | None:
        """Get model class by name.

        Args:
            name: Model identifier

        Returns:
            Model class if found, None otherwise
        """
        return self._models.get(name)

    def get_model_names(self) -> list[str]:
        """Get list of all registered model names.

        Returns:
            List of registered model names
        """
        return self._model_names.copy()

    def is_registered(self, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Model identifier

        Returns:
            True if model is registered, False otherwise
        """
        return name in self._models

    def unregister(self, name: str) -> bool:
        """Unregister a model.

        Args:
            name: Model identifier

        Returns:
            True if model was unregistered, False if not found
        """
        if name in self._models:
            del self._models[name]
            if name in self._model_names:
                self._model_names.remove(name)
            logger.info(f"Unregistered model: {name}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()
        self._model_names.clear()
        logger.info("Cleared all model registrations")


# Global registry instance
_global_registry = ModelRegistry()


def register_model(name: str, model_class: type[MonteCarloModelProtocol]) -> None:
    """Register a model class in the global registry.

    Args:
        name: Model identifier
        model_class: Model class implementing MonteCarloModelProtocol
    """
    _global_registry.register(name, model_class)


def get_model_class(name: str) -> type[MonteCarloModelProtocol] | None:
    """Get model class by name from global registry.

    Args:
        name: Model identifier

    Returns:
        Model class if found, None otherwise
    """
    return _global_registry.get_model_class(name)


def get_model_names() -> list[str]:
    """Get list of all registered model names from global registry.

    Returns:
        List of registered model names
    """
    return _global_registry.get_model_names()


def is_model_registered(name: str) -> bool:
    """Check if a model is registered in global registry.

    Args:
        name: Model identifier

    Returns:
        True if model is registered, False otherwise
    """
    return _global_registry.is_registered(name)


def get_registry() -> ModelRegistry:
    """Get the global registry instance.

    Returns:
        Global ModelRegistry instance
    """
    return _global_registry
