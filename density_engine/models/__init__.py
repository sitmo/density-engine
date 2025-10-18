"""
Monte Carlo simulation models for financial applications.

This package provides a protocol-based architecture for implementing
various Monte Carlo simulation models including GARCH, stochastic volatility
models, and other path-dependent financial models.
"""

from .base import MonteCarloModelBase, MonteCarloModelProtocol
from .garch import GJRGARCH, GJRGARCHNormalized
from .mdn import MDNModel, MixtureDensityNetwork
from .registry import ModelRegistry, get_model_class, register_model
from .rough_bergomi import RoughBergomi
from .rough_heston import RoughHeston

# Register all available models
register_model("gjrgarch_normalized", GJRGARCHNormalized)
register_model("gjrgarch", GJRGARCH)
register_model("rough_heston", RoughHeston)
register_model("rough_bergomi", RoughBergomi)
register_model("mdn", MDNModel)  # type: ignore[arg-type]

__all__ = [
    "MonteCarloModelProtocol",  # Protocol interface
    "MonteCarloModelBase",  # Base implementation class
    "ModelRegistry",
    "get_model_class",
    "register_model",
    "GJRGARCHNormalized",
    "GJRGARCH",
    "RoughHeston",
    "RoughBergomi",
    "MDNModel",
    "MixtureDensityNetwork",
]
