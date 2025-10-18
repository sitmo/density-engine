"""
Base protocol for Monte Carlo simulation models.

This module defines the MonteCarloModel protocol that all simulation models
must implement to ensure consistent interfaces across different model types.
"""

from typing import Dict, List, Protocol, Union, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class MonteCarloModelProtocol(Protocol):
    """Protocol for all Monte Carlo simulation models.

    This protocol defines the interface for models that generate
    sample paths and compute quantiles, including GARCH (dynamic volatility),
    stochastic volatility models (Heston, Bergomi), and other path-dependent
    models.

    All models implementing this protocol must provide:
    - Initialization with model-specific parameters
    - State reset for simulation runs
    - Quantile computation at specified time points
    - Parameter serialization
    - Model identification
    """

    def __init__(
        self, device: str | torch.device | None = None, **params: float | int
    ) -> None:
        """Initialize model with specific parameters.

        Args:
            device: PyTorch device for tensor placement (e.g., 'cpu', 'cuda', 'cuda:0')
            **params: Model-specific parameters (e.g., alpha, beta for GARCH)
                     All parameters must be numeric (float or int)

        Note:
            Implementations may use keyword-only arguments (*) for better API design
        """
        ...

    def reset(self, num_paths: int) -> None:
        """Reset simulation state for a new run.

        Args:
            num_paths: Number of Monte Carlo paths to simulate
        """
        ...

    def path_quantiles(
        self,
        t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        normalize: bool = True,
        center: bool = True,
    ) -> torch.Tensor:
        """Compute quantiles of cumulative returns at specified time points.

        Args:
            t: Time points at which to compute quantiles (must be sorted)
            lo: Lower quantile bound (e.g., 0.001 for 0.1th percentile)
            hi: Upper quantile bound (e.g., 0.999 for 99.9th percentile)
            size: Number of quantiles to compute between lo and hi
            normalize: If True, normalize quantiles by sqrt(time)
            center: If True, subtract mean before computing quantiles

        Returns:
            torch.Tensor: Shape (len(t), size) containing quantiles at each time point
        """
        ...

    @property
    def parameter_dict(self) -> dict[str, float]:
        """Return model parameters as dictionary.

        Returns:
            Dict mapping parameter names to their values
        """
        ...

    @property
    def model_name(self) -> str:
        """Return model identifier.

        Returns:
            String identifier for this model type (e.g., 'garch', 'rough_heston')
        """
        ...


class MonteCarloModelBase:
    """Base implementation providing common functionality for Monte Carlo models.

    This class provides default implementations and utilities that can be
    inherited by concrete model implementations.
    """

    def __init__(self, device: str | torch.device | None = None):
        """Initialize base model with device configuration.

        Args:
            device: PyTorch device for computations
        """
        self.device = torch.device(device) if device else torch.device("cpu")
        self._eps = 1e-8  # Small epsilon for numerical stability

    def _validate_time_points(
        self, t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """Validate and normalize time points.

        Args:
            t: Time points to validate

        Returns:
            numpy array of validated time points

        Raises:
            ValueError: If time points are invalid
        """
        t = np.asarray(t)

        if len(t) == 0:
            return t

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        return t

    def _validate_quantile_bounds(self, lo: float, hi: float) -> None:
        """Validate quantile bounds.

        Args:
            lo: Lower quantile bound
            hi: Upper quantile bound

        Raises:
            ValueError: If bounds are invalid
        """
        if not (0 <= lo < hi <= 1):
            raise ValueError("Quantile bounds must satisfy 0 <= lo < hi <= 1")

    def _generate_quantile_levels(
        self, lo: float, hi: float, size: int
    ) -> torch.Tensor:
        """Generate quantile levels for computation.

        Args:
            lo: Lower quantile bound
            hi: Upper quantile bound
            size: Number of quantiles

        Returns:
            Tensor of quantile levels
        """
        return torch.linspace(lo, hi, size, device=self.device)

    def _normalize_quantiles(
        self, quantiles: torch.Tensor, time_points: np.ndarray, normalize: bool
    ) -> torch.Tensor:
        """Normalize quantiles by sqrt(time) if requested.

        Args:
            quantiles: Raw quantiles
            time_points: Time points corresponding to quantiles
            normalize: Whether to normalize

        Returns:
            Normalized quantiles
        """
        if normalize:
            time_tensor = torch.tensor(
                time_points, device=self.device, dtype=quantiles.dtype
            )
            return quantiles / torch.sqrt(time_tensor.unsqueeze(1))
        return quantiles
