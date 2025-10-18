"""
Rough Heston model implementation.

This module provides a Rough Heston model implementation that conforms to the MonteCarloModel protocol.
The Rough Heston model extends the classical Heston model by introducing rough volatility paths.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..utils import FractionalBrownianMotion
from .base import MonteCarloModelBase


class RoughHeston(MonteCarloModelBase):
    """
    Rough Heston model simulator implementing MonteCarloModelProtocol.

    The Rough Heston model extends the classical Heston stochastic volatility model
    by introducing rough volatility paths with Hurst parameter H < 1/2.

    Model:
        dS_t = S_t * sqrt(V_t) * dW_t^S
        dV_t = κ(θ - V_t)dt + σ * V_t^α * dW_t^V
        dW_t^S * dW_t^V = ρ dt

    where the volatility process V_t exhibits rough behavior controlled by α.

    This is a mock implementation for the restructuring phase.
    """

    def __init__(
        self,
        *,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        alpha: float = 0.5,
        v0: float = 0.04,
        device: str | torch.device | None = None,
    ):
        """
        Initialize Rough Heston model.

        Parameters
        ----------
        kappa : float, optional
            Mean reversion speed. Default 2.0.
        theta : float, optional
            Long-term variance level. Default 0.04.
        sigma : float, optional
            Volatility of volatility. Default 0.3.
        rho : float, optional
            Correlation between price and volatility. Default -0.7.
        alpha : float, optional
            Roughness parameter (0 < alpha < 1). Default 0.5.
        v0 : float, optional
            Initial variance level. Default 0.04.
        device : str | torch.device | None, optional
            PyTorch device for computations. Default uses best available device.
        """
        super().__init__(device)

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.v0 = float(v0)

        # Validate parameters
        if not (0 < self.alpha < 1):
            raise ValueError("Alpha must be in (0, 1)")
        if not (-1 <= self.rho <= 1):
            raise ValueError("Rho must be in [-1, 1]")
        if self.kappa <= 0:
            raise ValueError("Kappa must be positive")
        if self.theta <= 0:
            raise ValueError("Theta must be positive")
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")
        if self.v0 <= 0:
            raise ValueError("V0 must be positive")

        # Runtime state
        self.ti: int = 0
        self.num_paths: int = 0
        self.variance: torch.Tensor | None = None
        self.cum_returns: torch.Tensor | None = None

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return "rough_heston"

    @property
    def parameter_dict(self) -> dict[str, float]:
        """Return model parameters as dictionary."""
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "alpha": self.alpha,
            "v0": self.v0,
        }

    def reset(self, num_paths: int) -> None:
        """
        Initialize (or reinitialize) the simulator state.

        Parameters
        ----------
        num_paths : int
            Number of independent paths to simulate in parallel.
        """
        self.ti = 0
        self.num_paths = int(num_paths)
        self.variance = torch.full((self.num_paths,), self.v0, device=self.device)
        self.cum_returns = torch.zeros((self.num_paths,), device=self.device)

    def step(self) -> tuple[torch.Tensor, int]:
        """
        Advance the simulation by one time step.

        This is a mock implementation that generates sample paths
        using a simplified discretization scheme.

        Returns
        -------
        cum_returns : torch.Tensor, shape (num_paths,)
            Cumulative returns after this step.
        t : int
            Current time index (1-based).
        """
        self.ti += 1

        # Mock implementation: simplified Euler scheme
        dt = 1.0  # Time step

        # Generate correlated Brownian motions
        dW_S = torch.randn(self.num_paths, device=self.device) * np.sqrt(dt)
        dW_V = self.rho * dW_S + np.sqrt(1 - self.rho**2) * torch.randn(
            self.num_paths, device=self.device
        ) * np.sqrt(dt)

        # Update variance (simplified)
        variance_sqrt = torch.sqrt(torch.clamp(self.variance, min=self._eps))
        dv = (
            self.kappa * (self.theta - self.variance) * dt
            + self.sigma * self.variance**self.alpha * dW_V
        )
        self.variance = torch.clamp(self.variance + dv, min=self._eps)

        # Update returns
        dr = variance_sqrt * dW_S
        self.cum_returns += dr

        return self.cum_returns, self.ti

    def path_quantiles(
        self,
        t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        normalize: bool = True,
        center: bool = True,
    ) -> torch.Tensor:
        """
        Compute quantiles of cumulative returns at specified time points.

        Parameters
        ----------
        t : array-like
            List or array of integers representing time steps (must be sorted).
        lo : float, default 0.001
            Lower quantile bound.
        hi : float, default 0.999
            Upper quantile bound.
        size : int, default 512
            Number of quantiles to compute between lo and hi.
        normalize : bool, default True
            If True, normalize quantiles by sqrt(time).
        center : bool, default True
            If True, subtract the average of cum_returns before computing quantiles.

        Returns
        -------
        torch.Tensor, shape (len(t), size)
            Quantiles at each specified time point.
        """
        # Validate inputs using base class methods
        t = self._validate_time_points(t)
        self._validate_quantile_bounds(lo, hi)

        if len(t) == 0:
            return torch.empty((0, size), device=self.device)

        # Generate quantile levels
        quantile_levels = self._generate_quantile_levels(lo, hi, size)

        # Reset to beginning
        self.reset(self.num_paths)

        # Initialize result tensor
        result = torch.empty((len(t), size), device=self.device)

        # Simulate up to each time point and compute quantiles
        for i, target_t in enumerate(t):
            # Simulate from current time to target time
            while self.ti < target_t:
                self.step()

            # Compute quantiles for current cumulative returns
            cum_returns_for_quantiles = self.cum_returns
            if center:
                cum_returns_for_quantiles = self.cum_returns - self.cum_returns.mean()

            # Compute raw quantiles
            raw_quantiles = torch.quantile(cum_returns_for_quantiles, quantile_levels)

            # Normalize if requested
            result[i] = self._normalize_quantiles(
                raw_quantiles, np.array([target_t]), normalize
            )[0]

        return result
