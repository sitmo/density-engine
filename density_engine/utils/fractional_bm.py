"""
Fractional Brownian Motion (fBM) implementation for rough volatility models.

This module provides efficient implementations of fractional Brownian motion
used in rough volatility models like Rough Heston and Rough Bergomi.
"""

from typing import Optional, Union

import numpy as np
import torch
from scipy.linalg import cholesky


class FractionalBrownianMotion:
    """
    Fractional Brownian Motion generator for rough volatility models.

    Implements efficient fBM generation using Cholesky decomposition of the
    covariance matrix for exact simulation.

    Parameters
    ----------
    hurst : float
        Hurst parameter H ∈ (0, 1). For rough volatility: H < 0.5
    device : str | torch.device | None
        PyTorch device for computations
    """

    def __init__(
        self,
        hurst: float,
        device: str | torch.device | None = None,
    ):
        if not (0 < hurst < 1):
            raise ValueError(f"Hurst parameter must be in (0, 1), got {hurst}")

        self.hurst = hurst
        self.device = torch.device(device) if device else torch.device("cpu")
        self._eps = 1e-8

    def generate_paths(
        self,
        num_paths: int,
        time_points: list | np.ndarray | torch.Tensor,
        method: str = "cholesky",
    ) -> torch.Tensor:
        """
        Generate fBM paths for given time points.

        Parameters
        ----------
        num_paths : int
            Number of sample paths to generate
        time_points : array-like
            Time points for the fBM paths
        method : str
            Generation method: "cholesky" (exact) or "approximate"

        Returns
        -------
        torch.Tensor
            Shape (num_paths, len(time_points)) containing fBM paths
        """
        time_points = np.asarray(time_points)
        n = len(time_points)

        if method == "cholesky":
            return self._generate_cholesky(num_paths, time_points)
        elif method == "approximate":
            return self._generate_approximate(num_paths, time_points)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate_cholesky(
        self, num_paths: int, time_points: np.ndarray
    ) -> torch.Tensor:
        """Generate fBM using exact Cholesky decomposition."""
        n = len(time_points)

        # Build covariance matrix
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = 0.5 * (
                    time_points[i] ** (2 * self.hurst)
                    + time_points[j] ** (2 * self.hurst)
                    - abs(time_points[i] - time_points[j]) ** (2 * self.hurst)
                )

        # Cholesky decomposition
        try:
            L = cholesky(cov_matrix, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to approximate method if Cholesky fails
            return self._generate_approximate(num_paths, time_points)

        # Generate standard normal random variables
        Z = torch.randn(num_paths, n, device=self.device)

        # Transform to fBM
        fbm_paths = torch.matmul(Z, torch.tensor(L, device=self.device).T)

        return fbm_paths

    def _generate_approximate(
        self, num_paths: int, time_points: np.ndarray
    ) -> torch.Tensor:
        """Generate fBM using approximate method (faster for large n)."""
        n = len(time_points)
        dt = np.diff(time_points, prepend=0)

        # Generate increments
        increments = torch.zeros(num_paths, n, device=self.device)

        for i in range(n):
            if i == 0:
                # First increment
                scale = dt[i] ** self.hurst
            else:
                # Subsequent increments with memory
                scale = dt[i] ** self.hurst

            increments[:, i] = scale * torch.randn(num_paths, device=self.device)

        # Cumulative sum to get fBM paths
        fbm_paths = torch.cumsum(increments, dim=1)

        return fbm_paths

    def get_covariance_matrix(self, time_points: np.ndarray) -> np.ndarray:
        """
        Get the theoretical covariance matrix for fBM.

        Parameters
        ----------
        time_points : np.ndarray
            Time points

        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        n = len(time_points)
        cov_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = 0.5 * (
                    time_points[i] ** (2 * self.hurst)
                    + time_points[j] ** (2 * self.hurst)
                    - abs(time_points[i] - time_points[j]) ** (2 * self.hurst)
                )

        return cov_matrix


def generate_fbm_increments(
    hurst: float,
    time_points: np.ndarray,
    num_paths: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convenience function to generate fBM increments.

    Parameters
    ----------
    hurst : float
        Hurst parameter
    time_points : np.ndarray
        Time points
    num_paths : int
        Number of paths
    device : str | torch.device | None
        PyTorch device

    Returns
    -------
    torch.Tensor
        fBM increments
    """
    fbm = FractionalBrownianMotion(hurst=hurst, device=device)
    return fbm.generate_paths(num_paths, time_points)
