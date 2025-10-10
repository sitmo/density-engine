from typing import List, Tuple, Union

import numpy as np
import torch

from .skew_student_t import HansenSkewedT_torch


class GJRGARCHReduced_torch:
    """
    Shape-only (normalized) GJR–GARCH(1,1) simulator.

    Model (normalized):
        r'_t = ε'_t,    ε'_t = σ'_t z_t,   z_t ~ S_t_{ν,λ}(0,1)
        (σ'_{t+1})^2 = ω' + (α + γ * I[ε'_t < 0]) * (ε'_t)^2 + β * (σ'_t)^2

    with:
        κ = α + β + γ * P0,      P0 = E[z^2 | z < 0]
        ω' = 1 - κ
        Initial variance (σ'_0)^2 = sigma0_sq  (typically 1.0 in the normalized model)

    Notes
    -----
    - This is the “reduced / shape-only” engine: mean = 0, long-run variance = 1.
    - It advances variance from t -> t+1 using the shock realized at time t.
    """

    def __init__(
        self,
        *,
        alpha: float,
        gamma: float,
        beta: float,
        sigma0_sq: float = 1.0,
        dist: HansenSkewedT_torch,
    ):
        """
        Parameters
        ----------
        alpha, gamma, beta : float
            GJR–GARCH coefficients.
        sigma0_sq : float, optional
            Initial normalized variance (σ'_0)^2. Default 1.0.
        dist : HansenSkewedT_torch
            Shock distribution with mean 0, var 1, and method to obtain P0.
        """
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.sigma0_sq = float(sigma0_sq)
        self.dist = dist

        self.device = dist.device
        self._eps = 1e-8

        P0 = dist.second_moment_left()  # E[z^2 | z<0]
        self.P0 = P0
        self.kappa = self.alpha + self.beta + self.gamma * P0
        if not (self.kappa < 1.0):
            raise ValueError(f"Stationarity violated: kappa={self.kappa} >= 1.")
        self.omega = 1.0 - self.kappa  # ω' in normalized model

        # Runtime state
        self.ti: int = 0
        self.num_paths: int = 0
        self.var: torch.Tensor | None = None  # (σ'_t)^2
        self.cum_returns: torch.Tensor | None = None  # sum of r'_t

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
        sigma0_sq = max(self.sigma0_sq, self._eps)
        self.var = torch.full((self.num_paths,), sigma0_sq, device=self.device)
        self.cum_returns = torch.zeros((self.num_paths,), device=self.device)

    def step(self) -> tuple[torch.Tensor, int]:
        """
        Advance the simulation by one time step.

        Returns
        -------
        cum_returns : torch.Tensor, shape (num_paths,)
            Cumulative normalized returns Σ_{s=1}^t r'_s after this step.
        t : int
            Current time index (1-based).
        """
        self.ti += 1

        # Draw standardized shocks z_t and form ε'_t
        z = self.dist.rvs(self.num_paths)  # shape (num_paths,)
        shock = z * torch.sqrt(self.var)  # ε'_t

        # Update cumulative return r'
        self.cum_returns += shock

        # Update variance to (t+1) using ε'_t
        is_negative = (shock < 0).to(self.var.dtype)
        self.var = (
            self.omega
            + (self.alpha + self.gamma * is_negative) * shock.pow(2)
            + self.beta * self.var
        )
        self.var = torch.clamp(self.var, min=self._eps)

        return self.cum_returns, self.ti

    def path(
        self, t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        Simulate paths up to specified time points and return cumulative returns.

        Parameters
        ----------
        t : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return cumulative returns after 4, 10, and 12 steps.

        Returns
        -------
        torch.Tensor, shape (len(t), num_paths)
            Cumulative returns at each specified time point.
            First row contains cum_returns after t[0] steps, etc.
        """
        import numpy as np

        t = np.asarray(t)
        if len(t) == 0:
            return torch.empty((0, self.num_paths), device=self.device)

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        # Reset to beginning
        self.reset(self.num_paths)

        # Initialize result tensor
        result = torch.empty((len(t), self.num_paths), device=self.device)

        # Simulate up to each time point
        for i, target_t in enumerate(t):
            # Simulate from current time to target time
            while self.ti < target_t:
                self.step()

            # Store cumulative returns at this time point
            result[i] = self.cum_returns

        return result

    def path_quantiles(
        self,
        t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute quantiles of cumulative returns at specified time points.

        Parameters
        ----------
        t : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return quantiles after 4, 10, and 12 steps.
        lo : float, default 0.001
            Lower quantile bound (e.g., 0.001 for 0.1th percentile).
        hi : float, default 0.999
            Upper quantile bound (e.g., 0.999 for 99.9th percentile).
        size : int, default 512
            Number of quantiles to compute between lo and hi.

        Returns
        -------
        torch.Tensor, shape (len(t), size)
            Quantiles at each specified time point.
            First row contains quantiles after t[0] steps, etc.
        """
        import numpy as np

        t = np.asarray(t)
        if len(t) == 0:
            return torch.empty((0, size), device=self.device)

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        # Validate quantile bounds
        if not (0 <= lo < hi <= 1):
            raise ValueError("Quantile bounds must satisfy 0 <= lo < hi <= 1")

        # Generate quantile levels
        quantile_levels = torch.linspace(lo, hi, size, device=self.device)

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
            if normalize:
                result[i] = torch.quantile(
                    self.cum_returns, quantile_levels
                ) / torch.sqrt(torch.tensor(target_t, device=self.device))
            else:
                result[i] = torch.quantile(self.cum_returns, quantile_levels)

        return result


class GJRGARCH_torch:
    """
    Standard (raw) GJR-GARCH(1,1) simulator built on the reduced engine.

    Raw model:
        r_t = μ + ε_t,    ε_t = σ_t z_t
        σ_{t+1}^2 = ω + α ε_t^2 + γ ε_t^2 I[ε_t < 0] + β σ_t^2

    Mapping to/from reduced (shape-only) model:
        κ     = α + β + γ * P0,      P0 = E[z^2 | z<0]
        v     = ω / (1 - κ)                             (long-run var)
        (σ'_0)^2 = σ_0^2 / v
        ω'    = 1 - κ
        r_t   = μ + √v * r'_t
        σ_t^2 = v * (σ'_t)^2
    """

    def __init__(
        self,
        *,
        mu: float,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
        sigma0_sq: float,
        dist: HansenSkewedT_torch,
        reduced_engine_cls: type = GJRGARCHReduced_torch,
    ):
        """
        Parameters
        ----------
        mu, omega, alpha, gamma, beta : float
            Raw GJR–GARCH parameters.
        sigma0_sq : float
            Initial raw variance σ_0^2.
        dist : object
            Shock distribution with .device and method for P0 = E[z^2 | z<0].
        reduced_engine_cls : type
            Class implementing the reduced model (default: GJRGARCHReduced_torch).
        """
        self.mu = float(mu)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.sigma0_sq = float(sigma0_sq)
        self.dist = dist

        P0 = dist.second_moment_left()
        self.kappa = self.alpha + self.beta + self.gamma * P0
        if not (self.kappa < 1.0):
            raise ValueError(f"Stationarity violated: kappa={self.kappa} >= 1.")

        # Long-run variance v
        self.v = self.omega / (1.0 - self.kappa)
        if not (self.v > 0.0):
            raise ValueError(f"Long-run variance must be positive; got v={self.v}.")
        self.sqrt_v = float(self.v**0.5)

        self.device = dist.device
        self._eps = 1e-8

        # Build reduced engine with normalized initial variance
        sigma0_sq_reduced = max(self.sigma0_sq / self.v, self._eps)
        self.reduced = reduced_engine_cls(
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            sigma0_sq=sigma0_sq_reduced,
            dist=self.dist,
        )

        # Raw state mirrors reduced state (but mapped back to raw units)
        self.ti: int = 0
        self.num_paths: int = 0
        self.var: torch.Tensor | None = None  # raw σ_t^2
        self.cum_returns: torch.Tensor | None = None  # raw cumulative returns Σ r_t

    def reset(self, num_paths: int) -> None:
        """
        Initialize (or reinitialize) the simulator state in raw units.

        Parameters
        ----------
        num_paths : int
            Number of independent paths to simulate in parallel.
        """
        self.reduced.reset(num_paths)
        self.ti = 0
        self.num_paths = int(num_paths)
        # Map reduced (σ'_0)^2 -> raw σ_0^2
        self.var = self.reduced.var * self.v
        self.cum_returns = torch.zeros((self.num_paths,), device=self.device)

    def step(self) -> tuple[torch.Tensor, int]:
        """
        Advance the simulation by one time step.

        Returns
        -------
        cum_returns : torch.Tensor, shape (num_paths,)
            Raw cumulative returns Σ_{s=1}^t r_s after this step.
        t : int
            Current time index (1-based).
        """
        cum_ret_reduced, ti = self.reduced.step()  # Σ r'_s
        self.ti = ti

        # Map cumulative returns to raw units: Σ r_s = t * μ + √v * Σ r'_s
        self.cum_returns = self.mu * self.ti + self.sqrt_v * cum_ret_reduced

        # Map variance to raw units: σ_t^2 = v * (σ'_t)^2
        self.var = self.reduced.var * self.v

        return self.cum_returns, self.ti

    def path(
        self, t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """
        Simulate paths up to specified time points and return cumulative returns.

        Parameters
        ----------
        t : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return cumulative returns after 4, 10, and 12 steps.

        Returns
        -------
        torch.Tensor, shape (len(t), num_paths)
            Cumulative returns at each specified time point.
            First row contains cum_returns after t[0] steps, etc.
        """
        import numpy as np

        t = np.asarray(t)
        if len(t) == 0:
            return torch.empty((0, self.num_paths), device=self.device)

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        # Reset to beginning
        self.reset(self.num_paths)

        # Initialize result tensor
        result = torch.empty((len(t), self.num_paths), device=self.device)

        # Simulate up to each time point
        for i, target_t in enumerate(t):
            # Simulate from current time to target time
            while self.ti < target_t:
                self.step()

            # Store cumulative returns at this time point
            result[i] = self.cum_returns

        return result

    def path_quantiles(
        self,
        t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute quantiles of cumulative returns at specified time points.

        Parameters
        ----------
        t : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return quantiles after 4, 10, and 12 steps.
        lo : float, default 0.001
            Lower quantile bound (e.g., 0.001 for 0.1th percentile).
        hi : float, default 0.999
            Upper quantile bound (e.g., 0.999 for 99.9th percentile).
        size : int, default 512
            Number of quantiles to compute between lo and hi.

        Returns
        -------
        torch.Tensor, shape (len(t), size)
            Quantiles at each specified time point.
            First row contains quantiles after t[0] steps, etc.
        """
        import numpy as np

        t = np.asarray(t)
        if len(t) == 0:
            return torch.empty((0, size), device=self.device)

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        # Validate quantile bounds
        if not (0 <= lo < hi <= 1):
            raise ValueError("Quantile bounds must satisfy 0 <= lo < hi <= 1")

        # Generate quantile levels
        quantile_levels = torch.linspace(lo, hi, size, device=self.device)

        # Store original num_paths to restore later
        original_num_paths = self.num_paths

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
            if normalize:
                result[i] = torch.quantile(
                    self.cum_returns, quantile_levels
                ) / torch.sqrt(torch.tensor(target_t, device=self.device))
            else:
                result[i] = torch.quantile(self.cum_returns, quantile_levels)

        return result
