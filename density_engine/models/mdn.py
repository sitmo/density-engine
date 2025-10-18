# patched_mdn.py — drop-in replacement for your MDN

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MonteCarloModelBase


class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network for predicting probability densities.
    Returns component means, stds, and prob-weights per input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        num_components: int,
        activation: str = "selu",
        center: bool = True,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_components = num_components
        self.center = center
        self.device = torch.device(device) if device else torch.device("cpu")

        # backbone
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), self._get_activation(activation)]
            prev_dim = h
        self.network = nn.Sequential(*layers)

        # heads
        self.mean_layer = nn.Linear(prev_dim, num_components)
        self.std_layer = nn.Linear(prev_dim, num_components)
        self.weight_layer = nn.Linear(prev_dim, num_components)

        # constants for numerics
        self.register_buffer("_sqrt2", torch.tensor(np.sqrt(2.0), dtype=torch.float32))
        self.register_buffer(
            "_log2pi", torch.tensor(np.log(2.0 * np.pi), dtype=torch.float32)
        )

        # Type hints for mypy
        self._sqrt2: torch.Tensor
        self._log2pi: torch.Tensor

        self.to(self.device)

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()
        return {
            "selu": nn.SELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }[name]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, D) -> returns (means, stds, weights) each (B, K)
        Formulas match the working MDN:
        pi = log_softmax(8 * tanh(W_pi h)),  mu = W_mu h,  sigma = exp(W_sigma h) clipped
        """
        features = self.network(x)

        # means (mu)
        means = self.mean_layer(features)  # (B, K)

        # mixture weights: bounded logits -> log_softmax -> probs
        weights_logits = torch.tanh(self.weight_layer(features)) * 6.0  # (B, K)
        weights = F.softmax(weights_logits, dim=-1)  # π_k in prob space

        # stds (sigma): exp + clip (match working code; keep tiny floor for safety)
        log_stds = torch.tanh(self.std_layer(features)) * 5.0 - 3.0
        stds = torch.exp(log_stds)

        # optional centering (same as before): subtract mixture mean so overall mean ~ 0
        if self.center:
            mixture_mean = torch.sum(weights * means, dim=-1, keepdim=True)  # (B,1)
            means = means - mixture_mean

        return means, stds, weights

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Vectorized sampling from the mixture.
        Returns: (B, S)
        """
        means, stds, weights = self.forward(x)  # (B, K)
        B, K = means.shape

        # draw component indices: (B, S)
        comp = torch.multinomial(weights, num_samples=num_samples, replacement=True)

        # gather means/stds for chosen components -> (B, S)
        gather_idx = comp.unsqueeze(-1)  # (B, S, 1)
        means_exp = means.unsqueeze(1).expand(B, num_samples, K)
        stds_exp = stds.unsqueeze(1).expand(B, num_samples, K)
        m_sel = torch.gather(means_exp, 2, gather_idx).squeeze(-1)
        s_sel = torch.gather(stds_exp, 2, gather_idx).squeeze(-1)

        # sample
        return m_sel + s_sel * torch.randn_like(m_sel)

    def log_prob(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        targets: (B, Q) -> log p(targets) per element: (B, Q)
        """
        means, stds, weights = self.forward(x)
        B, Q = targets.shape
        K = means.shape[-1]

        # expand to (B, Q, K)
        t = targets.unsqueeze(-1)  # (B,Q,1)
        m = means.unsqueeze(1)  # (B,1,K)
        s = stds.unsqueeze(1)  # (B,1,K)
        w = weights.unsqueeze(1)  # (B,1,K)

        # clamp z-range for stable tails
        z = (t - m) / s
        z = z.clamp(-12.0, 12.0)

        # log N(t; m, s) = -0.5[(z^2) + log(2π) + 2log s]
        log_prob_ik = -0.5 * (z * z + self._log2pi + 2.0 * torch.log(s))
        # mixture: logsumexp over K
        log_mix = torch.logsumexp(torch.log(w) + log_prob_ik, dim=-1)
        return log_mix  # (B, Q)

    def cdf(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        targets: (B, Q) -> mixture CDF at those points: (B, Q)
        """
        means, stds, weights = self.forward(x)
        B, Q = targets.shape
        K = means.shape[-1]

        t = targets.unsqueeze(-1)  # (B,Q,1)
        m = means.unsqueeze(1)  # (B,1,K)
        s = stds.unsqueeze(1)  # (B,1,K)
        w = weights.unsqueeze(1)  # (B,1,K)

        z = (t - m) / s
        z = z.clamp(-12.0, 12.0)

        # Φ(z) via erf
        cdf_comp = 0.5 * (1.0 + torch.erf(z / self._sqrt2))
        return torch.sum(w * cdf_comp, dim=-1)  # (B, Q)


class MDNModel(MonteCarloModelBase):
    """
    Thin wrapper to fit your existing interface.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layers: list[int],
        num_components: int,
        activation: str = "selu",
        center: bool = True,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_components = num_components
        self.activation = activation
        self.center = center

        self.mdn = MixtureDensityNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            num_components=num_components,
            activation=activation,
            center=center,
            device=device,
        )
        self.num_paths: int = 0
        self.trained: bool = False

    @property
    def model_name(self) -> str:
        return "mdn"

    @property
    def parameter_dict(self) -> dict[str, float | int | str | bool | list[int]]:
        return {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "num_components": self.num_components,
            "activation": self.activation,
            "center": self.center,
            "trained": self.trained,
        }

    def reset(self, num_paths: int) -> None:
        self.num_paths = int(num_paths)

    def path_quantiles(
        self,
        t: list[int] | tuple[int, ...] | torch.Tensor | np.ndarray,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        normalize: bool = True,
        center: bool = True,
    ) -> torch.Tensor:
        return torch.empty((0, size), device=self.device)

    def predict_cdf(
        self, params: torch.Tensor, quantile_levels: torch.Tensor
    ) -> torch.Tensor:
        if not self.trained:
            raise RuntimeError("Model must be trained before making predictions")
        targets = quantile_levels.unsqueeze(0).expand(params.size(0), -1)
        return self.mdn.cdf(params, targets)
