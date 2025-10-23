"""
Custom learning rate schedulers for PyTorch optimizers.

All schedulers are sample-aware and PyTorch-compliant, supporting linear warmup
followed by various decay strategies.
"""

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupExpDecayScheduler(_LRScheduler):
    """PyTorch-compliant scheduler with linear warmup and step exponential decay (sample-aware)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lrs: float,
        lre: float,
        lrd: float,
        lrds: int,
        warmup_samples: int,
        last_epoch: int = -1,
    ):
        self.lrs = float(lrs)
        self.lre = float(lre)
        self.lrd = float(lrd)
        self.lrds = int(max(1, lrds))
        self.warmup_samples = int(max(0, warmup_samples))
        self.cumulative_samples = 0
        super().__init__(optimizer, last_epoch)

    def lr_at(self, samples_processed: int) -> float:
        samples = max(0, int(samples_processed))
        if self.warmup_samples > 0 and samples < self.warmup_samples:
            return max(0.0, self.lrs * (samples / self.warmup_samples))
        samples_after_warmup = max(0, samples - self.warmup_samples)
        decay_steps = samples_after_warmup // self.lrds
        lr = self.lrs * (self.lrd**decay_steps)
        return max(self.lre, float(lr))

    def get_lr(self) -> list[float]:
        lr = self.lr_at(self.cumulative_samples)
        return [lr for _ in self.optimizer.param_groups]

    def step(self, samples_in_batch: int | None = None) -> None:
        if samples_in_batch is not None:
            self.cumulative_samples += int(samples_in_batch)
        # Update learning rates directly - no need to call super().step()
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LinearWarmupCosineDecayScheduler(_LRScheduler):
    """Sample-aware LR scheduler with linear warmup and cosine decay.

    - Warmup: linearly increase from 0 to lrs over `warmup_samples` samples.
    - Cosine decay: after warmup, decay smoothly from lrs -> lre over `decay_samples`.
      LR is clamped at `lre` after the decay window.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lrs: float,
        lre: float,
        decay_samples: int,
        warmup_samples: int,
        last_epoch: int = -1,
    ):
        self.lrs = float(lrs)
        self.lre = float(lre)
        self.decay_samples = int(max(1, decay_samples))
        self.warmup_samples = int(max(0, warmup_samples))
        self.cumulative_samples = 0
        super().__init__(optimizer, last_epoch)

    def lr_at(self, samples_processed: int) -> float:
        s = max(0, int(samples_processed))

        # Linear warmup 0 -> lrs
        if self.warmup_samples > 0 and s < self.warmup_samples:
            return self.lrs * (s / self.warmup_samples)

        # Cosine decay lrs -> lre over decay_samples
        t = s - self.warmup_samples
        if t >= self.decay_samples:
            return self.lre

        # progress in [0,1]
        p = t / self.decay_samples
        # half-cosine from 1 -> 0
        cos_part = 0.5 * (1.0 + math.cos(math.pi * p))
        lr = self.lre + (self.lrs - self.lre) * cos_part
        return float(lr)

    def get_lr(self) -> list[float]:
        lr = self.lr_at(self.cumulative_samples)
        return [lr for _ in self.optimizer.param_groups]

    def step(self, samples_in_batch: int | None = None) -> None:
        if samples_in_batch is not None:
            self.cumulative_samples += int(samples_in_batch)
        # Update learning rates directly - no need to call super().step()
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LinearWarmupPowerLawScheduler(_LRScheduler):
    """PyTorch-compliant scheduler: linear warmup to lrs, then power-law decay.

    Modes after warmup (total decay window = total_samples):
      A) Given alpha: lre is implied as lrs * (1 + total_samples/s0)^alpha
      B) Given lre: alpha is implied to reach lre at total_samples.

    lr(t) = lre + (lrs - lre) * (1 + t/s0)^alpha, with t in [0, total_samples]
    where alpha is negative (e.g., -0.5 for 1/sqrt(t) decay)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lrs: float,
        lre: float | None,
        s0: int,
        warmup_samples: int,
        total_samples: int,
        alpha: float | None = None,
        last_epoch: int = -1,
    ):
        assert total_samples > 0 and s0 > 0 and warmup_samples >= 0
        self.lrs = float(lrs)
        self.s0 = int(s0)
        self.warmup_samples = int(warmup_samples)
        self.total_samples = int(total_samples)
        if alpha is not None:
            self.alpha = float(alpha)
            # Ensure alpha is negative for decay (e.g., -0.5 for 1/sqrt(t))
            if self.alpha > 0:
                self.alpha = -self.alpha
            # Compute implied lre based on total_samples
            self.lre = self.lrs * (1.0 + self.total_samples / self.s0) ** self.alpha
        else:
            # Use provided lre and compute alpha
            assert lre is not None, "Either alpha or lre must be provided"
            self.lre = float(lre)
            ratio = max(1e-12, self.lrs / max(1e-12, self.lre))
            self.alpha = math.log(ratio) / math.log(1.0 + self.total_samples / self.s0)
            # Ensure alpha is negative for decay
            if self.alpha > 0:
                self.alpha = -self.alpha
        self.cumulative_samples = 0
        super().__init__(optimizer, last_epoch)

    def lr_at(self, samples_processed: int) -> float:
        s = max(0, int(samples_processed))
        if self.warmup_samples > 0 and s < self.warmup_samples:
            return self.lrs * (s / self.warmup_samples)
        # Power-law decay over total_samples window
        t = min(max(0, s - self.warmup_samples), self.total_samples)
        lr = self.lre + (self.lrs - self.lre) * (1.0 + (t / self.s0)) ** self.alpha
        return float(max(self.lre, float(lr)))

    def get_lr(self) -> list[float]:
        lr = self.lr_at(self.cumulative_samples)
        return [lr for _ in self.optimizer.param_groups]

    def step(self, samples_in_batch: int | None = None) -> None:
        if samples_in_batch is not None:
            self.cumulative_samples += int(samples_in_batch)
        # Update learning rates directly - no need to call super().step()
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CooldownSchedulerWrapper(_LRScheduler):
    """Wrapper scheduler that adds a cooldown phase to any base scheduler.

    During normal training, delegates to the base scheduler. When the cooldown
    phase begins (last cooldown_samples), returns a fixed cooldown_lr.

    Args:
        base_scheduler: The underlying scheduler (exp/cosine/power)
        cooldown_samples: Number of samples for cooldown phase
        cooldown_lr: Fixed learning rate during cooldown (if None, computed dynamically)
        cooldown_lr_factor: Factor to apply to LR when cooldown starts (used if cooldown_lr is None)
        total_samples: Total training samples (to calculate cooldown start)
    """

    def __init__(
        self,
        base_scheduler: _LRScheduler,
        cooldown_samples: int,
        cooldown_lr: float | None,
        total_samples: int,
        cooldown_lr_factor: float = 0.1,
    ):
        self.base_scheduler = base_scheduler
        self.cooldown_samples = int(cooldown_samples)
        self.cooldown_lr = float(cooldown_lr) if cooldown_lr is not None else None
        self.cooldown_lr_factor = float(cooldown_lr_factor)
        self.total_samples = int(total_samples)
        self.cooldown_start = self.total_samples - self.cooldown_samples
        self._cooldown_lr_computed = False
        # Don't call super().__init__ - we delegate to base_scheduler
        self.optimizer = base_scheduler.optimizer

    def lr_at(self, samples_processed: int) -> float:
        """Return cooldown LR if in cooldown phase, otherwise delegate to base."""
        s = max(0, int(samples_processed))
        if s >= self.cooldown_start:
            # Compute cooldown LR dynamically on first cooldown step
            if not self._cooldown_lr_computed and self.cooldown_lr is None:
                # Query the actual LR at the cooldown start point
                if hasattr(self.base_scheduler, "lr_at"):
                    actual_lr_at_cooldown_start = self.base_scheduler.lr_at(
                        self.cooldown_start
                    )
                else:
                    # Fallback: use current LR from optimizer
                    actual_lr_at_cooldown_start = self.optimizer.param_groups[0]["lr"]
                self.cooldown_lr = actual_lr_at_cooldown_start * self.cooldown_lr_factor
                self._cooldown_lr_computed = True
                print(f"Cooldown phase started at sample {self.cooldown_start:,}")
                print(f"  LR at cooldown start: {actual_lr_at_cooldown_start:.2e}")
                print(
                    f"  Cooldown LR (factor {self.cooldown_lr_factor}): {self.cooldown_lr:.2e}"
                )
            return float(self.cooldown_lr)
        if hasattr(self.base_scheduler, "lr_at"):
            return float(self.base_scheduler.lr_at(s))
        else:
            # Fallback: return current LR from optimizer
            return float(self.optimizer.param_groups[0]["lr"])

    def get_lr(self) -> list[float]:
        """Get current learning rate(s)."""
        lr = self.lr_at(getattr(self.base_scheduler, "cumulative_samples", 0))
        return [lr for _ in self.optimizer.param_groups]

    def step(self, samples_in_batch: int | None = None) -> None:
        """Step the scheduler by the given number of samples."""
        # Delegate stepping to base scheduler
        self.base_scheduler.step(samples_in_batch)
        # Update learning rates with cooldown-aware logic
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
