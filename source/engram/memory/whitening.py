"""Per-channel running whitening of the VAE latent manifold (Engram Sec 3.4).

Wan/Mirage VAE channels have very different scales; an unweighted or precision-weighted
average in raw latent space is dominated by high-variance channels and can drift off the
decoder's well-trained manifold. We standardise each channel with running population
statistics (Welford) before fusion and invert the transform at readout. The transform is
affine and per-channel, so it commutes with the inverse-variance weighted mean used by
the voxel store.
"""
from __future__ import annotations

import torch


class RunningWhitener:
    def __init__(self, channels: int, eps: float = 1e-5, device="cpu", dtype=torch.float32, momentum: float | None = None):
        self.C = channels
        self.eps = eps
        self.momentum = momentum  # None => exact running (Welford); else EMA
        self.mean = torch.zeros(channels, dtype=dtype, device=device)
        self.M2 = torch.zeros(channels, dtype=dtype, device=device)
        self.var = torch.ones(channels, dtype=dtype, device=device)
        self.count = 0
        self.frozen = False

    def set_stats(self, mean: torch.Tensor, var: torch.Tensor) -> "RunningWhitener":
        """Install precomputed (global) statistics, e.g. dataset-level VAE latent stats."""
        self.mean = mean.to(self.mean)
        self.var = var.to(self.var)
        return self

    def freeze(self) -> "RunningWhitener":
        """Freeze statistics. Required at inference so whiten/unwhiten are exact inverses
        (a drifting whitener would make readout features inconsistent with stored ones)."""
        self.frozen = True
        return self

    def unfreeze(self) -> "RunningWhitener":
        self.frozen = False
        return self

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        """feats: (N, C) raw latent features."""
        if feats.numel() == 0 or self.frozen:
            return
        n = feats.shape[0]
        batch_mean = feats.mean(dim=0)
        batch_var = feats.var(dim=0, unbiased=False)
        if self.momentum is not None:
            m = self.momentum
            self.mean = (1 - m) * self.mean + m * batch_mean
            self.var = (1 - m) * self.var + m * batch_var
            self.count += n
            return
        # exact parallel Welford merge
        new_count = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / new_count)
        self.M2 = self.M2 + batch_var * n + (delta * delta) * (self.count * n / new_count)
        self.count = new_count
        self.var = self.M2 / max(self.count, 1)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps)

    @torch.no_grad()
    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise along the last dimension (size C)."""
        return (x - self.mean) / self.std()

    @torch.no_grad()
    def unwhiten(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std() + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "M2": self.M2, "count": self.count}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]
        self.var = sd["var"]
        self.M2 = sd["M2"]
        self.count = sd["count"]
