"""Small procedural scenes used by CPU tests.

The scene is a static coloured latent point set. ``render`` projects those world points
into a target latent grid and returns the frontmost feature per cell. This gives tests a
deterministic world without depending on real datasets or a diffusion backbone.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.projection import project_to_latent_grid


@dataclass
class SyntheticView:
    latent: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor


@dataclass
class SyntheticScene:
    points: torch.Tensor
    feats: torch.Tensor

    @classmethod
    def random_grid(
        cls,
        n_side: int,
        spacing: float,
        depth0: float,
        channels: int,
        seed: int = 0,
        depth_jitter: float = 0.15,
    ) -> "SyntheticScene":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        coords = (torch.arange(n_side, dtype=torch.float32) - (n_side - 1) / 2.0) * float(spacing)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        z = torch.full_like(xx, float(depth0))
        if depth_jitter:
            z = z + depth_jitter * torch.sin(xx * 1.7) * torch.cos(yy * 1.3)
        points = torch.stack([xx.reshape(-1), yy.reshape(-1), z.reshape(-1)], dim=-1)

        rand = torch.randn(points.shape[0], channels, generator=gen)
        basis = torch.stack(
            [
                torch.sin(points[:, 0] * 1.3),
                torch.cos(points[:, 1] * 1.1),
                torch.sin(points[:, 2] * 0.7),
                points[:, 0],
                points[:, 1],
                points[:, 2],
            ],
            dim=-1,
        )
        repeat = (channels + basis.shape[1] - 1) // basis.shape[1]
        structured = basis.repeat(1, repeat)[:, :channels]
        feats = 0.7 * structured + 0.3 * rand
        return cls(points=points.contiguous(), feats=feats.contiguous())

    @torch.no_grad()
    def render(self, E_t: torch.Tensor, K_l: torch.Tensor, h: int, w: int) -> SyntheticView:
        device = E_t.device
        dtype = E_t.dtype
        points = self.points.to(device=device, dtype=dtype)
        feats = self.feats.to(device=device, dtype=dtype)
        cu, cv, z, in_front, in_bounds = project_to_latent_grid(points, E_t, K_l, h, w)
        keep = in_front & in_bounds & torch.isfinite(z) & (z > 0)

        latent = torch.zeros(feats.shape[1], h, w, dtype=dtype, device=device)
        depth = torch.full((h, w), float("inf"), dtype=dtype, device=device)
        valid = torch.zeros(h, w, dtype=torch.bool, device=device)
        if not keep.any():
            return SyntheticView(latent=latent, depth=depth, valid=valid)

        kept = torch.nonzero(keep, as_tuple=False).flatten()
        cell = cv[kept] * w + cu[kept]
        order_depth = torch.argsort(z[kept], stable=True)
        order_cell = torch.argsort(cell[order_depth], stable=True)
        order = order_depth[order_cell]
        sorted_cell = cell[order]
        first = torch.ones_like(sorted_cell, dtype=torch.bool)
        if sorted_cell.numel() > 1:
            first[1:] = sorted_cell[1:] != sorted_cell[:-1]
        chosen = kept[order[first]]
        u = cu[chosen]
        v = cv[chosen]
        latent[:, v, u] = feats[chosen].transpose(0, 1)
        depth[v, u] = z[chosen]
        valid[v, u] = True
        return SyntheticView(latent=latent, depth=depth, valid=valid)
