"""Depth handling for cache construction.

Two operations:
  * ``downsample_depth_to_latent`` reproduces Mirage's requirement that the dense depth
    map and the latent grid share a resolution (bilinear by default, the lowest-hole
    variant reported in Mirage Table 5).
  * ``top_m_depth_modes`` implements Engram's multi-hypothesis lift (Sec 3.3, H6 fix):
    instead of collapsing a 16x16 pixel block into a single depth, it clusters the
    block's depths into up to ``m`` modes so thin structures / occlusion boundaries
    land in multiple voxels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def resample_depth_to_latent_camera(depth, valid_mask, source_intrinsics, latent_intrinsics, height, width, mode="bilinear"):
    """Map OpenCV integer-centred depth pixels to Engram half-centred latent rays.

    Cropping is represented by the source intrinsics. A sample is valid only
    when its complete interpolation support is valid, so masked zero depths
    cannot turn into spurious near surfaces during resizing.
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError("Camera-aware depth sampling supports bilinear or nearest")
    depth = torch.as_tensor(depth).detach().float().cpu()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu() & torch.isfinite(depth) & (depth > 0)
    source_intrinsics = torch.as_tensor(source_intrinsics).detach().float().cpu()
    latent_intrinsics = torch.as_tensor(latent_intrinsics).detach().float().cpu()
    if depth.ndim != 2 or valid.shape != depth.shape or source_intrinsics.shape != (3, 3) or latent_intrinsics.shape != (3, 3):
        raise ValueError("Invalid depth, validity, or camera dimensions")
    rows, columns = torch.meshgrid(torch.arange(height).float() + .5, torch.arange(width).float() + .5, indexing="ij")
    target_pixels = torch.stack((columns, rows, torch.ones_like(rows)), dim=-1)
    source_pixels = target_pixels @ torch.linalg.inv(latent_intrinsics).T @ source_intrinsics.T
    source_pixels = source_pixels[..., :2] / source_pixels[..., 2:].clamp_min(1e-12)
    source_height, source_width = depth.shape
    grid = (source_pixels + .5) / torch.tensor([source_width, source_height]) * 2 - 1
    safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
    sampled = F.grid_sample(safe_depth[None, None], grid[None], mode=mode, padding_mode="zeros", align_corners=False)[0, 0]
    support = F.grid_sample(valid.float()[None, None], grid[None], mode=mode, padding_mode="zeros", align_corners=False)[0, 0]
    accepted = support >= 1 - 1e-5
    corrected = torch.where(accepted, sampled / support.clamp_min(1e-12), torch.zeros_like(sampled))
    return corrected, accepted


def downsample_depth_to_latent(depth_hw: torch.Tensor, h: int, w: int, mode: str = "bilinear") -> torch.Tensor:
    """Downsample a pixel-resolution depth map (H, W) to latent resolution (h, w).

    mode in {"bilinear", "nearest", "area", "median"} (Mirage Table 5). Bilinear is
    the default as it yields the lowest hole rate.
    """
    H, W = depth_hw.shape
    d = depth_hw.view(1, 1, H, W).float()
    if mode == "median":
        ky, kx = H // h, W // w
        if ky * h != H or kx * w != W:
            d = F.interpolate(d, size=(h * (H // h or 1), w * (W // w or 1)), mode="nearest")
            H2, W2 = d.shape[-2:]
            ky, kx = H2 // h, W2 // w
        patches = d.unfold(2, ky, ky).unfold(3, kx, kx)             # (1,1,h,w,ky,kx)
        patches = patches.contiguous().view(1, 1, h, w, -1)
        return patches.median(dim=-1).values.view(h, w)
    if mode == "area":
        return F.adaptive_avg_pool2d(d, (h, w)).view(h, w)
    if mode == "nearest":
        return F.interpolate(d, size=(h, w), mode="nearest").view(h, w)
    if mode == "bilinear":
        return F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False).view(h, w)
    raise ValueError(f"unknown downsample mode: {mode}")


def top_m_depth_modes(
    depth_hw: torch.Tensor,
    h: int,
    w: int,
    m: int = 2,
    rel_gap: float = 0.05,
    min_frac: float = 0.1,
):
    """Cluster each latent cell's pixel depths into up to ``m`` modes.

    For each latent cell (a stride-by-stride block of pixels) the depths are sorted and
    split wherever consecutive values jump by more than ``rel_gap`` times the cell's
    depth range; clusters covering at least ``min_frac`` of the block are kept, ranked
    by population, capped at ``m``. Each mode is summarised by its median depth and the
    fraction of pixels it owns (used downstream as a structural confidence prior).

    Returns:
        depths: (m, h, w) modal depths (nan where a mode is absent).
        weights: (m, h, w) population fraction in [0, 1] (0 where absent).
    """
    H, W = depth_hw.shape
    ky, kx = H // h, W // w
    assert ky >= 1 and kx >= 1, "pixel resolution must be >= latent resolution"
    d = depth_hw[: ky * h, : kx * w]
    blocks = d.unfold(0, ky, ky).unfold(1, kx, kx).contiguous().view(h, w, ky * kx)
    blocks, _ = torch.sort(blocks, dim=-1)
    n = blocks.shape[-1]

    depths = torch.full((m, h, w), float("nan"), dtype=depth_hw.dtype)
    weights = torch.zeros((m, h, w), dtype=depth_hw.dtype)

    # Vectorised over cells is awkward because cluster counts vary; the per-cell loop is
    # cheap (h*w on the order of 44*80) and keeps the logic exact and bug-free.
    for i in range(h):
        for j in range(w):
            vals = blocks[i, j]
            vals = vals[torch.isfinite(vals) & (vals > 0)]
            if vals.numel() == 0:
                continue
            rng = float(vals[-1] - vals[0])
            thr = rel_gap * rng if rng > 0 else 0.0
            # split into runs separated by gaps > thr
            gaps = vals[1:] - vals[:-1]
            split_idx = torch.nonzero(gaps > thr, as_tuple=False).flatten().tolist()
            bounds = [0] + [s + 1 for s in split_idx] + [vals.numel()]
            clusters = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                seg = vals[a:b]
                frac = seg.numel() / n
                if frac >= min_frac:
                    clusters.append((seg.numel(), float(seg.median()), frac))
            clusters.sort(key=lambda c: -c[0])
            for k, (_, med, frac) in enumerate(clusters[:m]):
                depths[k, i, j] = med
                weights[k, i, j] = frac
    return depths, weights
