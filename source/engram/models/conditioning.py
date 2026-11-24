"""Pack a sequence of layered readouts into a DiT-aligned conditioning volume.

For each target latent frame the memory produces a ``LayeredReadout`` with K depth
layers. We stack, per layer, the latent feature (C ch), the calibrated confidence (1 ch),
the existence mask (1 ch) and a normalised inverse depth (1 ch). Over a chunk of F frames
this yields a conditioning volume of shape (K*(C+3), F, h, w) that the side branch
patchifies on the SAME grid as the Wan latent video, so memory tokens align 1:1 with the
DiT tokens they condition.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..memory.readout import LayeredReadout


# Explicit camera-control channels appended once per frame (Plucker ray: direction 3 +
# moment 3). Independent of memory coverage, so an empty-memory chunk still gets camera
# motion. Appended AFTER the readout channels, so coverage-mask indices are unaffected.
CAMERA_RAY_CHANNELS = 6


def conditioning_coverage_mask(
    cond_volume: torch.Tensor,
    top_k: int,
    latent_channels: int,
    composite: bool = False,
) -> torch.Tensor:
    """Recover a per-cell memory-coverage mask from a packed conditioning volume.

    ``cond_volume`` is (B, Cin, T, h, w) as produced by :func:`pack_readout_sequence`.
    Layered packing repeats ``[feat(C), conf(1), mask(1), inv_depth(1)]`` per layer; a
    cell is covered if any layer's existence mask is set. Composite packing is
    ``[composite(C), conf(1)]``; a cell is covered where the confidence map is positive.
    Returns a bool tensor (B, T, h, w). Used only for diagnostics (covered vs uncovered
    flow loss), never inside the trained forward path.
    """
    if cond_volume.ndim != 5:
        raise ValueError(f"cond_volume must be [B,Cin,T,h,w], got {tuple(cond_volume.shape)}")
    C = int(latent_channels)
    if composite:
        conf = cond_volume[:, C]
        return conf > 0
    stride = C + 3
    covered = None
    for k in range(int(top_k)):
        mask_channel = k * stride + C + 1
        layer = cond_volume[:, mask_channel] > 0.5
        covered = layer if covered is None else (covered | layer)
    if covered is None:
        raise ValueError("layered conditioning requires top_k >= 1")
    return covered


def readout_channels(
    top_k: int, latent_channels: int, composite: bool = False, camera_cond: bool = False
) -> int:
    """Conditioning channel count.

    Layered (default): ``K*(C+3)`` -- per layer: feature (C) + confidence (1) + mask (1) +
    inverse depth (1). Composite (idea-1's "soft-composite 49ch" option / ablation):
    ``C+1`` -- the confidence-weighted soft composite over layers plus one confidence map.
    When ``camera_cond`` is set, ``CAMERA_RAY_CHANNELS`` explicit Plucker camera channels
    are appended once per frame.
    """
    base = (latent_channels + 1) if composite else top_k * (latent_channels + 3)
    return base + (CAMERA_RAY_CHANNELS if camera_cond else 0)


def _normalised_inverse_depth(
    depth: torch.Tensor,
    scene_scale: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Map camera depth to a bounded inverse-depth feature in [0, 1].

    ``scene_scale=None`` preserves the historical per-frame maximum normalisation. The
    scale-normalised camera protocol instead uses ``x=s/depth; x/(1+x)``, so depth metadata
    and camera translation share one dimensionless scene scale.
    """
    finite = torch.isfinite(depth) & (depth > 0)
    inv = torch.zeros_like(depth)
    if scene_scale is not None:
        scale = torch.as_tensor(scene_scale, device=depth.device, dtype=depth.dtype)
        if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(
                f"scene_scale must be one positive finite scalar, got {scene_scale!r}"
            )
        inv[finite] = scale.reshape(()) / depth[finite]
        return inv / (1.0 + inv)
    inv[finite] = 1.0 / depth[finite].clamp_min(1e-3)
    if finite.any():
        mx = inv[finite].max().clamp_min(1e-6)
        inv = inv / mx
    return inv


def _normalised_confidence(conf: torch.Tensor) -> torch.Tensor:
    """Map absolute precision mass ``W`` to a stable probability-like channel.

    Per-frame max normalisation made an entirely unreliable frame contain a cell with
    confidence one. ``W / (1 + W)`` is bounded and monotonic while preserving the
    absolute meaning of the held-out calibrated precision across frames.
    """
    finite = torch.nan_to_num(
        conf,
        nan=0.0,
        posinf=torch.finfo(conf.dtype).max,
        neginf=0.0,
    ).clamp_min(0)
    return finite / (1.0 + finite)


def pack_readout(
    readout: LayeredReadout,
    scene_scale: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Pack a single-frame readout into (K*(C+3), h, w)."""
    K, C, h, w = readout.feat.shape
    inv_depth = _normalised_inverse_depth(
        readout.depth, scene_scale=scene_scale
    )                                                               # (K, h, w)
    conf_norm = _normalised_confidence(readout.conf)                # (K, h, w) bounded to [0,1]
    parts = []
    for k in range(K):
        parts.append(readout.feat[k])                              # (C, h, w)
        parts.append(conf_norm[k].unsqueeze(0))                    # (1, h, w)
        parts.append(readout.mask[k].unsqueeze(0).to(readout.feat.dtype))
        parts.append(inv_depth[k].unsqueeze(0))
    return torch.cat(parts, dim=0)                                 # (K*(C+3), h, w)


def pack_readout_composite(readout: LayeredReadout) -> torch.Tensor:
    """Pack a single-frame readout into the soft-composite form (C+1, h, w).

    Uses the confidence-weighted composite over the K layers plus a single confidence map
    (total precision mass, normalised to [0, 1]). This is idea-1's optional 49-channel
    conditioning (C=48 -> 49) and the ablation control against the K*(C+3) layered form.
    """
    C, h, w = readout.composite.shape
    conf = readout.composite_conf
    conf_norm = _normalised_confidence(conf).unsqueeze(0)         # (1, h, w)
    return torch.cat([readout.composite, conf_norm], dim=0)        # (C+1, h, w)


def pack_readout_sequence(readouts: List[LayeredReadout], composite: bool = False) -> torch.Tensor:
    """Pack F per-frame readouts into a volume (Cin, F, h, w)."""
    fn = pack_readout_composite if composite else pack_readout
    frames = [fn(r) for r in readouts]
    return torch.stack(frames, dim=1)
