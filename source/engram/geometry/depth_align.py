"""Align a monocular depth prediction onto the scale already established in memory.

A metric monocular depth model run independently on each generated frame carries a
frame-local scale ambiguity, but back-projection uses absolute extrinsics. The scale error
therefore becomes a world-space displacement, and it compounds: displaced observations
either average into the existing anchor (smearing geometry) or land in a neighbouring
voxel as a ghost surface that can win the top-K depth sort and occlude the correct
content. The same drifting depth also moves the median used as the scene scale, so the
packed camera translation slowly loses its calibration.

The memory already answers the question. Reading it at the pose we are about to write
gives a depth prediction for the cells it covers, so a robust ratio against that readout
recovers the missing scale. This is the standard SLAM/TSDF scale association, expressed
against a latent voxel store instead of a point cloud.

The estimator is deliberately conservative: it uses a median (so a minority of grossly
wrong cells cannot move it), refuses to act without enough covered cells, clamps the
result to a bounded ratio, and smooths across anchors so a single bad frame cannot inject
a step change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DepthScaleEstimate:
    """Outcome of one alignment attempt.

    ``scale`` is what a caller should multiply its depth by. It is exactly 1.0 whenever
    the estimate was refused, so callers never need to special-case failure.
    """

    scale: float
    raw_scale: float
    coverage: float
    trusted: bool


def estimate_depth_scale_to_memory(
    memory_depth: torch.Tensor,
    memory_mask: torch.Tensor,
    memory_conf: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    static: Optional[torch.Tensor] = None,
    min_coverage: float = 0.05,
    max_ratio: float = 1.43,
    conf_floor: float = 0.0,
    previous: Optional[float] = None,
    momentum: float = 0.5,
) -> DepthScaleEstimate:
    """Median-ratio scale mapping ``depth`` onto ``memory_depth``.

    Args:
        memory_depth: (h, w) frontmost depth from the memory readout at this pose.
        memory_mask: (h, w) bool, frontmost layer present.
        memory_conf: (h, w) frontmost layer confidence.
        depth: (h, w) the predicted depth to be aligned.
        valid: (h, w) bool, cells with usable predicted depth.
        static: optional (h, w) bool, restrict to static content.
        min_coverage: refuse to estimate below this fraction of usable cells.
        max_ratio: clamp the scale to ``[1/max_ratio, max_ratio]``.
        conf_floor: ignore memory cells at or below this confidence.
        previous: previous accepted scale, for exponential smoothing.
        momentum: weight of ``previous`` in the smoothed result.
    """
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must lie in [0, 1]")
    if float(max_ratio) < 1.0:
        raise ValueError("max_ratio must be >= 1")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("momentum must lie in [0, 1)")

    usable = (
        memory_mask.bool()
        & valid.bool()
        & torch.isfinite(memory_depth)
        & (memory_depth > 0)
        & torch.isfinite(depth)
        & (depth > 0)
        & (memory_conf > float(conf_floor))
    )
    if static is not None:
        usable = usable & static.bool()
    coverage = float(usable.float().mean())
    if coverage < float(min_coverage):
        return DepthScaleEstimate(
            scale=1.0, raw_scale=1.0, coverage=coverage, trusted=False
        )

    ratio = (memory_depth[usable] / depth[usable]).clamp(1e-3, 1e3)
    raw = float(ratio.median())
    limit = float(max_ratio)
    scale = min(max(raw, 1.0 / limit), limit)
    if previous is not None:
        scale = float(momentum) * float(previous) + (1.0 - float(momentum)) * scale
    return DepthScaleEstimate(
        scale=scale, raw_scale=raw, coverage=coverage, trusted=True
    )


def depth_disagreement_mask(
    memory_depth: torch.Tensor,
    memory_mask: torch.Tensor,
    depth: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Cells whose depth contradicts the memory beyond a relative ``tolerance``.

    Used as a write gate. A cell that still disagrees after scale alignment is either a
    genuinely new surface or a drifted duplicate of an existing one; fusing it into the
    existing anchor is wrong in both cases. Unlike the learned confidence proxies this
    judgement is purely geometric, so it needs no calibrator to be meaningful.
    """
    if float(tolerance) <= 0:
        return torch.zeros_like(memory_mask, dtype=torch.bool)
    present = (
        memory_mask.bool() & torch.isfinite(memory_depth) & (memory_depth > 0)
    )
    relative = (depth - memory_depth).abs() / memory_depth.clamp_min(1e-6)
    return present & (relative > float(tolerance))
