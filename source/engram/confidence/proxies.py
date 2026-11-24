"""Per-cell precision proxies (Engram Sec 3.4).

Each proxy returns a non-negative, per-latent-cell precision-like quantity. They are
combined in log space by ``ConfidenceEstimator`` into a single raw log-precision, which
is then passed through an explicit calibrator (``engram.confidence.calibration``).

Proxies implemented (all from signals the pipeline already produces):
  * depth uncertainty   -> precision from DepthAnything3 confidence/variance;
  * reprojection / photometric consistency -> precision from cross-view residuals;
  * grazing angle       -> precision from the angle between surface normal and ray;
  * generation quality  -> precision from the diffusion head (per-token log-variance).
"""
from __future__ import annotations

from typing import Dict

import torch

from ..geometry.projection import backproject_depth_map


def depth_uncertainty_precision(depth_conf=None, depth_var=None, eps: float = 1e-4) -> torch.Tensor:
    """Precision from monocular-depth uncertainty.

    Pass either a confidence map in [0, 1] (DepthAnything3 returns one) or a variance map.
    Returns a precision-like map (higher == more reliable).
    """
    if depth_var is not None:
        return 1.0 / (depth_var + eps)
    if depth_conf is not None:
        return depth_conf.clamp_min(0.0) / (1.0 - depth_conf.clamp(0.0, 1.0 - eps))
    raise ValueError("provide depth_conf or depth_var")


def reprojection_precision(
    residual: torch.Tensor,
    scale: float = 1.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Precision from a cross-view photometric/latent residual map.

    residual >= 0 (e.g. L2 distance between a re-projected token and the observed token).
    Maps small residuals to high precision via an exponential reliability model
    reliability = exp(-residual / scale), then precision = reliability / (1 - reliability).
    """
    rel = torch.exp(-residual.clamp_min(0.0) / scale)
    rel = rel.clamp(1e-6, 1.0 - 1e-6)
    precision = rel / (1.0 - rel)
    if valid is not None:
        precision = torch.where(valid, precision, torch.ones_like(precision))
    return precision


def grazing_precision(depth_lat: torch.Tensor, K_l: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Precision from the grazing angle between the surface normal and the viewing ray.

    Surfaces seen edge-on (ray nearly parallel to the surface) are unreliable. We
    back-project the latent depth map into camera space, estimate normals by finite
    differences, and use |cos(angle between normal and ray)| as the precision proxy.

    The grazing angle is a camera-relative quantity: both the surface normal and the ray
    rotate with the camera, so |cos(.)| is invariant to the extrinsics. We therefore work
    directly in the camera frame (E = identity); no per-view pose is needed and feeding the
    real ``E_t`` would give the identical result.
    """
    # Geometry must stay FP32 even when model inference runs inside BF16 autocast.
    # Otherwise back-projection may return BF16 points that cannot be assigned to an
    # FP32 grid, and the normal estimate loses precision near depth discontinuities.
    with torch.autocast(device_type=depth_lat.device.type, enabled=False):
        depth = depth_lat.to(dtype=torch.float32)
        K = K_l.to(device=depth.device, dtype=torch.float32)
        h, w = depth.shape
        E = torch.eye(4, dtype=torch.float32, device=depth.device)
        pts, uu, vv = backproject_depth_map(depth, K, E)              # camera frame
        grid = torch.zeros(h, w, 3, dtype=torch.float32, device=depth.device)
        grid[vv, uu] = pts.to(dtype=torch.float32)
        # tangent vectors via central/forward differences
        dx = torch.zeros_like(grid)
        dy = torch.zeros_like(grid)
        dx[:, 1:] = grid[:, 1:] - grid[:, :-1]
        dy[1:, :] = grid[1:, :] - grid[:-1, :]
        normal = torch.cross(dx, dy, dim=-1)
        n_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(eps)
        normal = normal / n_norm
        ray = grid / torch.linalg.vector_norm(grid, dim=-1, keepdim=True).clamp_min(eps)
        cos = (normal * ray).sum(dim=-1).abs()
        return cos.clamp(eps, 1.0)


def generation_quality_precision(token_logvar: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Precision from the backbone's per-token predictive log-variance.

    Lower predicted variance == more confident generation == higher precision. Used when
    the backbone exposes an explicit predictive variance head.
    """
    return torch.exp(-token_logvar)


def generation_quality_from_residual(residual: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Generation-quality precision from the diffusion denoising residual (Sec 3.2).

    The proposal down-weights tokens the backbone is still uncertain about ("去噪残差越大
    则降权"): at the final (lowest-noise) sampling step a large remaining velocity update
    means the latent has not settled, i.e. a less confident generation. ``residual`` is the
    per-cell L2 magnitude of that final-step update (>= 0). We map it to a precision with the
    same reliability/odds link used elsewhere: reliability = exp(-residual/scale), then
    precision = reliability / (1 - reliability) -- monotone decreasing in the residual.
    """
    rel = torch.exp(-residual.clamp_min(0.0) / scale)
    rel = rel.clamp(1e-6, 1.0 - 1e-6)
    return rel / (1.0 - rel)


def combine_log_precision(proxies: Dict[str, torch.Tensor], weights: Dict[str, float], eps: float = 1e-8) -> torch.Tensor:
    """Log-linear pooling of proxy precisions: log p = sum_k w_k * log proxy_k."""
    out = None
    for name, val in proxies.items():
        wk = weights.get(name, 0.0)
        term = wk * torch.log(val.clamp_min(eps))
        out = term if out is None else out + term
    if out is None:
        raise ValueError("no proxies provided")
    return out
