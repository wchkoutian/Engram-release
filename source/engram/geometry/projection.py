"""Pinhole geometry at latent resolution (Mirage Eq.7-10, Appendix A).

Conventions (identical to Mirage):
  * Extrinsics E in SE(3) map WORLD points into the CAMERA frame: x_cam = E @ [x_world; 1].
  * Intrinsics K is the standard pinhole matrix (fx, fy, cx, cy) at PIXEL resolution.
  * The VAE has spatial stride s, so latent resolution is (h, w) = (H/s, W/s).
  * Latent cells are indexed by (u, v) with pixel-centre homogeneous coords [u+0.5, v+0.5, 1].

All functions are batched and operate on torch tensors; CPU and CUDA both work.
"""
from __future__ import annotations

import torch


LEGACY_CAMERA_GEOMETRY_CONTRACT = "absolute_plucker_v1"
RELATIVE_SCENE_SCALE_CAMERA_GEOMETRY_CONTRACT = "relative_prefix_scene_scale_v1"
SUPPORTED_CAMERA_GEOMETRY_CONTRACTS = frozenset(
    {
        LEGACY_CAMERA_GEOMETRY_CONTRACT,
        RELATIVE_SCENE_SCALE_CAMERA_GEOMETRY_CONTRACT,
    }
)


def validate_camera_geometry_contract(contract: str) -> str:
    """Validate and canonicalise the camera-conditioning geometry protocol."""
    contract = str(contract)
    if contract not in SUPPORTED_CAMERA_GEOMETRY_CONTRACTS:
        raise ValueError(
            f"unsupported camera geometry contract {contract!r}; expected one of "
            f"{sorted(SUPPORTED_CAMERA_GEOMETRY_CONTRACTS)}"
        )
    return contract


def estimate_scene_scale(
    depth: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the median positive depth used to make camera translation dimensionless."""
    usable = torch.isfinite(depth) & (depth > 0)
    if valid is not None:
        if valid.shape != depth.shape:
            raise ValueError(
                f"scene-scale valid mask shape {tuple(valid.shape)} differs from depth "
                f"shape {tuple(depth.shape)}"
            )
        usable = usable & valid.bool()
    if not usable.any():
        raise ValueError("cannot estimate scene scale without a positive finite depth")
    scale = depth[usable].median()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError(f"scene scale must be positive and finite, got {scale.item()!r}")
    return scale


def camera_conditioning_w2c(
    target_w2c: torch.Tensor,
    prefix_w2c: torch.Tensor | None = None,
    scene_scale: torch.Tensor | float | None = None,
    contract: str = LEGACY_CAMERA_GEOMETRY_CONTRACT,
) -> torch.Tensor:
    """Return the W2C pose whose Plucker rays encode the requested camera motion.

    The legacy protocol exposes the dataset's absolute world gauge. The scale-normalised
    protocol re-bases every target pose into the clean-prefix camera frame and divides its
    translation by the prefix view's median scene depth. This preserves rotation while
    making translation invariant to per-clip monocular scale.
    """
    contract = validate_camera_geometry_contract(contract)
    if contract == LEGACY_CAMERA_GEOMETRY_CONTRACT:
        return target_w2c
    if prefix_w2c is None or scene_scale is None:
        raise ValueError(
            f"{contract} requires both prefix_w2c and scene_scale"
        )
    if target_w2c.shape[-2:] != (4, 4) or prefix_w2c.shape[-2:] != (4, 4):
        raise ValueError(
            "camera poses must end in [4,4], got "
            f"target={tuple(target_w2c.shape)}, prefix={tuple(prefix_w2c.shape)}"
        )
    prefix_w2c = prefix_w2c.to(device=target_w2c.device, dtype=target_w2c.dtype)
    relative = target_w2c @ torch.linalg.inv(prefix_w2c)
    scale = torch.as_tensor(
        scene_scale, device=target_w2c.device, dtype=target_w2c.dtype
    )
    if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(f"scene_scale must be one positive finite scalar, got {scene_scale!r}")
    normalised = relative.clone()
    normalised[..., :3, 3] = normalised[..., :3, 3] / scale.reshape(())
    return normalised


def latent_intrinsics(K: torch.Tensor, H: int, W: int, h: int, w: int) -> torch.Tensor:
    """Scale pixel-resolution intrinsics K to latent resolution (Mirage Eq.7).

    K_l = diag(w/W, h/H, 1) @ K. Both focal lengths and the principal point are
    rescaled by the per-axis stride ratio so perspective stays consistent after the
    depth map is downsampled to latent resolution.

    Args:
        K: (..., 3, 3) pixel-resolution intrinsics.
    Returns:
        (..., 3, 3) latent-resolution intrinsics.
    """
    scale = torch.tensor([w / W, h / H, 1.0], dtype=K.dtype, device=K.device)
    return scale.view(*([1] * (K.dim() - 2)), 3, 1) * K


def plucker_rays(E: torch.Tensor, K_l: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Per-latent-cell Plucker ray embedding (6, h, w) in world coordinates.

    For each latent cell (u, v) the back-projected camera ray has a world-space unit
    direction ``d`` and shares the camera centre ``o`` (world). Its Plucker coordinates
    ``(d, m = o x d)`` encode the ray smoothly and independently of parametrisation, giving
    the side branch an explicit camera-control signal that does NOT depend on memory
    coverage (so the first chunk, whose memory is near empty, still receives the intended
    camera motion). Uses the same pinhole conventions as ``backproject``: ``E`` maps world
    into camera, ``K_l`` is latent-resolution intrinsics.

    Returns:
        (6, h, w) tensor [dx, dy, dz, mx, my, mz] in ``K_l``'s dtype/device.
    """
    dtype = K_l.dtype
    device = K_l.device
    E = E.to(device=device, dtype=dtype)
    vv, uu = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(uu)
    pix = torch.stack([uu + 0.5, vv + 0.5, ones], dim=-1)           # (h, w, 3)
    K_inv = _invert_intrinsics(K_l)                                 # (3, 3)
    dirs_cam = pix @ K_inv.transpose(-1, -2)                        # (h, w, 3) camera-ray dirs
    E_inv = torch.inverse(E)                                        # camera-to-world
    R_c2w = E_inv[:3, :3]
    o = E_inv[:3, 3]                                                # (3,) camera centre in world
    dirs_world = dirs_cam @ R_c2w.transpose(-1, -2)                # (h, w, 3)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    o_b = o.view(1, 1, 3).expand_as(dirs_world)
    moment = torch.cross(o_b, dirs_world, dim=-1)                  # (h, w, 3)
    plucker = torch.cat([dirs_world, moment], dim=-1)             # (h, w, 6)
    return plucker.permute(2, 0, 1).contiguous()                  # (6, h, w)


def _invert_intrinsics(K: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of an upper-triangular pinhole intrinsic matrix."""
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    skew = K[..., 0, 1]
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    inv = torch.stack(
        [
            torch.stack([1.0 / fx, -skew / (fx * fy), (skew * cy - cx * fy) / (fx * fy)], dim=-1),
            torch.stack([zeros, 1.0 / fy, -cy / fy], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return inv


def backproject(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    K_l: torch.Tensor,
    E: torch.Tensor,
) -> torch.Tensor:
    """Back-project latent cells (u, v) with depth into WORLD coordinates (Mirage Eq.8).

    p_world = E^{-1} @ [ d * K_l^{-1} @ [u+0.5, v+0.5, 1]^T ; 1 ] |_{1:3}

    Args:
        u, v: (N,) integer or float latent-cell indices.
        depth: (N,) metric depth at those cells.
        K_l: (3, 3) latent-resolution intrinsics.
        E: (4, 4) world-to-camera extrinsics.
    Returns:
        (N, 3) world-space points.
    """
    # MapAnything may return depth on CUDA while benchmark camera tensors remain
    # on CPU. Keep the complete projection calculation on the depth tensor's
    # device and dtype so evaluation does not fail at the first matrix product.
    u = u.to(device=depth.device, dtype=depth.dtype)
    v = v.to(device=depth.device, dtype=depth.dtype)
    K_l = K_l.to(device=depth.device, dtype=depth.dtype)
    E = E.to(device=depth.device, dtype=depth.dtype)
    ones = torch.ones_like(u)
    pix = torch.stack([u + 0.5, v + 0.5, ones], dim=-1)              # (N, 3)
    K_inv = _invert_intrinsics(K_l)                                  # (3, 3)
    rays = pix @ K_inv.transpose(-1, -2)                            # (N, 3) camera-ray dirs
    cam = rays * depth.unsqueeze(-1)                                # (N, 3) camera points
    cam_h = torch.cat([cam, torch.ones_like(depth).unsqueeze(-1)], dim=-1)  # (N, 4)
    E_inv = torch.inverse(E)                                        # camera-to-world
    world_h = cam_h @ E_inv.transpose(-1, -2)                      # (N, 4)
    return world_h[..., :3]


def backproject_depth_map(
    depth_lat: torch.Tensor,
    K_l: torch.Tensor,
    E: torch.Tensor,
    valid: torch.Tensor | None = None,
):
    """Back-project an entire latent-resolution depth map.

    Args:
        depth_lat: (h, w) metric depth at latent resolution.
        K_l: (3, 3) latent intrinsics.
        E: (4, 4) world-to-camera extrinsics.
        valid: optional (h, w) boolean mask of cells to lift.
    Returns:
        points: (M, 3) world points, uu: (M,), vv: (M,) the source latent indices.
    """
    h, w = depth_lat.shape
    vv, uu = torch.meshgrid(
        torch.arange(h, device=depth_lat.device),
        torch.arange(w, device=depth_lat.device),
        indexing="ij",
    )
    uu = uu.reshape(-1)
    vv = vv.reshape(-1)
    d = depth_lat.reshape(-1)
    if valid is None:
        valid = (d > 0) & torch.isfinite(d)
    else:
        valid = valid.reshape(-1) & (d > 0) & torch.isfinite(d)
    uu, vv, d = uu[valid], vv[valid], d[valid]
    pts = backproject(uu, vv, d, K_l, E)
    return pts, uu, vv


def project_to_latent_grid(
    points_world: torch.Tensor,
    E_t: torch.Tensor,
    K_l: torch.Tensor,
    h: int,
    w: int,
):
    """Project world points onto the target latent grid (Mirage Eq.9-10).

    q = (E_t @ [p; 1])_{1:3};  [x, y, 1]^T = K_l @ q / q_z;  cell = (floor(x), floor(y)).

    Args:
        points_world: (N, 3) world points.
        E_t: (4, 4) target world-to-camera extrinsics.
        K_l: (3, 3) latent intrinsics.
        h, w: latent resolution.
    Returns:
        cell_u: (N,) long, cell_v: (N,) long, depth_cam: (N,) float (q_z),
        in_front: (N,) bool, in_bounds: (N,) bool.
    """
    x, y, z, in_front, in_bounds = project_to_image_plane(
        points_world, E_t, K_l, h, w
    )
    cell_u = torch.floor(x).long()
    cell_v = torch.floor(y).long()
    return cell_u, cell_v, z, in_front, in_bounds


def project_to_image_plane(
    points_world: torch.Tensor,
    E_t: torch.Tensor,
    K_l: torch.Tensor,
    h: int,
    w: int,
):
    """Project world points without quantising away sub-cell camera motion.

    Returns continuous latent-grid coordinates ``x`` and ``y`` together with camera
    depth and visibility flags.  ``project_to_latent_grid`` remains the discrete memory
    lookup API and delegates its common projection math here.
    """
    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError(
            f"points_world must have shape [N,3], got {tuple(points_world.shape)}"
        )
    count = points_world.shape[0]
    ph = torch.cat(
        [
            points_world,
            torch.ones(
                count,
                1,
                dtype=points_world.dtype,
                device=points_world.device,
            ),
        ],
        dim=-1,
    )
    E_t = E_t.to(device=points_world.device, dtype=points_world.dtype)
    K_l = K_l.to(device=points_world.device, dtype=points_world.dtype)
    q = (ph @ E_t.transpose(-1, -2))[..., :3]
    z = q[..., 2]
    in_front = torch.isfinite(z) & (z > 0)
    z_safe = torch.where(in_front, z, torch.ones_like(z))
    proj = (q / z_safe.unsqueeze(-1)) @ K_l.transpose(-1, -2)
    x = proj[..., 0]
    y = proj[..., 1]
    finite = torch.isfinite(x) & torch.isfinite(y)
    cell_u = torch.floor(torch.where(finite, x, torch.zeros_like(x))).long()
    cell_v = torch.floor(torch.where(finite, y, torch.zeros_like(y))).long()
    in_bounds = (
        finite
        & (cell_u >= 0)
        & (cell_u < int(w))
        & (cell_v >= 0)
        & (cell_v < int(h))
    )
    return x, y, z, in_front, in_bounds


def reverse_relative_w2c(
    target_w2c: torch.Tensor, prefix_w2c: torch.Tensor
) -> torch.Tensor:
    """Invert target motion around the clean-prefix camera in the same world gauge.

    For world-to-camera extrinsics, ``target @ inv(prefix)`` maps prefix-camera
    coordinates to target-camera coordinates.  Inverting that relative transform and
    composing it with ``prefix`` produces a physically valid counterfactual pose while
    leaving an identity target unchanged.
    """
    if target_w2c.shape[-2:] != (4, 4) or prefix_w2c.shape[-2:] != (4, 4):
        raise ValueError(
            "camera poses must end in [4,4], got "
            f"target={tuple(target_w2c.shape)}, prefix={tuple(prefix_w2c.shape)}"
        )
    prefix_w2c = prefix_w2c.to(
        device=target_w2c.device, dtype=target_w2c.dtype
    )
    relative = target_w2c @ torch.linalg.inv(prefix_w2c)
    return torch.linalg.inv(relative) @ prefix_w2c


def camera_motion_map_from_prefix(
    prefix_depth: torch.Tensor,
    prefix_static_mask: torch.Tensor,
    prefix_K_l: torch.Tensor,
    prefix_w2c: torch.Tensor,
    target_K_l: torch.Tensor,
    target_w2c: torch.Tensor,
    *,
    quantile: float = 0.9,
    depth_tolerance: float = 1e-4,
) -> torch.Tensor:
    """Return a target-aligned, visible-static parallax map in ``[0,1]``.

    Static T0 cells are lifted with the training depth, projected continuously into the
    target view, and z-buffered.  Their sub-cell displacement is scattered onto the
    target grid and normalised by a per-view quantile.  This map is a training weight,
    not an optical-flow label: it only tells the flow objective where the requested
    camera motion should make reconstruction especially informative.
    """
    if prefix_depth.ndim != 2 or prefix_static_mask.shape != prefix_depth.shape:
        raise ValueError(
            "prefix depth/static mask must be matching [H,W] tensors, got "
            f"depth={tuple(prefix_depth.shape)}, mask={tuple(prefix_static_mask.shape)}"
        )
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError(f"motion-map quantile must lie in (0,1], got {quantile}")
    h, w = prefix_depth.shape
    valid_source = (
        prefix_static_mask.bool()
        & torch.isfinite(prefix_depth)
        & (prefix_depth > 0)
    )
    points, source_u, source_v = backproject_depth_map(
        prefix_depth, prefix_K_l, prefix_w2c, valid=valid_source
    )
    result = prefix_depth.new_zeros((h, w), dtype=torch.float32)
    if points.numel() == 0:
        return result

    x, y, z, in_front, in_bounds = project_to_image_plane(
        points, target_w2c, target_K_l, h, w
    )
    visible = in_front & in_bounds
    if not visible.any():
        return result
    x = x[visible]
    y = y[visible]
    z = z[visible]
    source_x = source_u[visible].to(x.dtype) + 0.5
    source_y = source_v[visible].to(y.dtype) + 0.5
    target_u = torch.floor(x).long()
    target_v = torch.floor(y).long()
    flat_index = target_v * int(w) + target_u

    nearest_depth = torch.full(
        (h * w,),
        float("inf"),
        device=z.device,
        dtype=z.dtype,
    )
    nearest_depth.scatter_reduce_(
        0, flat_index, z, reduce="amin", include_self=True
    )
    nearest = z <= nearest_depth[flat_index] + float(depth_tolerance)
    if not nearest.any():
        return result
    flat_index = flat_index[nearest]
    displacement = torch.sqrt(
        (x[nearest] - source_x[nearest]).square()
        + (y[nearest] - source_y[nearest]).square()
    ).float()
    raw = torch.zeros(h * w, device=z.device, dtype=torch.float32)
    raw.scatter_reduce_(
        0, flat_index, displacement, reduce="amax", include_self=True
    )
    occupied = torch.zeros(h * w, device=z.device, dtype=torch.int64)
    occupied.scatter_reduce_(
        0,
        flat_index,
        torch.ones_like(flat_index, dtype=torch.int64),
        reduce="amax",
        include_self=True,
    )
    active = occupied.bool()
    if not active.any() or raw[active].max() <= 1e-4:
        return result
    scale = torch.quantile(raw[active], float(quantile)).clamp_min(1e-4)
    normalised = (raw / scale).clamp_(0.0, 1.0)
    return normalised.reshape(h, w).to(device=prefix_depth.device)


def blender_c2w_to_relative_opencv_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """Convert Blender c2w poses to first-view-relative OpenCV w2c poses."""
    c2w_relative = torch.linalg.inv(c2w[0]) @ c2w
    w2c_blender = torch.linalg.inv(c2w_relative)
    blender_to_opencv = torch.diag(
        torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=c2w.dtype, device=c2w.device)
    )
    return blender_to_opencv @ w2c_blender @ blender_to_opencv
