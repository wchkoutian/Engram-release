"""Top-K layered latent readout (Engram Sec 3.5, Eq.5').

Replaces Mirage's hard z-buffer argmin (which keeps a single frontmost token per cell)
with a confidence-annotated, depth-ranked top-K readout. For every target latent cell
we keep the K frontmost memory slots that project to it, exposing per-layer features,
per-layer precision (confidence) and a per-layer existence mask. Setting K=1 recovers
Mirage's frontmost-only readout exactly.

Two readout products are returned:
  * layered tensors (K, C, h, w) + masks, for layered injection into the side branch;
  * a soft composite (C, h, w) + a single confidence map, for the 48ch composite mode.

We also expose the coverage map used to separate the two hole classes (Sec 1.1, H1):
  * geometric hole  = cell with zero projected slots (never observed);
  * occlusion layers = handled by keeping K frontmost rather than only the nearest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..geometry.projection import project_to_latent_grid


@dataclass
class LayeredReadout:
    feat: torch.Tensor       # (K, C, h, w) frontmost-to-back features, zero where absent
    conf: torch.Tensor       # (K, h, w) per-layer precision/confidence, 0 where absent
    depth: torch.Tensor      # (K, h, w) per-layer camera depth, +inf where absent
    mask: torch.Tensor       # (K, h, w) bool, layer present
    coverage: torch.Tensor   # (h, w) int, number of present layers (0 => geometric hole)
    composite: torch.Tensor  # (C, h, w) confidence-weighted soft composite over layers
    composite_conf: torch.Tensor  # (h, w) total precision mass at the cell
    used_slots: torch.Tensor # (P,) indices (into the gathered alive set) actually used


@torch.no_grad()
def frustum_filter(points_world, E_t, K_l, h, w, near=1e-3, far=float("inf")):
    """Return a boolean mask of points that fall inside the target view frustum."""
    cu, cv, z, in_front, in_bounds = project_to_latent_grid(points_world, E_t, K_l, h, w)
    keep = in_front & in_bounds & (z >= near) & (z <= far)
    return keep, cu, cv, z


@torch.no_grad()
def topk_layered_readout(
    points_world: torch.Tensor,
    feats: torch.Tensor,
    conf: torch.Tensor,
    E_t: torch.Tensor,
    K_l: torch.Tensor,
    h: int,
    w: int,
    top_k: int = 2,
    near: float = 1e-3,
    far: float = float("inf"),
    origin_real: Optional[torch.Tensor] = None,
    reserve_real_in_topk: bool = False,
) -> LayeredReadout:
    """Project memory slots into the target view and keep K frontmost per cell.

    Args:
        points_world: (N, 3) world positions of gathered alive slots.
        feats: (N, C) slot features (in whatever space the caller fuses in).
        conf: (N,) slot precision/confidence (e.g. accumulated W).
        E_t, K_l: target world-to-camera extrinsics and latent intrinsics.
        h, w, top_k: latent resolution and number of depth layers.
        origin_real: (N,) provenance mask aligned with ``points_world``.
        reserve_real_in_topk: keep the nearest visible real slot in each covered cell.
    """
    C = feats.shape[1]
    device = feats.device
    dtype = feats.dtype

    feat_out = torch.zeros(top_k, C, h, w, dtype=dtype, device=device)
    conf_out = torch.zeros(top_k, h, w, dtype=dtype, device=device)
    depth_out = torch.full((top_k, h, w), float("inf"), dtype=dtype, device=device)
    mask_out = torch.zeros(top_k, h, w, dtype=torch.bool, device=device)
    coverage = torch.zeros(h, w, dtype=torch.int64, device=device)

    keep, cu, cv, z = frustum_filter(points_world, E_t, K_l, h, w, near, far)
    if keep.any():
        cu, cv, z = cu[keep], cv[keep], z[keep]
        f = feats[keep]
        c = conf[keep]
        slot_ids = torch.nonzero(keep, as_tuple=False).flatten()
        cell_id = cv * w + cu

        # Stable two-pass sort: primary cell_id, secondary depth (ascending => frontmost first).
        idx1 = torch.argsort(z, stable=True)
        idx2 = torch.argsort(cell_id[idx1], stable=True)
        order = idx1[idx2]
        cs = cell_id[order]
        M = cs.shape[0]

        new_group = torch.ones(M, dtype=torch.bool, device=device)
        if M > 1:
            new_group[1:] = cs[1:] != cs[:-1]
        group_id = torch.cumsum(new_group.to(torch.int64), dim=0) - 1
        starts = torch.nonzero(new_group, as_tuple=False).flatten()
        rank = torch.arange(M, device=device) - starts[group_id]

        sel = rank < top_k
        if reserve_real_in_topk:
            real_sorted = origin_real[keep][order]
            positions = torch.arange(M, dtype=torch.int64, device=device)
            sentinel = torch.full_like(positions, M)
            first_real = torch.full(
                (int(starts.numel()),), M, dtype=torch.int64, device=device
            )
            first_real.scatter_reduce_(
                0,
                group_id,
                torch.where(real_sorted, positions, sentinel),
                reduce="amin",
                include_self=True,
            )
            selected_has_real = torch.zeros_like(first_real)
            selected_has_real.scatter_reduce_(
                0,
                group_id,
                (sel & real_sorted).to(torch.int64),
                reduce="amax",
                include_self=True,
            )
            replace_groups = torch.nonzero(
                (first_real < M) & (selected_has_real == 0), as_tuple=False
            ).flatten()
            if replace_groups.numel() > 0:
                sel[starts[replace_groups] + (top_k - 1)] = False
                sel[first_real[replace_groups]] = True
            selected_positions = torch.nonzero(sel, as_tuple=False).flatten()
            sorder = order[selected_positions]
            selected_cells = cell_id[sorder]
            selected_new_group = torch.ones(
                selected_positions.shape[0], dtype=torch.bool, device=device
            )
            if selected_positions.numel() > 1:
                selected_new_group[1:] = selected_cells[1:] != selected_cells[:-1]
            selected_starts = torch.nonzero(selected_new_group, as_tuple=False).flatten()
            selected_group_id = (
                torch.cumsum(selected_new_group.to(torch.int64), dim=0) - 1
            )
            srank = torch.arange(selected_positions.numel(), device=device) - selected_starts[
                selected_group_id
            ]
        else:
            sorder = order[sel]
            srank = rank[sel]
        su, sv = cu[sorder], cv[sorder]
        coverage.index_put_((cv, cu), torch.ones_like(cu), accumulate=True)
        # cap coverage display at top_k? keep true count for hole analysis; report raw.
        feat_out[srank, :, sv, su] = f[sorder]
        conf_out[srank, sv, su] = c[sorder]
        depth_out[srank, sv, su] = z[sorder]
        mask_out[srank, sv, su] = True
        used_slots = slot_ids[order[sel]]
    else:
        used_slots = torch.empty(0, dtype=torch.int64, device=device)

    total_conf = conf_out.sum(dim=0)                                # (h, w)
    safe = total_conf.clamp_min(1e-12)
    composite = (conf_out.unsqueeze(1) * feat_out).sum(dim=0) / safe.unsqueeze(0)
    composite = composite * (total_conf > 0).unsqueeze(0)

    return LayeredReadout(
        feat=feat_out,
        conf=conf_out,
        depth=depth_out,
        mask=mask_out,
        coverage=coverage,
        composite=composite,
        composite_conf=total_conf,
        used_slots=used_slots,
    )
