"""Step 1 (2026-10-14, inference-only diagnostic): verify-then-write for generated memory.

Replaces core.write_generated_chunk for one chunk.  Same MapAnything call, same latent views
(indices 1..8 with index*4 < valid_count), same commanded poses, same precision proxies, same
EngramMemory.update.  Before the update every generated cell is checked:
  * real-anchor veto: back-projected point lies more than REAL_MARGIN in front of the real I0
    surface (seed depth) when seen from I0 -> dropped (it would occlude real content at I0);
  * intra-chunk multi-view agreement against the other written views j of the same chunk:
    visible  = projects in-bounds, in front, onto a valid depth cell of j;
    agree    = |z_j - D_j| < DEPTH_REL * D_j  and  ||whiten(f_i) - whiten(f_j)||_2 < FEAT_TAU
               (FEAT_TAU = the memory's own consistency_tau, same whitened space as slot fusion);
    visible >= MIN_SUPPORT: written iff agree >= MIN_SUPPORT, else dropped;
    visible <  MIN_SUPPORT: (newly revealed / edge) written with precision x LOW_SUPPORT_WEIGHT.
Thresholds fixed before any run; no per-scene tuning.
mode='real_only' (B6 ablation, e6_realveto): only the real-anchor veto; every other valid cell written at weight 1.
"""
import time

import torch

DEPTH_REL = .05
FEAT_TAU = 3.
MIN_SUPPORT = 2
LOW_SUPPORT_WEIGHT = .25
REAL_MARGIN = .10
ALIGN_MIN_CELLS = 20  # E8: minimum real/generated overlap cells for a scale estimate


@torch.no_grad()
def gated_write_chunk(memory, estimator, bundle, rgb_frames, latents, poses, start, valid_count, mode='full'):
    from engram import core
    from engram.geometry.projection import backproject, project_to_latent_grid

    latent_indices = [index for index in range(1, 9) if index * 4 < valid_count]
    if not latent_indices:
        return {'geometry_seconds': 0., 'fusion_seconds': 0., 'gate': None}
    metric_poses = core.poses_in_metres(poses, memory.unit_metres)
    started = time.monotonic()
    results = estimator.depth_for_views([rgb_frames[i * 4] for i in latent_indices], [bundle['intrinsics'][0] for _ in latent_indices],
                                        [metric_poses[i] for i in latent_indices], bundle['height'] // 16, bundle['width'] // 16)
    geometry_seconds = time.monotonic() - started
    started = time.monotonic()
    K = bundle['intrinsics_latent'][0].float()
    whitener = memory.memory.whitener
    depths, valids, feats, extr = [], [], [], []
    for index, result in zip(latent_indices, results):
        d = (result['depth'] / memory.unit_metres).float().cpu()
        depths.append(d)
        valids.append(torch.isfinite(d) & (d > 0))
        f = latents[0, :, index].float().cpu()                        # (C, h, w) raw latent
        feats.append(whitener.whiten(f.permute(1, 2, 0)))              # (h, w, C) whitened
        extr.append(poses[index].float().cpu())
    h, w = depths[0].shape
    seed = bundle['seed_depth'].float().cpu().reshape(h, w)
    seed_valid = torch.isfinite(seed) & (seed > 0)
    E0 = bundle['poses']['nvs'][0].float().cpu()
    vv, uu = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    align = []
    if mode == 'real_align':  # E8: scale each generated depth map to the real I0 surface rendered into its view
        P0 = backproject(uu[seed_valid], vv[seed_valid], seed[seed_valid], K, E0)
        for i in range(len(latent_indices)):
            cu, cv, z, front, inb = project_to_latent_grid(P0, extr[i], K, h, w)
            sel = front & inb
            zbuf = torch.full((h * w,), float('inf')).scatter_reduce(0, cv[sel] * w + cu[sel], z[sel].float(), reduce='amin').reshape(h, w)
            ov = torch.isfinite(zbuf) & valids[i]
            s = float(torch.median(zbuf[ov] / depths[i][ov])) if int(ov.sum()) >= ALIGN_MIN_CELLS else 1.
            depths[i] = depths[i] * s
            align.append(round(s, 4))
    stats = {'cells': 0, 'drop_real': 0, 'drop_inconsistent': 0, 'low_support': 0, 'kept': 0, 'align_scale': align}
    weights = []
    for i in range(len(latent_indices)):
        m = valids[i]
        weight = torch.zeros(h, w)
        u, v, d = uu[m], vv[m], depths[i][m]
        pts = backproject(u, v, d, K, extr[i])
        fi = feats[i][m]
        # real-anchor veto
        cu, cv, z, front, inb = project_to_latent_grid(pts, E0, K, h, w)
        ok = front & inb
        real_d = torch.zeros_like(z)
        real_d[ok] = seed[cv[ok], cu[ok]]
        ok = ok & (real_d > 0) & seed_valid[cv.clamp(0, h - 1), cu.clamp(0, w - 1)]
        occlude = ok & (z < (1 - REAL_MARGIN) * real_d)
        visible = torch.zeros(len(d), dtype=torch.long)
        support = torch.zeros(len(d), dtype=torch.long)
        for j in range(len(latent_indices)):
            if j == i:
                continue
            cu, cv, z, front, inb = project_to_latent_grid(pts, extr[j], K, h, w)
            sel = front & inb
            cuc, cvc = cu.clamp(0, w - 1), cv.clamp(0, h - 1)
            sel = sel & valids[j][cvc, cuc]
            dj = depths[j][cvc, cuc]
            geo = (z - dj).abs() < DEPTH_REL * dj
            app = torch.linalg.vector_norm(fi - feats[j][cvc, cuc], dim=-1) < FEAT_TAU
            visible += sel.long()
            support += (sel & geo & app).long()
        if mode in ('real_only', 'real_align'):
            keep_full = ~occlude
            keep_low = torch.zeros_like(occlude)
        else:
            keep_full = (visible >= MIN_SUPPORT) & (support >= MIN_SUPPORT) & ~occlude
            keep_low = (visible < MIN_SUPPORT) & ~occlude
        vals = torch.where(keep_full, torch.ones_like(d), torch.where(keep_low, torch.full_like(d, LOW_SUPPORT_WEIGHT), torch.zeros_like(d)))
        weight[m] = vals
        weights.append(weight)
        stats['cells'] += int(m.sum())
        stats['drop_real'] += int(occlude.sum())
        stats['drop_inconsistent'] += int((~keep_full & ~keep_low & ~occlude).sum())
        stats['low_support'] += int(keep_low.sum())
        stats['kept'] += int(keep_full.sum())
    for i, index in enumerate(latent_indices):
        _update(memory, core, latents[0, :, index], depths[i], weights[i], K, poses[index], start + index * 4)
    return {'geometry_seconds': geometry_seconds, 'fusion_seconds': time.monotonic() - started, 'gate': stats}


def _update(memory, core, feature, depth, weight, intrinsics, pose, frame_index):
    """core.ProtectedMemory.update for origin='generated' with precision scaled by the gate weight."""
    if not memory.writeback:
        return
    dev = memory.device
    depth = depth.to(dev).float()
    intrinsics = intrinsics.to(dev).float()
    valid = torch.isfinite(depth) & (depth > 0) & (weight.to(dev) > 0)
    proxies = {'depth': core.depth_uncertainty_precision(depth_conf=torch.full_like(depth, .5)), 'graze': core.grazing_precision(depth, intrinsics)}
    precision = memory.confidence.precision(proxies, calibrate=False) * weight.to(dev)
    anchor_index = int(frame_index) // 4
    memory.memory.update(feature.to(dev).float(), depth, precision, pose.to(dev).float(), intrinsics, valid=valid, t=anchor_index + 1,
                         bias_discount=core.Recipe().bias_gamma ** anchor_index, origin='generated')
    memory.generated_frames.append(int(frame_index))
