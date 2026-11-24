"""EngramMemory: the orchestrator tying together whitening, the unbounded voxel-hash
store, confidence-weighted fusion and the top-K layered readout (Engram idea-1, Sec 3.2-3.5).

Pipeline per generated chunk (Algorithm 1):
  update(): back-project confident latent tokens to world space, standardise them,
            discount autoregressive bias, and fuse into the store with inverse-variance
            weights (Eq.6');
  read():   project memory into the next target view, keep K frontmost layers per cell
            (Eq.5'), and return features in *original* latent space for injection into
            the side branch.

idea-1 keeps every observed layer: the store is unbounded and has no eviction/forgetting.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..geometry.projection import backproject
from .voxel_hash import GENERATED_ORIGIN, VoxelHashMemory
from .whitening import RunningWhitener
from .readout import topk_layered_readout, LayeredReadout


class EngramMemory:
    def __init__(
        self,
        channels: int,
        voxel_size: float = 0.05,
        consistency_tau: Optional[float] = 3.0,
        max_layers: int = 8,
        fuse: bool = True,
        whiten_momentum: Optional[float] = None,
        update_whitener: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        w_max: Optional[float] = None,
        real_weight_scale: float = 1.0,
        gen_into_real_scale: float = 1.0,
        gen_into_real_budget: Optional[float] = None,
        tau_geom: Optional[float] = None,
        fuse_position: bool = True,
        drop_on_layer_budget: bool = False,
        conf_report_max: Optional[float] = None,
        immutable_real_memory: bool = False,
        reserve_real_in_topk: bool = False,
    ):
        """``conf_report_max`` caps the confidence *reported* by :meth:`read` without
        touching the fusion accumulator. Training builds a memory from a handful of views
        (accumulated precision of order a few), while a closed-loop rollout accumulates
        tens of views, so the packed ``W/(1+W)`` confidence channel saturates to 1 at
        inference and carries no information the side branch was trained to read. Capping
        the reported value restores that channel's training semantics; capping the
        accumulator instead (``w_max``) would also change the estimator."""
        if conf_report_max is not None and float(conf_report_max) <= 0:
            raise ValueError(
                f"conf_report_max must be positive when set, got {conf_report_max}"
            )
        self.C = channels
        self.update_whitener = update_whitener
        self.device = torch.device(device)
        self.dtype = dtype
        self.conf_report_max = None if conf_report_max is None else float(conf_report_max)
        self.reserve_real_in_topk = bool(reserve_real_in_topk)
        self.store = VoxelHashMemory(
            channels=channels,
            voxel_size=voxel_size,
            consistency_tau=consistency_tau,
            max_layers=max_layers,
            fuse=fuse,
            device=device,
            dtype=dtype,
            w_max=w_max,
            real_weight_scale=real_weight_scale,
            gen_into_real_scale=gen_into_real_scale,
            gen_into_real_budget=gen_into_real_budget,
            tau_geom=tau_geom,
            fuse_position=fuse_position,
            drop_on_layer_budget=drop_on_layer_budget,
            immutable_real_memory=immutable_real_memory,
        )
        self.whitener = RunningWhitener(channels, device=device, dtype=dtype, momentum=whiten_momentum)
        self._t = 0

    def __len__(self):
        return len(self.store)

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def _lift_and_write(self, latent, depth_lat, conf, E, K_l, valid, weight_scale, t,
                        origin=GENERATED_ORIGIN):
        h, w = depth_lat.shape
        vv, uu = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        m = valid & (depth_lat > 0) & torch.isfinite(depth_lat) & (conf > 0)
        if not m.any():
            return
        uu, vv = uu[m], vv[m]
        d = depth_lat[m]
        feats = latent[:, vv, uu].transpose(0, 1).contiguous()       # (M, C) raw latent
        weights = (conf[m] * weight_scale).to(self.dtype)
        pts = backproject(uu, vv, d, K_l, E)
        if self.update_whitener:
            self.whitener.update(feats)
        feats_w = self.whitener.whiten(feats)
        self.store.write(pts, feats_w, weights, t=t, origin=origin)

    @torch.no_grad()
    def update(
        self,
        latent: torch.Tensor,           # (C, h, w) generated latent for this view
        depth_lat: torch.Tensor,        # (h, w) latent-resolution depth
        conf: torch.Tensor,             # (h, w) calibrated precision per cell (>=0)
        E: torch.Tensor,                # (4, 4) world-to-camera extrinsics
        K_l: torch.Tensor,              # (3, 3) latent intrinsics
        valid: Optional[torch.Tensor] = None,
        t: Optional[int] = None,
        bias_discount: float = 1.0,     # autoregressive bias discount in (0, 1]
        modes_depth: Optional[torch.Tensor] = None,   # (m, h, w) optional multi-hypothesis
        modes_weight: Optional[torch.Tensor] = None,  # (m, h, w) population fractions
        origin: str = GENERATED_ORIGIN,               # "real" for observed imagery
    ) -> None:
        h, w = depth_lat.shape
        if t is None:
            t = self._t + 1
        self._t = max(self._t, int(t))
        if valid is None:
            valid = torch.ones(h, w, dtype=torch.bool, device=self.device)

        if modes_depth is None:
            self._lift_and_write(latent, depth_lat, conf, E, K_l, valid, bias_discount, t,
                                 origin=origin)
        else:
            # Multi-hypothesis lift: each pixel block may resolve into several depth modes.
            # ``modes_weight`` is the per-mode population fraction; absent modes carry weight
            # 0 (and/or nan depth) so they are dropped by the validity/positivity mask in
            # ``_lift_and_write`` (conf * 0 == 0 is filtered). Missing weights default to 1.
            for k in range(modes_depth.shape[0]):
                dk = modes_depth[k]
                wk = modes_weight[k] if modes_weight is not None else torch.ones_like(dk)
                vk = valid & torch.isfinite(dk)
                self._lift_and_write(latent, dk, conf * wk, E, K_l, vk, bias_discount, t,
                                     origin=origin)

    # -------------------------------------------------------------------- read
    @torch.no_grad()
    def read(
        self,
        E_t: torch.Tensor,
        K_l: torch.Tensor,
        h: int,
        w: int,
        top_k: int = 2,
        near: float = 1e-3,
        far: float = float("inf"),
    ) -> LayeredReadout:
        idx, pos, feat_w, W = self.store.gather_alive()
        if idx.numel() == 0:
            C = self.C
            return LayeredReadout(
                feat=torch.zeros(top_k, C, h, w, dtype=self.dtype, device=self.device),
                conf=torch.zeros(top_k, h, w, dtype=self.dtype, device=self.device),
                depth=torch.full((top_k, h, w), float("inf"), dtype=self.dtype, device=self.device),
                mask=torch.zeros(top_k, h, w, dtype=torch.bool, device=self.device),
                coverage=torch.zeros(h, w, dtype=torch.int64, device=self.device),
                composite=torch.zeros(C, h, w, dtype=self.dtype, device=self.device),
                composite_conf=torch.zeros(h, w, dtype=self.dtype, device=self.device),
                used_slots=torch.empty(0, dtype=torch.int64, device=self.device),
            )
        origin_real = self.store.alive_origin_real() if self.reserve_real_in_topk else None
        ro = topk_layered_readout(
            pos,
            feat_w,
            W,
            E_t,
            K_l,
            h,
            w,
            top_k=top_k,
            near=near,
            far=far,
            origin_real=origin_real,
            reserve_real_in_topk=self.reserve_real_in_topk,
        )
        if self.conf_report_max is not None:
            # Bound only what the conditioning packer sees. ``composite`` keeps the true
            # inverse-variance layer weights; the capped mass is what ``W/(1+W)`` reads.
            ro.conf = ro.conf.clamp_max(self.conf_report_max)
            ro.composite_conf = ro.composite_conf.clamp_max(self.conf_report_max)

        # invert whitening on present features only (affine, commutes with the mean).
        K = ro.feat.shape[0]
        feat = self.whitener.unwhiten(ro.feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = feat * ro.mask.unsqueeze(1)
        comp = self.whitener.unwhiten(ro.composite.permute(1, 2, 0)).permute(2, 0, 1)
        comp = comp * (ro.composite_conf > 0).unsqueeze(0)
        ro.feat = feat
        ro.composite = comp
        return ro

    # --------------------------------------------------------------- utilities
    @torch.no_grad()
    def fit_whitener(self, feats: torch.Tensor, freeze: bool = True) -> "EngramMemory":
        """Fit whitening stats on a sample of raw latents and (by default) freeze them.

        In deployment this is run once on dataset-level VAE latents; stats are stationary,
        so freezing keeps whiten/unwhiten exact inverses throughout the rollout.
        """
        was_frozen = self.whitener.frozen
        self.whitener.unfreeze()
        self.whitener.update(feats)
        if freeze:
            self.whitener.freeze()
        elif was_frozen:
            self.whitener.freeze()
        return self

    def memory_bytes(self) -> int:
        return self.store.memory_bytes()

    def num_slots(self) -> int:
        return len(self.store)

    @torch.no_grad()
    def real_coverage_fraction(
        self, E_t: torch.Tensor, K_l: torch.Tensor, h: int, w: int, top_k: int = 2
    ) -> float:
        """Fraction of covered cells whose frontmost layer came from a real observation.

        This is the closed-loop diagnostic that matters: when the camera returns to its
        starting viewpoint, how much of what it reads back is still the seed frame rather
        than the model's own predictions.
        """
        idx, pos, feat_w, W = self.store.gather_alive()
        if idx.numel() == 0:
            return 0.0
        ro = topk_layered_readout(pos, feat_w, W, E_t, K_l, h, w, top_k=top_k)
        if ro.used_slots.numel() == 0:
            return 0.0
        front = ro.mask[0]
        covered = int(front.sum())
        if covered == 0:
            return 0.0
        real = self.store.alive_origin_real()
        real_map = torch.zeros(h, w, dtype=torch.bool, device=ro.mask.device)
        # ``used_slots`` is ordered by (cell, depth); recover the frontmost per cell by
        # re-running the same projection and keeping rank-0 hits.
        from .readout import frustum_filter

        keep, cu, cv, z = frustum_filter(pos, E_t, K_l, h, w)
        if bool(keep.any()):
            cu, cv, z = cu[keep], cv[keep], z[keep]
            slot_ids = torch.nonzero(keep, as_tuple=False).flatten()
            cell_id = cv * w + cu
            order = torch.argsort(z, stable=True)
            order = order[torch.argsort(cell_id[order], stable=True)]
            cs = cell_id[order]
            new_group = torch.ones(cs.shape[0], dtype=torch.bool, device=cs.device)
            if cs.shape[0] > 1:
                new_group[1:] = cs[1:] != cs[:-1]
            first = order[new_group]
            real_map[cv[first], cu[first]] = real[slot_ids[first]]
        return float((real_map & front).sum()) / float(covered)

    def provenance_stats(self) -> dict:
        return self.store.provenance_stats()
