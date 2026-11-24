"""Sparse voxel-hash store for latent spatial memory (Engram idea-1, Sec 3.3-3.5).

Each *slot* holds one fused latent anchor: a world position, a fused latent feature
(stored in whitened space by the orchestrator) and an accumulated precision weight ``W``.
A voxel may hold several slots so genuinely distinct surfaces (occlusion layers) are not
averaged together.

Two write behaviours are supported, so the union baseline and the fusion method share one
read path:
  * ``fuse=True``  (default, Engram): observations falling in the same voxel layer are
    combined by an online inverse-variance update (Eq.6'): ``f <- (W f + w f_i)/(W + w)``.
  * ``fuse=False`` (Mirage union baseline): every observation is appended as its own slot;
    the top-K depth readout then reproduces Mirage's frontmost z-buffer at K=1.

Bounded provenance-aware fusion
-------------------------------
Pure unbounded inverse-variance fusion is the minimum-variance estimator only when the
observation errors are independent. During a closed-loop rollout they are not: after the
seed frame every observation is *generated*, and their errors are strongly correlated
through the model that produced them. Under that correlation, averaging n observations
drives the single real observation's contribution to 1/n, so a camera that returns to its
starting viewpoint finds the evidence it needs has been averaged away by its own
predictions. The store therefore optionally tracks provenance and bounds how much
generated evidence may displace a real observation:

  * ``real_weight_scale``    -- multiplies the write weight of real observations;
  * ``gen_into_real_scale``  -- attenuates generated weight fused into a real-seeded slot;
  * ``gen_into_real_budget`` -- caps the *cumulative* generated weight injected into a
    real-seeded slot at ``budget * W_real``, so the real observation keeps at least
    ``1/(1+budget)`` of the estimate no matter how long the rollout runs;
  * ``w_max``                -- optional saturating cap on the fusion accumulator. This
    turns fusion into an exponential moving average with time constant ``w_max/w``; it
    bounds the confidence channel but *increases* recency bias, so it is off by default
    and exists to be swept against ``EngramMemory(conf_report_max=...)``, which bounds the
    reported confidence without touching the estimator.

Geometry gating
---------------
The consistency gate originally compared *features* only. Two observations of the same
surface whose depths disagree have near-identical features, so the feature gate declares
them consistent and their world positions get averaged, smearing the geometry. When
``tau_geom`` is set the gate additionally requires the positions to agree to within
``tau_geom`` voxel lengths, and ``fuse_position=False`` stops averaging positions
altogether (a drifted observation can then no longer drag a settled anchor).

idea-1 scope: this store has no capacity bound and no eviction (those belong to the
separate bounded-memory idea and are deliberately excluded here). The voxel key is used
purely as a fusion index. Storage is preallocated and grows by doubling when full, so the
write path stays vectorised without an explicit free list.

Efficiency notes:
  * Within a single ``write`` call, observations falling in the same voxel are
    pre-aggregated with a single vectorised inverse-variance reduction (``index_add_``).
  * Brand-new voxels (the dominant case while exploring) and the whole union baseline are
    written with one vectorised batch append; only voxels that already hold slots run the
    per-voxel consistency gate. This is the natural place for a CUDA spatial-hash kernel;
    the Python version here is exact and bug-checked.

Whitening is the orchestrator's responsibility (``engram.memory.EngramMemory``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


REAL_ORIGIN = "real"
GENERATED_ORIGIN = "generated"
_ORIGINS = (REAL_ORIGIN, GENERATED_ORIGIN)


class VoxelHashMemory:
    def __init__(
        self,
        channels: int,
        voxel_size: float,
        consistency_tau: Optional[float] = None,
        max_layers: int = 8,
        fuse: bool = True,
        initial_capacity: int = 4096,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        w_max: Optional[float] = None,
        real_weight_scale: float = 1.0,
        gen_into_real_scale: float = 1.0,
        gen_into_real_budget: Optional[float] = None,
        tau_geom: Optional[float] = None,
        fuse_position: bool = True,
        drop_on_layer_budget: bool = False,
        immutable_real_memory: bool = False,
    ):
        assert channels > 0 and voxel_size > 0 and initial_capacity > 0
        if w_max is not None and float(w_max) <= 0:
            raise ValueError(f"w_max must be positive when set, got {w_max}")
        if float(real_weight_scale) <= 0:
            raise ValueError(f"real_weight_scale must be positive, got {real_weight_scale}")
        if not 0.0 <= float(gen_into_real_scale) <= 1.0:
            raise ValueError(
                f"gen_into_real_scale must lie in [0, 1], got {gen_into_real_scale}"
            )
        if gen_into_real_budget is not None and float(gen_into_real_budget) < 0:
            raise ValueError(
                f"gen_into_real_budget must be non-negative when set, got {gen_into_real_budget}"
            )
        if tau_geom is not None and float(tau_geom) <= 0:
            raise ValueError(f"tau_geom must be positive when set, got {tau_geom}")
        self.C = channels
        self.voxel_size = float(voxel_size)
        self.tau = consistency_tau
        self.max_layers = int(max_layers)
        self.fuse = bool(fuse)
        self.device = torch.device(device)
        self.dtype = dtype
        self.w_max = None if w_max is None else float(w_max)
        self.real_weight_scale = float(real_weight_scale)
        self.gen_into_real_scale = float(gen_into_real_scale)
        self.gen_into_real_budget = (
            None if gen_into_real_budget is None else float(gen_into_real_budget)
        )
        self.tau_geom = None if tau_geom is None else float(tau_geom)
        self.fuse_position = bool(fuse_position)
        self.drop_on_layer_budget = bool(drop_on_layer_budget)
        self.immutable_real_memory = bool(immutable_real_memory)

        self._cap = int(initial_capacity)
        self.pos = torch.zeros(self._cap, 3, dtype=dtype, device=self.device)
        self.feat = torch.zeros(self._cap, channels, dtype=dtype, device=self.device)
        self.W = torch.zeros(self._cap, dtype=dtype, device=self.device)
        self.vox = torch.zeros(self._cap, 3, dtype=torch.int64, device=self.device)
        # Provenance / drift bookkeeping. ``pos0`` and ``birth`` are diagnostics only;
        # ``origin_real``, ``W_real`` and ``W_gen`` drive the bounded fusion policy.
        self.origin_real = torch.zeros(self._cap, dtype=torch.bool, device=self.device)
        self.W_real = torch.zeros(self._cap, dtype=dtype, device=self.device)
        self.W_gen = torch.zeros(self._cap, dtype=dtype, device=self.device)
        self.pos0 = torch.zeros(self._cap, 3, dtype=dtype, device=self.device)
        self.birth = torch.zeros(self._cap, dtype=torch.int64, device=self.device)

        self.index: Dict[Tuple[int, int, int], List[int]] = {}
        self._n = 0          # number of slots in use; slots [0, _n) are alive
        self._t = 0
        self.stats: Dict[str, int] = {}
        self.reset_stats()

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return self._n

    def reset_stats(self) -> None:
        self.stats = {
            "appended": 0,
            "fused": 0,
            "new_layer": 0,
            "gated_feature": 0,
            "gated_geometry": 0,
            "dropped_layer_budget": 0,
            "dropped_gen_budget": 0,
            "dropped_immutable_real": 0,
        }

    def voxel_of(self, pos: torch.Tensor) -> torch.Tensor:
        return torch.floor(pos / self.voxel_size).to(torch.int64)

    def memory_bytes(self) -> int:
        per_slot = (3 + self.C + 1 + 3) * 4  # pos + feat + W + vox, approximated at 4B
        return self._n * per_slot

    def _grow(self, needed: int) -> None:
        if needed <= self._cap:
            return
        new_cap = self._cap
        while new_cap < needed:
            new_cap *= 2

        def _resize(t: torch.Tensor) -> torch.Tensor:
            shape = list(t.shape)
            shape[0] = new_cap
            nt = torch.zeros(shape, dtype=t.dtype, device=t.device)
            nt[: self._cap] = t
            return nt

        self.pos = _resize(self.pos)
        self.feat = _resize(self.feat)
        self.W = _resize(self.W)
        self.vox = _resize(self.vox)
        self.origin_real = _resize(self.origin_real)
        self.W_real = _resize(self.W_real)
        self.W_gen = _resize(self.W_gen)
        self.pos0 = _resize(self.pos0)
        self.birth = _resize(self.birth)
        self._cap = new_cap

    def _alloc(self, key: Tuple[int, int, int], pos, feat, W, is_real: bool = False) -> int:
        s = self._n
        if s >= self._cap:
            self._grow(s + 1)
        self.pos[s] = pos
        self.pos0[s] = pos
        self.feat[s] = feat
        self.W[s] = W if self.w_max is None else torch.clamp(W, max=self.w_max)
        self.vox[s] = torch.tensor(key, dtype=torch.int64, device=self.device)
        self.origin_real[s] = bool(is_real)
        self.W_real[s] = W if is_real else 0
        self.W_gen[s] = 0
        self.birth[s] = int(self._t)
        self.index.setdefault(key, []).append(s)
        self._n += 1
        self.stats["appended"] += 1
        return s

    def _append_batch(
        self, keys: List[Tuple[int, int, int]], pos, feat, W, is_real: bool = False
    ) -> None:
        """Vectorised append of ``n`` slots (one per key). Keys must be distinct *within*
        this call (guaranteed: they come from ``torch.unique`` of the write's voxels)."""
        n = len(keys)
        if n == 0:
            return
        s0 = self._n
        if s0 + n > self._cap:
            self._grow(s0 + n)
        sl = slice(s0, s0 + n)
        self.pos[sl] = pos
        self.pos0[sl] = pos
        self.feat[sl] = feat
        self.W[sl] = W if self.w_max is None else torch.clamp(W, max=self.w_max)
        self.vox[sl] = torch.tensor(keys, dtype=torch.int64, device=self.device)
        self.origin_real[sl] = bool(is_real)
        self.W_real[sl] = W if is_real else torch.zeros_like(W)
        self.W_gen[sl] = 0
        self.birth[sl] = int(self._t)
        for i, key in enumerate(keys):
            self.index.setdefault(key, []).append(s0 + i)
        self._n += n
        self.stats["appended"] += n

    def _effective_weight(self, s: int, W: torch.Tensor, is_real: bool) -> Optional[torch.Tensor]:
        """Apply the provenance policy to one generated write into an existing slot.

        Returns ``None`` when the write is fully absorbed by an exhausted budget.
        """
        if is_real or not bool(self.origin_real[s]):
            return W
        if self.immutable_real_memory:
            self.stats["dropped_immutable_real"] += 1
            return None
        w_eff = W
        if self.gen_into_real_scale != 1.0:
            w_eff = w_eff * self.gen_into_real_scale
        if self.gen_into_real_budget is not None:
            remaining = self.gen_into_real_budget * self.W_real[s] - self.W_gen[s]
            remaining = torch.clamp(remaining, min=0)
            w_eff = torch.minimum(w_eff, remaining)
            if float(w_eff) <= 0:
                self.stats["dropped_gen_budget"] += 1
                return None
        self.W_gen[s] = self.W_gen[s] + w_eff
        return w_eff

    def _fuse_into(self, s: int, pos, feat, W, is_real: bool = False) -> None:
        w_eff = self._effective_weight(s, W, is_real)
        if w_eff is None:
            return
        Wold = self.W[s]
        denom = Wold + w_eff
        if float(denom) <= 0:
            return
        self.feat[s] = (Wold * self.feat[s] + w_eff * feat) / denom
        if self.fuse_position:
            self.pos[s] = (Wold * self.pos[s] + w_eff * pos) / denom
        elif not bool(self.origin_real[s]) and float(w_eff) > float(Wold):
            # Strictly stronger evidence relocates the anchor; weaker evidence may refine
            # the feature but must not drag settled geometry.
            self.pos[s] = pos
        self.W[s] = denom if self.w_max is None else torch.clamp(denom, max=self.w_max)
        if is_real:
            self.W_real[s] = self.W_real[s] + w_eff
            self.origin_real[s] = True
        self.stats["fused"] += 1

    def _match_slot(
        self, slots: List[int], feat, pos, is_real: bool = False
    ) -> Optional[int]:
        """Pick the layer this observation should fuse into, or ``None`` to open a new one.

        A layer is eligible when it agrees in appearance (``tau``) *and*, when
        ``tau_geom`` is set, in position. Among eligible layers the closest in feature
        space wins. Requiring both keeps a depth-drifted duplicate of the same surface out
        of an existing layer instead of averaging it into place.
        """
        if self.immutable_real_memory and not is_real:
            slots = [slot for slot in slots if not bool(self.origin_real[slot])]
        if not slots:
            return None
        slot_idx = torch.tensor(slots, device=self.device)
        d_feat = torch.linalg.vector_norm(self.feat[slot_idx] - feat.unsqueeze(0), dim=-1)
        eligible = (
            torch.ones_like(d_feat, dtype=torch.bool)
            if self.tau is None
            else d_feat < self.tau
        )
        if not bool(eligible.any()):
            self.stats["gated_feature"] += 1
            return None
        if self.tau_geom is not None:
            d_geom = torch.linalg.vector_norm(
                self.pos[slot_idx] - pos.unsqueeze(0), dim=-1
            ) / self.voxel_size
            geom_ok = eligible & (d_geom < self.tau_geom)
            if not bool(geom_ok.any()):
                self.stats["gated_geometry"] += 1
                return None
            eligible = geom_ok
        masked = torch.where(eligible, d_feat, torch.full_like(d_feat, float("inf")))
        return slots[int(torch.argmin(masked))]

    # -------------------------------------------------------------------- write
    @torch.no_grad()
    def write(
        self,
        pos: torch.Tensor,
        feat: torch.Tensor,
        weight: torch.Tensor,
        t: Optional[int] = None,
        origin: str = GENERATED_ORIGIN,
    ) -> None:
        """Insert/fuse N observations. ``feat`` is assumed already whitened.

        ``origin`` marks the provenance of this batch: ``"real"`` for observations lifted
        from an actually observed image, ``"generated"`` for anything the model produced.
        """
        if origin not in _ORIGINS:
            raise ValueError(f"origin must be one of {_ORIGINS}, got {origin!r}")
        if pos.numel() == 0:
            return
        if t is not None:
            self._t = max(self._t, int(t))
        is_real = origin == REAL_ORIGIN
        if is_real and self.real_weight_scale != 1.0:
            weight = weight * self.real_weight_scale

        vc = self.voxel_of(pos)
        uniq, inv = torch.unique(vc, dim=0, return_inverse=True)
        G = uniq.shape[0]
        wsum = torch.zeros(G, dtype=self.dtype, device=self.device).index_add_(0, inv, weight)
        fsum = torch.zeros(G, self.C, dtype=self.dtype, device=self.device).index_add_(
            0, inv, weight.unsqueeze(-1) * feat
        )
        psum = torch.zeros(G, 3, dtype=self.dtype, device=self.device).index_add_(
            0, inv, weight.unsqueeze(-1) * pos
        )
        wsafe = wsum.clamp_min(torch.finfo(self.dtype).tiny)
        agg_feat = fsum / wsafe.unsqueeze(-1)
        agg_pos = psum / wsafe.unsqueeze(-1)

        # Worst case each touched voxel allocates one new slot this call.
        self._grow(self._n + G)

        keys = [(int(uniq[g, 0]), int(uniq[g, 1]), int(uniq[g, 2])) for g in range(G)]

        if not self.fuse:
            # Union baseline: every observation is its own slot -> one vectorised append.
            self._append_batch(keys, agg_pos, agg_feat, wsum, is_real=is_real)
            return

        # Split into brand-new voxels (vectorised append) and voxels that already hold
        # slots (per-voxel consistency gate). Within this call keys are distinct, so the
        # batch append cannot collide with itself.
        new_g = [g for g, key in enumerate(keys) if key not in self.index]
        existing_g = [g for g, key in enumerate(keys) if key in self.index]

        if new_g:
            sel = torch.tensor(new_g, device=self.device)
            self._append_batch(
                [keys[g] for g in new_g],
                agg_pos[sel],
                agg_feat[sel],
                wsum[sel],
                is_real=is_real,
            )

        for g in existing_g:
            key = keys[g]
            af, ap, aw = agg_feat[g], agg_pos[g], wsum[g]
            slots = self.index[key]
            match = self._match_slot(slots, af, ap, is_real=is_real)
            if match is not None:
                # Consistent with an existing layer -> inverse-variance fusion.
                self._fuse_into(match, ap, af, aw, is_real=is_real)
            elif len(slots) < self.max_layers:
                # Inconsistent surface -> open a new depth sublayer rather than averaging.
                self._alloc(key, ap, af, aw, is_real=is_real)
                self.stats["new_layer"] += 1
            elif self.drop_on_layer_budget:
                # Every layer disagrees and there is no room for another: forcing the
                # observation into the nearest layer would blend distinct surfaces, so
                # discard it instead.
                self.stats["dropped_layer_budget"] += 1
            else:
                # Legacy fallback: fuse the closest layer rather than overwriting, to
                # bound layers per voxel.
                mutable_slots = (
                    [slot for slot in slots if not bool(self.origin_real[slot])]
                    if self.immutable_real_memory and not is_real
                    else slots
                )
                if not mutable_slots:
                    self.stats["dropped_immutable_real"] += 1
                    continue
                slot_idx = torch.tensor(mutable_slots, device=self.device)
                d = torch.linalg.vector_norm(self.feat[slot_idx] - af.unsqueeze(0), dim=-1)
                self._fuse_into(
                    mutable_slots[int(torch.argmin(d))], ap, af, aw, is_real=is_real
                )

    # --------------------------------------------------------------------- read
    @torch.no_grad()
    def gather_alive(self):
        idx = torch.arange(self._n, device=self.device)
        return idx, self.pos[: self._n], self.feat[: self._n], self.W[: self._n]

    @torch.no_grad()
    def alive_origin_real(self) -> torch.Tensor:
        return self.origin_real[: self._n]

    @torch.no_grad()
    def alive_birth(self) -> torch.Tensor:
        return self.birth[: self._n]

    @torch.no_grad()
    def position_drift(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-slot ``||pos - pos_at_creation||``, optionally restricted to ``mask``."""
        drift = torch.linalg.vector_norm(
            self.pos[: self._n] - self.pos0[: self._n], dim=-1
        )
        return drift if mask is None else drift[mask]

    def provenance_stats(self) -> Dict[str, float]:
        """Slot-level provenance and drift summary for rollout diagnostics."""
        if self._n == 0:
            return {"num_slots": 0, "real_slots": 0, "real_fraction": 0.0}
        real = self.origin_real[: self._n]
        real_count = int(real.sum())
        out: Dict[str, float] = {
            "num_slots": int(self._n),
            "real_slots": real_count,
            "real_fraction": real_count / float(self._n),
            **{f"write_{key}": int(value) for key, value in self.stats.items()},
        }
        if real_count:
            real_drift = self.position_drift(real)
            out["real_pos_drift_mean"] = float(real_drift.mean())
            out["real_pos_drift_max"] = float(real_drift.max())
            out["real_pos_drift_voxels"] = float(real_drift.mean() / self.voxel_size)
            out["real_gen_weight_ratio"] = float(
                (self.W_gen[: self._n][real].sum()
                 / self.W_real[: self._n][real].sum().clamp_min(1e-12))
            )
        return out
