"""Autoregressive rollout loop -- Engram Algorithm 1.

Wires the latent spatial memory, the confidence estimator, a depth estimator and a
backbone generator into the read -> generate -> estimate -> update cycle. The backbone
and depth estimator are injected through light Protocols so the same loop drives:
  * the CPU integration test (a procedural ``SyntheticScene`` renderer as the world);
  * the real run (Wan2.2-TI2V-5B + DepthAnything3 on GPU, see ``engram.models``).

Per step t (target pose E_t):
  1. read  : R_t = memory.read(E_t)        -- top-K layered latent + confidence;
  2. gen   : z_t = backbone.generate(R_t)  -- conditioned latent for the chunk;
  3. depth : d_t, c_t = depth(z_t)         -- metric depth + uncertainty;
  4. conf  : w_t = confidence(proxies)     -- calibrated per-cell precision;
  5. update: memory.update(z_t, d_t, w_t)  -- discounted by gamma**t (autoregressive bias).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import torch

from .memory import EngramMemory, LayeredReadout
from .confidence import (
    ConfidenceEstimator, grazing_precision, reprojection_precision, depth_uncertainty_precision,
    generation_quality_precision, generation_quality_from_residual,
)
from .geometry.depth_align import (
    depth_disagreement_mask,
    estimate_depth_scale_to_memory,
)


class Backbone(Protocol):
    def generate(self, readout: LayeredReadout, E_t: torch.Tensor, K_l: torch.Tensor,
                 step: int, context: Optional[dict]) -> torch.Tensor:
        ...


class DepthEstimator(Protocol):
    """Returns (depth, depth_conf) or, for the multi-hypothesis lift, additionally
    (modes_depth (m,h,w), modes_weight (m,h,w))."""

    def __call__(self, latent: torch.Tensor, E_t: torch.Tensor, K_l: torch.Tensor):
        ...


@dataclass
class StepOutput:
    step: int
    latent: torch.Tensor
    depth: torch.Tensor
    confidence: torch.Tensor
    readout: LayeredReadout


@dataclass
class EngramRollout:
    memory: EngramMemory
    backbone: Backbone
    depth_estimator: DepthEstimator
    confidence: ConfidenceEstimator
    K_l: torch.Tensor
    h: int
    w: int
    top_k: int = 2
    bias_gamma: float = 0.98          # autoregressive bias discount base in (0, 1]
    reproj_scale: float = 1.0
    # Memory write gating (pollution mitigation). Cells below
    # ``min_write_precision`` are not written; a whole step is skipped when its mean genq
    # precision is below ``min_write_genq``. 0 == disabled (legacy behaviour).
    min_write_precision: float = 0.0
    min_write_genq: float = 0.0
    # Depth scale alignment against the memory readout.
    # Without it a per-frame monocular scale error turns into world-space drift that
    # compounds over the rollout. 0 / False keep the legacy unaligned behaviour.
    depth_align_to_memory: bool = False
    depth_align_min_coverage: float = 0.05
    depth_align_max_ratio: float = 1.43
    depth_align_momentum: float = 0.5
    depth_align_conf_floor: float = 0.0
    # Geometric write gate: skip cells still contradicting the memory after alignment.
    geom_write_reject_tol: float = 0.0
    proxy_history: List[float] = field(default_factory=list)
    depth_scale_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._depth_scale_ema: Optional[float] = None

    @torch.no_grad()
    def step(self, E_t: torch.Tensor, step: int, context: Optional[dict] = None) -> StepOutput:
        readout = self.memory.read(E_t, self.K_l, self.h, self.w, top_k=self.top_k)
        latent = self.backbone.generate(readout, E_t, self.K_l, step, context)
        assert latent.shape == (self.memory.C, self.h, self.w), "backbone latent shape mismatch"

        dout = self.depth_estimator(latent, E_t, self.K_l)
        if len(dout) == 4:
            depth, depth_conf, modes_depth, modes_weight = dout
        else:
            depth, depth_conf = dout
            modes_depth = modes_weight = None

        # Align before anything consumes the depth, so the write and the geometric gate
        # both see one scale for this frame.
        depth_valid = torch.isfinite(depth) & (depth > 0)
        estimate = estimate_depth_scale_to_memory(
            readout.depth[0],
            readout.mask[0],
            readout.conf[0],
            depth,
            depth_valid,
            min_coverage=self.depth_align_min_coverage,
            max_ratio=self.depth_align_max_ratio,
            conf_floor=self.depth_align_conf_floor,
            previous=self._depth_scale_ema,
            momentum=self.depth_align_momentum,
        )
        if estimate.trusted:
            self._depth_scale_ema = estimate.scale
        if self.depth_align_to_memory and estimate.trusted and estimate.scale != 1.0:
            depth = depth * estimate.scale
            if modes_depth is not None:
                modes_depth = modes_depth * estimate.scale
        self.depth_scale_history.append(estimate.scale)

        proxies: Dict[str, torch.Tensor] = {
            "depth": depth_uncertainty_precision(depth_conf=depth_conf.clamp(1e-4, 1 - 1e-4)),
            "graze": grazing_precision(depth, self.K_l),
        }
        confidence_weights = getattr(self.confidence, "weights", {})
        # reprojection consistency where memory already had coverage at this view
        covered = readout.composite_conf > 0
        if confidence_weights.get("reproj", 0.0) and covered.any():
            residual = torch.linalg.vector_norm(latent - readout.composite, dim=0)
            proxies["reproj"] = reprojection_precision(
                residual, scale=self.reproj_scale, valid=covered
            )
        # generation-quality proxy (Sec 3.2): prefer an explicit per-token predictive
        # log-variance if the backbone exposes one, else use the final-step denoising residual.
        logvar = getattr(self.backbone, "last_token_logvar", None)
        if confidence_weights.get("genq", 0.0) and logvar is not None:
            proxies["genq"] = generation_quality_precision(logvar.to(latent.device))
        elif confidence_weights.get("genq", 0.0):
            gres = getattr(self.backbone, "last_generation_residual", None)
            if gres is not None:
                gres = gres.to(latent.device)
                if gres.dim() == 3:               # (T,h,w) chunk -> this step's frame
                    gres = gres[0]
                proxies["genq"] = generation_quality_from_residual(gres)

        precision = self.confidence.precision(proxies, calibrate=self.confidence.calibrator is not None)
        discount = self.bias_gamma ** step
        write_step = True
        if self.min_write_genq > 0.0 and "genq" in proxies:
            write_step = float(proxies["genq"].mean()) >= self.min_write_genq
        write_valid = None
        if self.min_write_precision > 0.0:
            write_valid = precision >= self.min_write_precision
        if self.geom_write_reject_tol > 0.0:
            keep = ~depth_disagreement_mask(
                readout.depth[0], readout.mask[0], depth, self.geom_write_reject_tol
            )
            write_valid = keep if write_valid is None else (write_valid & keep)
        if write_step:
            self.memory.update(latent, depth, precision, E_t, self.K_l, t=step + 1,
                               bias_discount=discount, valid=write_valid,
                               modes_depth=modes_depth, modes_weight=modes_weight)
        return StepOutput(step=step, latent=latent, depth=depth, confidence=precision, readout=readout)

    @torch.no_grad()
    def run(self, poses: List[torch.Tensor], contexts: Optional[List[dict]] = None) -> List[StepOutput]:
        outs = []
        for i, E in enumerate(poses):
            ctx = contexts[i] if contexts is not None else None
            outs.append(self.step(E, i, ctx))
        return outs
