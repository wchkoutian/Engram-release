"""DepthAnything3 wrapper (ByteDance-Seed/Depth-Anything-3).

Provides metric depth + confidence used by Engram's confidence proxies and the
back-projection. Official API:
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device)
    pred = model.inference([image])      # pred.depth (N,H,W), pred.conf (N,H,W)
Metric scaling: DA3NESTED-GIANT-LARGE is already in metres; DA3METRIC-LARGE requires
metric_depth = focal * net_output / 300, with focal measured on DA3's output grid.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..geometry.depth import downsample_depth_to_latent, top_m_depth_modes


_FOCAL_SCALED_MODEL_NAMES = {"da3metric-large"}


def _model_name(repo: str | os.PathLike[str]) -> str:
    """Return the final component of either a local path or a Hub model ID."""
    normalized = os.fspath(repo).rstrip("/\\").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].casefold()


def _needs_focal_scale(repo: str | os.PathLike[str]) -> bool:
    return _model_name(repo) in _FOCAL_SCALED_MODEL_NAMES


class DepthAnything3Estimator:
    def __init__(self, repo: str = "depth-anything/DA3METRIC-LARGE", device: str = "cuda",
                 ensemble_conf: bool = True):
        from depth_anything_3.api import DepthAnything3

        self.repo = repo
        self.device = device
        self.needs_focal_scale = _needs_focal_scale(repo)
        # When the model exposes no native confidence, derive one from flip-augmentation
        # ensemble consistency (the multi-scale/ensemble agreement proxy of Sec 3.2), never a
        # constant. Set ensemble_conf=False to require a native confidence map instead.
        self.ensemble_conf = ensemble_conf
        self.model = DepthAnything3.from_pretrained(repo).to(device)
        self.model.eval()

    def _scale(
        self,
        depth: np.ndarray,
        intrinsics: Optional[np.ndarray],
        image_shapes: List[Tuple[int, int]],
    ) -> np.ndarray:
        if not self.needs_focal_scale:
            return depth
        if intrinsics is None:
            raise ValueError("DA3METRIC-LARGE needs focal length for metric scaling")
        if depth.ndim < 2:
            raise ValueError(f"depth must have at least two dimensions, got {depth.shape}")

        batch = 1 if depth.ndim == 2 else int(depth.shape[0])
        if len(image_shapes) != batch:
            raise ValueError(
                f"received {len(image_shapes)} source image shapes for a depth batch of {batch}"
            )
        Ks = np.asarray(intrinsics)
        if Ks.shape == (3, 3):
            Ks = np.broadcast_to(Ks, (batch, 3, 3))
        elif Ks.shape != (batch, 3, 3):
            raise ValueError(
                "intrinsics must have shape (3, 3) or (N, 3, 3), "
                f"got {Ks.shape} for N={batch}"
            )

        depth_h, depth_w = depth.shape[-2:]
        scale_factors = []
        for K, (image_h, image_w) in zip(Ks, image_shapes):
            if image_h <= 0 or image_w <= 0:
                raise ValueError(f"invalid source image shape {(image_h, image_w)}")
            # K is expressed on the source-image grid while DA3 predicts on its
            # processed/output grid. Convert each focal axis before applying the
            # DA3METRIC-LARGE focal/300 contract.
            fx = float(K[0, 0]) * depth_w / image_w
            fy = float(K[1, 1]) * depth_h / image_h
            scale_factors.append(0.5 * (fx + fy) / 300.0)

        factors = np.asarray(scale_factors, dtype=depth.dtype)
        if depth.ndim == 2:
            return depth * factors[0]
        return depth * factors.reshape((batch,) + (1,) * (depth.ndim - 1))

    @torch.no_grad()
    def infer(self, images: List[np.ndarray], intrinsics: Optional[np.ndarray] = None):
        """images: list of HxWx3 uint8/float arrays. Returns (depth (N,H,W) m, conf (N,H,W) in (0,1])."""
        pred = self.model.inference(images)
        image_shapes = [(int(image.shape[0]), int(image.shape[1])) for image in images]
        depth = self._scale(
            np.asarray(pred.depth, dtype=np.float32), intrinsics, image_shapes
        )
        if hasattr(pred, "conf") and pred.conf is not None:
            conf = np.asarray(pred.conf, dtype=np.float32)
            return depth, conf
        if not self.ensemble_conf:
            raise RuntimeError(
                "DepthAnything3 returned no confidence; enable ensemble_conf or use a model "
                "that outputs a confidence map.")
        # Flip-TTA ensemble: high agreement between the original and the horizontally
        # flipped-then-unflipped prediction == high confidence.
        flipped = [np.ascontiguousarray(im[:, ::-1]) for im in images]
        pred2 = self.model.inference(flipped)
        depth2 = self._scale(
            np.asarray(pred2.depth, dtype=np.float32)[:, :, ::-1],
            intrinsics,
            image_shapes,
        )
        denom = np.abs(depth) + np.abs(depth2) + 1e-6
        rel = 2.0 * np.abs(depth - depth2) / denom            # symmetric relative disagreement
        conf = np.exp(-rel).astype(np.float32)                # in (0, 1]
        return depth, conf

    @torch.no_grad()
    def depth_for_view(
        self,
        image: np.ndarray,
        K_pixel: np.ndarray,
        h: int,
        w: int,
        H: int,
        W: int,
        mode: str = "bilinear",
        top_m: int = 0,
    ):
        """Estimate metric depth for one frame and return it at latent resolution.

        Returns (depth_lat (h,w), conf_lat (h,w)) and, if top_m>0, multi-hypothesis
        (modes_depth (m,h,w), modes_weight (m,h,w)) for the occlusion-aware lift.
        """
        depth, conf = self.infer([image], intrinsics=K_pixel)
        depth_t = torch.from_numpy(depth[0])
        conf_t = torch.from_numpy(conf[0])
        depth_lat = downsample_depth_to_latent(depth_t, h, w, mode=mode)
        conf_lat = downsample_depth_to_latent(conf_t, h, w, mode="area")
        if top_m > 0:
            modes_d, modes_w = top_m_depth_modes(depth_t, h, w, m=top_m)
            return depth_lat, conf_lat, modes_d, modes_w
        return depth_lat, conf_lat
