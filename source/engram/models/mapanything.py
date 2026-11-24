"""Chunk-wise MapAnything depth estimation for generated-memory writes."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..geometry.depth import resample_depth_to_latent_camera


def build_eval_depth_estimator(cfg: dict, device: str):
    """Build the explicitly selected evaluation geometry backend."""
    backend = str(cfg.get("depth_backend", "da3")).lower()
    if backend == "mapanything":
        return MapAnythingEstimator(
            repo=cfg.get("mapanything_repo", "facebook/map-anything"),
            device=device,
            torch_home=cfg.get("mapanything_torch_home"),
        )
    if backend == "da3":
        from .depth_anything3 import DepthAnything3Estimator

        return DepthAnything3Estimator(
            repo=cfg.get("depth_repo", "depth-anything/DA3METRIC-LARGE"),
            device=device,
        )
    raise ValueError(f"depth_backend must be 'mapanything' or 'da3', got {backend!r}")


def resolve_eval_multi_hypothesis_m(cfg: dict, variant_value: int) -> int:
    """Resolve the explicit evaluation depth-mode count for the selected backend."""
    value = int(cfg.get("multi_hypothesis_m_override", variant_value))
    if value < 0:
        raise ValueError("multi_hypothesis_m_override must be non-negative")
    backend = str(cfg.get("depth_backend", "da3")).lower()
    if backend == "mapanything" and value != 0:
        raise ValueError(
            "MapAnything requires multi_hypothesis_m_override: 0; "
            f"the selected variant requests {value}"
        )
    return value


class MapAnythingEstimator:
    """Run one joint reconstruction over all latent-aligned views in a chunk.

    MapAnything exposes an uncalibrated ranking score rather than the [0, 1]
    probability expected by Engram's depth-confidence proxy. Mirage does not use
    this score, so the aligned Engram contract uses a neutral probability of 0.5.
    """

    def __init__(
        self,
        repo: str = "facebook/map-anything",
        device: str = "cuda",
        torch_home: str | None = None,
    ) -> None:
        if torch_home is not None:
            torch.hub.set_dir(str(Path(torch_home).expanduser() / "hub"))
        from mapanything.models import MapAnything

        self.repo = repo
        self.torch_home = torch_home
        self.device = torch.device(device)
        self.model = MapAnything.from_pretrained(repo).to(self.device)
        self.model.eval()

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _depth_2d(prediction: dict) -> torch.Tensor:
        depth = torch.as_tensor(prediction["depth_z"]).detach().float().cpu()
        while depth.ndim > 2 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        while depth.ndim > 2 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.ndim != 2:
            raise ValueError(
                f"MapAnything depth_z must reduce to (H, W), got {tuple(depth.shape)}"
            )
        return depth

    @torch.no_grad()
    def depth_for_views(
        self,
        images: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray | torch.Tensor],
        poses_w2c: Sequence[np.ndarray | torch.Tensor],
        h: int,
        w: int,
        *,
        mode: str = "bilinear",
        top_m: int = 0,
    ) -> list[dict[str, torch.Tensor]]:
        """Return latent-grid depth/confidence for a jointly reconstructed chunk."""
        if top_m != 0:
            raise ValueError("MapAnything chunk inference does not support top_m depth modes")
        from mapanything.utils.image import preprocess_inputs

        if not (len(images) == len(intrinsics) == len(poses_w2c)):
            raise ValueError("images, intrinsics, and poses_w2c must have equal length")
        if not images:
            return []

        views = []
        for image, K, E in zip(images, intrinsics, poses_w2c):
            E_tensor = torch.as_tensor(E, dtype=torch.float32)
            views.append(
                {
                    "img": np.asarray(image),
                    "intrinsics": self._as_numpy(K),
                    "camera_poses": self._as_numpy(torch.linalg.inv(E_tensor)),
                    "is_metric_scale": torch.tensor([True]),
                }
            )

        processed_views = preprocess_inputs(views)
        predictions = self.model.infer(
            processed_views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
        if len(predictions) != len(views):
            raise ValueError(
                "MapAnything returned a different number of predictions and input views: "
                f"{len(predictions)} != {len(views)}"
            )

        output = []
        for prediction, processed_view, original_intrinsics, image in zip(predictions, processed_views, intrinsics, images):
            depth = self._depth_2d(prediction)
            valid = prediction.get("mask")
            valid = torch.isfinite(depth) & (depth > 0) if valid is None else torch.as_tensor(valid).detach().cpu().reshape(depth.shape).bool()
            target_intrinsics = torch.as_tensor(self._as_numpy(original_intrinsics)).clone()
            target_intrinsics[0] *= w / np.asarray(image).shape[1]
            target_intrinsics[1] *= h / np.asarray(image).shape[0]
            depth_lat, valid_lat = resample_depth_to_latent_camera(depth, valid, processed_view["intrinsics"][0], target_intrinsics, h, w, mode=mode)
            output.append(
                {
                    "depth": depth_lat,
                    "conf": torch.full_like(depth_lat, 0.5),
                    "mask": valid_lat,
                }
            )
        return output
