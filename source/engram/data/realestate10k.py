"""RealEstate10K camera parsing and lightweight precomputed-cache access."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.wan_latents import (
    WAN_LATENT_CONTRACT_NAME,
    WAN_VAE_TEMPORAL_STRIDE,
    validate_wan_latent_contract,
)


LATENT_CONTRACT_KEY = "__latent_contract__"
DATASET_ID_KEY = "__latent_dataset_id__"


@dataclass
class CameraFrame:
    timestamp: int
    K_norm: np.ndarray
    w2c: np.ndarray


def parse_cameras(path: str) -> Tuple[str, List[CameraFrame]]:
    """Parse one RealEstate10K camera text file.

    Expected format: first line URL, then rows
    ``timestamp fx fy cx cy 0 0 e00 ... e23`` where intrinsics are normalized and the
    final 12 values are a row-major 3x4 world-to-camera matrix.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"empty camera file: {path}")
    url = lines[0]
    frames: List[CameraFrame] = []
    for line in lines[1:]:
        vals = [float(x) for x in line.split()]
        if len(vals) < 19:
            raise ValueError(f"camera row must have at least 19 values, got {len(vals)}: {line}")
        timestamp = int(vals[0])
        k_norm = np.asarray(vals[1:5], dtype=np.float64)
        w2c = np.asarray(vals[7:19], dtype=np.float64).reshape(3, 4)
        frames.append(CameraFrame(timestamp=timestamp, K_norm=k_norm, w2c=w2c))
    return url, frames


def denorm_intrinsics(K_norm: np.ndarray | Sequence[float], W: int, H: int) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(K_norm, dtype=np.float64).tolist()
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx * W
    K[1, 1] = fy * H
    K[0, 2] = cx * W
    K[1, 2] = cy * H
    return K


def w2c_to_4x4(w2c: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=np.asarray(w2c).dtype)
    E[:3, :4] = np.asarray(w2c).reshape(3, 4)
    return E


class PrecomputedStore:
    """Tiny LMDB-backed dictionary for per-frame precomputed tensors."""

    def __init__(self, path: str, readonly: bool = True, map_size: int = 1 << 40):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("PrecomputedStore requires the optional 'lmdb' package") from exc
        self._lmdb = lmdb
        self.path = path
        self.readonly = bool(readonly)
        self.env = lmdb.open(
            path,
            readonly=self.readonly,
            lock=not self.readonly,
            readahead=False,
            meminit=False,
            map_size=map_size,
            subdir=True,
        )

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if self.readonly:
            raise RuntimeError("cannot put into a readonly PrecomputedStore")
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def get(self, key: str) -> Dict[str, Any]:
        with self.env.begin(write=False) as txn:
            raw = txn.get(key.encode("utf-8"))
        if raw is None:
            raise KeyError(key)
        return pickle.loads(raw)

    def put_meta(self, clips_meta: Sequence[Tuple[str, int]]) -> None:
        self.put("__meta__", list(clips_meta))

    def get_meta(self) -> List[Tuple[str, int]]:
        try:
            return list(self.get("__meta__"))
        except KeyError:
            return []

    def put_latent_contract(self, contract: Dict[str, Any]) -> None:
        self.put(LATENT_CONTRACT_KEY, dict(contract))

    def get_latent_contract(self) -> Optional[Dict[str, Any]]:
        try:
            return dict(self.get(LATENT_CONTRACT_KEY))
        except KeyError:
            return None

    def require_latent_contract(
        self, expected: Dict[str, Any] | str = WAN_LATENT_CONTRACT_NAME
    ) -> Dict[str, Any]:
        return validate_wan_latent_contract(self.get_latent_contract(), expected, source=self.path)

    def put_dataset_id(self, dataset_id: str) -> None:
        if not dataset_id:
            raise ValueError("latent dataset ID must not be empty")
        self.put(DATASET_ID_KEY, {"dataset_id": str(dataset_id)})

    def get_dataset_id(self) -> Optional[str]:
        try:
            return str(self.get(DATASET_ID_KEY)["dataset_id"])
        except KeyError:
            return None

    def require_dataset_id(self) -> str:
        dataset_id = self.get_dataset_id()
        if not dataset_id:
            raise ValueError(
                f"{self.path} has no latent dataset ID and cannot be paired with latent stats"
            )
        return dataset_id

    def close(self) -> None:
        self.env.close()


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


class PrecomputedWindowDataset(Dataset):
    """Windows over records written by ``PrecomputedStore``.

    Each frame record is expected to hold at least ``latent``, ``depth``, ``conf``,
    ``static_mask``, ``K_l`` and ``E``. Optional multi-hypothesis depth fields are stacked
    when present in every frame of a window.
    """

    def __init__(self, store_path: str, window: int, stride: int = 1,
                 index_path: Optional[str] = None, require_caption: bool = False,
                 required_latent_contract: Optional[Dict[str, Any] | str] = WAN_LATENT_CONTRACT_NAME):
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        self.store = PrecomputedStore(store_path, readonly=True)
        try:
            self.latent_contract = (
                self.store.require_latent_contract(required_latent_contract)
                if required_latent_contract is not None else self.store.get_latent_contract()
            )
            self.dataset_id = self.store.require_dataset_id()
        except Exception:
            self.store.close()
            raise
        self.window = int(window)
        self.stride = int(stride)
        self.require_caption = bool(require_caption)
        self.index_path = index_path
        if index_path:
            self.index = self._load_index(index_path)
        else:
            self.index: List[Tuple[str, int]] = []
            for cid, n in self.store.get_meta():
                for start in range(0, max(0, int(n) - self.window + 1), self.stride):
                    self.index.append((cid, start))

    def _load_index(self, index_path: str) -> List[Tuple[str, int]]:
        meta = {str(cid): int(n) for cid, n in self.store.get_meta()}
        index: List[Tuple[str, int]] = []
        with open(index_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = str(row["clip_id"])
                start = int(row["start"])
                row_window = int(row.get("window", self.window))
                if row_window != self.window:
                    raise ValueError(
                        f"{index_path}:{line_no} has window={row_window}, expected {self.window}"
                    )
                if cid not in meta:
                    raise ValueError(f"{index_path}:{line_no} references missing clip_id={cid!r}")
                if start < 0 or start + self.window > meta[cid]:
                    raise ValueError(
                        f"{index_path}:{line_no} window {cid}:{start}+{self.window} "
                        f"exceeds clip length {meta[cid]}"
                    )
                index.append((cid, start))
        if not index:
            raise ValueError(f"empty window index: {index_path}")
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        cid, start = self.index[i]
        frames = [self.store.get(f"{cid}:{k}") for k in range(start, start + self.window)]
        keys = ["latent", "depth", "conf", "static_mask", "K_l", "E"]
        out = {key: torch.stack([_to_tensor(fr[key]) for fr in frames], dim=0) for key in keys}
        if not all("source_rgb_index" in fr for fr in frames):
            raise ValueError(
                f"precomputed window {cid}:{start} is missing source_rgb_index provenance"
            )
        source_indices = torch.stack(
            [_to_tensor(fr["source_rgb_index"]).long() for fr in frames], dim=0
        )
        # 2026-07-14 native-chunk alignment gate: the LMDB key ``clip:k``
        # defines the RGB source frame exactly.  Checking only a stride of four
        # would let a consistently shifted sequence such as [1, 5, 9, ...]
        # silently pair every latent with the wrong depth and camera pose.
        expected = (
            torch.arange(start, start + len(frames), dtype=torch.long)
            * WAN_VAE_TEMPORAL_STRIDE
        )
        if not torch.equal(source_indices.cpu(), expected.cpu()):
            raise ValueError(
                f"precomputed window {cid}:{start} has misaligned source_rgb_index "
                f"values {source_indices.tolist()}, expected {expected.tolist()}"
            )
        out["source_rgb_index"] = source_indices
        for opt in ("modes_depth", "modes_weight"):
            if all(opt in fr for fr in frames):
                out[opt] = torch.stack([_to_tensor(fr[opt]) for fr in frames], dim=0)
        captions = [str(frame.get("caption", "")).strip() for frame in frames]
        has_caption = any(captions)
        if (self.require_caption or has_caption) and (
            not captions[0] or any(caption != captions[0] for caption in captions[1:])
        ):
            raise ValueError(f"precomputed window {cid}:{start} has missing or inconsistent captions")
        out["caption"] = captions[0] if has_caption else ""
        out["static_mask"] = out["static_mask"].bool()
        return out
