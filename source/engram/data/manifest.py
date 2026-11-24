"""Unified manifest support for already-extracted RGB-frame datasets.

The project can train from several RE10K-style sources.  Some use Google's original
camera txt files, while others, such as extracted Hugging Face mirrors, store one JSON
metadata object per clip with frame paths, pixel intrinsics and w2c matrices.  This
module normalizes those variants into a small clip/frame schema used by precompute.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .realestate10k import denorm_intrinsics, w2c_to_4x4


@dataclass
class ManifestFrame:
    path: str
    timestamp: int
    w2c: np.ndarray
    K_norm: Optional[np.ndarray] = None
    K_pix: Optional[np.ndarray] = None
    source_size: Optional[tuple[int, int]] = None

    def intrinsics_for_size(self, target_h: int, target_w: int,
                            source_size: Optional[tuple[int, int]] = None) -> np.ndarray:
        """Return pixel intrinsics after resizing this frame to ``target_h x target_w``."""
        if self.K_norm is not None:
            return denorm_intrinsics(self.K_norm, target_w, target_h)
        if self.K_pix is None:
            raise ValueError(f"frame has no intrinsics: {self.path}")

        K = np.asarray(self.K_pix, dtype=np.float64).copy()
        src = source_size or self.source_size
        if src is not None:
            src_h, src_w = src
            if src_h <= 0 or src_w <= 0:
                raise ValueError(f"invalid source_size for {self.path}: {src}")
            scale = np.diag([target_w / src_w, target_h / src_h, 1.0])
            K = scale @ K
        return K

    @property
    def E(self) -> np.ndarray:
        return w2c_to_4x4(self.w2c)


@dataclass
class ManifestClip:
    clip_id: str
    frames: List[ManifestFrame]
    split: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None


def _as_mapping(obj: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{context} must be a JSON object, got {type(obj).__name__}")
    return obj


def _as_array(value: Any, shape: tuple[int, ...] | None, context: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        if arr.size == int(np.prod(shape)):
            arr = arr.reshape(shape)
        else:
            raise ValueError(f"{context} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{context} contains non-finite values")
    return arr


def _matrix3x4_or_4x4(value: Any, context: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr[:3, :4]
    if arr.shape == (3, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)[:3, :4]
    if arr.size == 12:
        return arr.reshape(3, 4)
    raise ValueError(f"{context} must be 3x4 or 4x4, got {arr.shape}")


def _intrinsics_from_fxfycxcy(value: Any) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    vals = _as_array(value, (4,), "fxfycxcy")
    if np.max(np.abs(vals)) <= 2.0:
        return vals, None
    K = np.eye(3, dtype=np.float64)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = vals.tolist()
    return None, K


def _intrinsics_from_frame(frame: Mapping[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "K_norm" in frame:
        return _as_array(frame["K_norm"], (4,), "K_norm"), None
    if "intrinsics_norm" in frame:
        return _as_array(frame["intrinsics_norm"], (4,), "intrinsics_norm"), None
    if "fxfycxcy_norm" in frame:
        return _as_array(frame["fxfycxcy_norm"], (4,), "fxfycxcy_norm"), None
    if "fxfycxcy" in frame:
        return _intrinsics_from_fxfycxcy(frame["fxfycxcy"])
    if "K" in frame:
        return None, _as_array(frame["K"], (3, 3), "K")
    if "intrinsics" in frame:
        intr = np.asarray(frame["intrinsics"], dtype=np.float64)
        if intr.size == 4:
            return _intrinsics_from_fxfycxcy(intr)
        return None, _as_array(intr, (3, 3), "intrinsics")
    raise ValueError("frame has no intrinsics; expected K_norm, fxfycxcy or K")


def _source_size(frame: Mapping[str, Any], clip: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    value = frame.get("source_size", frame.get("image_size", clip.get("source_size", clip.get("image_size"))))
    if value is None:
        return None
    vals = [int(x) for x in value]
    if len(vals) != 2:
        raise ValueError(f"source/image size must be [H, W], got {value}")
    return vals[0], vals[1]


def _timestamp(frame: Mapping[str, Any], path: str, index: int) -> int:
    for key in ("timestamp", "time", "frame_id", "frame_index", "index"):
        if key in frame:
            return int(frame[key])
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem)
    return int(index)


def _path_from_frame(frame: Mapping[str, Any]) -> str:
    for key in ("image_path", "frame_path", "path", "file_path", "rgb_path"):
        if key in frame and frame[key]:
            return str(frame[key])
    raise ValueError("frame has no image path; expected image_path/path/frame_path")


def _w2c_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    if "w2c" in frame:
        return _matrix3x4_or_4x4(frame["w2c"], "w2c")
    if "E" in frame:
        return _matrix3x4_or_4x4(frame["E"], "E")
    if "extrinsics" in frame:
        return _matrix3x4_or_4x4(frame["extrinsics"], "extrinsics")
    if "c2w" in frame:
        c2w = _as_array(frame["c2w"], None, "c2w")
        if c2w.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float64)
            tmp[:3, :4] = c2w
            c2w = tmp
        elif c2w.size == 16:
            c2w = c2w.reshape(4, 4)
        else:
            raise ValueError(f"c2w must be 3x4 or 4x4, got {c2w.shape}")
        return np.linalg.inv(c2w)[:3, :4]
    raise ValueError("frame has no extrinsics; expected w2c/E/extrinsics/c2w")


def _resolve_frame_path(path: str, root: Optional[str]) -> str:
    if not root or os.path.isabs(path):
        return path
    resolved = os.path.normpath(os.path.join(str(root), path))
    if os.path.exists(resolved):
        return resolved
    normalized = path.replace("\\", "/")
    if normalized.startswith("re10k-tv/"):
        candidate = os.path.normpath(
            os.path.join(str(root), "RealEstate10K-tv", normalized[len("re10k-tv/"):])
        )
        if os.path.exists(candidate):
            return candidate
    return resolved


def _infer_clip_id(payload: Mapping[str, Any], frames: Sequence[Mapping[str, Any]],
                   key: Optional[str]) -> str:
    for field in ("clip_id", "scene_name", "scene_id", "sequence_id", "video_id", "id"):
        if payload.get(field):
            return str(payload[field])
    if key:
        return Path(str(key)).name
    if frames:
        try:
            path = Path(_path_from_frame(frames[0]))
            parent = path.parent.name
            if parent:
                return parent
        except ValueError:
            pass
    raise ValueError("cannot infer clip_id")


def _infer_split(payload: Mapping[str, Any], frames: Sequence[Mapping[str, Any]],
                 key: Optional[str]) -> Optional[str]:
    if payload.get("split"):
        return str(payload["split"])
    candidates = [str(key or "")]
    if frames:
        try:
            candidates.append(_path_from_frame(frames[0]))
        except ValueError:
            pass
    joined = "/".join(candidates).lower().replace("\\", "/")
    for split in ("train", "validation", "val", "test"):
        if f"/{split}/" in f"/{joined}/":
            return "val" if split == "validation" else split
    return None


def _frame_dicts(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    frames = payload.get("frames")
    if frames is None:
        raise ValueError("clip payload has no frames")
    if not isinstance(frames, list):
        raise ValueError(f"frames must be a list, got {type(frames).__name__}")
    if not frames:
        raise ValueError("frames must not be empty")
    if isinstance(frames[0], Mapping):
        return [_as_mapping(fr, "frame") for fr in frames]
    if isinstance(frames[0], str):
        out = []
        for i, p in enumerate(frames):
            fr = {"image_path": p}
            for key in ("fxfycxcy", "K_norm", "K", "w2c", "E", "c2w"):
                vals = payload.get(key)
                if isinstance(vals, list) and len(vals) == len(frames):
                    fr[key] = vals[i]
            out.append(fr)
        return out
    raise ValueError(f"unsupported frame entry type: {type(frames[0]).__name__}")


def _unwrap_payload(obj: Mapping[str, Any]) -> tuple[Mapping[str, Any], Optional[str], Optional[str]]:
    key = obj.get("__key__")
    url = obj.get("__url__")
    payload = obj.get("json", obj)
    if isinstance(payload, str):
        payload = json.loads(payload)
    return _as_mapping(payload, "clip payload"), (str(key) if key is not None else None), (
        str(url) if url is not None else None
    )


def clip_from_payload(obj: Mapping[str, Any], *, frames_root: Optional[str] = None,
                      manifest_dir: Optional[str] = None) -> ManifestClip:
    """Normalize one JSON/HF row into a ``ManifestClip``.

    Supports ghuijo/realestate10k rows whose payload shape is roughly:
    ``{"json": {"frames": [{"image_path", "fxfycxcy", "w2c"}, ...]}, "__key__": ...}``.
    """
    payload, key, url = _unwrap_payload(obj)
    raw_frames = _frame_dicts(payload)
    clip_id = _infer_clip_id(payload, raw_frames, key)
    split = _infer_split(payload, raw_frames, key)
    root = frames_root or payload.get("frames_root") or manifest_dir

    frames: List[ManifestFrame] = []
    for i, raw in enumerate(raw_frames):
        path = _resolve_frame_path(_path_from_frame(raw), root)
        K_norm, K_pix = _intrinsics_from_frame(raw)
        frames.append(
            ManifestFrame(
                path=path,
                timestamp=_timestamp(raw, path, i),
                w2c=_w2c_from_frame(raw),
                K_norm=K_norm,
                K_pix=K_pix,
                source_size=_source_size(raw, payload),
            )
        )

    return ManifestClip(
        clip_id=clip_id,
        frames=frames,
        split=split,
        source=str(payload.get("source", obj.get("source", "")) or "") or None,
        url=url or payload.get("url"),
    )


def _iter_manifest_objects(path: str) -> Iterable[Mapping[str, Any]]:
    suffix = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        yield _as_mapping(json.loads(line), f"{path}:{line_no}")
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON on {path}:{line_no}: {exc}") from exc
        else:
            obj = json.load(f)
            if isinstance(obj, Mapping) and "clips" in obj:
                for clip in obj["clips"]:
                    yield _as_mapping(clip, "clip")
            elif isinstance(obj, list):
                for clip in obj:
                    yield _as_mapping(clip, "clip")
            else:
                yield _as_mapping(obj, "manifest")


def load_manifest(path: str, *, frames_root: Optional[str] = None,
                  split: Optional[str] = None, max_clips: int = 0) -> List[ManifestClip]:
    manifest_dir = str(Path(path).resolve().parent)
    clips: List[ManifestClip] = []
    for obj in _iter_manifest_objects(path):
        clip = clip_from_payload(obj, frames_root=frames_root, manifest_dir=manifest_dir)
        if split and clip.split and clip.split != split:
            continue
        clips.append(clip)
        if max_clips and len(clips) >= max_clips:
            break
    return clips


def manifest_clip_to_json(clip: ManifestClip, *, path_root: Optional[str] = None) -> dict[str, Any]:
    frames = []
    for frame in clip.frames:
        path = frame.path
        if path_root and os.path.isabs(path):
            try:
                path = os.path.relpath(path, path_root)
            except ValueError:
                pass
        fr: dict[str, Any] = {
            "image_path": path,
            "timestamp": int(frame.timestamp),
            "w2c": np.asarray(frame.w2c, dtype=float).tolist(),
        }
        if frame.K_norm is not None:
            fr["K_norm"] = np.asarray(frame.K_norm, dtype=float).tolist()
        elif frame.K_pix is not None:
            fr["K"] = np.asarray(frame.K_pix, dtype=float).tolist()
        if frame.source_size is not None:
            fr["source_size"] = list(frame.source_size)
        frames.append(fr)
    out: dict[str, Any] = {"clip_id": clip.clip_id, "frames": frames}
    if clip.split:
        out["split"] = clip.split
    if clip.source:
        out["source"] = clip.source
    if clip.url:
        out["url"] = clip.url
    return out


def write_manifest_jsonl(clips: Sequence[ManifestClip], path: str, *,
                         path_root: Optional[str] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(manifest_clip_to_json(clip, path_root=path_root), ensure_ascii=True) + "\n")
