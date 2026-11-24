"""SAM3 wrapper (facebookresearch/sam3) for dynamic-object masking.

Mirage's failure mode includes baked-in moving objects; Engram excludes movable entities
from the *static* memory by masking them out before back-projection (Sec 3.3). SAM3
segments open-vocabulary entities from text prompts and tracks them across the video.

Official video API (SAM 3.1, Object Multiplex):
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor()
    r = predictor.handle_request({"type": "start_session", "resource_path": video})
    predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": "person"})
    for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}): ...
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


DEFAULT_DYNAMIC_PROMPTS = ["person", "people", "car", "vehicle", "animal", "bicycle", "moving object"]
STATIC_MEMORY_ALWAYS_MASK_PROMPTS = ["sky"]


def static_memory_prompts(movable_prompts: Optional[Iterable[str]] = None) -> List[str]:
    """Return deduplicated SAM prompts for content excluded from static memory."""
    prompts = DEFAULT_DYNAMIC_PROMPTS if movable_prompts is None else movable_prompts
    result: List[str] = []
    seen = set()
    for prompt in [*prompts, *STATIC_MEMORY_ALWAYS_MASK_PROMPTS]:
        normalized = str(prompt).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


class SAM3Segmenter:
    def __init__(self, prompts: List[str] = None, gpus_to_use=None,
                 checkpoint_path: Optional[str] = None):
        from sam3.model_builder import build_sam3_video_predictor

        self.prompts = static_memory_prompts(prompts)
        kwargs = {}
        if gpus_to_use is not None:
            kwargs["gpus_to_use"] = gpus_to_use
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path
        default_dtype = torch.get_default_dtype()
        autocast_enabled = {
            device_type: torch.is_autocast_enabled(device_type)
            for device_type in ("cpu", "cuda")
        }
        try:
            self.predictor = build_sam3_video_predictor(**kwargs)
        finally:
            # SAM3 enters a process-wide CUDA BF16 autocast context in its constructor.
            # Restore both process defaults here; dynamic_masks opens a bounded autocast
            # context only while SAM3 itself is running.
            torch.set_default_dtype(default_dtype)
            for device_type, enabled in autocast_enabled.items():
                torch.set_autocast_enabled(device_type, enabled)

    @staticmethod
    def _union_masks(outputs: dict) -> Optional[np.ndarray]:
        masks = outputs.get("out_binary_masks")
        if masks is None:
            return None
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        array = np.asarray(masks, dtype=bool)
        if array.size == 0:
            return None
        return np.any(array.reshape(-1, *array.shape[-2:]), axis=0)

    def dynamic_masks(self, video_path: str) -> Dict[int, np.ndarray]:
        """Return per-frame union masks for dynamic entities and sky.

        SAM3 resets its text state on every ``add_prompt`` call, so each prompt must be
        propagated independently and merged on CPU.
        """
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=autocast_device == "cuda",
        ):
            res = self.predictor.handle_request(
                {"type": "start_session", "resource_path": video_path}
            )
            sid = res["session_id"]
            out: Dict[int, np.ndarray] = {}
            try:
                for text in static_memory_prompts(self.prompts):
                    response = self.predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": sid,
                            "frame_index": 0,
                            "text": text,
                        }
                    )
                    union = self._union_masks(response["outputs"])
                    if union is not None:
                        out[0] = union if 0 not in out else (out[0] | union)
                    for frame in self.predictor.handle_stream_request(
                        {"type": "propagate_in_video", "session_id": sid}
                    ):
                        idx = int(frame["frame_index"])
                        union = self._union_masks(frame["outputs"])
                        if union is not None:
                            out[idx] = union if idx not in out else (out[idx] | union)
            finally:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": sid}
                )
        return out

    @staticmethod
    def static_mask_for_latent(dynamic_mask: np.ndarray, h: int, w: int) -> torch.Tensor:
        """Downsample a pixel dynamic mask to a latent-resolution STATIC mask (True=keep)."""
        m = torch.from_numpy(dynamic_mask.astype(np.float32)).view(1, 1, *dynamic_mask.shape)
        pooled = torch.nn.functional.adaptive_max_pool2d(m, (h, w))[0, 0]
        return pooled < 0.5
