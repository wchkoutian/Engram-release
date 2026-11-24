"""Qwen3-VL wrapper for open-vocabulary entity extraction.

Used to derive the dynamic-entity prompt set per clip (which categories to mask out of
static memory) instead of a fixed list. Returns short noun phrases describing removable
foreground objects across sampled clip frames, which are then fed to SAM3 as text prompts.

Official API via transformers:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
"""
from __future__ import annotations

import re
from typing import List

import numpy as np


_PROMPT = (
    "These are frames from a real-estate walkthrough. Decide whether any removable "
    "foreground object is visibly present in the supplied frames. A removable "
    "foreground object means an actually visible person, animal, vehicle, or object "
    "being carried or moved. Most clips contain only an empty static room. Never "
    "infer a person from camera motion or from an unseen camera operator. Do not count "
    "furniture, appliances, plants, decorations, photographs, reflections, shadows, "
    "windows, architecture, or scenery. Output exactly Nothing when no removable "
    "foreground is visibly present. Otherwise output only a comma-separated list of "
    "at most 3 singular category names; use person for any visible human."
)

_NO_ENTITY = (
    "no moving", "no motion", "no movement", "no dynamic", "no foreground",
    "nothing", "none", "static scene",
)
_PERSON_TERMS = {
    "person", "people", "human", "man", "woman", "child", "pedestrian",
    "hand", "arm", "leg", "face",
}
_ANSWER_MARKER = re.compile(
    r"(?:the\s+)?(?:correct\s+)?(?:answer|output)(?:\s+is)?\s*:\s*",
    re.IGNORECASE,
)
_ENTITY_ITEM = re.compile(
    r"[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'/-]*){0,3}"
)
_NARRATIVE_TERMS = {
    "are", "frame", "frames", "is", "moving", "provided", "scene", "seen",
    "visible", "was", "were",
}


def parse_entities(text: str) -> List[str]:
    """Parse the constrained Strict20K Qwen response into SAM text prompts."""
    cleaned = text.strip().strip("[]")
    lowered = cleaned.lower()
    if not cleaned or any(
        re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in _NO_ENTITY
    ):
        return []

    markers = list(_ANSWER_MARKER.finditer(cleaned))
    if markers:
        cleaned = cleaned[markers[-1].end():].strip()
    else:
        cleaned = cleaned.rstrip(".").strip()
        if "." in cleaned:
            return []

    entities: List[str] = []
    for part in re.split(r"[,;\n]+", cleaned.rstrip(".")):
        part = re.sub(r"^[\d\s\-\)\(\.]+", "", part).strip()
        if not part:
            continue
        words = set(re.findall(r"[a-z]+", part.lower()))
        if len(part) > 32 or not _ENTITY_ITEM.fullmatch(part) or words & _NARRATIVE_TERMS:
            return []
        if part.lower() not in {item.lower() for item in entities}:
            entities.append(part)
    if len(entities) > 3:
        return []

    canonical: List[str] = []
    for entity in entities:
        words = set(re.findall(r"[a-z]+", entity.lower()))
        entity = "person" if words & _PERSON_TERMS else entity
        if entity.lower() not in {item.lower() for item in canonical}:
            canonical.append(entity)
    return canonical


class Qwen3VLEntityExtractor:
    def __init__(self, repo: str, device: str = "cuda", dtype="bfloat16",
                 local_files_only: bool = True):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        td = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        default_dtype = torch.get_default_dtype()
        try:
            self.processor = AutoProcessor.from_pretrained(
                repo, local_files_only=local_files_only
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                repo, torch_dtype=td, device_map=device,
                local_files_only=local_files_only,
            )
        finally:
            # Transformers may temporarily set the process-wide default dtype while
            # materializing BF16 weights. Later WorldScore camera matrices must stay
            # FP32 because torch.linalg.inv does not support BF16.
            torch.set_default_dtype(default_dtype)
        self.model.eval()

    def movable_entities(self, image: np.ndarray, max_new_tokens: int = 64) -> List[str]:
        return self.movable_entities_from_frames(
            [image], max_frames=1, max_new_tokens=max_new_tokens
        )

    def movable_entities_from_frames(
        self, frames, max_frames: int = 10, max_new_tokens: int = 128
    ) -> List[str]:
        from PIL import Image
        import torch

        if isinstance(frames, np.ndarray):
            frames = [frames] if frames.ndim == 3 else list(frames)
        frames = list(frames)
        if not frames:
            return []
        indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)
        selected = [frames[int(index)] for index in indices]
        pil_frames = [
            Image.fromarray(frame.astype(np.uint8)) if isinstance(frame, np.ndarray) else frame
            for frame in selected
        ]
        content = [{"type": "text", "text": _PROMPT}]
        content.extend({"type": "image", "image": frame} for frame in pil_frames)
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=pil_frames, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return parse_entities(decoded)
