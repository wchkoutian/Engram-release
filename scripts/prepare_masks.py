#!/usr/bin/env python3
"""Prepare dynamic-object and sky masks using Qwen3-VL-2B and SAM3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import re
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.data.preprocessing import samples, sha256, validate_clip, validate_masks, write_json


def load_runtime():
    global cv2, np, torch, Image
    import cv2
    import numpy as np
    import torch
    from PIL import Image


ENTITY_PROMPT = (
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

NO_ENTITY = (
    "no moving",
    "no motion",
    "no movement",
    "no dynamic",
    "no foreground",
    "nothing",
    "none",
    "static scene",
)
PERSON_TERMS = (
    "person",
    "people",
    "human",
    "man",
    "woman",
    "child",
    "pedestrian",
    "hand",
    "arm",
    "leg",
    "face",
)
ANSWER_MARKER = re.compile(
    r"(?:the\s+)?(?:correct\s+)?(?:answer|output)(?:\s+is)?\s*:\s*",
    re.IGNORECASE,
)
ENTITY_ITEM = re.compile(
    r"[A-Za-z][A-Za-z0-9'/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'/-]*){0,3}"
)
NARRATIVE_TERMS = {
    "are",
    "frame",
    "frames",
    "is",
    "moving",
    "provided",
    "scene",
    "seen",
    "visible",
    "was",
    "were",
}


def parse_entities(text: str) -> list[str]:
    cleaned = text.strip().strip("[]")
    lowered = cleaned.lower()
    if not cleaned or any(
        re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in NO_ENTITY
    ):
        return []

    markers = list(ANSWER_MARKER.finditer(cleaned))
    if markers:
        cleaned = cleaned[markers[-1].end() :].strip()
    else:
        cleaned = cleaned.rstrip(".。").strip()
        if "." in cleaned or "。" in cleaned:
            return []

    entities = []
    for part in re.split(r"[,;\n]+", cleaned.rstrip(".。")):
        part = re.sub(r"^[\d\s\-\)\(\.]+", "", part).strip()
        if not part:
            continue
        words = set(re.findall(r"[a-z]+", part.lower()))
        if (
            len(part) > 32
            or not ENTITY_ITEM.fullmatch(part)
            or words & NARRATIVE_TERMS
        ):
            return []
        if part.lower() not in {item.lower() for item in entities}:
            entities.append(part)
    if len(entities) > 3:
        return []

    canonical = []
    for entity in entities:
        words = set(re.findall(r"[a-z]+", entity.lower()))
        entity = "person" if any(term in words for term in PERSON_TERMS) else entity
        if entity.lower() not in {item.lower() for item in canonical}:
            canonical.append(entity)
    return canonical


def sample_frames(video_path: Path, count: int = 10) -> tuple[list[Image.Image], int, int, int]:
    from lsm_world.data_process.video_io import get_video_frame_count
    capture = cv2.VideoCapture(str(video_path))
    frame_count = get_video_frame_count(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = np.linspace(0, frame_count - 1, count).round().astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {index} from {video_path}")
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames, frame_count, height, width


class EntityExtractor:
    def __init__(self, model_path: str):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
        ).eval()

    def extract(self, frames: list[Image.Image]) -> tuple[list[str], str]:
        content = [{"type": "text", "text": ENTITY_PROMPT}]
        content.extend({"type": "image", "image": frame} for frame in frames)
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=frames, return_tensors="pt"
        ).to("cuda:0")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        return parse_entities(raw), raw


def union_masks(outputs: dict, height: int, width: int) -> np.ndarray:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return np.zeros((height, width), dtype=bool)
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    array = np.asarray(masks, dtype=bool)
    if array.size == 0:
        return np.zeros((height, width), dtype=bool)
    return np.any(array.reshape(-1, height, width), axis=0)


def track_masks(predictor, video_path: Path, prompts: list[str], frame_count: int, height: int, width: int) -> list[np.ndarray]:
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(video_path)}
    )
    session_id = response["session_id"]
    try:
        masks = [np.zeros((height, width), dtype=bool) for _ in range(frame_count)]
        for prompt in prompts:
            response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                }
            )
            masks[0] |= union_masks(response["outputs"], height, width)
            for result in predictor.handle_stream_request(
                {"type": "propagate_in_video", "session_id": session_id}
            ):
                index = int(result["frame_index"])
                masks[index] |= union_masks(result["outputs"], height, width)
        return masks
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )


def write_mask_zip(path: Path, masks: list[np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, mask in enumerate(masks):
            ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
            if not ok:
                raise RuntimeError(f"cannot encode mask frame {index}")
            archive.writestr(f"{index:05d}.png", encoded.tobytes())
    temporary.replace(path)


def prepare(args):
    pending, reused = [], []
    model = args.qwen_model.resolve()
    checkpoint = args.sam_checkpoint.resolve()
    if not model.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError('Provide a local Qwen3-VL-2B model directory and SAM3 checkpoint')
    for row, folder in samples(args.manifest, args.sample_root, args.shard_id, args.num_shards):
        digest = validate_clip(folder / 'clip.mp4', row.get('clip_sha256'))
        output, metadata = folder / 'mask.zip', folder / 'mask_metadata.json'
        binding = {'clip_sha256': digest, 'qwen_model': str(model), 'sam_checkpoint': str(checkpoint)}
        if output.exists() or metadata.exists():
            if not args.skip_existing or not output.is_file() or not metadata.is_file():
                raise FileExistsError(f'Existing mask output; use a new output or --skip-existing: {folder}')
            old = json.loads(metadata.read_text(encoding='utf-8'))
            if any(old.get(key) != value for key, value in binding.items()) or old.get('mask_sha256') != sha256(output):
                raise ValueError(f'Existing masks no longer match their clip/settings: {folder}')
            validate_masks(output)
            reused.append(row['sample_id'])
        else:
            pending.append((row, folder, binding))
    if not pending:
        return {'written': [], 'reused': reused}
    load_runtime()
    from sam3.model_builder import build_sam3_video_predictor
    extractor = EntityExtractor(str(model))
    predictor = build_sam3_video_predictor(checkpoint_path=str(checkpoint), gpus_to_use=[0])
    written = []
    for row, folder, binding in pending:
        frames, frame_count, height, width = sample_frames(folder / 'clip.mp4')
        entities, raw = extractor.extract(frames)
        prompts = entities + ['sky']
        masks = track_masks(predictor, folder / 'clip.mp4', prompts, frame_count, height, width)
        output = folder / 'mask.zip'
        write_mask_zip(output, masks)
        validate_masks(output)
        write_json(folder / 'mask_metadata.json', {
            **binding, 'mask_sha256': sha256(output), 'qwen_raw': raw, 'entities': entities,
            'sam_prompts': prompts, 'frame_count': frame_count,
            'masked_ratio_mean': float(np.mean([float(mask.mean()) for mask in masks])),
            'masked_ratio_max': float(max(float(mask.mean()) for mask in masks)),
        })
        written.append(row['sample_id'])
        print(json.dumps({'sample_id': row['sample_id'], 'prompts': prompts}), flush=True)
    return {'written': written, 'reused': reused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--qwen-model', type=Path, required=True)
    parser.add_argument('--sam-checkpoint', type=Path, required=True)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--skip-existing', action='store_true')
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == '__main__':
    main()
