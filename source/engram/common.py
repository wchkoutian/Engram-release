from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch


MODEL_ROOT = os.environ.get("ENGRAM_WAN_MODEL", "")
MAP_ROOT = os.environ.get("ENGRAM_MAPANYTHING_MODEL", "")
TORCH_HOME = os.environ.get("TORCH_HOME", "")
LATENT_STATS = os.environ.get("ENGRAM_LATENT_STATS", "")
BASE_EXPERIMENT = str(Path(__file__).resolve().parents[2])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cuda(fraction):
    fraction = float(os.environ.get('ENGRAM_MEMORY_FRACTION', fraction))
    if not 0 < fraction <= 1:
        raise ValueError('ENGRAM_MEMORY_FRACTION must be greater than 0 and at most 1')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_digest(tensors):
    result = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def save_video(path, frames, fps=16):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    frame_count, height, width, channels = frames.shape
    if channels != 3 or not frame_count:
        raise ValueError("Expected nonempty RGB video")
    command = [ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    result = subprocess.run(command, input=frames.tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2000:])


def rgb_tensor(frames):
    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1)


def decoded_rgb(decoded):
    return decoded[0].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()


def build_depth_estimator():
    from engram.models.mapanything import MapAnythingEstimator

    original_load = torch.hub.load
    local_repository = Path(TORCH_HOME) / "hub/facebookresearch_dinov2_main"

    def offline_load(repository, model, *arguments, **keywords):
        if str(repository).startswith("facebookresearch/dinov2"):
            repository = str(local_repository)
            keywords["source"] = "local"
            keywords.pop("force_reload", None)
        return original_load(repository, model, *arguments, **keywords)

    torch.hub.load = offline_load
    try:
        return MapAnythingEstimator(repo=MAP_ROOT, device="cuda", torch_home=TORCH_HOME)
    finally:
        torch.hub.load = original_load
