"""Shared file contracts for clip, geometry, and mask preparation."""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'third_party'))

FRAMES, FPS, WIDTH, HEIGHT = 81, 16, 1280, 704


def read_scene_manifest(path: Path, max_samples: int = 0) -> list[dict]:
    if max_samples < 0:
        raise ValueError('max_samples must be nonnegative')
    text = Path(path).read_text(encoding='utf-8')
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = [payload] if 'sample_id' in payload else payload.get('scenes')
    if not isinstance(payload, list) or not payload:
        raise ValueError('Manifest must contain a nonempty scene list or JSONL rows')
    ids = []
    for row in payload:
        sample = row.get('sample_id') if isinstance(row, dict) else None
        if not isinstance(sample, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sample):
            raise ValueError('Each scene requires a safe string sample_id')
        if 'clip_sha256' in row and not re.fullmatch(r'[0-9a-fA-F]{64}', str(row['clip_sha256'])):
            raise ValueError('Invalid source clip SHA256')
        ids.append(sample)
    if len(set(ids)) != len(ids):
        raise ValueError('Manifest contains duplicate sample IDs')
    return payload[:max_samples] if max_samples else payload



def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def samples(manifest, sample_root, shard_id=0, num_shards=1):
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError('Require num-shards >= 1 and 0 <= shard-id < num-shards')
    root = Path(sample_root).resolve()
    rows = read_scene_manifest(Path(manifest))
    result = []
    for index, row in enumerate(rows):
        if index % num_shards != shard_id:
            continue
        folder = root / row['sample_id']
        if not folder.resolve().is_relative_to(root):
            raise ValueError('Sample directory escapes sample-root')
        result.append((row, folder))
    return result


def validate_clip(path, expected_sha=None):
    from lsm_world.data_process.clip_sampling import check_clip_file_complete
    if not check_clip_file_complete(Path(path), FRAMES, FPS, (WIDTH, HEIGHT)):
        raise ValueError(f'Expected an {FRAMES}-frame, {FPS}-fps, {WIDTH}x{HEIGHT} clip: {path}')
    digest = sha256(path)
    if expected_sha and digest.lower() != expected_sha.lower():
        raise ValueError(f'Clip SHA256 differs from manifest: {path}')
    return digest


def validate_geometry(paths):
    import numpy as np
    import OpenEXR
    import Imath
    for name, shape in [('pose', (FRAMES, 4, 4)), ('intrinsics', (FRAMES, 4))]:
        with np.load(paths[name], allow_pickle=False) as data:
            indices = np.asarray(data['inds'])
            values = np.asarray(data['data'])
        if indices.shape != (FRAMES,) or not np.array_equal(indices, np.arange(FRAMES)):
            raise ValueError(f'Invalid {name} indices: {paths[name]}')
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name} values: {paths[name]}')
        if name == 'pose' and not np.allclose(values[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError('Pose matrices must be homogeneous camera-to-world transforms')
        if name == 'intrinsics' and not (values[:, :2] > 0).all():
            raise ValueError('Pixel-space focal lengths must be positive')
    with zipfile.ZipFile(paths['depth']) as archive:
        members = [p for p in archive.namelist() if not p.endswith('/')]
        if sorted(members) != [f'{i:05d}.exr' for i in range(FRAMES)] or archive.testzip():
            raise ValueError('Depth archive must contain frames 00000.exr through 00080.exr')
        for name in members:
            exr = OpenEXR.InputFile(io.BytesIO(archive.read(name)))
            try:
                header = exr.header()
                window = header['dataWindow']
                shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
                if shape != (HEIGHT, WIDTH) or 'Z' not in header['channels']:
                    raise ValueError(f'Expected a {WIDTH}x{HEIGHT} EXR Z depth channel: {name}')
                depth = np.frombuffer(exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                if not (np.isfinite(depth) & (depth > 0)).any():
                    raise ValueError(f'Depth frame has no finite positive pixels: {name}')
            finally:
                exr.close()


def validate_masks(path):
    import cv2
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        names = [p for p in archive.namelist() if not p.endswith('/')]
        if sorted(names) != [f'{i:05d}.png' for i in range(FRAMES)]:
            raise ValueError('Mask archive must contain frames 00000.png through 00080.png')
        for name in names:
            mask = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (HEIGHT, WIDTH):
                raise ValueError(f'Expected a {WIDTH}x{HEIGHT} grayscale mask: {name}')
