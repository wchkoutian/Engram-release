import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SCENES = tuple(json.loads((Path(__file__).resolve().parents[1] / 'configs/re10k_test100_scenes.json').read_text()))
FRAME_COUNTS = {'nvs': 81, 'closed_loop': 153}
IDENTIFIER = 'engram_re10k_test100_own_v1'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Asset path escapes the RE10K test-100 copy: {relative}')
    return path


def validate_manifest(manifest):
    if manifest.get('identifier') != IDENTIFIER or manifest.get('source_split') != 'RealEstate10K/test':
        raise ValueError('Expected the fixed RE10K test-100 test data')
    if manifest.get('frame_counts') != FRAME_COUNTS or manifest.get('resolution') != [1280, 704]:
        raise ValueError('RE10K test-100 requires NVS81/CL153 at 1280x704')
    if tuple(case['scene_name'] for case in manifest['cases']) != SCENES:
        raise ValueError('RE10K test-100 identity/order mismatch')
    for index, case in enumerate(manifest['cases']):
        if case['source_index'] != index or case['name'] != f'{index:03d}_{SCENES[index]}' or len(case['frames']) != 81:
            raise ValueError('Invalid RE10K test-100 case')


def trajectory(path, protocol):
    values = np.asarray([ast.literal_eval(line) for line in path.read_text().splitlines() if line.strip()], dtype=np.float32)
    if values.shape != (FRAME_COUNTS[protocol], 4, 4) or not np.isfinite(values).all():
        raise ValueError('Invalid fixed trajectory')
    if not np.allclose(values[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid homogeneous pose')
    if protocol == 'closed_loop' and not np.allclose(values[0], values[-1], atol=1e-5):
        raise ValueError('Closed-loop endpoint differs from start')
    return values


def crop_seed(image, intrinsics):
    width, height = 1280, 704
    source = image.convert('RGB')
    source_width, source_height = source.size
    if source_height / source_width < height / width:
        crop_width, crop_height = int(source_height / height * width), source_height
        left, top = (source_width - crop_width) // 2, 0
    else:
        crop_width, crop_height = source_width, int(source_width / width * height)
        left, top = 0, (source_height - crop_height) // 2
    image = source.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.Resampling.LANCZOS)
    focal_x, focal_y, center_x, center_y = intrinsics
    camera = np.asarray([[focal_x * width / crop_width, 0, (center_x - left) * width / crop_width], [0, focal_y * height / crop_height, (center_y - top) * height / crop_height], [0, 0, 1]], dtype=np.float32)
    return image, camera


def score_pairs(protocol):
    if protocol == 'nvs':
        return [(index, index) for index in range(1, 81)]
    if protocol == 'closed_loop':
        return [(152, 0)]
    raise ValueError('Unknown protocol')


def chunk_indices(protocol, start):
    count = FRAME_COUNTS[protocol]
    if start not in range(0, count - 1, 32):
        raise ValueError('Invalid chunk start')
    return np.minimum(np.arange(start, start + 33), count - 1), min(33, count - start)


def verify_assets(root, check_hashes=True):
    manifest = json.loads((root / 'selected_10.json').read_text())
    validate_manifest(manifest)
    if check_hashes:
        for relative, record in manifest['copied_files'].items():
            path = inside(root, relative)
            if not path.is_file() or digest(path) != record['sha256']:
                raise ValueError(f'Copied asset changed: {relative}')
    rows = []
    for case in manifest['cases']:
        for frame in case['frames']:
            with Image.open(inside(root, frame['image_path'])) as image:
                image.verify()
        with Image.open(inside(root, case['frames'][0]['image_path'])) as image:
            seed, camera = crop_seed(image, case['frames'][0]['fxfycxcy'])
        with Image.open(inside(root, case['conditioning_image'])) as expected:
            if not np.array_equal(np.asarray(seed), np.asarray(expected.convert('RGB'))):
                raise ValueError('Seed crop differs from the old V10 preprocessing')
        poses = {name: trajectory(inside(root, relative), name) for name, relative in case['trajectories'].items()}
        rows.append({'name': case['name'], 'images': 81, 'seed_crop_matches_v10': True, 'trajectory_frames': {name: len(value) for name, value in poses.items()}, 'first65_poses_equal': bool(np.array_equal(poses['nvs'][:65], poses['closed_loop'][:65]))})
    return manifest, {'pass': True, 'copied_files': len(manifest['copied_files']), 'images': 81 * len(SCENES), 'cases': rows}


def select_case(manifest, sample):
    matches = [case for case in manifest['cases'] if sample in [case['name'], f'{case["source_index"]:03d}', case['scene_name']]]
    if len(matches) != 1:
        raise ValueError('Select an existing RE10K test-100 case, e.g. 000')
    return matches[0]
