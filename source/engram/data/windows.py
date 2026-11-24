import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


BASE = Path(__file__).resolve().parents[3]
DATASET = Path(os.environ.get('ENGRAM_STRICT30K_LMDB_DIR', ''))
SAMPLES = Path(os.environ.get('ENGRAM_STRICT30K_SAMPLES', ''))
SOURCE_MANIFEST = Path(os.environ.get('ENGRAM_SELECTION_MANIFEST', ''))
QUOTAS = {'other': 128, 'yaw_left': 64, 'yaw_right': 64}  # V14: x8 of V11 (16/8/8)
EXCLUDED_CLIPS = {'67203c7bd53e260b', '853ca19311254f8a', '7f602fd34aa315b0',
                  # V14: all Local-10 clips (scenes and parent videos kept out of training)
                  'a374f27587ad2d52', '8ab1900fbe301245', '4f88a0a88784bbf1', '8371a8888df622aa', 'bc1571b6ae57de68', '84a7bc993f35b6b4', 'ce9e0dbe761cd99c'}


class SceneRejected(RuntimeError):
    pass


def select_rows(rows, motions):
    excluded_parents = {row['parent_video_id'] for row in rows if row['clip_id'] in EXCLUDED_CLIPS}
    candidates = []
    for row in rows:
        motion = motions.get(row['sample_id'])
        if motion is None or row['clip_id'] in EXCLUDED_CLIPS or row['parent_video_id'] in excluded_parents:
            continue
        if not .41 <= motion['baseline_ratio'] <= 1.0 or motion['rotation_degrees'] > 45:
            continue
        candidates.append({**row, 'motion': motion, 'motion_group': motion['bucket'].split('|')[0]})
    candidates.sort(key=lambda row: hashlib.sha256(('v11-strict32-20917-' + row['sample_id']).encode()).hexdigest())
    pools = {group: [] for group in QUOTAS}
    parents, clips = set(), set()
    for row in candidates:
        group = row['motion_group']
        if group not in pools or len(pools[group]) >= QUOTAS[group] * 2:
            continue
        if row['parent_video_id'] in parents or row['clip_id'] in clips:
            continue
        pools[group].append(row)
        parents.add(row['parent_video_id'])
        clips.add(row['clip_id'])
    if any(len(pools[group]) < QUOTAS[group] for group in QUOTAS):
        raise RuntimeError('Insufficient distinct long-motion scenes')
    return pools, sorted(excluded_parents)


def history_groups(prefix_frame, direction):
    if direction != 1:
        raise ValueError('Reverse training windows are not supported')
    if prefix_frame < 0 or prefix_frame > 80 or prefix_frame % 4:
        raise ValueError('Invalid forward history prefix')
    prefix_time = prefix_frame
    groups = []
    for start, end in ((0, 32), (32, 64), (64, 80)):
        times = list(range(start + 4, min(end, prefix_time) + 1, 4))
        if times:
            groups.append([(time, time) for time in times])
    return prefix_time, groups


def history_feature(windows, time):
    if time <= 0 or time > 80 or time % 4:
        raise ValueError('History feature must be a forward latent anchor')
    if time <= 32:
        window, start = 0, 0
    elif time <= 64:
        window, start = 2, 32
    else:
        window, start = 4, 64
    return windows[window]['latent'][0, :, (time - start) // 4]


def epoch_order(size, epoch, seed=44111):
    return np.random.default_rng(seed + epoch).permutation(size).tolist()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
