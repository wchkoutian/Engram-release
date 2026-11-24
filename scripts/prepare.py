"""Prepare the fixed 256-scene forward training cache from LMDB and source clips."""
import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path
import sys

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.core import (MODEL_ROOT, LATENT_STATS, ProtectedMemory, build_depth_estimator,
                  conditioning, digest, poses_in_metres, rgb_tensor, save_tensor,
                  setup_cuda, window_indices, write_json)
from engram.data.contracts import (SCENE_COUNT, WINDOW_STARTS, forward_data_contract,
                       load_training_selection,
                       validate_forward_manifest, validate_forward_scene_cache,
                       validate_prepared_window, validate_scene_cache)
from engram.models.wan_latents import encode_wan_latents, wan_latent_contract
from engram.data.windows import SceneRejected, history_feature, history_groups

SAMPLES = Path(os.environ.get('ENGRAM_TRAINING_SAMPLES', ''))


def verify_scene_files(root, scene, *, forward=False):
    (validate_forward_scene_cache if forward else validate_scene_cache)(scene)
    root = Path(root).resolve()
    for record in scene['windows']:
        path = (root / record['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != record['sha256']:
            raise RuntimeError(f"Prepared window changed: {record['file']}")


def initialize_root(root, contract):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'DATA_CONTRACT.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')).get('data_contract') != contract:
            raise RuntimeError('Output contains a different cache format; use a new root')
    elif any(root.iterdir()):
        raise RuntimeError('Use an empty output directory for a new preparation')
    else:
        write_json(path, {'data_contract': contract})
    if (root / 'runs').exists():
        raise RuntimeError('Preparation output must be separate from training runs')


def load_rgb(path):
    capture = cv2.VideoCapture(str(path))
    images = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    array = np.stack(images)
    if array.shape != (81, 704, 1280, 3):
        raise SceneRejected(f'Unexpected source shape {array.shape}')
    return array


@torch.no_grad()
def prepare_scene(row, root, store, estimator, vae, text):
    scene = row['sample_id']
    clip_hash = digest(SAMPLES / scene / 'clip.mp4')
    pose_hash = digest(SAMPLES / scene / 'pose.npz')
    if clip_hash != row['clip_sha256']:
        raise RuntimeError(f'Source video differs from the training selection: {scene}')
    destination = root / 'data' / scene
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'COMPLETE.json'
    if marker.exists():
        complete = json.loads(marker.read_text())
        verify_scene_files(root, complete, forward=True)
        if any(complete[key] != row[key] for key in ('sample_id', 'clip_id', 'parent_video_id')):
            raise RuntimeError('Prepared scene differs from the fixed training selection')
        if complete['clip_sha256'] != clip_hash or complete['source_pose_sha256'] != pose_hash:
            raise RuntimeError('Source video or poses changed after preparation')
        for name in ('inference.pt', 'history_depths.pt'):
            if digest(destination / name) != complete['support_sha256'][name]:
                raise RuntimeError(f'Forward support cache changed: {name}')
        return complete
    started = time.monotonic()
    images = load_rgb(SAMPLES / scene / 'clip.mp4')
    records = []
    with store.begin() as transaction:
        for anchor in range(21):
            raw = transaction.get(f'{scene}:{anchor}'.encode())
            if raw is None:
                raise SceneRejected('Missing LMDB anchor')
            record = pickle.loads(raw)
            if record['source_rgb_index'] != anchor * 4:
                raise SceneRejected('RGB/latent anchor misalignment')
            records.append(record)
    cached = torch.stack([torch.as_tensor(record['latent']) for record in records], dim=1)[None].float()
    if cached.shape != (1, 48, 21, 44, 80) or not torch.isfinite(cached).all():
        raise SceneRejected('Invalid cached latent')
    latent_camera = torch.stack([torch.as_tensor(record['K_l']).float() for record in records])
    if not torch.allclose(latent_camera, latent_camera[0].expand_as(latent_camera), atol=1e-4):
        raise SceneRejected('Training cache requires fixed intrinsics')
    camera = latent_camera[0].clone()
    camera[:2] *= 16
    pose_asset = np.load(SAMPLES / scene / 'pose.npz')
    if not np.array_equal(pose_asset['inds'], np.arange(81)):
        raise SceneRejected('Incomplete source poses')
    absolute = torch.linalg.inv(torch.from_numpy(pose_asset['data']).float())
    stored = torch.stack([torch.as_tensor(record['E']).float() for record in records])
    if not torch.allclose(absolute[::4], stored, atol=1e-3, rtol=1e-4):
        raise SceneRejected('Pose archive and LMDB conventions differ')
    seed_depth = estimator.depth_for_views([images[0]], [camera], [torch.eye(4)], 44, 80)[0]['depth'].cpu()
    valid = torch.isfinite(seed_depth) & (seed_depth > 0)
    if valid.float().mean() < .5:
        raise SceneRejected('Seed MapAnything coverage below 50 percent')
    vipe_depth = torch.as_tensor(records[0]['depth']).float()
    shared = valid & torch.isfinite(vipe_depth) & (vipe_depth > 0)
    if shared.float().mean() < .4:
        raise SceneRejected('Insufficient shared seed geometry for scale alignment')
    unit = float(seed_depth[valid].median())
    map_to_vipe = float((seed_depth[shared] / vipe_depth[shared]).median())
    poses = absolute @ torch.linalg.inv(absolute[0])
    poses[:, :3, 3] *= map_to_vipe / unit
    standalone = encode_wan_latents(vae, rgb_tensor(images[:1]).to('cuda')).cpu()
    prefix_cached_error = float((standalone - cached[:, :, :1]).abs().max())
    bundle = {'scene': row['clip_id'], 'sample_id': scene, 'height': 704, 'width': 1280, 'seed_rgb': images[0], 'seed_latent': standalone, 'seed_depth': seed_depth / unit, 'seed_depth_metric_median': unit, 'intrinsics': camera.repeat(81, 1, 1), 'intrinsics_latent': latent_camera[0].repeat(81, 1, 1), 'poses': {'nvs': poses}, 'latent_contract': wan_latent_contract(vae), 'text': text, 'geometry_calibration': 'offline_training_ViPE_poses_scaled_by_I0_MapAnything_depth', 'map_to_vipe_depth_ratio': map_to_vipe}
    windows, latent_audit = [], []
    for number, window in enumerate(window_indices()):
        indices = window['indices']
        fresh = encode_wan_latents(vae, rgb_tensor(images[indices]).to('cuda')).cpu()
        candidate = cached[:, :, np.array(indices[::4]) // 4]
        maximum = float((candidate - fresh).abs().max())
        compatible = torch.allclose(candidate, fresh, atol=.02, rtol=.005)
        latent = candidate if compatible else fresh
        if number == 0:
            short = encode_wan_latents(vae, rgb_tensor(images[indices[:17]]).to('cuda')).cpu()
            if not torch.allclose(short, latent[:, :, :5], atol=.02, rtol=.005):
                raise RuntimeError('Temporal causality check failed')
        windows.append({**window, 'latent': latent.to(torch.bfloat16), 'valid_mask': torch.tensor(window['valid'])})
        latent_audit.append({'window': number, 'max_abs_cached_vs_native': maximum, 'reuse_cached': bool(compatible), 'reason': 'matched_native_window' if compatible else 'native_window_reencoded_for_prefix_contract'})
    depths, window_records = {}, []
    for number, window in enumerate(windows):
        memory = ProtectedMemory(bundle, 'cpu', writeback=True)
        real_hash = memory.fingerprint(True)
        prefix_time, groups = history_groups(window['start'], window['direction'])
        history = []
        for group in groups:
            key = tuple(group)
            if key not in depths:
                camera_poses = torch.stack([poses[frame] for _, frame in group])
                metric = poses_in_metres(camera_poses, unit)
                results = estimator.depth_for_views([images[frame].copy() for _, frame in group], [camera for _ in group], list(metric), 44, 80)
                depths[key] = [result['depth'].cpu() / unit for result in results]
            for (time_index, frame), depth in zip(group, depths[key]):
                assert time_index <= prefix_time
                memory.update(history_feature(windows, time_index), depth, latent_camera[0], poses[frame], time_index, 'generated')
                history.append({'trajectory_time': time_index, 'rgb_frame': frame, 'rgb_source': 'GT_past_only', 'origin': 'generated'})
        assert memory.fingerprint(True) == real_hash and len(history) == prefix_time // 4
        indices = window['indices'][::4]
        condition = conditioning(poses[indices], bundle['intrinsics_latent'][indices], 44, 80, memory).to(torch.bfloat16)
        item = {**window, 'data_contract': forward_data_contract(), 'condition': condition, 'history': history, 'sample_id': scene, 'prefix_trajectory_time': prefix_time}
        validate_prepared_window(item, {'sample_id': scene, 'start': window['start'],
                                        'direction': window['direction']}, forward=True)
        path = destination / f'window_{number:02d}.pt'
        save_tensor(path, item)
        window_records.append({'file': path.relative_to(root).as_posix(), 'sha256': digest(path), 'sample_id': scene, 'window': number, 'start': window['start'], 'direction': window['direction'], 'history_count': len(history), 'target_hole_fraction': float((~((condition[0, 49, 1:] > .5) | (condition[0, 100, 1:] > .5))).float().mean())})
    save_tensor(destination / 'inference.pt', bundle)
    Image.fromarray(images[0]).save(destination / 'seed.jpg', quality=95)
    Image.fromarray(images[-1]).save(destination / 'last.jpg', quality=95)
    save_tensor(destination / 'history_depths.pt', depths)
    result = {'sample_id': scene, 'clip_id': row['clip_id'], 'parent_video_id': row['parent_video_id'], 'motion_group': row['motion_group'], 'motion': row['motion'], 'clip_sha256': clip_hash, 'unit_metres': unit, 'map_to_vipe_ratio': map_to_vipe, 'seed_coverage': float(valid.float().mean()), 'prefix_cached_error': prefix_cached_error, 'latent_audit': latent_audit, 'windows': window_records, 'seconds': time.monotonic() - started}
    result['data_contract'] = forward_data_contract()
    result['source_pose_sha256'] = pose_hash
    result['support_sha256'] = {name: digest(destination / name)
                                for name in ('inference.pt', 'history_depths.pt')}
    validate_forward_scene_cache(result)
    write_json(marker, result)
    print(json.dumps({'event': 'SCENE_PREPARED', 'sample_id': scene, 'seconds': result['seconds'], 'latent_reused_windows': sum(item['reuse_cached'] for item in latent_audit)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    selection_path = Path(os.environ['ENGRAM_TRAINING_SCENES'])
    selection = load_training_selection(selection_path)
    rows = selection['scenes']
    initialize_root(root, forward_data_contract())
    inputs = {'selection_sha256': digest(selection_path),
              'latent_stats_sha256': digest(LATENT_STATS),
              'text_bundle_sha256': digest(Path(os.environ['ENGRAM_TEXT_BUNDLE']))}
    inputs_path = root / 'PREPARATION_INPUTS.json'
    if inputs_path.exists() and json.loads(inputs_path.read_text(encoding='utf-8')) != inputs:
        raise RuntimeError('Preparation inputs changed; use a new output root')
    write_json(inputs_path, inputs)
    if (root / 'DATA_READY.json').exists():
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        ready = json.loads((root / 'DATA_READY.json').read_text(encoding='utf-8'))
        validate_forward_manifest(manifest)
        if ready['manifest_sha256'] != digest(root / 'manifest.json'):
            raise RuntimeError('Prepared manifest changed')
        for scene in manifest['scenes']:
            verify_scene_files(root, scene, forward=True)
        print('FORWARD_DATA_ALREADY_READY', flush=True)
        return
    setup_cuda(.24)
    store = lmdb.open(os.environ['ENGRAM_TRAINING_LMDB'], readonly=True, lock=False, readahead=False)
    try:
        with store.begin() as transaction:
            raw = transaction.get(b'__latent_contract__')
            if raw is None:
                raise ValueError('LMDB has no Wan latent normalization metadata')
            latent_contract = pickle.loads(raw)
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(MODEL_ROOT, subfolder='vae', torch_dtype=torch.float32,
                                              local_files_only=True).to('cuda').eval()
        vae.enable_tiling()
        vae.enable_slicing()
        if wan_latent_contract(vae) != latent_contract:
            raise RuntimeError('LMDB and Wan VAE latent normalization differ')
        estimator = build_depth_estimator()
        text = torch.load(Path(os.environ['ENGRAM_TEXT_BUNDLE']), map_location='cpu', weights_only=False)['text']
        save_tensor(root / 'text.pt', text)
        accepted = []
        for row in rows:
            accepted.append(prepare_scene(row, root, store, estimator, vae, text))
            gc.collect()
            torch.cuda.empty_cache()
        manifest = {'version': 'ENGRAM-forward', 'data_contract': forward_data_contract(),
                    'scene_count': SCENE_COUNT, 'window_count': SCENE_COUNT * len(WINDOW_STARTS),
                    'scenes': accepted, 'records': [record for scene in accepted for record in scene['windows']],
                    'latent_contract': latent_contract, 'selection_sha256': inputs['selection_sha256'],
                    'source_training_manifest_sha256': selection['source_manifest_sha256'],
                    'geometry': 'MapAnything seed and past history with ViPE pose scale alignment',
                    'history': 'causal_GT_under_generated_provenance'}
        validate_forward_manifest(manifest)
        write_json(root / 'manifest.json', manifest)
        write_json(root / 'DATA_READY.json', {'pass': True, 'data_contract': forward_data_contract(),
                   'scenes': SCENE_COUNT, 'windows': manifest['window_count'],
                   'manifest_sha256': digest(root / 'manifest.json')})
    finally:
        store.close()


if __name__ == '__main__':
    main()
