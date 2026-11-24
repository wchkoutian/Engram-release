"""Add four return windows per scene to the prepared forward cache."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.core import (MODEL_ROOT, LATENT_STATS, ProtectedMemory, build_depth_estimator,
                  conditioning, digest, poses_in_metres, rgb_tensor, save_tensor,
                  setup_cuda, write_json)
from engram.data.contracts import (RETURN_ENDS, SCENE_COUNT, WINDOW_COUNT, data_contract,
                       forward_data_contract, return_indices,
                       validate_forward_manifest, validate_prepared_window,
                       validate_scene_cache, validate_training_manifest)
from engram.models.wan_latents import encode_wan_latents
from prepare import SAMPLES, initialize_root, load_rgb, verify_scene_files
from engram.data.windows import history_feature, history_groups


def scene_digest(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True).encode('utf-8')).hexdigest()


@torch.no_grad()
def build_scene(scene, source, root, estimator, vae):
    sample = scene['sample_id']
    verify_scene_files(source, scene, forward=True)
    if digest(SAMPLES / sample / 'pose.npz') != scene['source_pose_sha256']:
        raise RuntimeError('Source poses differ from forward preparation')
    if digest(SAMPLES / sample / 'clip.mp4') != scene['clip_sha256']:
        raise RuntimeError('Source video differs from forward preparation')
    for name in ('inference.pt', 'history_depths.pt'):
        if digest(source / 'data' / sample / name) != scene['support_sha256'][name]:
            raise RuntimeError(f'Forward support cache changed: {name}')
    source_dir, destination = source / 'data' / sample, root / 'data' / sample
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'COMPLETE.json'
    if marker.exists():
        complete = json.loads(marker.read_text())
        verify_scene_files(root, complete)
        if complete['forward_scene_sha256'] != scene_digest(scene):
            raise RuntimeError('Forward cache differs from the prepared return windows')
        return complete
    started = time.monotonic()
    bundle = torch.load(source_dir / 'inference.pt', map_location='cpu', weights_only=False)
    depths = torch.load(source_dir / 'history_depths.pt', map_location='cpu', weights_only=False)
    forward = []
    records = []
    for record in scene['windows']:
        item = torch.load(source / record['file'], map_location='cpu', weights_only=False)
        if item['data_contract'] != forward_data_contract() or item['direction'] != 1:
            raise RuntimeError('Expected a forward training window')
        validate_prepared_window(item, record, forward=True)
        forward.append(item)
        item = {**item, 'data_contract': data_contract()}
        path = destination / f'window_{record["window"]:02d}.pt'
        save_tensor(path, item)
        records.append({**record, 'file': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    images = load_rgb(SAMPLES / sample / 'clip.mp4')
    poses = bundle['poses']['nvs']
    camera = bundle['intrinsics'][0]
    latent_camera = bundle['intrinsics_latent'][0]
    unit = bundle['seed_depth_metric_median']
    new_depth_groups = 0
    for offset, end in enumerate(RETURN_ENDS):
        indices = return_indices(end)
        latent = encode_wan_latents(vae, rgb_tensor(images[indices]).to('cuda')).cpu().to(torch.bfloat16)
        valid = [step > 0 for step in range(0, 33, 4)]
        memory = ProtectedMemory(bundle, 'cpu', writeback=True)
        real_hash = memory.fingerprint(True)
        prefix_time, groups = history_groups(end, 1)
        history = []
        for group in groups:
            key = tuple(group)
            if key not in depths:
                metric = poses_in_metres(torch.stack([poses[frame] for _, frame in group]), unit)
                results = estimator.depth_for_views([images[frame].copy() for _, frame in group], [camera for _ in group], list(metric), 44, 80)
                depths[key] = [result['depth'].cpu() / unit for result in results]
                new_depth_groups += 1
            for (time_index, frame), depth in zip(group, depths[key]):
                assert time_index <= prefix_time
                memory.update(history_feature(forward, time_index), depth, latent_camera, poses[frame], time_index, 'generated')
                history.append({'trajectory_time': time_index, 'rgb_frame': frame, 'rgb_source': 'GT_outbound_past', 'origin': 'generated'})
        assert memory.fingerprint(True) == real_hash and len(history) == end // 4
        condition = conditioning(poses[indices[::4]], bundle['intrinsics_latent'][indices[::4]], 44, 80, memory).to(torch.bfloat16)
        number = len(forward) + offset
        item = {'start': end, 'direction': -1, 'indices': indices, 'valid': valid, 'latent': latent, 'valid_mask': torch.tensor(valid), 'data_contract': data_contract(), 'condition': condition, 'history': history, 'sample_id': sample, 'prefix_trajectory_time': end}
        validate_prepared_window(item, {'sample_id': sample, 'start': end, 'direction': -1})
        path = destination / f'window_{number:02d}.pt'
        save_tensor(path, item)
        records.append({'file': path.relative_to(root).as_posix(), 'sha256': digest(path), 'sample_id': sample, 'window': number, 'start': end, 'direction': -1, 'history_count': len(history), 'target_hole_fraction': float((~((condition[0, 49, 1:] > .5) | (condition[0, 100, 1:] > .5))).float().mean())})
    save_tensor(destination / 'inference.pt', bundle)
    save_tensor(destination / 'history_depths.pt', depths)
    result = {**{key: value for key, value in scene.items() if key not in ('windows', 'data_contract')}, 'windows': records, 'data_contract': data_contract(), 'return_new_depth_groups': new_depth_groups, 'return_seconds': time.monotonic() - started, 'forward_scene_sha256': scene_digest(scene)}
    result['support_sha256'] = {name: digest(destination / name)
                                for name in ('inference.pt', 'history_depths.pt')}
    validate_scene_cache(result)
    write_json(marker, result)
    print(json.dumps({'event': 'SCENE_PREPARED', 'sample_id': sample, 'seconds': result['return_seconds'], 'new_depth_groups': new_depth_groups}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    if source == root:
        raise ValueError('Forward and final caches must use separate directories')
    forward = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    ready = json.loads((source / 'DATA_READY.json').read_text(encoding='utf-8'))
    validate_forward_manifest(forward)
    if ready['manifest_sha256'] != digest(source / 'manifest.json'):
        raise RuntimeError('Forward manifest hash mismatch')
    forward_inputs = json.loads((source / 'PREPARATION_INPUTS.json').read_text(encoding='utf-8'))
    if forward_inputs['latent_stats_sha256'] != digest(LATENT_STATS):
        raise RuntimeError('Return preparation must use the same latent statistics as the forward cache')
    scenes = forward['scenes']
    initialize_root(root, data_contract())
    inputs = {'forward_manifest_sha256': digest(source / 'manifest.json'),
              'latent_stats_sha256': digest(LATENT_STATS)}
    inputs_path = root / 'PREPARATION_INPUTS.json'
    if inputs_path.exists() and json.loads(inputs_path.read_text(encoding='utf-8')) != inputs:
        raise RuntimeError('Forward preparation inputs changed; use a new output root')
    write_json(inputs_path, inputs)
    if (root / 'DATA_READY.json').exists():
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        ready = json.loads((root / 'DATA_READY.json').read_text(encoding='utf-8'))
        validate_training_manifest(manifest)
        if ready['manifest_sha256'] != digest(root / 'manifest.json'):
            raise RuntimeError('Prepared manifest changed')
        for scene in manifest['scenes']:
            verify_scene_files(root, scene)
        print('TRAINING_DATA_ALREADY_READY', flush=True)
        return
    setup_cuda(.24)
    estimator = build_depth_estimator()
    from diffusers import AutoencoderKLWan
    from engram.models.wan_latents import wan_latent_contract
    vae = AutoencoderKLWan.from_pretrained(MODEL_ROOT, subfolder='vae', torch_dtype=torch.float32,
                                          local_files_only=True).to('cuda').eval()
    vae.enable_tiling()
    vae.enable_slicing()
    if wan_latent_contract(vae) != forward['latent_contract']:
        raise RuntimeError('Forward cache and Wan VAE latent normalization differ')
    save_tensor(root / 'text.pt', torch.load(source / 'text.pt', map_location='cpu', weights_only=False))
    accepted = [build_scene(scene, source, root, estimator, vae) for scene in scenes]
    manifest = {**{key: value for key, value in forward.items()
                   if key not in ('scenes', 'records', 'data_contract', 'window_count', 'history', 'version')},
                'version': 'ENGRAM', 'data_contract': data_contract(), 'scenes': accepted,
                'records': [record for scene in accepted for record in scene['windows']],
                'window_count': WINDOW_COUNT, 'history': data_contract()['history'],
                'forward_manifest_sha256': inputs['forward_manifest_sha256']}
    validate_training_manifest(manifest)
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'DATA_READY.json', {'pass': True, 'data_contract': data_contract(),
               'scenes': SCENE_COUNT, 'windows': WINDOW_COUNT,
               'manifest_sha256': digest(root / 'manifest.json')})


if __name__ == '__main__':
    main()
