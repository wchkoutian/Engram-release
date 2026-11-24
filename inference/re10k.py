"""RE10K preparation and scoring, with portable checkpoint locations."""
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from re10k_protocol import (FRAME_COUNTS, IDENTIFIER, SCENES, crop_seed, digest,
                           inside, score_pairs, select_case, trajectory, validate_manifest, verify_assets)
HERE = Path(__file__).resolve().parent
CODE = HERE.parent

def infer(args, assets):
    import torch
    import fcntl
    from engram.core import build_depth_estimator
    from model_loader import load_model
    from re10k_rollout import run_case
    manifest = read_run(args)
    case = select_case(manifest, args.sample)
    with (args.root / f'{case["name"]}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, training_sha = checkpoint(assets)
        if training_sha != manifest['training_manifest_sha256'] or assets['checkpoint_sha256'] != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from prepared evaluation')
        # Inference may use I0, never future reference frames.
        forbidden = {str(inside(Path(manifest['data_root']), frame['image_path']))
                     for row in manifest['cases'] for frame in row['frames'][1:]}
        def guard(event, arguments):
            if event == 'open' and arguments and isinstance(arguments[0], (str, bytes)):
                if str(Path(os.fsdecode(arguments[0])).resolve()) in forbidden:
                    raise PermissionError('Inference must not read future GT frames')
        sys.addaudithook(guard)
        model = load_model(state)
        del state
        estimator = build_depth_estimator()
        with torch.no_grad():
            run_case(args.root, case, manifest, model, estimator)

def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

def checkpoint(assets):
    import torch
    from engram.data.contracts import require_data_contract, validate_checkpoint_config, validate_training_manifest
    from engram.core import recipe_contract

    if 'checkpoint_sha256' not in assets:
        raise ValueError('Set the SHA256 of the selected checkpoint')
    path = Path(assets['checkpoint'])
    if digest(path) != assets['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    training_path = Path(assets['training_root']) / 'manifest.json'
    training = json.loads(training_path.read_text())
    validate_training_manifest(training)
    if set(SCENES) & {scene['clip_id'] for scene in training['scenes']}:
        raise ValueError('Training clip IDs overlap the RE10K test-100 test set')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state['config'].get('version') == 'V22-stage2-bbfull':  # V22 (10-20): stage-2 full backbone fine-tune of the V14 stage-1 weights
        c = state['config']
        require_data_contract(c)
        if c.get('manifest_sha256') != digest(training_path) or c['recipe'] != recipe_contract() or c.get('lora') is not False:
            raise ValueError('V22 checkpoint does not match the V14 training manifest / recipe')
        if digest(Path(assets['stage1_checkpoint'])) != c['stage1_sha256']:
            raise ValueError('V22 stage-1 checkpoint changed')
        if not isinstance(state['step'], int) or not 0 < state['step'] <= c['steps'] or 'backbone_blocks' not in state:
            raise ValueError('Invalid V22 checkpoint')
        return state, digest(training_path)
    validate_checkpoint_config(state['config'], digest(training_path))
    if state['config']['recipe'] != recipe_contract() or state['config'].get('arm') != 'G':
        raise ValueError('Expected V11 original G weights')
    if not isinstance(state['step'], int) or not 0 < state['step'] <= state['config']['steps']:
        raise ValueError('Invalid checkpoint step')
    return state, digest(training_path)


def prepare(args, assets):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan
    from engram.core import build_depth_estimator, rgb_tensor, save_tensor, setup_cuda, write_json
    from engram.models.wan_latents import encode_wan_latents, wan_latent_contract

    manifest, audit = verify_assets(args.data_root)
    state, training_sha = checkpoint(assets)
    step = state['step']
    del state
    if args.root.exists() and any(args.root.iterdir()):
        raise ValueError('Use a new evaluation root; do not replace an existing run')
    args.root.mkdir(parents=True, exist_ok=True)
    setup_cuda(.24)
    estimator = build_depth_estimator()
    vae = AutoencoderKLWan.from_pretrained(assets['wan_model'], subfolder='vae', torch_dtype=torch.float32, local_files_only=True).to('cuda').eval()
    vae.enable_tiling()
    vae.enable_slicing()
    text = torch.load(assets['text_bundle'], map_location='cpu', weights_only=False)['text']
    prepared = []
    with torch.no_grad():
        for case in manifest['cases']:
            with Image.open(inside(args.data_root, case['frames'][0]['image_path'])) as image:
                seed, camera_array = crop_seed(image, case['frames'][0]['fxfycxcy'])
            rgb = np.asarray(seed).copy()
            camera = torch.from_numpy(camera_array)
            depth = estimator.depth_for_views([rgb], [camera], [torch.eye(4)], 44, 80)[0]['depth'].cpu()
            valid = torch.isfinite(depth) & (depth > 0)
            if valid.float().mean() < .5:
                raise ValueError(f'Insufficient seed geometry: {case["name"]}')
            unit = float(depth[valid].median())
            poses = {}
            for protocol, relative in case['trajectories'].items():
                raw = torch.from_numpy(trajectory(inside(args.data_root, relative), protocol))
                relative_pose = raw @ torch.linalg.inv(raw[0])
                relative_pose[:, :3, 3] /= unit
                poses[protocol] = relative_pose
            latent = encode_wan_latents(vae, rgb_tensor(rgb[None]).to('cuda')).cpu()
            latent_camera = camera.clone()
            latent_camera[:2] /= 16
            bundle = {'scene': case['scene_name'], 'height': 704, 'width': 1280, 'seed_rgb': rgb, 'seed_latent': latent, 'seed_depth': depth / unit, 'seed_depth_metric_median': unit, 'intrinsics': camera[None], 'intrinsics_latent': latent_camera[None], 'poses': poses, 'latent_contract': wan_latent_contract(vae), 'text': text, 'geometry_calibration': 'fixed_legacy_SfM_command_no_extra_scale_fit; poses_and_MapAnything_depth_divided_by_same_seed_median'}
            folder = args.root / 'data' / case['name']
            save_tensor(folder / 'inference.pt', bundle)
            seed.save(folder / 'conditioning_1280x704.png')
            prepared.append({**case, 'input_sha256': digest(folder / 'inference.pt'), 'seed_sha256': digest(folder / 'conditioning_1280x704.png')})
            print(json.dumps({'prepared': case['name']}), flush=True)
    write_json(args.root / 'MANIFEST.json', {'identifier': IDENTIFIER, 'frame_counts': FRAME_COUNTS, 'resolution': [1280, 704], 'source_split': manifest['source_split'], 'cases': prepared, 'data_root': str(args.data_root), 'selection_sha256': digest(args.data_root / 'selected_10.json'), 'checkpoint': assets['checkpoint'], 'checkpoint_sha256': assets['checkpoint_sha256'], 'checkpoint_step': step, 'training_root': assets['training_root'], 'training_manifest_sha256': training_sha, 'code_sha256': {path.name: digest(path) for path in HERE.glob('*.py')}, 'frozen_core_manifest_sha256': digest(CODE / 'PACKAGE_MANIFEST.json'), 'asset_audit': audit, 'scope': 'RE10K test-100; NVS81 and closed-loop153'})


def read_run(args):
    manifest = json.loads((args.root / 'MANIFEST.json').read_text())
    validate_manifest(manifest)
    if digest(CODE / 'PACKAGE_MANIFEST.json') != manifest['frozen_core_manifest_sha256']:
        raise ValueError('Frozen package manifest changed')
    for name, expected in manifest['code_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError('RE10K test-100 adapter changed after preparation')
    return manifest


def score(args, assets):
    import math
    import numpy as np
    import torch
    import lpips
    from PIL import Image
    from engram.core import setup_cuda, write_json

    manifest = read_run(args)
    case = select_case(manifest, args.sample)
    data_root = Path(manifest['data_root'])
    if digest(data_root / 'selected_10.json') != manifest['selection_sha256']:
        raise ValueError('Scoring data manifest changed')
    copied = json.loads((data_root / 'selected_10.json').read_text())['copied_files']
    if digest(args.root / 'data' / case['name'] / 'conditioning_1280x704.png') != case['seed_sha256']:
        raise ValueError('Scoring seed changed')
    for relative in ['reference_code/metric_helper.py', 'reference_code/metrics.py'] + [frame['image_path'] for frame in case['frames']]:
        if digest(inside(data_root, relative)) != copied[relative]['sha256']:
            raise ValueError('Scoring asset changed')
    frozen_metrics = load_module('local10_frozen_metrics', data_root / 'reference_code/metrics.py')
    helper = load_module('local10_frozen_helper', data_root / 'reference_code/metric_helper.py')
    helper.psnr, helper.ssim = frozen_metrics.psnr, frozen_metrics.ssim
    device = torch.device(args.device)
    if device.type == 'cuda':
        setup_cuda(.08)
    perceptual = lpips.LPIPS(net='alex').eval().to(device)
    with torch.no_grad():
        for protocol in ['nvs', 'closed_loop']:
            folder = args.root / 'evaluation' / case['name'] / protocol
            complete = json.loads((folder / 'COMPLETE.json').read_text())
            if complete['checkpoint_sha256'] != manifest['checkpoint_sha256'] or complete['input_sha256'] != case['input_sha256']:
                raise ValueError('Output provenance mismatch')
            video = helper._decode_video(folder / 'engram_raw.mp4')
            if len(video) != FRAME_COUNTS[protocol]:
                raise ValueError('Generated video frame count mismatch')
            rows = []
            for generated_index, reference_index in score_pairs(protocol):
                reference_path = inside(data_root, case['frames'][reference_index]['image_path']) if protocol == 'nvs' else args.root / 'data' / case['name'] / 'conditioning_1280x704.png'
                with Image.open(reference_path) as image:
                    reference = np.asarray(helper._crop_and_resize(image, 704, 1280) if protocol == 'nvs' else image.convert('RGB'))
                values = helper._pair_metrics(helper._to_tensor(video[generated_index], device), helper._to_tensor(reference, device), perceptual)
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError('Nonfinite score')
                rows.append({'frame': generated_index, **values})
            write_json(folder / 'scores.json', {'case': case['name'], 'protocol': protocol, 'checkpoint_sha256': manifest['checkpoint_sha256'], 'mean': helper._mean(rows), 'evaluated_frames': len(rows), 'rows': rows, 'scoring_source': 'decoded MP4, frozen RE10K test-100 helper and metric primitives; matches historical scoring domain'})


def export(args):
    import numpy as np
    from engram.core import write_json

    manifest = read_run(args)
    result = {'identifier': IDENTIFIER, 'checkpoint_sha256': manifest['checkpoint_sha256'], 'scenes': list(SCENES), 'scope': manifest['scope'], 'metrics': {}}
    csv_rows = []
    for protocol in ['nvs', 'closed_loop']:
        rows = []
        for case in manifest['cases']:
            folder = args.root / 'evaluation' / case['name'] / protocol
            score_row = json.loads((folder / 'scores.json').read_text())
            if score_row['checkpoint_sha256'] != manifest['checkpoint_sha256']:
                raise ValueError('Mixed checkpoints in aggregate')
            row = {'source_index': case['source_index'], 'scene_name': case['scene_name'], **score_row['mean']}
            rows.append(row)
            csv_rows.append({'protocol': protocol, **row})
        result['metrics'][protocol] = {'scene_count': len(manifest['cases']), 'per_scene': rows, 'mean': {key: float(np.mean([row[key] for row in rows])) for key in ['psnr', 'ssim', 'lpips']}}
    write_json(args.root / 'table2_metrics.json', result)
    with (args.root / 'SUMMARY.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(args.root / 'COMPLETE.json', {'scored_videos': 2 * len(manifest['cases']), 'quality_accepted': False})
