"""Portable release entrypoint. Asset checks and --help need only Python."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
MODELS = ['wan_model', 'mapanything_model', 'torch_home', 'latent_stats']
WEIGHTS = ['checkpoint', 'checkpoint_sha256', 'stage1_checkpoint', 'training_root']
REQUIRED = {
    'build-lmdb': ['wan_model', 'training_samples', 'training_scenes'],
    'merge-lmdb': ['training_scenes'],
    'validate-lmdb': ['training_samples', 'training_scenes'],
    'prepare-training': MODELS + ['text_bundle', 'training_lmdb', 'training_samples', 'training_scenes'],
    'prepare-training-return': MODELS + ['training_samples'],
    'train-stage1': ['wan_model'],
    'train-stage2': ['wan_model', 'stage1_checkpoint'],
    'caption-training': ['wan_model', 'caption_model', 'text_bundle'],
    'prepare-re10k': MODELS + WEIGHTS + ['text_bundle', 're10k_data'],
    'infer-re10k': MODELS + WEIGHTS,
    'score-re10k': ['torch_home'],
    'export-re10k': [],
    'prepare-worldscore': MODELS + WEIGHTS + ['text_bundle'],
    'infer-worldscore': MODELS + WEIGHTS,
}
FILE_KEYS = {'latent_stats', 'checkpoint', 'stage1_checkpoint', 'text_bundle', 'training_scenes'}


def load_assets(path, action=None):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('Assets must be a JSON object')
    allowed = {key for required in REQUIRED.values() for key in required}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError('Unknown asset keys: ' + ', '.join(sorted(unknown)))
    result = {}
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Asset {key} must be a nonempty string')
        if key == 'checkpoint_sha256':
            required = action is None or key in REQUIRED[action]
            if required and (len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value)):
                raise ValueError('Invalid checkpoint SHA256; hash the actual checkpoint')
            result[key] = value.lower()
        else:
            location = Path(os.path.expandvars(value)).expanduser()
            result[key] = str((path.parent / location).resolve() if not location.is_absolute() else location.resolve())
    return result


def check_assets(values, action):
    errors = []
    for key in REQUIRED[action]:
        if key not in values:
            errors.append(f'Missing {key}: not configured')
        elif key != 'checkpoint_sha256':
            path = Path(values[key])
            valid = path.is_file() if key in FILE_KEYS else path.is_dir()
            if not valid:
                errors.append(f'Missing {key}: {path}')
    if errors:
        raise ValueError('\n'.join(errors))


def configure(values):
    for key, value in values.items():
        os.environ['TORCH_HOME' if key == 'torch_home' else 'ENGRAM_' + key.upper()] = value
    for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'DIFFUSERS_OFFLINE']:
        os.environ[name] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    sys.dont_write_bytecode = True
    for folder in reversed(['scripts', 'source', 'runtime_overlay', 'inference']):
        sys.path.insert(0, str(ROOT / folder))


def isolate_rank_gpu():
    """The archived trainers address cuda:0 with one visible GPU per process."""
    if int(os.environ.get('WORLD_SIZE', '1')) <= 1:
        return
    rank = int(os.environ['LOCAL_RANK'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = [item.strip() for item in visible.split(',')] if visible else None
    if devices is not None and (rank >= len(devices) or not devices[rank]):
        raise ValueError('LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[rank] if devices else str(rank)


def main():
    parser = argparse.ArgumentParser(description='ENGRAM training and camera-conditioned video inference')
    parser.add_argument('action', choices=[*REQUIRED, 'check-assets'])
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--root', type=Path, help='LMDB output, prepared training root, or evaluation output root')
    parser.add_argument('--source', type=Path, help='Forward cache for prepare-training-return')
    parser.add_argument('--sample', help='RE10K case identifier, e.g. 000')
    parser.add_argument('--steps', type=int, help='Defaults: Stage 1 = 4000; Stage 2 = 8000')
    parser.add_argument('--accumulation', type=int, default=4, help='Global microbatch count per update')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda', help='Device for LMDB encoding or scoring')
    parser.add_argument('--memory-fraction', type=float,
                        help='Override the per-process CUDA memory limit (0 < value <= 1)')
    parser.add_argument('--for-action', choices=REQUIRED, default='infer-re10k')
    parser.add_argument('--phases', default='frames,caption,encode')
    parser.add_argument('--manifest', type=Path, help='WorldScore input manifest or LMDB scene-list override')
    parser.add_argument('--num-shards', type=int, default=1, help='Number of LMDB encoding shards')
    parser.add_argument('--shard-id', type=int, default=0, help='LMDB shard index, starting at zero')
    parser.add_argument('--map-size-gib', type=int, default=1024, help='LMDB virtual map size in GiB')
    parser.add_argument('--audit-scenes', type=int, default=8, help='Source scenes checked by validate-lmdb')
    args = parser.parse_args()
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            parser.error('--memory-fraction must be greater than 0 and at most 1')
        os.environ['ENGRAM_MEMORY_FRACTION'] = str(args.memory_fraction)
    if args.assets is None:
        parser.error('--assets is required; copy configs/assets.example.json and configure local paths')
    action = args.for_action if args.action == 'check-assets' else args.action
    values = load_assets(args.assets, action)
    check_assets(values, action)
    if args.action == 'check-assets':
        print(json.dumps({'pass': True, 'action': action, 'checked': REQUIRED[action], 'gpu_run': False}))
        return
    if args.root is None:
        parser.error('--root is required')
    args.root = args.root.resolve()
    if args.action in ['infer-re10k', 'score-re10k'] and not args.sample:
        parser.error('--sample is required')
    if args.steps is not None and args.steps <= 0 or args.accumulation <= 0:
        parser.error('steps and accumulation must be positive')
    if args.action == 'caption-training' and any(x not in ('frames', 'caption', 'encode') for x in args.phases.split(',')):
        parser.error('--phases must be a comma-separated subset of frames,caption,encode')
    if args.action == 'prepare-worldscore' and args.manifest is None:
        parser.error('--manifest is required')
    if args.action == 'prepare-training-return' and args.source is None:
        parser.error('--source is required for prepare-training-return')
    if args.action.startswith('train-'):
        isolate_rank_gpu()  # must run before any torch import
    configure(values)
    if args.action in ('build-lmdb', 'merge-lmdb', 'validate-lmdb'):
        script = ROOT / 'scripts/build_lmdb.py'
        command = {'build-lmdb': 'encode', 'merge-lmdb': 'merge', 'validate-lmdb': 'validate'}[args.action]
        manifest = args.manifest.resolve() if args.manifest else values['training_scenes']
        sys.argv = [str(script), command, '--manifest', str(manifest), '--output-dir', str(args.root)]
        if command != 'validate':
            sys.argv.extend(['--num-shards', str(args.num_shards), '--map-size-gib', str(args.map_size_gib)])
        if command != 'merge':
            sys.argv.extend(['--sample-root', values['training_samples']])
        if command == 'encode':
            sys.argv.extend(['--model-id', values['wan_model'], '--device', args.device, '--shard-id', str(args.shard_id)])
        if command == 'validate':
            sys.argv.extend(['--audit-scenes', str(args.audit_scenes)])
        runpy.run_path(str(script), run_name='__main__')
    elif args.action in ('prepare-training', 'prepare-training-return'):
        script = ROOT / 'scripts' / ('prepare_return.py' if args.action.endswith('return') else 'prepare.py')
        sys.argv = [str(script), '--root', str(args.root)]
        if args.source is not None:
            sys.argv.extend(['--source', str(args.source.resolve())])
        runpy.run_path(str(script), run_name='__main__')
    elif args.action.startswith('train-'):
        if not args.root.is_dir():
            parser.error('Training root does not exist; prepare its cache and captions first')
        stage2 = args.action == 'train-stage2'
        script = ROOT / 'scripts' / ('train_bb.py' if stage2 else 'train.py')
        sys.argv = [str(script), '--root', str(args.root), '--steps', str(args.steps or (8000 if stage2 else 4000)), '--accumulation', str(args.accumulation)]
        if args.resume:
            sys.argv.append('--resume')
        runpy.run_path(str(script), run_name='__main__')
    elif args.action == 'caption-training':
        from engram.train import captions
        captions.run(args, values)
    elif args.action.endswith('worldscore'):
        import worldscore
        getattr(worldscore, 'prepare' if args.action.startswith('prepare') else 'infer')(args, values)
    else:
        import re10k
        if args.action == 'prepare-re10k':
            args.data_root = Path(values['re10k_data'])
            re10k.prepare(args, values)
        elif args.action == 'infer-re10k':
            re10k.infer(args, values)
        elif args.action == 'score-re10k':
            re10k.score(args, values)
        else:
            re10k.export(args)


if __name__ == '__main__':
    main()
