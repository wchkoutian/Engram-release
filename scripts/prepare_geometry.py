"""Run an external ViPE installation and export geometry beside each source clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.data.preprocessing import samples, sha256, validate_clip, validate_geometry, write_json

ASSETS = {'pose': ('pose', '.npz', 'pose.npz'),
          'intrinsics': ('intrinsics', '.npz', 'intrinsics.npz'),
          'depth': ('depth', '.zip', 'depth.zip')}


def run_vipe(args, pending, raw):
    with tempfile.TemporaryDirectory(prefix='inputs_', dir=args.output_root) as temporary:
        inputs = Path(temporary)
        for row, folder, _ in pending:
            target = inputs / (row['sample_id'] + '.mp4')
            try:
                target.symlink_to((folder / 'clip.mp4').resolve())
            except OSError:
                shutil.copy2(folder / 'clip.mp4', target)
        command = [str(args.vipe_python), str(args.vipe_root / 'run.py'),
                   'pipeline=dav3', 'streams=raw_mp4_stream', f'streams.base_path={inputs}',
                   'streams.frame_start=0', 'streams.frame_end=81', 'streams.frame_skip=1',
                   'pipeline.init.instance=null', f'pipeline.output.path={raw}',
                   'pipeline.output.skip_exists=true', 'pipeline.output.save_artifacts=true',
                   'pipeline.output.save_viz=false', 'pipeline.output.save_slam_map=false']
        subprocess.run(command, cwd=args.vipe_root, check=True)


def prepare(args):
    args.vipe_root = args.vipe_root.resolve()
    args.vipe_python = args.vipe_python.resolve()
    args.output_root = args.output_root.resolve()
    if not (args.vipe_root / 'run.py').is_file():
        raise FileNotFoundError(args.vipe_root / 'run.py')
    if not args.vipe_python.is_file():
        raise FileNotFoundError(args.vipe_python)
    if args.output_root == args.sample_root.resolve():
        raise ValueError('Use a separate ViPE output directory')
    pending, reused, inputs = [], [], {}
    for row, folder in samples(args.manifest, args.sample_root, args.shard_id, args.num_shards):
        digest = validate_clip(folder / 'clip.mp4', row.get('clip_sha256'))
        inputs[row['sample_id']] = digest
        destinations = {key: folder / value[2] for key, value in ASSETS.items()}
        metadata = folder / 'geometry_metadata.json'
        if any(path.exists() for path in destinations.values()) or metadata.exists():
            if not args.skip_existing or not metadata.is_file() or not all(p.is_file() for p in destinations.values()):
                raise FileExistsError(f'Existing geometry; use a new sample root or a verified --skip-existing result: {folder}')
            old = json.loads(metadata.read_text(encoding='utf-8'))
            hashes = {key: sha256(path) for key, path in destinations.items()}
            if old.get('clip_sha256') != digest or old.get('geometry_sha256') != hashes:
                raise ValueError(f'Existing geometry no longer matches its clip/metadata: {folder}')
            validate_geometry(destinations)
            reused.append(row['sample_id'])
        else:
            pending.append((row, folder, digest))
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw = args.output_root / f'shard_{args.shard_id:03d}'
    binding_path = args.output_root / f'shard_{args.shard_id:03d}_inputs.json'
    binding = {'clips': inputs, 'num_shards': args.num_shards, 'shard_id': args.shard_id,
               'pipeline': 'dav3', 'frame_range': [0, 81], 'vipe_run_sha256': sha256(args.vipe_root / 'run.py')}
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding='utf-8')) != binding:
            raise ValueError('ViPE output directory belongs to different clips or settings')
    elif raw.exists():
        raise FileExistsError('ViPE output exists without an input binding; use a new output directory')
    else:
        write_json(binding_path, binding)
    if pending:
        run_vipe(args, pending, raw)
    written = []
    for row, folder, digest in pending:
        source = {key: raw / directory / (row['sample_id'] + suffix)
                  for key, (directory, suffix, _) in ASSETS.items()}
        validate_geometry(source)
        hashes = {key: sha256(path) for key, path in source.items()}
        for key, path in source.items():
            target = folder / ASSETS[key][2]
            temporary = target.with_suffix(target.suffix + '.tmp')
            shutil.copy2(path, temporary)
            temporary.replace(target)
        write_json(folder / 'geometry_metadata.json', {'clip_sha256': digest,
                   'geometry_sha256': hashes, 'pipeline': 'dav3', 'frame_range': [0, 81]})
        written.append(row['sample_id'])
    report = {'written': written, 'reused': reused}
    write_json(args.output_root / f'shard_{args.shard_id:03d}_complete.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--vipe-root', type=Path, required=True)
    parser.add_argument('--vipe-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--skip-existing', action='store_true')
    print(json.dumps(prepare(parser.parse_args())))


if __name__ == '__main__':
    main()
