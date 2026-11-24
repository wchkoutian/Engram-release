"""Prepare 81-frame source clips for ENGRAM from a video-window manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.data.preprocessing import FRAMES, FPS, WIDTH, HEIGHT, samples, validate_clip, write_json


def resolve_source(row, manifest):
    value = row.get('source_path')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Each preprocessing row requires source_path')
    path = Path(value).expanduser()
    return (manifest.resolve().parent / path).resolve()


def extract_video(row, source, output, seed):
    from lsm_world.data_process.clip_sampling import (
        ClipExtractionSpec, build_clip_extraction_spec, build_ffmpeg_extract_command,
        get_video_info, resolve_ffmpeg_executable, validate_source_dimensions,
    )
    if 'source_start_seconds' in row:
        import math
        start = float(row['source_start_seconds'])
        info = get_video_info(source)
        if info is None or not math.isfinite(start) or start < 0:
            raise ValueError(f'Invalid source video or start time: {source}')
        fps, count, width, height = info
        if fps + 1e-6 < FPS or start + FRAMES / FPS > count / fps + 1e-6:
            raise ValueError(f'Source video cannot cover the requested clip: {source}')
        spec = ClipExtractionSpec(start, FRAMES / FPS, fps, count, width, height)
    else:
        spec, reason = build_clip_extraction_spec(source, FRAMES, FPS, seed)
        if spec is None:
            raise ValueError(f'{source}: {reason}')
    valid, reason = validate_source_dimensions(spec.source_width, spec.source_height, (WIDTH, HEIGHT))
    if not valid:
        raise ValueError(reason)
    command = build_ffmpeg_extract_command(resolve_ffmpeg_executable(), source, output, spec,
                                           FRAMES, FPS, (WIDTH, HEIGHT))
    # Match the local source-clip materializer's H.264 encoding settings.
    command[-1:-1] = ['-preset', 'medium', '-crf', '18', '-r', str(FPS), '-movflags', '+faststart']
    subprocess.run(command, check=True)
    return spec.start_time_sec


def extract_frames(row, source, output):
    from lsm_world.data_process.clip_sampling import resolve_ffmpeg_executable
    from PIL import Image
    indices = row.get('selected_source_timestamps_us')
    if not isinstance(indices, list) or len(indices) != FRAMES:
        raise ValueError('Frame-directory rows require 81 selected_source_timestamps_us values')
    if any(isinstance(i, bool) or not isinstance(i, int) for i in indices) or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError('Source timestamps must be strictly increasing integers')
    paths = {}
    for path in source.iterdir():
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png') and path.stem.isdigit():
            key = int(path.stem)
            if key in paths:
                raise ValueError(f'Duplicate frame timestamp: {key}')
            paths[key] = path
    with tempfile.TemporaryDirectory(prefix='source_frames_', dir=output.parent) as temporary:
        work = Path(temporary)
        for index, timestamp in enumerate(indices):
            path = paths[timestamp]
            with Image.open(path) as image:
                if image.width < WIDTH or image.height < HEIGHT:
                    raise ValueError(f'Frame is too small for the target crop: {path}')
                left, top = (image.width - WIDTH) // 2, (image.height - HEIGHT) // 2
                image.convert('RGB').crop((left, top, left + WIDTH, top + HEIGHT)).save(work / f'{index:05d}.png')
        subprocess.run([resolve_ffmpeg_executable(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                        '-y', '-framerate', str(FPS), '-i', str(work / '%05d.png'), '-frames:v', str(FRAMES),
                        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-r', str(FPS), '-movflags', '+faststart', str(output)], check=True)


def prepare(args):
    completed = []
    for row, folder in samples(args.manifest, args.sample_root, args.shard_id, args.num_shards):
        source = resolve_source(row, args.manifest)
        if not source.exists():
            raise FileNotFoundError(source)
        folder.mkdir(parents=True, exist_ok=True)
        clip, metadata = folder / 'clip.mp4', folder / 'clip_metadata.json'
        binding = {'source_path': str(source), 'requested_start_seconds': row.get('source_start_seconds'),
                   'selected_source_timestamps_us': row.get('selected_source_timestamps_us'), 'seed': args.seed}
        if clip.exists() or metadata.exists():
            if not args.skip_existing or not clip.is_file() or not metadata.is_file():
                raise FileExistsError(f'Existing sample; use a new output or --skip-existing: {folder}')
            old = json.loads(metadata.read_text(encoding='utf-8'))
            if old.get('source') != binding:
                raise ValueError(f'Existing clip belongs to different source settings: {folder}')
            digest = validate_clip(clip, row.get('clip_sha256'))
            if old.get('clip_sha256') != digest:
                raise ValueError(f'Existing clip changed: {clip}')
            completed.append({'sample_id': row['sample_id'], 'status': 'reused'})
            continue
        with tempfile.TemporaryDirectory(prefix='.clip_', dir=folder) as temporary:
            output = Path(temporary) / 'clip.mp4'
            start = extract_video(row, source, output, args.seed) if source.is_file() else None
            if source.is_dir():
                extract_frames(row, source, output)
            digest = validate_clip(output, row.get('clip_sha256'))
            output.replace(clip)
        write_json(metadata, {'sample_id': row['sample_id'], 'source': binding,
                              'source_start_seconds': start, 'clip_sha256': digest,
                              'frames': FRAMES, 'fps': FPS, 'resolution': [WIDTH, HEIGHT]})
        completed.append({'sample_id': row['sample_id'], 'status': 'written'})
        print(json.dumps(completed[-1]), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sample-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--skip-existing', action='store_true')
    print(json.dumps({'samples': prepare(parser.parse_args())}))


if __name__ == '__main__':
    main()
