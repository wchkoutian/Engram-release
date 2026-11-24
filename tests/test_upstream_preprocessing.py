"""CPU checks of vendored provenance and the source-to-LMDB file handoffs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import engram.data.preprocessing as common
import prepare_clips as clips
import prepare_geometry as geometry
import prepare_masks as masks


def test_vendored_lsm_files_and_microsoft_license_are_preserved():
    expected = {
        'LICENSE': 'c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383',
        'data_process/clip_sampling.py': 'e483d77e3b82c6ed20beb86b6b138275685b1b985702f4a38239a134fb1ddf75',
        'data_process/video_io.py': 'e15d7ad036df4fe16c774c03084ad1c3937c2d31cab4767cb9ea2206956d6776',
    }
    for relative, digest in expected.items():
        assert common.sha256(ROOT / 'third_party/lsm_world' / relative) == digest


@pytest.mark.parametrize('name', ['prepare_clips', 'prepare_geometry', 'prepare_masks'])
def test_preprocessing_help_does_not_import_gpu_dependencies(name):
    result = subprocess.run([sys.executable, '-B', '-S', str(ROOT / 'scripts' / (name + '.py')), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--manifest' in result.stdout


def test_manifest_shards_cover_only_the_requested_safe_samples(tmp_path):
    manifest = tmp_path / 'source.jsonl'
    manifest.write_text('\n'.join(json.dumps({'sample_id': str(i)}) for i in range(5)))
    shards = [common.samples(manifest, tmp_path / 'samples', i, 2) for i in range(2)]
    assert [[r['sample_id'] for r, _ in shard] for shard in shards] == [['0', '2', '4'], ['1', '3']]
    with pytest.raises(ValueError, match='shard'):
        common.samples(manifest, tmp_path / 'samples', 2, 2)
    manifest.write_text(json.dumps([{'sample_id': '../outside'}]))
    with pytest.raises(ValueError, match='safe'):
        common.samples(manifest, tmp_path / 'samples')


def test_real_video_extraction_and_verified_reuse(tmp_path):
    pytest.importorskip('cv2')
    ffmpeg = pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    raw = tmp_path / 'raw.mp4'
    subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:size=1280x720:rate=16', '-frames:v', '90',
                    '-c:v', 'libx264', '-threads', '1', '-preset', 'ultrafast', str(raw)], check=True)
    manifest = tmp_path / 'sources.json'
    row = {'sample_id': '00000000', 'source_path': 'raw.mp4', 'source_start_seconds': 0.125}
    manifest.write_text(json.dumps([row]))
    args = SimpleNamespace(manifest=manifest, sample_root=tmp_path / 'samples', seed=3407,
                           shard_id=0, num_shards=1, skip_existing=False)
    assert clips.prepare(args)[0]['status'] == 'written'
    output = args.sample_root / '00000000/clip.mp4'
    metadata = json.loads(output.with_name('clip_metadata.json').read_text())
    assert metadata['source_start_seconds'] == 0.125
    assert metadata['clip_sha256'] == common.validate_clip(output)
    args.skip_existing = True
    assert clips.prepare(args)[0]['status'] == 'reused'
    row['source_start_seconds'] = 0.25
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match='different source settings'):
        clips.prepare(args)
    row['source_start_seconds'] = 0.125
    manifest.write_text(json.dumps([row]))
    with output.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='clip changed'):
        clips.prepare(args)


def write_geometry(tmp_path, raw, sample):
    np = pytest.importorskip('numpy')
    exr = pytest.importorskip('OpenEXR')
    imath = pytest.importorskip('Imath')
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 81, axis=0)
    intrinsics = np.repeat(np.array([[640, 640, 640, 352]], dtype=np.float32), 81, axis=0)
    for name, values in [('pose', poses), ('intrinsics', intrinsics)]:
        (raw / name).mkdir(parents=True, exist_ok=True)
        np.savez(raw / name / (sample + '.npz'), data=values, inds=np.arange(81))
    header = exr.Header(1280, 704)
    header['channels'] = {'Z': imath.Channel(imath.PixelType(imath.PixelType.HALF))}
    path = tmp_path / 'depth.exr'
    writer = exr.OutputFile(str(path), header)
    writer.writePixels({'Z': np.full((704, 1280), 2, dtype=np.float16).tobytes()})
    writer.close()
    (raw / 'depth').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / 'depth' / (sample + '.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for i in range(81):
            archive.writestr(f'{i:05d}.exr', path.read_bytes())


def test_vipe_command_export_and_clip_binding(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy')
    pytest.importorskip('OpenEXR')
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps([{'sample_id': '00000000'}]))
    folder = tmp_path / 'samples/00000000'
    folder.mkdir(parents=True)
    (folder / 'clip.mp4').write_bytes(b'source clip')
    vipe = tmp_path / 'vipe'
    vipe.mkdir()
    (vipe / 'run.py').write_text('# external ViPE entrypoint')
    args = SimpleNamespace(manifest=manifest, sample_root=folder.parent, vipe_root=vipe,
                           vipe_python=Path(sys.executable), output_root=tmp_path / 'outputs',
                           shard_id=0, num_shards=1, skip_existing=False)
    monkeypatch.setattr(geometry, 'validate_clip', lambda path, expected: common.sha256(path))
    calls = []
    def fake_vipe(command, cwd, check):
        calls.append(command)
        assert cwd == vipe and check
        assert 'streams.frame_end=81' in command and 'pipeline=dav3' in command
        inputs = Path(next(x.split('=', 1)[1] for x in command if x.startswith('streams.base_path=')))
        assert (inputs / '00000000.mp4').read_bytes() == b'source clip'
        raw = Path(next(x.split('=', 1)[1] for x in command if x.startswith('pipeline.output.path=')))
        write_geometry(tmp_path, raw, '00000000')
    monkeypatch.setattr(geometry.subprocess, 'run', fake_vipe)
    assert geometry.prepare(args)['written'] == ['00000000']
    assert np.load(folder / 'pose.npz')['data'].shape == (81, 4, 4)
    common.validate_geometry({name: folder / item[2] for name, item in geometry.ASSETS.items()})
    args.skip_existing = True
    assert geometry.prepare(args)['reused'] == ['00000000']
    assert len(calls) == 1
    (folder / 'clip.mp4').write_bytes(b'changed clip')
    with pytest.raises(ValueError, match='no longer matches'):
        geometry.prepare(args)


def test_generated_masks_match_lmdb_exclusion_convention(tmp_path, monkeypatch):
    np = pytest.importorskip('numpy')
    cv2 = pytest.importorskip('cv2')
    pytest.importorskip('torch')
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps([{'sample_id': '00000000'}]))
    folder = tmp_path / 'samples/00000000'
    folder.mkdir(parents=True)
    (folder / 'clip.mp4').write_bytes(b'source clip')
    model, checkpoint = tmp_path / 'qwen', tmp_path / 'sam3.pt'
    model.mkdir()
    checkpoint.write_bytes(b'checkpoint')
    args = SimpleNamespace(manifest=manifest, sample_root=folder.parent, qwen_model=model,
                           sam_checkpoint=checkpoint, shard_id=0, num_shards=1, skip_existing=False)
    monkeypatch.setattr(masks, 'validate_clip', lambda path, expected: common.sha256(path))
    masks.load_runtime()
    monkeypatch.setattr(masks, 'EntityExtractor', lambda path: SimpleNamespace(extract=lambda frames: ([], 'Nothing')))
    monkeypatch.setattr(masks, 'sample_frames', lambda path: ([], 81, 704, 1280))
    keep_out = np.zeros((1, 704, 1280), dtype=bool)
    keep_out[:, 0:16, 0:16] = True
    class Predictor:
        def handle_request(self, request):
            if request['type'] == 'start_session':
                return {'session_id': 'session'}
            if request['type'] == 'add_prompt':
                assert request['text'] == 'sky'
                return {'outputs': {'out_binary_masks': keep_out}}
            return {}
        def handle_stream_request(self, request):
            for i in range(81):
                yield {'frame_index': i, 'outputs': {'out_binary_masks': keep_out}}
    import types
    package, module = types.ModuleType('sam3'), types.ModuleType('sam3.model_builder')
    module.build_sam3_video_predictor = lambda **kwargs: Predictor()
    monkeypatch.setitem(sys.modules, 'sam3', package)
    monkeypatch.setitem(sys.modules, 'sam3.model_builder', module)
    assert masks.prepare(args)['written'] == ['00000000']
    spec = importlib.util.spec_from_file_location('preprocess_builder', ROOT / 'scripts/build_lmdb.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.np, builder.cv2 = np, cv2
    static = builder._load_anchor_static_masks(folder / 'mask.zip')
    assert static.shape == (21, 44, 80)
    assert not static[:, 0, 0].any() and static[:, 1:, :].all()
    args.skip_existing = True
    assert masks.prepare(args)['reused'] == ['00000000']
    with (folder / 'mask.zip').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='no longer match'):
        masks.prepare(args)
