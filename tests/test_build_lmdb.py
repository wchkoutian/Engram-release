import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('release_lmdb_builder', 'scripts/build_lmdb.py')


@pytest.mark.parametrize('format', ['list', 'object', 'jsonl', 'single_jsonl'])
def test_scene_manifest_formats_keep_source_order(tmp_path, format):
    rows = [{'sample_id': '00000002'}, {'sample_id': '00000001'}]
    if format == 'single_jsonl':
        rows = rows[:1]
    text = (json.dumps({'scenes': rows}) if format == 'object' else
            '\n'.join(json.dumps(row) for row in rows) if format in ('jsonl', 'single_jsonl') else json.dumps(rows))
    path = tmp_path / 'scenes.json'
    path.write_text(text)
    assert builder._read_manifest(path) == rows


@pytest.mark.parametrize('rows', [[], [{'sample_id': '../outside'}],
                                  [{'sample_id': 'x'}, {'sample_id': 'x'}],
                                  [{'sample_id': 123}]])
def test_invalid_scene_manifest_rejected(tmp_path, rows):
    path = tmp_path / 'scenes.json'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        builder._read_manifest(path)


def test_dataset_identity_does_not_depend_on_machine_paths(tmp_path):
    rows = [{'sample_id': '00000001'}]
    one, two = tmp_path / 'one.json', tmp_path / 'two.json'
    for path in (one, two):
        path.write_text(json.dumps(rows))
    assert builder._dataset_id(one, rows, {'vae': 'same'}) == builder._dataset_id(two, rows, {'vae': 'same'})


def test_standalone_help_is_available_without_model_imports():
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/build_lmdb.py'), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '{encode,merge,validate}' in result.stdout


@pytest.mark.parametrize('action,command', [('build-lmdb', 'encode'), ('merge-lmdb', 'merge'),
                                          ('validate-lmdb', 'validate')])
def test_lmdb_entrypoint_uses_configured_inputs_without_checkpoints(tmp_path, monkeypatch, action, command):
    entry = load('lmdb_entry', 'run.py')
    values = {}
    for key in entry.REQUIRED[action]:
        path = tmp_path / key
        if key in entry.FILE_KEYS:
            path.write_text('{}')
        else:
            path.mkdir()
        values[key] = str(path)
    assets = tmp_path / 'assets.json'
    assets.write_text(json.dumps(values))
    monkeypatch.setattr(sys, 'argv', ['run.py', action, '--assets', str(assets), '--root', str(tmp_path / 'out')])
    calls = []
    monkeypatch.setattr(entry, 'configure', lambda configured: calls.append(configured))
    monkeypatch.setattr(entry.runpy, 'run_path', lambda path, run_name: calls.append(sys.argv.copy()))
    entry.main()
    assert calls[0] == values
    assert Path(calls[1][0]).name == 'build_lmdb.py'
    assert calls[1][1] == command
    assert not any('checkpoint' in key for key in values)


def test_encode_merge_validate_with_real_lmdb_geometry_and_statistics(tmp_path, monkeypatch):
    """Use real EXR/NPZ/mask inputs and LMDB; substitute the video decoder and neural VAE."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('lmdb')
    OpenEXR = pytest.importorskip('OpenEXR')
    Imath = pytest.importorskip('Imath')
    torch.set_num_threads(2)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    builder.load_runtime()
    from engram.data.realestate10k import PrecomputedWindowDataset
    from engram.models.wan_latents import load_wan_latent_stats

    # A half-precision Z channel matches the prepared ViPE depth archive format.
    exr_path = tmp_path / 'depth.exr'
    header = OpenEXR.Header(1280, 704)
    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    writer = OpenEXR.OutputFile(str(exr_path), header)
    writer.writePixels({'Z': np.full((704, 1280), 2, dtype=np.float16).tobytes()})
    writer.close()
    assert builder._decode_exr(exr_path.read_bytes()).dtype == np.float32
    mask = np.zeros((704, 1280), dtype=np.uint8)
    mask[16:32, 32:48] = 1
    success, mask_png = builder.cv2.imencode('.png', mask)
    assert success
    samples = tmp_path / 'samples'
    rows = []
    for index in (2, 1):
        sample = f'{index:08d}'
        folder = samples / sample
        folder.mkdir(parents=True)
        (folder / 'clip.mp4').write_bytes(bytes([index]))
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 81, axis=0)
        intrinsics = np.repeat(np.array([[640, 640, 640, 352]], dtype=np.float32), 81, axis=0)
        np.savez(folder / 'pose.npz', data=poses, inds=np.arange(81))
        np.savez(folder / 'intrinsics.npz', data=intrinsics, inds=np.arange(81))
        for name, extension, payload in (('depth', 'exr', exr_path.read_bytes()), ('mask', 'png', mask_png.tobytes())):
            with zipfile.ZipFile(folder / f'{name}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for frame in builder.ANCHOR_INDICES:
                    archive.writestr(f'{frame:05d}.{extension}', payload)
        rows.append({'sample_id': sample, 'clip_sha256': hashlib.sha256(bytes([index])).hexdigest()})
    manifest = tmp_path / 'scenes.json'
    manifest.write_text(json.dumps({'scenes': rows}))

    class VAE:
        config = types.SimpleNamespace(z_dim=48)
        def encode(self, frames):
            size = 48 * 21 * 44 * 80
            latent = (torch.arange(size).float().remainder(97) / 97).reshape(1, 48, 21, 44, 80)
            latent += frames[0, 0, 0, 0, 0]
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: latent))

    monkeypatch.setattr(builder, '_load_vae', lambda *args: VAE())
    monkeypatch.setattr(builder, '_load_video', lambda path: [np.full((1, 1, 3), path.read_bytes()[0], np.uint8)] * 81)
    output = tmp_path / 'output'
    args = types.SimpleNamespace(manifest=manifest, sample_root=samples, output_dir=output,
                                 model_id='test-vae', device='cpu', num_shards=2, shard_id=0,
                                 max_samples=0, map_size_gib=1, audit_scenes=2)
    builder.encode(args)
    args.shard_id = 1
    builder.encode(args)
    with pytest.raises(FileExistsError):
        builder.encode(args)
    builder.merge(args)
    report = builder.validate(args)
    assert report['scenes'] == 2 and report['source_scenes_checked'] == 2
    with pytest.raises(FileExistsError):
        builder.merge(args)

    statistics = load_wan_latent_stats(str(output / builder.STATS_NAME))
    store = builder.PrecomputedStore(str(output / builder.STORE_NAME))
    try:
        assert store.get_meta() == [('00000002', 21), ('00000001', 21)]
        assert store.env.stat()['entries'] == 46
        all_latents = []
        for row in rows:
            latents = []
            for anchor in range(21):
                record = store.get(f"{row['sample_id']}:{anchor}")
                assert record['source_rgb_index'] == anchor * 4
                assert record['caption'] == ''
                assert record['depth'].shape == (44, 80)
                assert not record['static_mask'][1, 2]
                assert np.allclose(record['K_l'], [[40, 0, 40], [0, 40, 22], [0, 0, 1]])
                assert np.allclose(record['E'], np.eye(4))
                latents.append(torch.from_numpy(record['latent']))
            all_latents.append(torch.stack(latents, dim=1))
        flat = torch.cat(all_latents, dim=1).permute(1, 2, 3, 0).reshape(-1, 48).double()
        assert torch.allclose(statistics['mean'].double(), flat.mean(0), atol=1e-7, rtol=1e-6)
        assert torch.allclose(statistics['var'].double(), flat.var(0, unbiased=False), atol=1e-7, rtol=1e-6)
        assert statistics['count'] == flat.shape[0]
    finally:
        store.close()
    dataset = PrecomputedWindowDataset(str(output / builder.STORE_NAME), window=9)
    try:
        item = dataset[0]
        assert tuple(item['latent'].shape) == (9, 48, 44, 80)
        assert tuple(item['E'].shape) == (9, 4, 4)
    finally:
        dataset.store.close()

    # The validator rejects a structurally valid but mismatched statistics file.
    stats = torch.load(output / builder.STATS_NAME, weights_only=False)
    stats['dataset_id'] = 'wrong-dataset'
    torch.save(stats, output / builder.STATS_NAME)
    with pytest.raises(ValueError, match='matching pair'):
        builder.validate(args)
    stats['dataset_id'] = report['dataset_id']
    torch.save(stats, output / builder.STATS_NAME)

    # Source geometry drift is detected against records already in the LMDB.
    changed = np.repeat(np.eye(4, dtype=np.float32)[None], 81, axis=0)
    changed[:, 0, 3] = 1
    np.savez(samples / rows[0]['sample_id'] / 'pose.npz', data=changed, inds=np.arange(81))
    with pytest.raises(ValueError, match='source geometry'):
        builder.validate(args)
