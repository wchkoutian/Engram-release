import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contracts = load('preparation_contracts', 'source/engram/data/contracts.py')


def test_fixed_training_selection_has_256_distinct_source_videos():
    selection = contracts.load_training_selection(ROOT / 'configs/re10k_train256_scenes.json')
    assert len(selection['scenes']) == 256
    assert len(selection['source_manifest_sha256']) == 64
    assert selection['scenes'][0]['sample_id'] == '00036138'


@pytest.mark.parametrize('change', ['duplicate_parent', 'unsafe_sample'])
def test_invalid_training_selection_rejected(tmp_path, change):
    selection = json.loads((ROOT / 'configs/re10k_train256_scenes.json').read_text())
    if change == 'duplicate_parent':
        selection['scenes'][1]['parent_video_id'] = selection['scenes'][0]['parent_video_id']
    else:
        selection['scenes'][0]['sample_id'] = '../escape'
    path = tmp_path / 'scenes.json'
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        contracts.load_training_selection(path)


def test_forward_cache_is_distinct_from_training_cache():
    scene = {'sample_id': '00036138', 'data_contract': contracts.forward_data_contract(),
             'windows': [{'sample_id': '00036138', 'window': i, 'start': start,
                          'direction': 1, 'history_count': start // 4}
                         for i, start in enumerate(contracts.WINDOW_STARTS)]}
    contracts.validate_forward_scene_cache(scene)
    with pytest.raises(ValueError):
        contracts.validate_scene_cache(scene)
    scene['data_contract'] = contracts.data_contract()
    with pytest.raises(ValueError):
        contracts.validate_scene_cache(scene)


@pytest.mark.parametrize('action', ['prepare-training', 'prepare-training-return'])
def test_preparation_cli_configures_assets_before_running_script(tmp_path, monkeypatch, action):
    entry = load('preparation_entry', 'run.py')
    source, output = tmp_path / 'forward', tmp_path / 'out'
    required = {}
    for key in entry.REQUIRED[action]:
        path = tmp_path / key
        if key in entry.FILE_KEYS:
            path.write_text('{}')
        else:
            path.mkdir()
        required[key] = str(path)
    assets = tmp_path / 'assets.json'
    assets.write_text(json.dumps(required))
    argv = ['run.py', action, '--assets', str(assets), '--root', str(output)]
    if action.endswith('return'):
        argv += ['--source', str(source)]
    monkeypatch.setattr(sys, 'argv', argv)
    calls = []

    def dispatch(path, run_name):
        import os
        assert run_name == '__main__'
        assert os.environ['ENGRAM_TRAINING_SAMPLES'] == required['training_samples']
        assert str(ROOT / 'scripts') in sys.path
        calls.append((Path(path).name, sys.argv.copy()))

    monkeypatch.setattr(entry.runpy, 'run_path', dispatch)
    # Restore environment and search path after configure() is exercised.
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    for key in required:
        env_key = 'TORCH_HOME' if key == 'torch_home' else 'ENGRAM_' + key.upper()
        monkeypatch.setenv(env_key, '')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'DIFFUSERS_OFFLINE',
                 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        monkeypatch.setenv(name, '')
    entry.main()
    assert len(calls) == 1
    assert calls[0][0] == ('prepare_return.py' if action.endswith('return') else 'prepare.py')
    assert ('--source' in calls[0][1]) == action.endswith('return')


def test_forward_to_return_cache_with_real_tensor_serialization(tmp_path, monkeypatch):
    """Exercise both cache builders; replace GPU geometry/VAE calls with deterministic CPU doubles."""
    torch = pytest.importorskip('torch')
    import pickle

    torch.set_num_threads(2)
    monkeypatch.syspath_prepend(str(ROOT / 'source'))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    samples = tmp_path / 'samples'
    sample = '00036138'
    sample_dir = samples / sample
    sample_dir.mkdir(parents=True)
    (sample_dir / 'clip.mp4').write_bytes(b'unit-test-source-video')
    np.savez(sample_dir / 'pose.npz', inds=np.arange(81),
             data=np.repeat(np.eye(4, dtype=np.float32)[None], 81, axis=0))
    monkeypatch.setenv('ENGRAM_TRAINING_SAMPLES', str(samples))
    records = {}

    class Store:
        def begin(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def get(self, key):
            return records.get(key)
        def close(self):
            pass

    store = Store()
    monkeypatch.setitem(sys.modules, 'lmdb', types.SimpleNamespace(open=lambda *a, **k: store))
    prepare = importlib.import_module('prepare')
    returns = importlib.import_module('prepare_return')
    rules = importlib.import_module('engram.data.contracts')
    latent_api = importlib.import_module('engram.models.wan_latents')
    # One scene keeps this integration fixture small; the shipped 256-scene list is tested above.
    for module in (rules, prepare, returns):
        monkeypatch.setattr(module, 'SCENE_COUNT', 1)
    monkeypatch.setattr(rules, 'WINDOW_COUNT', 9)
    monkeypatch.setattr(returns, 'WINDOW_COUNT', 9)
    latent_contract = {'name': 'test-wan-latents'}
    records[b'__latent_contract__'] = pickle.dumps(latent_contract)
    for anchor in range(21):
        records[f'{sample}:{anchor}'.encode()] = pickle.dumps({
            'latent': np.full((48, 44, 80), anchor * 4, dtype=np.float32),
            'source_rgb_index': anchor * 4, 'K_l': np.eye(3, dtype=np.float32),
            'E': np.eye(4, dtype=np.float32), 'depth': np.ones((44, 80), dtype=np.float32)})
    images = np.repeat(np.arange(81, dtype=np.uint8).reshape(81, 1, 1, 1), 3, axis=3)

    class Frames:
        def __init__(self, rgb):
            self.rgb = rgb
        def to(self, *args, **kwargs):
            return self

    def encode(vae, frames):
        values = torch.as_tensor(frames.rgb[::4, 0, 0, 0].copy()).float()
        return values.reshape(1, 1, -1, 1, 1).expand(1, 48, -1, 44, 80).clone()

    class VAE:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def to(self, *args, **kwargs):
            return self
        def eval(self):
            return self
        def enable_tiling(self):
            pass
        def enable_slicing(self):
            pass

    class Geometry:
        def depth_for_views(self, rgb, *args):
            return [{'depth': torch.ones(44, 80)} for _ in rgb]

    class Memory:
        def __init__(self, *args, **kwargs):
            self.frames = []
        def fingerprint(self, *args):
            return 'unchanged-real-memory'
        def update(self, feature, depth, camera, pose, time_index, origin):
            assert float(feature[0, 0, 0]) == time_index
            self.frames.append(time_index)

    def condition(poses, intrinsics, height, width, memory):
        value = torch.zeros(1, 108, len(poses), height, width)
        value[:, 0] = len(memory.frames)
        value[:, (49, 100)] = 1
        return value

    monkeypatch.setitem(sys.modules, 'diffusers', types.SimpleNamespace(AutoencoderKLWan=VAE))
    monkeypatch.setattr(latent_api, 'wan_latent_contract', lambda vae: latent_contract)
    for module in (prepare, returns):
        monkeypatch.setattr(module, 'load_rgb', lambda path: images)
        monkeypatch.setattr(module, 'rgb_tensor', Frames)
        monkeypatch.setattr(module, 'encode_wan_latents', encode)
        monkeypatch.setattr(module, 'ProtectedMemory', Memory)
        monkeypatch.setattr(module, 'conditioning', condition)
        monkeypatch.setattr(module, 'setup_cuda', lambda fraction: None)
        monkeypatch.setattr(module, 'build_depth_estimator', Geometry)
    monkeypatch.setattr(prepare, 'wan_latent_contract', lambda vae: latent_contract)
    statistics, text_bundle = tmp_path / 'statistics.pt', tmp_path / 'text.pt'
    statistics.write_bytes(b'unchanged-statistics')
    torch.save({'text': torch.zeros(1, 2, 3)}, text_bundle)
    for module in (prepare, returns):
        monkeypatch.setattr(module, 'LATENT_STATS', str(statistics))
    scene = {'sample_id': sample, 'clip_id': 'e9f7e73140682cec', 'parent_video_id': 'video1',
             'motion_group': 'other', 'motion': {}, 'clip_sha256': prepare.digest(sample_dir / 'clip.mp4')}
    selection_path = tmp_path / 'selection.json'
    selection_path.write_text(json.dumps({'scene_count': 1, 'scenes': [scene],
                                          'source_manifest_sha256': 'a' * 64}))
    monkeypatch.setenv('ENGRAM_TRAINING_SCENES', str(selection_path))
    monkeypatch.setenv('ENGRAM_TRAINING_LMDB', str(tmp_path / 'lmdb'))
    monkeypatch.setenv('ENGRAM_TEXT_BUNDLE', str(text_bundle))
    forward_root, final_root = tmp_path / 'forward', tmp_path / 'final'
    monkeypatch.setattr(sys, 'argv', ['prepare.py', '--root', str(forward_root)])
    prepare.main()
    prepare.main()  # A completed cache can be checked again without regenerating it.
    monkeypatch.setattr(sys, 'argv', ['prepare_return.py', '--source', str(forward_root), '--root', str(final_root)])
    returns.main()
    returns.main()
    manifest = json.loads((final_root / 'manifest.json').read_text())
    rules.validate_training_manifest(manifest)
    assert manifest['window_count'] == 9
    assert [r['direction'] for r in manifest['records']] == [1] * 5 + [-1] * 4
    for record in manifest['records']:
        item = torch.load(final_root / record['file'], map_location='cpu', weights_only=False)
        rules.validate_prepared_window(item, record)
        assert float(item['condition'][0, 0, 0, 0, 0]) == record['history_count']
        expected = [min(record['start'] + offset, 80) if record['direction'] == 1
                    else record['start'] - offset for offset in range(0, 33, 4)]
        assert item['latent'][0, 0, :, 0, 0].tolist() == expected
    for old, new in zip(json.loads((forward_root / 'manifest.json').read_text())['records'], manifest['records']):
        a = torch.load(forward_root / old['file'], map_location='cpu', weights_only=False)
        b = torch.load(final_root / new['file'], map_location='cpu', weights_only=False)
        assert torch.equal(a['latent'], b['latent'])
        assert torch.equal(a['condition'], b['condition'])
    captions = {r['file']: torch.zeros(1, 2, 3) for r in manifest['records']}
    (final_root / 'captions.json').write_text(json.dumps({key: 'A room.' for key in captions}))
    torch.save({'captions': captions, 'empty': torch.zeros(1, 2, 3)}, final_root / 'caption_text.pt')
    linker = importlib.import_module('link_training_data')
    stage = tmp_path / 'stage1'
    linker.link(final_root, stage)
    first = manifest['records'][0]
    assert prepare.digest(stage / first['file']) == first['sha256']
    damaged = final_root / manifest['records'][-1]['file']
    with damaged.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(RuntimeError, match='Prepared window changed'):
        returns.main()
