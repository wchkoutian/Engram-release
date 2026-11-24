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

entry = load('release_entry', 'run.py')
protocol = load('release_protocol', 'inference/re10k_protocol.py')
contracts = load('release_contracts', 'source/engram/data/contracts.py')
inputs = load('release_inputs', 'inference/input_validation.py')

def test_asset_paths_are_relative_to_json_not_cwd(tmp_path, monkeypatch):
    config = tmp_path / 'config' / 'assets.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'wan_model': '../models/wan'}))
    monkeypatch.chdir(ROOT)
    assert entry.load_assets(config)['wan_model'] == str(tmp_path / 'models' / 'wan')

@pytest.mark.parametrize('value', [{'password': 'secret'}, {'checkpoint_sha256': 'TODO'}, {'wan_model': ''}, {'wan_model': 1}])
def test_invalid_configuration_rejected(tmp_path, value):
    config = tmp_path / 'assets.json'
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        entry.load_assets(config)

def test_asset_check_rejects_file_as_model_directory(tmp_path):
    file = tmp_path / 'wan'
    file.touch()
    with pytest.raises(ValueError, match='wan_model'):
        entry.check_assets({'wan_model': str(file)}, 'train-stage1')
    file.unlink()
    file.mkdir()
    entry.check_assets({'wan_model': str(file)}, 'train-stage1')

def test_torchrun_rank_isolation(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,3,5,7')
    entry.isolate_rank_gpu()
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5'

def test_invalid_rank_fails_before_cuda_import(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('LOCAL_RANK', '3')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(ValueError):
        entry.isolate_rank_gpu()

@pytest.mark.parametrize('name,count', [('nvs', 81), ('closed_loop', 153)])
def test_chunk_stitching_covers_every_frame_once(name, count):
    stitched = []
    for start in range(0, count - 1, 32):
        indices, valid_count = protocol.chunk_indices(name, start)
        assert len(indices) == 33
        stitched.extend(indices[:valid_count] if start == 0 else indices[1:valid_count])
    assert stitched == list(range(count))
    assert protocol.score_pairs(name) == ([(i, i) for i in range(1, 81)] if name == 'nvs' else [(152, 0)])

def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        protocol.inside(tmp_path, '../outside.png')

def test_trajectory_requires_closed_loop(tmp_path):
    trajectory = np.repeat(np.eye(4)[None], 153, axis=0)
    path = tmp_path / 'camera.txt'
    path.write_text('\n'.join(repr(x.tolist()) for x in trajectory))
    assert protocol.trajectory(path, 'closed_loop').shape == (153, 4, 4)
    trajectory[-1, 0, 3] = 1
    path.write_text('\n'.join(repr(x.tolist()) for x in trajectory))
    with pytest.raises(ValueError, match='endpoint'):
        protocol.trajectory(path, 'closed_loop')

def training_manifest():
    scenes = []
    windows = [(start, 1) for start in contracts.WINDOW_STARTS] + [(end, -1) for end in contracts.RETURN_ENDS]
    for i in range(256):
        sample = str(i)
        records = [{'sample_id': sample, 'window': n, 'start': start, 'direction': direction, 'history_count': start // 4}
                   for n, (start, direction) in enumerate(windows)]
        scenes.append({'sample_id': sample, 'clip_id': sample, 'parent_video_id': sample,
                       'data_contract': contracts.data_contract(), 'windows': records})
    return {'data_contract': contracts.data_contract(), 'scene_count': 256, 'window_count': 2304,
            'scenes': scenes, 'records': [r for s in scenes for r in s['windows']]}

def test_training_contract_accepts_full_256_scene_cache():
    contracts.validate_training_manifest(training_manifest())

def test_training_contract_rejects_wrong_return_direction():
    m = training_manifest()
    m['scenes'][0]['windows'][5]['direction'] = 1
    with pytest.raises(ValueError):
        contracts.validate_training_manifest(m)

def test_training_contract_rejects_duplicate_scene():
    m = training_manifest()
    m['scenes'][1]['clip_id'] = m['scenes'][0]['clip_id']
    with pytest.raises(ValueError, match='Duplicate'):
        contracts.validate_training_manifest(m)

@pytest.mark.parametrize('shard', ['-1/4', '4/4', '0/0', '1', 'a/2'])
def test_invalid_worldscore_shards(shard):
    with pytest.raises(ValueError):
        inputs.parse_shard(shard)

def test_worldscore_rejects_output_escape_before_writing(tmp_path):
    p = tmp_path / 'input.json'
    p.write_text(json.dumps({'cases': [{'name': '../escape', 'input_directory': '.', 'positive_prompt': 'room'}]}))
    with pytest.raises(ValueError):
        inputs.validate_worldscore_inputs(p)

def test_worldscore_defaults_second_prompt_and_checks_files(tmp_path):
    p = tmp_path / 'input.json'
    row = {'name': 'static_001', 'input_directory': 'case', 'positive_prompt': 'A room.'}
    p.write_text(json.dumps({'cases': [row]}))
    with pytest.raises(FileNotFoundError):
        inputs.validate_worldscore_inputs(p)
    folder = tmp_path / 'case'
    folder.mkdir()
    for name in ['input_image.png', 'camera_data.json', 'image_data.json']:
        (folder / name).touch()
    assert inputs.validate_worldscore_inputs(p)[0]['positive_prompt_2'] == 'A room.'

def test_final_model_loader_loads_both_weight_groups_strictly(monkeypatch):
    loader = load('release_model_loader', 'inference/model_loader.py')
    calls = []
    class Component:
        def __init__(self, name):
            self.name = name
        def load_state_dict(self, state, strict):
            calls.append((self.name, state, strict))
    model = types.SimpleNamespace(side_branch=Component('branch'), transformer=Component('transformer'))
    monkeypatch.setitem(sys.modules, 'engram.core', types.SimpleNamespace(build_backbone=lambda *a: model))
    state = {'config': {'version': 'V22-stage2-bbfull'}, 'side_branch': {'branch': 1}, 'backbone_blocks': {'full': 2}}
    assert loader.load_model(state) is model
    assert calls == [('branch', {'branch': 1}, True), ('transformer', {'full': 2}, True)]

