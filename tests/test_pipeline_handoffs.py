import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load('handoff_entry', 'run.py')


@pytest.mark.parametrize('action', ['build-lmdb', 'prepare-training', 'train-stage1', 'train-stage2', 'caption-training'])
def test_data_and_training_actions_ignore_unrelated_checkpoint_placeholder(tmp_path, action):
    values = json.loads((ROOT / 'configs/assets.example.json').read_text())
    for key in entry.REQUIRED[action]:
        path = tmp_path / key
        if key in entry.FILE_KEYS:
            path.write_text('{}')
        else:
            path.mkdir()
        values[key] = str(path)
    config = tmp_path / 'assets.local.json'
    config.write_text(json.dumps(values))
    assets = entry.load_assets(config, action)
    entry.check_assets(assets, action)


def test_inference_still_requires_valid_checkpoint_sha256(tmp_path):
    config = tmp_path / 'assets.local.json'
    config.write_text((ROOT / 'configs/assets.example.json').read_text())
    with pytest.raises(ValueError, match='checkpoint SHA256'):
        entry.load_assets(config, 'infer-re10k')


def test_example_scene_list_resolves_after_copying_config_to_repo_root(tmp_path):
    config = tmp_path / 'assets.local.json'
    config.write_text((ROOT / 'configs/assets.example.json').read_text())
    assets = entry.load_assets(config, 'build-lmdb')
    assert Path(assets['training_scenes']) == tmp_path / 'configs/re10k_train256_scenes.json'
    assert Path(assets['checkpoint']) == tmp_path / 'checkpoints/ENGRAM/stage2_step_8000.pt'


def test_training_manifest_and_two_stage_weights_are_accepted_by_inference(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(ROOT / 'source'))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.syspath_prepend(str(ROOT / 'inference'))
    from engram.data import contracts
    from engram.core import recipe_contract
    re10k = load('handoff_re10k', 'inference/re10k.py')
    scenes = []
    windows = [(start, 1) for start in contracts.WINDOW_STARTS] + [(end, -1) for end in contracts.RETURN_ENDS]
    for i in range(contracts.SCENE_COUNT):
        sample = f'{i:08d}'
        records = [{'sample_id': sample, 'window': number, 'start': start,
                    'direction': direction, 'history_count': start // 4}
                   for number, (start, direction) in enumerate(windows)]
        scenes.append({'sample_id': sample, 'clip_id': f'training-clip-{i}',
                       'parent_video_id': f'training-parent-{i}',
                       'data_contract': contracts.data_contract(), 'windows': records})
    manifest = {'data_contract': contracts.data_contract(), 'scene_count': contracts.SCENE_COUNT,
                'window_count': contracts.WINDOW_COUNT, 'scenes': scenes,
                'records': [r for scene in scenes for r in scene['windows']]}
    training_root = tmp_path / 'training'
    training_root.mkdir()
    manifest_path = training_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    stage1 = tmp_path / 'stage1.pt'
    torch.save({'side_branch': {'weight': torch.ones(1)}}, stage1)
    checkpoint = tmp_path / 'stage2.pt'
    torch.save({'step': 8000, 'side_branch': {'weight': torch.ones(1)},
                'backbone_blocks': {'weight': torch.full((1,), 2)},
                'config': {'version': 'V22-stage2-bbfull', 'steps': 8000,
                           'data_contract': contracts.data_contract(), 'recipe': recipe_contract(),
                           'manifest_sha256': sha(manifest_path), 'stage1_sha256': sha(stage1),
                           'lora': False}}, checkpoint)
    assets = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
              'stage1_checkpoint': str(stage1), 'training_root': str(training_root)}
    state, training_sha = re10k.checkpoint(assets)
    assert training_sha == sha(manifest_path)
    assert state['backbone_blocks']['weight'].item() == 2
    stage1.write_bytes(stage1.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='stage-1 checkpoint changed'):
        re10k.checkpoint(assets)
