import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('override,expected', [(None, 0.4), ('0.65', 0.65)])
def test_cuda_setup_applies_requested_budget_without_allocating(monkeypatch, override, expected):
    common = load('cuda_budget_common', 'source/engram/common.py')
    if override is None:
        monkeypatch.delenv('ENGRAM_MEMORY_FRACTION', raising=False)
    else:
        monkeypatch.setenv('ENGRAM_MEMORY_FRACTION', override)
    calls = []
    monkeypatch.setattr(common.torch.cuda, 'set_per_process_memory_fraction',
                        lambda fraction, device: calls.append((fraction, device)))
    monkeypatch.setattr(common.torch, 'set_num_threads', lambda count: None)
    common.setup_cuda(0.4)
    assert calls == [(expected, 0)]


@pytest.mark.parametrize('value', ['0', '-0.1', '1.01', 'nan'])
def test_invalid_budget_fails_before_cuda_setup(monkeypatch, value):
    common = load('cuda_budget_invalid', 'source/engram/common.py')
    monkeypatch.setenv('ENGRAM_MEMORY_FRACTION', value)
    def unexpected(*args):
        pytest.fail('Invalid budget must not reach the CUDA API')
    monkeypatch.setattr(common.torch.cuda, 'set_per_process_memory_fraction', unexpected)
    with pytest.raises(ValueError, match='ENGRAM_MEMORY_FRACTION'):
        common.setup_cuda(0.4)


def test_cli_budget_reaches_runtime_and_handoff(monkeypatch, tmp_path):
    entry = load('cuda_budget_entry', 'run.py')
    monkeypatch.setenv('ENGRAM_MEMORY_FRACTION', '0.4')
    monkeypatch.setenv('INV_MEM_FRACTION', '0.47')
    assets = tmp_path / 'assets.json'
    assets.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['run.py', 'check-assets', '--assets', str(assets),
                                    '--for-action', 'export-re10k', '--memory-fraction', '0.65'])
    entry.main()
    handoff = load('cuda_budget_handoff', 'inference/inv_handoff.py')
    assert handoff.MEM_FRACTION == 0.65


def test_cli_rejects_invalid_budget_before_loading_assets(monkeypatch):
    entry = load('cuda_budget_invalid_entry', 'run.py')
    monkeypatch.setattr(sys, 'argv', ['run.py', 'check-assets', '--memory-fraction', '1.5'])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
