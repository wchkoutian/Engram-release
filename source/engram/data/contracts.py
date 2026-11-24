SCENE_COUNT = 256  # V14 (was 32)
WINDOW_STARTS = (0, 16, 32, 48, 64)
RETURN_ENDS = (32, 48, 64, 80)
WINDOWS_PER_SCENE = len(WINDOW_STARTS) + len(RETURN_ENDS)
WINDOW_COUNT = SCENE_COUNT * WINDOWS_PER_SCENE


def data_contract():
    return {
        'identifier': 'v12_a3_forward_plus_return_v1',
        'source_frames': 81,
        'window_frames': 33,
        'stride': 16,
        'window_starts': list(WINDOW_STARTS),
        'reverse_windows': True,
        'return_ends': list(RETURN_ENDS),
        'history': 'past_original_timeline; return windows read the outbound history up to the turning frame',
        'padding': 'repeat_last_frame_mask_invalid_targets',
    }


def require_data_contract(metadata):
    if metadata.get('data_contract') != data_contract():
        raise ValueError('V12-A3 forward+return data contract required; rebuild other caches in a new root')


def return_indices(end):
    return [end - offset for offset in range(33)]


def validate_scene_cache(scene):
    require_data_contract(scene)
    records = scene['windows']
    if len(records) != WINDOWS_PER_SCENE:
        raise ValueError('Expected five forward and four return windows per scene')
    expected = [(start, 1) for start in WINDOW_STARTS] + [(end, -1) for end in RETURN_ENDS]
    for number, (record, (start, direction)) in enumerate(zip(records, expected)):
        if (record.get('sample_id'), record.get('window'), record.get('start'), record.get('direction'), record.get('history_count')) != (scene['sample_id'], number, start, direction, start // 4):
            raise ValueError('Invalid training window record')


def validate_training_manifest(manifest):
    require_data_contract(manifest)
    scenes = manifest['scenes']
    records = manifest['records']
    if manifest.get('scene_count') != SCENE_COUNT or len(scenes) != SCENE_COUNT:
        raise ValueError(f'Expected {SCENE_COUNT} training scenes')
    if manifest.get('window_count') != WINDOW_COUNT or len(records) != WINDOW_COUNT:
        raise ValueError(f'Expected {WINDOW_COUNT} forward+return training windows')
    for key in ('sample_id', 'clip_id', 'parent_video_id'):
        if len({scene[key] for scene in scenes}) != SCENE_COUNT:
            raise ValueError(f'Duplicate training scene {key}')
    for scene in scenes:
        validate_scene_cache(scene)
    if records != [record for scene in scenes for record in scene['windows']]:
        raise ValueError('Training records differ from scene windows')


def validate_training_window(item, record):
    require_data_contract(item)
    start, direction = record['start'], record['direction']
    if item.get('start') != start or item.get('direction') != direction or item.get('sample_id') != record['sample_id']:
        raise ValueError('Training tensor does not match its window record')
    if direction == 1:
        if start not in WINDOW_STARTS or item.get('indices') != [min(start + offset, 80) for offset in range(33)]:
            raise ValueError('Training frame indices are not forward-only')
        expected_valid = [offset > 0 and start + offset <= 80 for offset in range(0, 33, 4)]
    elif direction == -1:
        if start not in RETURN_ENDS or item.get('indices') != return_indices(start):
            raise ValueError('Return window indices must walk back 32 frames from the turning frame')
        expected_valid = [offset > 0 for offset in range(0, 33, 4)]
    else:
        raise ValueError('Unknown window direction')
    if item.get('valid') != expected_valid or item['valid_mask'].tolist() != expected_valid:
        raise ValueError('Invalid padding loss mask')
    history = [(observation['trajectory_time'], observation['rgb_frame']) for observation in item['history']]
    if item.get('prefix_trajectory_time') != start or history != [(frame, frame) for frame in range(4, start + 1, 4)]:
        raise ValueError('Training history must be the real outbound past up to the window start/turning frame')


def validate_checkpoint_config(config, manifest_sha256):
    require_data_contract(config)
    if config.get('manifest_sha256') != manifest_sha256:
        raise ValueError('Checkpoint training manifest mismatch')
    if config.get('initialization') != 'direct_wan_copy_no_engram_checkpoint' or config.get('engram_checkpoint_loaded') is not False:
        raise ValueError('Expected fresh Wan-copy training without an Engram checkpoint')
    if config.get('scene_count') != SCENE_COUNT or config.get('window_count') != WINDOW_COUNT:
        raise ValueError('Checkpoint is not from the V12-A3 training protocol')


def forward_data_contract():
    return {**data_contract(), 'identifier': 'engram_forward_training_v1',
            'reverse_windows': False, 'return_ends': [],
            'history': 'past_original_timeline'}


def validate_forward_scene_cache(scene):
    if scene.get('data_contract') != forward_data_contract():
        raise ValueError('Expected a forward training cache')
    records = scene['windows']
    if len(records) != len(WINDOW_STARTS):
        raise ValueError('Expected five forward windows per scene')
    for number, (record, start) in enumerate(zip(records, WINDOW_STARTS)):
        if (record.get('sample_id'), record.get('window'), record.get('start'),
                record.get('direction'), record.get('history_count')) != (
                scene['sample_id'], number, start, 1, start // 4):
            raise ValueError('Invalid forward training window record')


def validate_forward_manifest(manifest):
    if manifest.get('data_contract') != forward_data_contract():
        raise ValueError('Expected a forward training manifest')
    scenes, records = manifest['scenes'], manifest['records']
    if manifest.get('scene_count') != SCENE_COUNT or len(scenes) != SCENE_COUNT:
        raise ValueError(f'Expected {SCENE_COUNT} forward training scenes')
    count = SCENE_COUNT * len(WINDOW_STARTS)
    if manifest.get('window_count') != count or len(records) != count:
        raise ValueError(f'Expected {count} forward training windows')
    for key in ('sample_id', 'clip_id', 'parent_video_id'):
        if len({scene[key] for scene in scenes}) != SCENE_COUNT:
            raise ValueError(f'Duplicate training scene {key}')
    for scene in scenes:
        validate_forward_scene_cache(scene)
    if records != [record for scene in scenes for record in scene['windows']]:
        raise ValueError('Forward records differ from scene windows')


def validate_prepared_window(item, record, *, forward=False):
    import torch

    expected = forward_data_contract() if forward else data_contract()
    if item.get('data_contract') != expected:
        raise ValueError('Prepared window has the wrong data contract')
    validate_training_window({**item, 'data_contract': data_contract()}, record)
    for key, shape in (('latent', (1, 48, 9, 44, 80)),
                       ('condition', (1, 108, 9, 44, 80))):
        tensor = item[key]
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid prepared {key}: expected finite tensor {shape}')


def load_training_selection(path):
    import json
    from pathlib import Path
    import re

    selection = json.loads(Path(path).read_text(encoding='utf-8'))
    scenes = selection['scenes']
    if selection.get('scene_count') != SCENE_COUNT or len(scenes) != SCENE_COUNT:
        raise ValueError(f'Expected a fixed selection of {SCENE_COUNT} training scenes')
    for scene in scenes:
        if not re.fullmatch(r'[0-9]{8}', scene.get('sample_id', '')):
            raise ValueError('Training sample_id must be eight digits')
        if not re.fullmatch(r'[0-9a-f]{16}', scene.get('clip_id', '')):
            raise ValueError('Invalid training clip_id')
        if not re.fullmatch(r'[0-9a-f]{64}', scene.get('clip_sha256', '')):
            raise ValueError('Missing source video SHA256')
        if not scene.get('parent_video_id'):
            raise ValueError('Missing parent video identifier')
    for key in ('sample_id', 'clip_id', 'parent_video_id'):
        if len({scene[key] for scene in scenes}) != SCENE_COUNT:
            raise ValueError(f'Duplicate training scene {key}')
    return selection


