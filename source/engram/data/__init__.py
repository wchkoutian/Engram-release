"""Data formats and preprocessing utilities, loaded on demand."""
from importlib import import_module


_EXPORTS = {
    'synthetic': ('SyntheticScene', 'SyntheticView'),
    'realestate10k': ('CameraFrame', 'PrecomputedStore', 'PrecomputedWindowDataset',
                     'denorm_intrinsics', 'parse_cameras', 'w2c_to_4x4'),
    'manifest': ('ManifestClip', 'ManifestFrame', 'clip_from_payload',
                 'load_manifest', 'write_manifest_jsonl'),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module_name, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module('.' + module_name, __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
