"""Backbone integration models.

The side branch and conditioning packer are pure torch and always importable. The Wan2.2
wrapper and the DepthAnything3 / SAM3 / Qwen3-VL wrappers import heavy third-party
packages, so they are exposed lazily and only imported when explicitly requested.
"""
from .side_branch import ControlNetSideBranch
from .conditioning import readout_channels, pack_readout, pack_readout_sequence

__all__ = [
    "ControlNetSideBranch",
    "readout_channels",
    "pack_readout",
    "pack_readout_sequence",
]


def __getattr__(name):
    if name in {"WanEngramBackbone", "load_wan_backbone"}:
        from . import wan_wrapper
        return getattr(wan_wrapper, name)
    if name in {"DepthAnything3Estimator"}:
        from . import depth_anything3
        return getattr(depth_anything3, name)
    if name in {"MapAnythingEstimator"}:
        from . import mapanything
        return getattr(mapanything, name)
    if name in {"SAM3Segmenter"}:
        from . import segmenter
        return getattr(segmenter, name)
    if name in {"Qwen3VLEntityExtractor"}:
        from . import entity
        return getattr(entity, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
