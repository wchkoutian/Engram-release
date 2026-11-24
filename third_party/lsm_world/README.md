# LSM-World preprocessing utilities

Source: [microsoft/LatentSpatialMemory](https://github.com/microsoft/LatentSpatialMemory).
License: [MIT, Copyright (c) Microsoft Corporation](LICENSE).

The following files are copied without modification from the LSM-World source
snapshot used for this release. SHA256 values identify the included snapshot:

| File | SHA256 |
| --- | --- |
| `data_process/clip_sampling.py` | `e483d77e3b82c6ed20beb86b6b138275685b1b985702f4a38239a134fb1ddf75` |
| `data_process/video_io.py` | `e15d7ad036df4fe16c774c03084ad1c3937c2d31cab4767cb9ea2206956d6776` |

Empty `__init__.py` files provide the `lsm_world` package namespace. ENGRAM's
entrypoints are `scripts/prepare_clips.py`, `scripts/prepare_geometry.py`, and
`scripts/prepare_masks.py`. ViPE, Qwen3-VL, and SAM3 are external dependencies.
