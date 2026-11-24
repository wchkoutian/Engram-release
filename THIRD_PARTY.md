# Third-party components

The authors' original code is licensed under MIT. Third-party components retain their own licenses and notices; the root MIT license does not replace those licenses or grant rights to external model weights or datasets.

- Wan2.2 / pretrained Wan2.2-TI2V-5B: https://github.com/Wan-Video/Wan2.2 . Model files are external and subject to their upstream terms.
- Hugging Face Diffusers: https://github.com/huggingface/diffusers . Installed as an external dependency.
- MapAnything 1.1.2: https://github.com/facebookresearch/map-anything . The frozen runtime copy and its distribution license are retained under `runtime_overlay/mapanything*`.
- UniCeption 0.1.7: https://github.com/castacks/UniCeption . The frozen code and BSD 3-Clause distribution license are retained under `runtime_overlay/uniception*`.
- jaxtyping 0.3.7 and bundled typeguard: https://github.com/patrick-kidger/jaxtyping . Existing license files are retained.
- wadler_lindig 0.1.7: https://github.com/patrick-kidger/wadler_lindig . Existing license file is retained.
- DINOv2: https://github.com/facebookresearch/dinov2 . Required offline hub repository and weights are external.
- LPIPS: https://github.com/richzhang/PerceptualSimilarity . AlexNet weights are external.
- LSM-World / Latent Spatial Memory: https://github.com/microsoft/LatentSpatialMemory . Unmodified clip-sampling and video-I/O utilities are included under `third_party/lsm_world/`, with Microsoft's MIT license. Source file hashes are listed in `third_party/lsm_world/README.md`.
- ViPE: https://github.com/nv-tlabs/vipe . External geometry-preprocessing dependency.
- Qwen3-VL and SAM3: https://github.com/QwenLM/Qwen3-VL and https://github.com/facebookresearch/sam3 . External dependencies for dynamic-object and sky masks. Model files remain external.

Dataset clips, `.pt` checkpoints, credentials, model weights, Python bytecode caches, and historical launcher scripts are not included. The small upstream `colors_fps_5k.npz` color lookup resource is retained; it is not a model or dataset cache. Installation `RECORD` and `direct_url.json` files are omitted to avoid publishing machine-local installation paths. Redundant `INSTALLER`, `REQUESTED`, `WHEEL` and `top_level.txt` files are also omitted; version metadata, entry points and license files are preserved. Review upstream licensing before any public redistribution.
