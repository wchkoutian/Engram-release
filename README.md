<div align="center">

<h1>ENGRAM: Reliability-Weighted Latent Fusion and<br>Occlusion-Aware Layered Readout for Video World Models</h1>

<p>
  Chenhao Wu<sup>1,*</sup>&nbsp;&nbsp;
  Zhaolong Su<sup>2,*</sup>&nbsp;&nbsp;
  Yang Liu<sup>3,*</sup>&nbsp;&nbsp;
  Tianrui Su<sup>4</sup><br>
  Yingguang Yang<sup>3</sup>&nbsp;&nbsp;
  Shang Luo<sup>3</sup>&nbsp;&nbsp;
  Yancheng Chen<sup>1</sup>&nbsp;&nbsp;
  Chongyang Zhang<sup>5</sup><br>
  Hao Zheng<sup>5</sup>&nbsp;&nbsp;
  Kefu Xu<sup>3</sup>&nbsp;&nbsp;
  Congjing Ran<sup>6</sup>&nbsp;&nbsp;
  Bin Chong<sup>3,†</sup>
</p>

<p>
  <sup>1</sup>University of the Chinese Academy of Sciences&nbsp;&nbsp;
  <sup>2</sup>Jilin University&nbsp;&nbsp;
  <sup>3</sup>Peking University<br>
  <sup>4</sup>University of New South Wales&nbsp;&nbsp;
  <sup>5</sup>Fullive-AI&nbsp;&nbsp;
  <sup>6</sup>Wuhan University<br>
  <small><sup>*</sup>Equal contribution.&nbsp;&nbsp;<sup>†</sup>Corresponding author.</small>
</p>

<p>
  <a href="https://arxiv.org/abs/2611.28136"><img src="https://img.shields.io/badge/arXiv-2611.28136-B31B1B?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
  &nbsp;
  <a href="https://wchkoutian.github.io/engram-website/"><img src="https://img.shields.io/badge/Project-Page-2563EB?style=flat&amp;logo=githubpages&amp;logoColor=white" alt="Project page"></a>
  &nbsp;
  <a href="https://huggingface.co/Mousky/ENGRAM/tree/main"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-FFB300?style=flat&logo=huggingface&logoColor=white" alt="Hugging Face weights"></a>
  &nbsp;
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-4F46E5?style=flat" alt="BibTeX citation"></a>
</p>

</div>

ENGRAM maintains persistent latent spatial memory for camera-conditioned video generation. Reliability-weighted latent fusion consolidates repeated observations, while occlusion-aware layered readout provides depth-ordered memory candidates to a Wan2.2-TI2V-5B video generator.

## Installation

Use Linux, Python 3.10 or newer, and a CUDA GPU with bfloat16 support. Install PyTorch for your CUDA version in the environment below, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Models

Download the ENGRAM checkpoints and supporting files from [Hugging Face](https://huggingface.co/Mousky/ENGRAM). Sign in with an account that has access to the repository:

```bash
pip install -U huggingface_hub
hf auth login
hf download Mousky/ENGRAM --local-dir checkpoints/ENGRAM
```

The model package contains `stage1_step_4000.pt`, `stage2_step_8000.pt`, `wan_latent_stats.pt`, `empty-text-bundle.pt`, and `training/manifest.json`.

Also download [Wan2.2-TI2V-5B in Diffusers format](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), [MapAnything](https://github.com/facebookresearch/map-anything), and its [DINOv2](https://github.com/facebookresearch/dinov2) dependencies.

```bash
cp configs/assets.example.json assets.local.json
```

Set the paths in `assets.local.json`:

| Key | Path |
| --- | --- |
| `wan_model` | Wan2.2 Diffusers model directory |
| `mapanything_model` | MapAnything model directory |
| `torch_home` | Torch cache containing `hub/facebookresearch_dinov2_main`; RE10K scoring also uses LPIPS/AlexNet weights |
| `stage1_checkpoint` | `checkpoints/ENGRAM/stage1_step_4000.pt` |
| `checkpoint` | `checkpoints/ENGRAM/stage2_step_8000.pt` |
| `checkpoint_sha256` | Stage-2 checkpoint SHA256 from the downloaded `SHA256SUMS` |
| `latent_stats` | `checkpoints/ENGRAM/wan_latent_stats.pt` |
| `text_bundle` | `checkpoints/ENGRAM/empty-text-bundle.pt` |
| `training_root` | `checkpoints/ENGRAM/training` for inference, or the prepared training cache for training |
| `training_lmdb` | Training LMDB directory containing `data.mdb` |
| `training_samples` | Directory containing the source training clips and poses |
| `training_scenes` | `configs/re10k_train256_scenes.json` |
| `re10k_data` | Prepared RealEstate10K test directory |
| `caption_model` | Qwen2.5-VL-7B-Instruct directory, used for training captions |

The example paths assume `assets.local.json` is in the repository root; paths are resolved relative to this JSON file. Inference uses both checkpoints and the accompanying training manifest. The checkpoint SHA256 is checked for inference; data preparation and training do not require it.

```bash
python run.py check-assets --assets assets.local.json --for-action infer-re10k
```

## Inference

### RealEstate10K

Set `re10k_data` to the prepared test directory containing `selected_10.json` (the 100-scene manifest), images, camera trajectories, and `reference_code/metric_helper.py` and `reference_code/metrics.py`. The test scene list is in [configs/re10k_test100_scenes.json](configs/re10k_test100_scenes.json); the manifest and trajectory formats are defined in [inference/re10k_protocol.py](inference/re10k_protocol.py).

Prepare the inputs, then generate and score one scene:

```bash
python run.py prepare-re10k --assets assets.local.json --root outputs/re10k
python run.py infer-re10k --assets assets.local.json --root outputs/re10k --sample 000
python run.py score-re10k --assets assets.local.json --root outputs/re10k --sample 000
```

Repeat inference and scoring with `--sample 001` through `--sample 099`, then export the results:

```bash
python run.py export-re10k --assets assets.local.json --root outputs/re10k
```

NVS generates 81 frames; closed-loop inference generates 153 frames. Both use 33-frame chunks, a 32-frame stride, 40 sampling steps, CFG 1, and seed 20917. Outputs include videos and per-scene scores.

### WorldScore

Use the [WorldScore toolkit](https://github.com/haoyi-duan/WorldScore) to prepare each case directory with `input_image.png`, `camera_data.json`, and `image_data.json`. Inputs use a 1280×704 center crop and 81 Blender camera-to-world poses in `camera_data.json` under `cameras_interp`.

Copy [configs/worldscore.example.json](configs/worldscore.example.json) to `worldscore.local.json` and set each case's `name`, `input_directory`, `positive_prompt`, and `positive_prompt_2`. Input directories are relative to this manifest. The first prompt conditions the first chunk; the second conditions later chunks.

The paper's 200-case list is in [configs/worldscore_test200_cases.json](configs/worldscore_test200_cases.json). To use these cases, map each `id` to the inference manifest's `name` and supply the prepared input directory. Use the listed first prompt; for static cases, use the second `source_row.prompt_list` entry plus style as the second prompt, and for dynamic cases repeat the first prompt.

```bash
python run.py prepare-worldscore --assets assets.local.json \
  --manifest worldscore.local.json --root outputs/worldscore
python run.py infer-worldscore --assets assets.local.json --root outputs/worldscore
```

WorldScore generation uses 81 frames, 16 fps, 40 sampling steps, CFG 1, and seed 42. Results are saved under `outputs/worldscore/model_outputs/Engram/worldscore_output/`. Use the official WorldScore evaluator for scoring.

## Training

Training uses 256 RealEstate10K scenes and 2,304 prepared windows. Stage 1 freezes the video backbone and trains the memory-conditioning side branch for 4,000 updates. Stage 2 freezes the side branch and fine-tunes the full Wan transformer for 8,000 updates. Both stages use an effective batch size of 4.

The training scenes are listed in [configs/re10k_train256_scenes.json](configs/re10k_train256_scenes.json). Set `training_samples` and `training_scenes` in `assets.local.json`.

### Source clips and geometry

Start with locally downloaded RealEstate10K videos or timestamp-named image sequences. Copy [configs/preprocessing.example.json](configs/preprocessing.example.json) to `preprocessing.local.json` and fill in the source paths, sample IDs, and clip start times. Source paths are relative to this JSON file. Frame-directory rows use `selected_source_timestamps_us`, a list of 81 strictly increasing timestamps, in place of `source_start_seconds`.

For the fixed training split, use the source windows corresponding to the IDs and clip SHA256 values in `re10k_train256_scenes.json`. The preprocessing manifest supplies source paths and windows; the training scene list supplies the selection and expected clip hashes.

```bash
python scripts/prepare_clips.py --manifest preprocessing.local.json \
  --sample-root datasets/re10k/samples
```

This writes 81-frame clips at 16 fps and 1280 × 704 using the LSM-World utilities in [third_party/lsm_world](third_party/lsm_world). Install [ViPE](https://github.com/nv-tlabs/vipe) in its own environment, then run geometry estimation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_geometry.py \
  --manifest preprocessing.local.json --sample-root datasets/re10k/samples \
  --vipe-root /path/to/vipe --vipe-python /path/to/vipe-env/bin/python \
  --output-root work/vipe
```

Geometry preparation uses ViPE's `dav3` pipeline and exports `pose.npz`, `intrinsics.npz`, and `depth.zip` into each sample directory. Generate dynamic-object and sky masks with Qwen3-VL-2B and [SAM3](https://github.com/facebookresearch/sam3). In the SAM3 environment, install `requirements-preprocessing.txt`, download the Qwen3-VL-2B-Instruct model and SAM3 checkpoint, and run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_masks.py \
  --manifest preprocessing.local.json --sample-root datasets/re10k/samples \
  --qwen-model /path/to/Qwen3-VL-2B-Instruct \
  --sam-checkpoint /path/to/sam3.pt
```

All three scripts accept `--num-shards N`, `--shard-id I`, and `--skip-existing` for independent shards and verified reuse. The prepared files have this layout:

```text
training_samples/
  00036138/
    clip.mp4
    pose.npz
    intrinsics.npz
    depth.zip
    mask.zip
  ...
```

`pose.npz` stores camera-to-world matrices of shape `(81, 4, 4)` under `data`, with frame indices `0..80` under `inds`. `intrinsics.npz` uses the same indices and stores pixel-space `(fx, fy, cx, cy)` values of shape `(81, 4)`. Depth maps and poses use the same scale. `depth.zip` contains 1280 × 704 EXR images with a `Z` depth channel; `mask.zip` contains matching PNG masks, where nonzero pixels mark excluded dynamic-object or sky regions. The builder reads frames `00000`, `00004`, …, `00080` from both archives.

### LMDB and training windows

Build and validate the LMDB:

```bash
python run.py build-lmdb --assets assets.local.json --root datasets/re10k
python run.py merge-lmdb --assets assets.local.json --root datasets/re10k
python run.py validate-lmdb --assets assets.local.json --root datasets/re10k
```

The output contains `precomputed_train_wan_model_v1.lmdb` and `latent_stats_train_wan_model_v1.pt`. Set `training_lmdb` to the generated LMDB. For the paper's training setup, use the released `wan_latent_stats.pt` as `latent_stats`. Each scene contains 21 latent anchors, depth maps, reliability values, static masks, intrinsics, and camera extrinsics. Training captions are generated after the training windows are prepared.

For parallel encoding, run `build-lmdb` once per GPU with a shared `--num-shards N` and distinct `--shard-id 0` through `N-1`, then run `merge-lmdb` with the same `--num-shards N`. To use another scene list, pass `--manifest path/to/scenes.json` to all three commands; JSON lists, objects with a `scenes` list, and JSONL files with a `sample_id` per row are supported. The standalone interface is `python scripts/build_lmdb.py --help`.

Build the five forward training windows per scene, then add the four return windows:

```bash
python run.py prepare-training --assets assets.local.json --root work/forward_cache
python run.py prepare-training-return --assets assets.local.json \
  --source work/forward_cache --root work/training_cache
```

The training cache uses this layout:

```text
training_cache/
  manifest.json
  DATA_READY.json
  data/.../window_*.pt
  captions.json
  caption_text.pt
```

Each window contains `latent`, `condition`, `valid_mask`, and history metadata in the format specified by [source/engram/data/contracts.py](source/engram/data/contracts.py). Set `training_root` to `work/training_cache` and `caption_model` to the local Qwen2.5-VL-7B-Instruct model, then generate captions and text embeddings:

```bash
python run.py caption-training --assets assets.local.json --root work/training_cache
```

Create separate training output directories linked to the cache:

```bash
python scripts/link_training_data.py work/training_cache work/stage1
python scripts/link_training_data.py work/training_cache work/stage2
```

Train Stage 1:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py \
  train-stage1 --assets assets.local.json --root work/stage1
```

Set `stage1_checkpoint` in `assets.local.json` to `work/stage1/runs/step_4000.pt`, then train Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py \
  train-stage2 --assets assets.local.json --root work/stage2
```

The final checkpoint is saved to `work/stage2/runs/step_8000.pt`. Add `--resume` to either training command to continue that stage from its saved state. To use the trained model for inference, update `checkpoint` and `checkpoint_sha256` in the asset configuration.

Use `--memory-fraction` to adjust the per-process CUDA memory limit for the available GPU. For example, add `--memory-fraction 0.65` when training Stage 1 on an H20 with sufficient free memory. The option also applies to inference and chunk handoff; it changes the allocation limit without changing the model or sampling settings.

## Acknowledgements

We thank [Wan2.2](https://github.com/Wan-Video/Wan2.2), [MapAnything](https://github.com/facebookresearch/map-anything), [DINOv2](https://github.com/facebookresearch/dinov2), and [WorldScore](https://github.com/haoyi-duan/WorldScore) for their open-source contributions.

## Citation

```bibtex
@misc{wu2026engram,
  title = {{ENGRAM}: Reliability-Weighted Latent Fusion and Occlusion-Aware Layered Readout for Video World Models},
  author = {Chenhao Wu and Zhaolong Su and Yang Liu and Tianrui Su and Yingguang Yang and Shang Luo and Yancheng Chen and Chongyang Zhang and Hao Zheng and Kefu Xu and Congjing Ran and Bin Chong},
  year = {2026},
  eprint = {2611.28136},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2611.28136}
}
```

## License

ENGRAM code is released under the [MIT License](LICENSE). Bundled third-party code retains its original licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
