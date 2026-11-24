from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from .common import BASE_EXPERIMENT, LATENT_STATS, MODEL_ROOT, append_json, build_depth_estimator, decoded_rgb, digest, ffmpeg_executable, rgb_tensor, save_tensor, save_video, seed_all, setup_cuda, tensor_digest, write_json
from engram.confidence.estimator import ConfidenceEstimator
from engram.confidence.proxies import depth_uncertainty_precision, grazing_precision
from engram.geometry.projection import plucker_rays
from engram.memory.engram_memory import EngramMemory
from engram.memory.policy import MemoryFusionPolicy
from engram.models.conditioning import pack_readout


LITE_LAYERS = (0, 8, 16, 24)
REFERENCE_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)
RECIPE_ID = "engram_lite_native_voxel_v1"
FUSION = MemoryFusionPolicy(w_max=20., conf_report_max=5., real_weight_scale=4., gen_into_real_scale=.25, gen_into_real_budget=.5, tau_geom=1., fuse_position=False, drop_on_layer_budget=True, immutable_real_memory=True, reserve_real_in_topk=True)


@dataclass(frozen=True)
class Recipe:
    identifier: str = RECIPE_ID
    top_k: int = 2
    channels: int = 108
    target_latents: int = 9
    rgb_frames: int = 33
    clean_prefix: int = 1
    sampling_steps: int = 40
    guidance_scale: float = 1.
    noise_mode: str = "shared_sigma"
    temporal_mode: str = "native_chunk_v1"
    initialization: str = "wan_copy"
    voxel_size_metres: float = .005  # V13: 0.05 -> 0.005 (Local-10 e4 / WS20 B4 diagnostics)
    bias_gamma: float = .98
    static_scene: bool = True


def recipe_contract():
    return {**asdict(Recipe()), "memory_fusion": FUSION.to_metadata(), "depth_weights": {"depth": 1., "graze": .5}, "masking": "finite_positive_depth_static_scene", "scene_scale": "fixed_original_I0_unit", "default_write_policy": "protected_voxel_writeback", "prefix": "last_generated_RGB_reencoded", "excluded_default_modules": ["PTR", "SAM3", "Qwen_entity", "motion_weighting", "ranking", "LoRA", "per_target_noise", "multi_hypothesis_depth"]}


def layers_for(profile):
    if profile == "lite4":
        return LITE_LAYERS
    if profile == "reference8":
        return REFERENCE_LAYERS
    raise ValueError(f"Unknown model profile: {profile}")


def derive_lite_initialization(reference_state):
    result = {}
    block_indices = [REFERENCE_LAYERS.index(layer) for layer in LITE_LAYERS]
    for name, tensor in reference_state.items():
        pieces = name.split(".")
        if pieces[0] in ("blocks", "zero_out"):
            reference_index = int(pieces[1])
            if reference_index not in block_indices:
                continue
            pieces[1] = str(block_indices.index(reference_index))
            name = ".".join(pieces)
        result[name] = tensor
    return result


def build_backbone(profile, training=False, memory_fraction=.22):
    setup_cuda(memory_fraction)
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    from engram.models.wan_wrapper import WanEngramBackbone

    layers = layers_for(profile)
    transformer = WanTransformer3DModel.from_pretrained(MODEL_ROOT, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    vae = AutoencoderKLWan.from_pretrained(MODEL_ROOT, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).eval()
    vae.enable_tiling()
    vae.enable_slicing()
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ROOT, subfolder="scheduler", local_files_only=True)
    model = WanEngramBackbone(transformer, vae, scheduler, top_k=2, num_control_layers=len(layers), control_layers=list(layers), camera_cond=True, clean_prefix_len=1, chunk_frames=9, enable_lora=False, temporal_mode="native_chunk_v1", side_branch_initialization="wan_copy", timestep_sampling="uniform", device="cuda", dtype=torch.bfloat16, model_id=MODEL_ROOT)
    if training:
        model.side_branch.float()
        model.enable_gradient_checkpointing()
        model.train()
        model.vae.eval()
    else:
        model.eval()
        model.vae.to("cuda")
    return model


class ProtectedMemory:
    def __init__(self, bundle, device="cpu", writeback=True, stats=None):
        from engram.models.wan_latents import load_wan_latent_stats

        self.device = torch.device(device)
        self.writeback = bool(writeback)
        self.unit_metres = float(bundle["seed_depth_metric_median"])
        self.memory = EngramMemory(channels=48, voxel_size=Recipe().voxel_size_metres / self.unit_metres, consistency_tau=3., max_layers=8, fuse=True, update_whitener=False, device=self.device, **FUSION.memory_kwargs())
        statistics = stats or load_wan_latent_stats(LATENT_STATS, bundle["latent_contract"])
        self.memory.whitener.set_stats(statistics["mean"], statistics["var"]).freeze()
        self.confidence = ConfidenceEstimator(weights={"depth": 1., "graze": .5})
        self.initialized = False
        self.generated_frames = []
        self.update(bundle["seed_latent"][0, :, 0], bundle["seed_depth"], bundle["intrinsics_latent"][0], bundle["poses"]["nvs"][0], 0, "real")

    def update(self, feature, depth, intrinsics, pose, frame_index, origin):
        if origin not in ("real", "generated"):
            raise ValueError("Only real I0 and generated observations are admitted")
        if origin == "real" and (self.initialized or frame_index != 0):
            raise ValueError("Only original I0 may initialize real memory")
        if origin == "generated" and not self.writeback:
            return False
        depth = depth.to(self.device).float()
        intrinsics = intrinsics.to(self.device).float()
        valid = torch.isfinite(depth) & (depth > 0)
        proxies = {"depth": depth_uncertainty_precision(depth_conf=torch.full_like(depth, .5)), "graze": grazing_precision(depth, intrinsics)}
        precision = self.confidence.precision(proxies, calibrate=False)
        anchor_index = int(frame_index) // 4
        self.memory.update(feature.to(self.device).float(), depth, precision, pose.to(self.device).float(), intrinsics, valid=valid, t=anchor_index + 1, bias_discount=Recipe().bias_gamma ** anchor_index, origin=origin)
        if origin == "real":
            self.initialized = True
        else:
            self.generated_frames.append(int(frame_index))
        return True

    def fingerprint(self, real_only=False):
        indices, positions, features, precision = self.memory.store.gather_alive()
        origins = self.memory.store.alive_origin_real()
        if real_only:
            indices, positions, features, precision, origins = (value[origins] for value in (indices, positions, features, precision, origins))
        return tensor_digest((indices, positions, features, precision, origins))

    def read(self, pose, intrinsics, height, width):
        readout = self.memory.read(pose.to(self.device).float(), intrinsics.to(self.device).float(), height, width, top_k=2)
        return pack_readout(readout, scene_scale=1.)

    def stats(self):
        return {**self.memory.store.provenance_stats(), "generated_observation_count": len(self.generated_frames)}


def conditioning(poses, intrinsics, height, width, memory):
    prefix_inverse = torch.linalg.inv(poses[0])
    outputs = []
    for pose, camera in zip(poses, intrinsics):
        readout = memory.read(pose, camera, height, width)
        rays = plucker_rays(pose @ prefix_inverse, camera, height, width)
        outputs.append(torch.cat((readout, rays), dim=0))
    return torch.stack(outputs, dim=1).unsqueeze(0)


def poses_in_metres(poses, unit_metres):
    converted = poses.clone()
    converted[..., :3, 3] *= float(unit_metres)
    return converted


@torch.no_grad()
def write_generated_chunk(memory, estimator, bundle, rgb_frames, latents, poses, start, valid_count):
    latent_indices = [index for index in range(1, 9) if index * 4 < valid_count]
    if not latent_indices:
        return {"geometry_seconds": 0., "fusion_seconds": 0.}
    metric_poses = poses_in_metres(poses, memory.unit_metres)
    started = time.monotonic()
    results = estimator.depth_for_views([rgb_frames[index * 4] for index in latent_indices], [bundle["intrinsics"][0] for _ in latent_indices], [metric_poses[index] for index in latent_indices], bundle["height"] // 16, bundle["width"] // 16)
    geometry_seconds = time.monotonic() - started
    started = time.monotonic()
    for index, result in zip(latent_indices, results):
        depth = result["depth"] / memory.unit_metres
        memory.update(latents[0, :, index], depth, bundle["intrinsics_latent"][0], poses[index], start + index * 4, "generated")
    return {"geometry_seconds": geometry_seconds, "fusion_seconds": time.monotonic() - started}


def flow_loss(model, target, text, condition, valid_mask):
    noise = torch.randn_like(target)
    sample = torch.rand((target.shape[0],), device=target.device).clamp(1e-4, 1 - 1e-4)
    shift = float(model.scheduler.config.flow_shift)
    sigma = shift * sample / (1 + (shift - 1) * sample)
    noisy = (1 - sigma[:, None, None, None, None]) * target + sigma[:, None, None, None, None] * noise
    noisy[:, :, :1] = target[:, :, :1]
    timestep = (sigma * model.scheduler.config.num_train_timesteps).round().long().clamp(0, model.scheduler.config.num_train_timesteps - 1)
    timestep = model._patch_token_timesteps(timestep, target.shape, 1)
    prediction = model.denoise(noisy.to(model.dtype), timestep, text, condition)
    errors = (prediction.float() - (noise - target).float()).square().mean((0, 1, 3, 4))
    mask = valid_mask.to(errors)
    return (errors * mask).sum() / mask.sum().clamp_min(1), float(sigma.mean().detach())


def window_indices(frame_count=81, stride=16):
    if frame_count < 2 or stride <= 0:
        raise ValueError("frame_count must exceed one and stride must be positive")
    rows = []
    for start in range(0, frame_count - 1, stride):
        indices = [min(start + offset, frame_count - 1) for offset in range(33)]
        valid = [offset > 0 and start + offset < frame_count for offset in range(0, 33, 4)]
        rows.append({"start": start, "direction": 1, "indices": indices, "valid": valid})
    return rows
