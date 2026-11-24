"""Canonical conversion between Wan VAE latents and Wan transformer latents."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import torch


WAN_LATENT_CONTRACT_NAME = "wan_model_latent_v1"
WAN_VAE_TEMPORAL_STRIDE = 4
WAN_CHUNK_LATENT_FRAMES = 9
WAN_CHUNK_RGB_FRAMES = 1 + (WAN_CHUNK_LATENT_FRAMES - 1) * WAN_VAE_TEMPORAL_STRIDE


def wan_rgb_frames_for_latents(
    latent_frames: int,
    temporal_stride: int = WAN_VAE_TEMPORAL_STRIDE,
) -> int:
    """Return the RGB span represented by a continuous causal Wan latent sequence."""
    latent_frames = int(latent_frames)
    temporal_stride = int(temporal_stride)
    if latent_frames <= 0 or temporal_stride <= 0:
        raise ValueError(
            f"latent_frames and temporal_stride must be positive, got "
            f"{latent_frames} and {temporal_stride}"
        )
    return 1 + (latent_frames - 1) * temporal_stride


def validate_wan_chunk_contract(
    latent_frames: int,
    temporal_stride: int,
    source: str = "Wan chunk",
) -> None:
    """Require Mirage/Wan's native 9-latent, 33-RGB temporal chunk contract."""
    latent_frames = int(latent_frames)
    temporal_stride = int(temporal_stride)
    rgb_frames = wan_rgb_frames_for_latents(latent_frames, temporal_stride)
    if (
        latent_frames != WAN_CHUNK_LATENT_FRAMES
        or temporal_stride != WAN_VAE_TEMPORAL_STRIDE
        or rgb_frames != WAN_CHUNK_RGB_FRAMES
    ):
        raise ValueError(
            f"{source} must use {WAN_CHUNK_LATENT_FRAMES} latent frames with temporal "
            f"stride {WAN_VAE_TEMPORAL_STRIDE} ({WAN_CHUNK_RGB_FRAMES} RGB frames), got "
            f"{latent_frames} latent frames, stride {temporal_stride} ({rgb_frames} RGB frames)"
        )


def wan_temporal_chunk_contract() -> Dict[str, int]:
    """Serializable native temporal contract stored with checkpoints and datasets."""
    return {
        "latent_frames": WAN_CHUNK_LATENT_FRAMES,
        "rgb_frames": WAN_CHUNK_RGB_FRAMES,
        "vae_temporal_stride": WAN_VAE_TEMPORAL_STRIDE,
    }


def _wan_latent_stats(vae, z: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return broadcast-ready channel statistics for a batched latent tensor."""
    if z.ndim != 5:
        raise ValueError(f"Wan latents must have shape [B, C, T, H, W], got {tuple(z.shape)}")
    if not z.is_floating_point():
        raise TypeError(f"Wan latents must use a floating dtype, got {z.dtype}")

    mean_cfg = getattr(vae.config, "latents_mean", None)
    std_cfg = getattr(vae.config, "latents_std", None)
    if (mean_cfg is None) != (std_cfg is None):
        raise ValueError("Wan VAE config must define latents_mean and latents_std together")
    if mean_cfg is None:
        return None, None

    mean_flat = torch.as_tensor(mean_cfg, device=z.device, dtype=z.dtype).flatten()
    std_flat = torch.as_tensor(std_cfg, device=z.device, dtype=z.dtype).flatten()
    channels = int(z.shape[1])
    configured_channels = int(getattr(vae.config, "z_dim", channels) or channels)
    if configured_channels != channels:
        raise ValueError(
            f"Wan VAE config z_dim={configured_channels} does not match latent channels={channels}"
        )
    if mean_flat.numel() != channels or std_flat.numel() != channels:
        raise ValueError(
            "Wan VAE latent statistics do not match the latent channel count: "
            f"channels={channels}, mean={mean_flat.numel()}, std={std_flat.numel()}"
        )
    if not torch.isfinite(mean_flat).all() or not torch.isfinite(std_flat).all():
        raise ValueError("Wan VAE latent statistics must be finite")
    if not torch.all(std_flat > 0):
        raise ValueError("Wan VAE latents_std must be strictly positive")

    view_shape = (1, channels) + (1,) * (z.ndim - 2)
    return mean_flat.view(view_shape), std_flat.view(view_shape)


def normalize_wan_latents(vae, z_raw: torch.Tensor) -> torch.Tensor:
    """Convert raw VAE latents to the latent space consumed by the Wan transformer."""
    mean, std = _wan_latent_stats(vae, z_raw)
    if mean is not None:
        return (z_raw - mean) / std

    scaling = getattr(vae.config, "scaling_factor", None)
    if scaling is None:
        return z_raw
    scaling_t = torch.as_tensor(scaling, device=z_raw.device, dtype=z_raw.dtype)
    if not torch.isfinite(scaling_t).all() or not torch.all(scaling_t != 0):
        raise ValueError("Wan VAE scaling_factor must be finite and non-zero")
    return z_raw * scaling_t


def denormalize_wan_latents(vae, z_model: torch.Tensor) -> torch.Tensor:
    """Convert Wan transformer latents back to the raw latent space decoded by the VAE."""
    mean, std = _wan_latent_stats(vae, z_model)
    if mean is not None:
        return z_model * std + mean

    scaling = getattr(vae.config, "scaling_factor", None)
    if scaling is None:
        return z_model
    scaling_t = torch.as_tensor(scaling, device=z_model.device, dtype=z_model.dtype)
    if not torch.isfinite(scaling_t).all() or not torch.all(scaling_t != 0):
        raise ValueError("Wan VAE scaling_factor must be finite and non-zero")
    return z_model / scaling_t


@torch.no_grad()
def encode_wan_latents(vae, frames: torch.Tensor) -> torch.Tensor:
    """Encode frames deterministically and return Wan transformer-space latents."""
    vae_dtype = getattr(vae, "dtype", frames.dtype)
    posterior = vae.encode(frames.to(dtype=vae_dtype).contiguous()).latent_dist
    z_raw = posterior.mode()
    return normalize_wan_latents(vae, z_raw)


@torch.no_grad()
def decode_wan_latents(vae, z_model: torch.Tensor) -> torch.Tensor:
    """Decode Wan transformer-space latents through the inverse VAE affine transform."""
    vae_dtype = getattr(vae, "dtype", z_model.dtype)
    # Match the official Wan pipeline: restore raw channel statistics in the
    # VAE dtype (fp32 here), not in the transformer's bf16 dtype.
    z_raw = denormalize_wan_latents(vae, z_model.to(dtype=vae_dtype))
    decoded = vae.decode(z_raw)
    if hasattr(decoded, "sample"):
        return decoded.sample
    if isinstance(decoded, (tuple, list)):
        return decoded[0]
    raise TypeError(f"unsupported Wan VAE decode output type: {type(decoded)!r}")


def _unpatchify_wan_frames(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return frames
    batch, packed_channels, time, height, width = frames.shape
    channels = packed_channels // (patch_size * patch_size)
    frames = frames.view(
        batch,
        channels,
        patch_size,
        patch_size,
        time,
        height,
        width,
    )
    frames = frames.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return frames.view(
        batch,
        channels,
        time,
        height * patch_size,
        width * patch_size,
    )


class WanVAEStreamDecoder:
    """Keep Wan decoder feature state across overlapping 9-latent chunks.

    The cache is owned here rather than by ``AutoencoderKLWan`` so an intervening VAE
    encode cannot clear the decoder history. Call ``reset`` once at each new scene, then
    pass only the newly generated, de-duplicated latent frames to ``decode``.
    """

    def __init__(self, vae):
        self.vae = vae
        self.reset()

    def reset(self) -> None:
        counts = getattr(self.vae, "_cached_conv_counts", None)
        if not counts or "decoder" not in counts:
            raise ValueError("Wan VAE does not expose its decoder causal-cache size")
        self._feat_map = [None] * int(counts["decoder"])
        self._started = False

    @torch.no_grad()
    def decode(self, z_model: torch.Tensor) -> torch.Tensor:
        if z_model.ndim != 5 or z_model.shape[2] <= 0:
            raise ValueError(
                f"streaming Wan decode expects [B,C,T,H,W] with T>0, got "
                f"{tuple(z_model.shape)}"
            )
        vae_dtype = getattr(self.vae, "dtype", z_model.dtype)
        z_raw = denormalize_wan_latents(self.vae, z_model.to(dtype=vae_dtype))
        hidden = self.vae.post_quant_conv(z_raw)
        outputs = []
        for index in range(hidden.shape[2]):
            feat_index = [0]
            kwargs = {
                "feat_cache": self._feat_map,
                "feat_idx": feat_index,
            }
            if not self._started:
                kwargs["first_chunk"] = True
            output = self.vae.decoder(
                hidden[:, :, index : index + 1],
                **kwargs,
            )
            outputs.append(output)
            self._started = True

        frames = torch.cat(outputs, dim=2)
        patch_size = int(getattr(self.vae.config, "patch_size", 1) or 1)
        frames = _unpatchify_wan_frames(frames, patch_size)
        return frames.clamp(-1.0, 1.0)


def wan_latent_contract(vae) -> Dict[str, Any]:
    """Describe the serialized latent representation produced by this module."""
    mean_cfg = getattr(vae.config, "latents_mean", None)
    std_cfg = getattr(vae.config, "latents_std", None)
    if (mean_cfg is None) != (std_cfg is None):
        raise ValueError("Wan VAE config must define latents_mean and latents_std together")
    if mean_cfg is not None:
        normalization = "latents_mean_std"
        channels = len(mean_cfg)
        signature_payload = {
            "latents_mean": [float(value) for value in mean_cfg],
            "latents_std": [float(value) for value in std_cfg],
        }
    else:
        scaling = getattr(vae.config, "scaling_factor", None)
        normalization = "scaling_factor" if scaling is not None else "identity"
        channels = int(getattr(vae.config, "z_dim", 0) or 0)
        signature_payload = {"scaling_factor": None if scaling is None else float(scaling)}
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "name": WAN_LATENT_CONTRACT_NAME,
        "posterior": "mode",
        "normalization": normalization,
        "channels": int(channels),
        "vae_affine_sha256": signature,
    }


def validate_wan_latent_contract(
    contract: Dict[str, Any] | None,
    expected: Dict[str, Any] | str = WAN_LATENT_CONTRACT_NAME,
    source: str = "latent data",
) -> Dict[str, Any]:
    """Validate serialized latent metadata against a name or a full VAE contract."""
    if contract is None:
        raise ValueError(
            f"{source} has no latent contract metadata and is treated as legacy raw VAE data; "
            "migrate or re-precompute it before training/evaluation"
        )
    required = {"name", "posterior", "normalization", "channels", "vae_affine_sha256"}
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(f"{source} latent contract is missing fields: {missing}")
    if contract["posterior"] != "mode":
        raise ValueError(f"{source} uses posterior={contract['posterior']!r}, expected 'mode'")
    if int(contract["channels"]) <= 0:
        raise ValueError(f"{source} latent contract has invalid channels={contract['channels']!r}")
    signature = str(contract["vae_affine_sha256"])
    if len(signature) != 64 or any(char not in "0123456789abcdef" for char in signature.lower()):
        raise ValueError(f"{source} latent contract has an invalid VAE affine signature")

    if isinstance(expected, str):
        if contract["name"] != expected:
            raise ValueError(f"{source} uses latent contract {contract['name']!r}, expected {expected!r}")
    else:
        mismatched = {
            key: (contract.get(key), expected.get(key))
            for key in required
            if contract.get(key) != expected.get(key)
        }
        if mismatched:
            raise ValueError(f"{source} latent contract does not match the active Wan VAE: {mismatched}")
    return dict(contract)


def load_wan_latent_stats(
    path: str,
    expected_contract: Dict[str, Any] | str = WAN_LATENT_CONTRACT_NAME,
) -> Dict[str, Any]:
    """Load memory-whitening statistics with latent-contract validation."""
    stats = torch.load(path, map_location="cpu")
    contract = validate_wan_latent_contract(
        stats.get("latent_contract"), expected_contract, source=path
    )
    for key in ("mean", "var"):
        if key not in stats:
            raise ValueError(f"{path} is missing latent statistic {key!r}")
    mean = torch.as_tensor(stats["mean"])
    var = torch.as_tensor(stats["var"])
    channels = int(contract["channels"])
    if mean.ndim != 1 or var.ndim != 1 or mean.numel() != channels or var.numel() != channels:
        raise ValueError(
            f"{path} latent statistics must both have shape [{channels}], got "
            f"mean={tuple(mean.shape)}, var={tuple(var.shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
        raise ValueError(f"{path} latent statistics must be finite")
    if not torch.all(var >= 0):
        raise ValueError(f"{path} latent variance must be non-negative")
    if "count" in stats and int(stats["count"]) <= 0:
        raise ValueError(f"{path} latent statistic count must be positive")
    if not stats.get("dataset_id"):
        raise ValueError(f"{path} has no latent dataset ID")
    return stats
