"""Wan2.2-TI2V-5B backbone wrapper with the Engram memory side branch (Sec 3.6).

Strategy: keep the pretrained DiT *frozen*, add (a) rank-64 LoRA adapters on the attention
projections for light task adaptation and (b) a trainable ControlNet-style side branch that
injects memory conditioning as zero-initialised residuals into the first ``num_control_layers``
transformer blocks via forward hooks. Training uses Wan's flow-matching objective.

Requires a GPU and the heavy stack (``diffusers`` from main, ``transformers``, ``peft``).
This module is import-light only when actually instantiated; the algorithmic core never
imports it. Wan2.2 TI2V-5B config: dim=3072, heads=24, layers=30, patch=(1,2,2),
VAE stride=(4,16,16).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn

from .side_branch import ControlNetSideBranch
from .conditioning import readout_channels, conditioning_coverage_mask, CAMERA_RAY_CHANNELS
from .checkpoint_contract import validate_lora_checkpoint, wan_checkpoint_model_contract
from .wan_latents import (
    WAN_CHUNK_LATENT_FRAMES,
    WAN_VAE_TEMPORAL_STRIDE,
    WanVAEStreamDecoder,
    decode_wan_latents,
    encode_wan_latents,
    validate_wan_chunk_contract,
)


WAN_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NATIVE_TEMPORAL_MODE = "native_chunk_v1"
PTR_TEMPORAL_MODE = "ptr_clean_tokens_v1"
PTR_TARGET_LATENT_FRAMES = WAN_CHUNK_LATENT_FRAMES
# The four-reference PTR contract uses independently positioned negative-time RoPE tokens.
PTR_MAX_REFERENCE_LATENTS = 4


def load_wan_backbone(model_id: str = WAN_TI2V_5B, dtype=torch.bfloat16, device="cuda"):
    """Load the Wan2.2 pipeline components (transformer, VAE, scheduler, text encoder)."""
    from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    pipe.to(device)
    if not isinstance(pipe.scheduler, UniPCMultistepScheduler):
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return {
        "pipe": pipe,
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "scheduler": pipe.scheduler,
        "text_encoder": getattr(pipe, "text_encoder", None),
        "tokenizer": getattr(pipe, "tokenizer", None),
    }


class WanEngramBackbone(nn.Module):
    def __init__(
        self,
        transformer,
        vae,
        scheduler,
        top_k: int = 2,
        num_control_layers: int = 8,
        control_layers: Optional[List[int]] = None,
        composite_cond: bool = False,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        clean_prefix_len: int = 0,
        chunk_frames: int = WAN_CHUNK_LATENT_FRAMES,
        vae_temporal_stride: int = WAN_VAE_TEMPORAL_STRIDE,
        model_id: Optional[str] = None,
        device="cuda",
        dtype=torch.bfloat16,
        enable_lora: bool = True,
        timestep_sampling: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        camera_cond: bool = False,
        temporal_mode: str = NATIVE_TEMPORAL_MODE,
        # Reference latents admitted per PTR call. The manifest and inference selector
        # preserve descending IoU rank in this axis.
        ptr_max_references: int = PTR_MAX_REFERENCE_LATENTS,
        side_branch_initialization: str = "wan_copy",
        motion_loss_alpha: float = 0.0,
        counterfactual_loss_weight: float = 0.0,
        counterfactual_loss_margin: float = 0.01,
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.top_k = top_k
        self.composite_cond = composite_cond
        self.camera_cond = bool(camera_cond)
        self.temporal_mode = str(temporal_mode)
        if self.temporal_mode not in (NATIVE_TEMPORAL_MODE, PTR_TEMPORAL_MODE):
            raise ValueError(
                "temporal_mode must be "
                f"{NATIVE_TEMPORAL_MODE!r} or {PTR_TEMPORAL_MODE!r}, "
                f"got {self.temporal_mode!r}"
            )
        self.ptr_max_references = int(ptr_max_references)
        if not 0 <= self.ptr_max_references <= PTR_MAX_REFERENCE_LATENTS:
            raise ValueError(
                "ptr_max_references must lie in "
                f"[0, {PTR_MAX_REFERENCE_LATENTS}], got {ptr_max_references}"
            )
        self.side_branch_initialization = str(side_branch_initialization)
        if self.side_branch_initialization not in ("wan_copy", "scratch"):
            raise ValueError(
                "side_branch_initialization must be 'wan_copy' or 'scratch', got "
                f"{self.side_branch_initialization!r}"
            )
        self.motion_loss_alpha = float(motion_loss_alpha)
        self.counterfactual_loss_weight = float(counterfactual_loss_weight)
        self.counterfactual_loss_margin = float(counterfactual_loss_margin)
        if self.motion_loss_alpha < 0.0:
            raise ValueError("motion_loss_alpha must be non-negative")
        if self.counterfactual_loss_weight < 0.0:
            raise ValueError("counterfactual_loss_weight must be non-negative")
        if self.counterfactual_loss_margin < 0.0:
            raise ValueError("counterfactual_loss_margin must be non-negative")
        self.clean_prefix_len = int(clean_prefix_len)
        if self.clean_prefix_len not in (0, 1, 2):
            raise ValueError(f"clean_prefix_len must be 0, 1 or 2, got {self.clean_prefix_len}")
        if self.temporal_mode == PTR_TEMPORAL_MODE and self.clean_prefix_len != 1:
            raise ValueError(
                f"{PTR_TEMPORAL_MODE} requires clean_prefix_len=1 (T0), "
                f"got {self.clean_prefix_len}"
            )
        if timestep_sampling not in ("uniform", "logit_normal", "uniform_per_target"):
            raise ValueError(
                "timestep_sampling must be 'uniform', 'logit_normal' or "
                f"'uniform_per_target', got {timestep_sampling!r}"
            )
        self.timestep_sampling = timestep_sampling
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)
        # Per-microbatch flow-matching diagnostics (sigma-aligned loss, covered vs
        # uncovered loss). Set by ``flow_matching_loss`` and read by the trainer logger.
        self.last_flow_diagnostics: Optional[dict] = None
        validate_wan_chunk_contract(
            chunk_frames, vae_temporal_stride, source="WanEngramBackbone"
        )
        self.chunk_frames = int(chunk_frames)
        self.vae_temporal_stride = int(vae_temporal_stride)

        cfg = transformer.config
        self.latent_channels = cfg.in_channels
        self.hidden = cfg.num_attention_heads * cfg.attention_head_dim
        self.num_heads = cfg.num_attention_heads
        self.patch_size = tuple(cfg.patch_size)
        # Explicit V10 configs use all even Wan depths {0,2,...,28}.  The fallback remains
        # evenly spaced for legacy constructors, but formal configs must record the list.
        if control_layers is None:
            if num_control_layers <= 1:
                control_layers = [0]
            else:
                endpoint = max(0, cfg.num_layers - 2)
                control_layers = [
                    round(index * endpoint / (num_control_layers - 1))
                    for index in range(num_control_layers)
                ]
        self.control_layers = [int(layer) for layer in control_layers]
        if any(layer < 0 or layer >= cfg.num_layers for layer in self.control_layers):
            raise ValueError(
                f"control_layers must lie in [0,{cfg.num_layers}), got "
                f"{self.control_layers}"
            )
        if self.control_layers != sorted(set(self.control_layers)):
            raise ValueError(
                "control_layers must be strictly increasing and unique, got "
                f"{self.control_layers}"
            )
        self.num_control_layers = len(self.control_layers)

        for p in self.transformer.parameters():
            p.requires_grad_(False)
        # 2026-07-10 post-DDP trainable-set fix: the design always described the VAE as
        # frozen, but its parameters previously retained requires_grad=True. DDP then
        # waited for gradients from a VAE that is not part of the flow-matching forward.
        for p in self.vae.parameters():
            p.requires_grad_(False)

        ray_ch = CAMERA_RAY_CHANNELS if self.camera_cond else 0
        in_ch = readout_channels(
            top_k, self.latent_channels, composite=composite_cond, camera_cond=self.camera_cond
        )
        source_blocks = [self._blocks()[layer] for layer in self.control_layers]
        block_init_kwargs = None
        if self.side_branch_initialization == "scratch":
            block_init_kwargs = {
                "dim": self.hidden,
                "ffn_dim": int(cfg.ffn_dim),
                "num_heads": int(cfg.num_attention_heads),
                "qk_norm": str(getattr(cfg, "qk_norm", "rms_norm_across_heads")),
                "cross_attn_norm": bool(getattr(cfg, "cross_attn_norm", False)),
                "eps": float(getattr(cfg, "eps", 1e-6)),
                "added_kv_proj_dim": getattr(cfg, "added_kv_proj_dim", None),
            }
        self.side_branch = ControlNetSideBranch(
            in_channels=in_ch,
            latent_channels=self.latent_channels,
            hidden=self.hidden,
            source_blocks=source_blocks,
            source_patch_embedding=self._wan_model().patch_embedding,
            top_k=self.top_k,
            composite_cond=self.composite_cond,
            patch_size=self.patch_size,
            camera_ray_channels=ray_ch,
            initialization=self.side_branch_initialization,
            block_init_kwargs=block_init_kwargs,
        ).to(device=device, dtype=dtype)
        self._side_branch_frozen = False

        # Copy the pristine Wan blocks before LoRA is attached to the main transformer.
        if enable_lora:
            self._add_lora(lora_rank, lora_alpha, lora_dropout)

        # per-cell generation-quality signal from the final sampling step, set by
        # ``sample_chunk`` and consumed as the ``genq`` confidence proxy (Sec 3.2).
        self.last_generation_residual: Optional[torch.Tensor] = None
        self._stream_decoder = WanVAEStreamDecoder(self.vae)

    # ------------------------------------------------------------------ LoRA
    def _add_lora(self, rank: int, alpha: int, dropout: float):
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=(
                r".*\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)$"
                r"|.*\.ffn\.(net\.0\.proj|net\.2)$"
            ),
            lora_dropout=float(dropout),
            bias="none",
        )
        self.transformer = get_peft_model(self.transformer, config)

    def _wan_model(self):
        m = self.transformer
        base = getattr(m, "base_model", None)
        if base is not None:
            m = base.model
        return m

    def _blocks(self):
        return self._wan_model().blocks

    # ---------------------------------------------------------- PTR contract
    def _require_ptr_mode(self, source: str) -> None:
        if getattr(self, "temporal_mode", NATIVE_TEMPORAL_MODE) != PTR_TEMPORAL_MODE:
            raise ValueError(
                f"{source} requires temporal_mode={PTR_TEMPORAL_MODE!r}, got "
                f"{getattr(self, 'temporal_mode', NATIVE_TEMPORAL_MODE)!r}"
            )

    @staticmethod
    def _uniform_ptr_count(count, batch_size: int, available: int, name: str) -> int:
        """Resolve a padded PTR count, rejecting variable lengths within a microbatch."""
        if count is None:
            value = int(available)
        elif torch.is_tensor(count):
            flat = count.detach().reshape(-1).to(device="cpu", dtype=torch.long)
            if flat.numel() not in (1, batch_size):
                raise ValueError(
                    f"{name} must be scalar or shaped [B], got {tuple(count.shape)}"
                )
            if flat.numel() == batch_size and not torch.equal(flat, flat[:1].expand_as(flat)):
                raise ValueError(
                    f"{name} must be uniform within a PTR microbatch; use per-device "
                    f"batch size 1 for variable P/R lengths, got {flat.tolist()}"
                )
            value = int(flat[0].item())
        else:
            value = int(count)
        if value < 0 or value > int(available):
            raise ValueError(
                f"{name}={value} is outside the padded temporal extent [0,{available}]"
            )
        return value

    def _trim_ptr_context(
        self,
        value: Optional[torch.Tensor],
        count,
        target: torch.Tensor,
        name: str,
        maximum: int,
    ) -> torch.Tensor:
        """Return the active prefix of a padded P/R latent tensor."""
        batch, channels, _, height, width = target.shape
        if value is None:
            active = self._uniform_ptr_count(count, batch, 0, f"{name}_count")
            if active:
                raise ValueError(f"{name} is None but {name}_count={active}")
            return target[:, :, :0]
        if value.ndim != 5:
            raise ValueError(f"{name} must be [B,C,T,H,W], got {tuple(value.shape)}")
        if value.shape[:2] != (batch, channels) or value.shape[3:] != (height, width):
            raise ValueError(
                f"{name} batch/channel/spatial shape differs from target: "
                f"target={tuple(target.shape)}, {name}={tuple(value.shape)}"
            )
        active = self._uniform_ptr_count(
            count, batch, value.shape[2], f"{name}_count"
        )
        if active > int(maximum):
            raise ValueError(
                f"{name}_count must be <= {maximum} for {PTR_TEMPORAL_MODE}, got {active}"
            )
        return value[:, :, :active]

    def _prepare_ptr_inputs(
        self,
        target: torch.Tensor,
        cond_volume: torch.Tensor,
        ptr_preceding_latent: Optional[torch.Tensor],
        ptr_reference_latent: Optional[torch.Tensor],
        ptr_preceding_count=None,
        ptr_reference_count=None,
        source: str = "PTR",
    ):
        self._require_ptr_mode(source)
        if target.ndim != 5:
            raise ValueError(f"{source} target must be [B,C,T,H,W], got {tuple(target.shape)}")
        validate_wan_chunk_contract(
            target.shape[2], WAN_VAE_TEMPORAL_STRIDE, source=f"{source} target"
        )
        preceding = self._trim_ptr_context(
            ptr_preceding_latent,
            ptr_preceding_count,
            target,
            "ptr_preceding_latent",
            maximum=2,
        )
        reference = self._trim_ptr_context(
            ptr_reference_latent,
            ptr_reference_count,
            target,
            "ptr_reference_latent",
            maximum=self.ptr_max_references,
        )
        if cond_volume.ndim != 5:
            raise ValueError(
                f"{source} conditioning must be [B,C,T,H,W], got {tuple(cond_volume.shape)}"
            )
        expected_cond_shape = (
            target.shape[0],
            target.shape[2] + preceding.shape[2],
            target.shape[3],
            target.shape[4],
        )
        actual_cond_shape = (
            cond_volume.shape[0],
            cond_volume.shape[2],
            cond_volume.shape[3],
            cond_volume.shape[4],
        )
        if actual_cond_shape != expected_cond_shape:
            raise ValueError(
                f"{source} conditioning must follow [T,P] and match "
                f"[B,T,H,W]={expected_cond_shape}, got {actual_cond_shape}"
            )
        combined = torch.cat([reference, target, preceding], dim=2)
        patch_t, patch_h, patch_w = self.patch_size
        if (
            combined.shape[2] % patch_t
            or combined.shape[3] % patch_h
            or combined.shape[4] % patch_w
        ):
            raise ValueError(
                f"{source} combined [R,T,P] shape must be divisible by patch size: "
                f"shape={tuple(combined.shape[2:])}, patch={self.patch_size}"
            )
        return reference, target, preceding, combined

    @staticmethod
    def _slice_rotary_tokens(rotary_emb, start: int):
        if isinstance(rotary_emb, tuple):
            return tuple(value[:, start:] for value in rotary_emb)
        if isinstance(rotary_emb, list):
            return [value[:, start:] for value in rotary_emb]
        return rotary_emb[:, start:]

    def _ptr_rotary_emb(self, hidden_states: torch.Tensor, reference_frames: int):
        """Mirage-style RoPE: negative time and a separate height bank for R tokens."""
        wan = self._wan_model()
        if reference_frames <= 0:
            return wan.rope(hidden_states)
        rope = wan.rope
        required = ("freqs_cos", "freqs_sin", "t_dim", "h_dim", "w_dim")
        if any(not hasattr(rope, name) for name in required):
            raise RuntimeError(
                f"{PTR_TEMPORAL_MODE} requires diffusers WanRotaryPosEmbed buffers "
                f"{required}; active rope is {type(rope).__name__}"
            )

        _, _, frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        if reference_frames % patch_t:
            raise ValueError(
                f"reference frames {reference_frames} must be divisible by patch_t={patch_t}"
            )
        grid_t, grid_h, grid_w = frames // patch_t, height // patch_h, width // patch_w
        ref_t = reference_frames // patch_t
        regular_t = grid_t - ref_t
        split_sizes = [int(rope.t_dim), int(rope.h_dim), int(rope.w_dim)]
        cos_t, cos_h, cos_w = rope.freqs_cos.split(split_sizes, dim=1)
        sin_t, sin_h, sin_w = rope.freqs_sin.split(split_sizes, dim=1)
        if ref_t >= cos_t.shape[0] or regular_t > cos_t.shape[0]:
            raise ValueError(
                "PTR temporal RoPE exceeds Wan's frequency bank: "
                f"reference={ref_t}, regular={regular_t}, max={cos_t.shape[0]}"
            )
        if 2 * grid_h > cos_h.shape[0] or grid_w > cos_w.shape[0]:
            raise ValueError(
                "PTR separated spatial RoPE exceeds Wan's frequency bank: "
                f"height requires {2 * grid_h}, width requires {grid_w}, "
                f"max={(cos_h.shape[0], cos_w.shape[0])}"
            )

        def expand_bank(time, vertical, horizontal):
            time = time.view(-1, 1, 1, time.shape[-1]).expand(-1, grid_h, grid_w, -1)
            vertical = vertical.view(1, grid_h, 1, -1).expand(time.shape[0], -1, grid_w, -1)
            horizontal = horizontal.view(1, 1, grid_w, -1).expand(
                time.shape[0], grid_h, -1, -1
            )
            return torch.cat([time, vertical, horizontal], dim=-1).reshape(
                1, time.shape[0] * grid_h * grid_w, 1, -1
            )

        # R positions are [-R,...,-1]. cos(-x)=cos(x), sin(-x)=-sin(x).
        negative_indices = torch.arange(
            ref_t, 0, -1, device=cos_t.device, dtype=torch.long
        )
        ref_cos = expand_bank(
            cos_t.index_select(0, negative_indices),
            cos_h[grid_h : 2 * grid_h],
            cos_w[:grid_w],
        )
        ref_sin = expand_bank(
            -sin_t.index_select(0, negative_indices),
            sin_h[grid_h : 2 * grid_h],
            sin_w[:grid_w],
        )
        regular_cos = expand_bank(
            cos_t[:regular_t], cos_h[:grid_h], cos_w[:grid_w]
        )
        regular_sin = expand_bank(
            sin_t[:regular_t], sin_h[:grid_h], sin_w[:grid_w]
        )
        return (
            torch.cat([ref_cos, regular_cos], dim=1).to(hidden_states.device),
            torch.cat([ref_sin, regular_sin], dim=1).to(hidden_states.device),
        )

    # ------------------------------------------------------- transformer call
    def denoise(self, latent_noisy, timestep, encoder_hidden_states, cond_volume):
        """One transformer call producing the flow-matching velocity prediction.

        latent_noisy: (B, C, T, h, w); cond_volume: (B, Cin, T, h, w);
        timestep: (B,) or Wan patch-token timesteps (B, N).
        """
        wan = self._wan_model()
        batch_size, _, num_frames, height, width = latent_noisy.shape
        patch_t, patch_h, patch_w = self.patch_size
        post_patch_frames = num_frames // patch_t
        post_patch_height = height // patch_h
        post_patch_width = width // patch_w
        rotary_emb = wan.rope(latent_noisy)
        hidden_states = wan.patch_embedding(latent_noisy).flatten(2).transpose(1, 2)
        control_tokens, control_grid = self.side_branch.prepare_control_tokens(
            cond_volume.to(self.dtype), wan.patch_embedding
        )
        expected_grid = (post_patch_frames, post_patch_height, post_patch_width)
        if control_grid != expected_grid:
            raise ValueError(
                f"conditioning token grid {control_grid} differs from Wan grid "
                f"{expected_grid}"
            )
        control_embedding = self.side_branch.project_control_tokens(control_tokens)
        if hidden_states.shape != control_embedding.shape:
            raise ValueError(
                f"main/control token shapes differ: {tuple(hidden_states.shape)} vs "
                f"{tuple(control_embedding.shape)}"
            )
        control_hidden = hidden_states + control_embedding

        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            timestep_input = timestep.flatten()
        else:
            timestep_seq_len = None
            timestep_input = timestep
        temb, timestep_proj, side_encoder_hidden_states, image_hidden_states = (
            wan.condition_embedder(
                timestep_input,
                encoder_hidden_states,
                None,
                timestep_seq_len=timestep_seq_len,
            )
        )
        if timestep_seq_len is None:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        if image_hidden_states is not None:
            side_encoder_hidden_states = torch.cat(
                [image_hidden_states, side_encoder_hidden_states], dim=1
            )

        control_by_layer = {
            layer: control_index
            for control_index, layer in enumerate(self.control_layers)
        }
        for layer_index, block in enumerate(wan.blocks):
            control_index = control_by_layer.get(layer_index)
            residual = None
            if control_index is not None:
                control_hidden, residual = self.side_branch(
                    control_index,
                    control_hidden,
                    side_encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
            if torch.is_grad_enabled() and wan.gradient_checkpointing:
                hidden_states = wan._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    side_encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    side_encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
            if residual is not None:
                hidden_states = hidden_states + residual.to(hidden_states.dtype)

        if temb.ndim == 3:
            shift, scale = (
                wan.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (
                wan.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (
            wan.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).to(hidden_states.dtype)
        hidden_states = wan.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_frames,
            post_patch_height,
            post_patch_width,
            patch_t,
            patch_h,
            patch_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def denoise_ptr(
        self,
        latent_noisy,
        timestep,
        encoder_hidden_states,
        cond_volume,
        num_reference_frames: int,
    ):
        """Denoise combined ``[R,T,P]`` while spatial control covers only ``[T,P]``.

        ``timestep`` is always a Wan patch-token tensor ``[B,N]``. Reference tokens use
        a separate RoPE bank and receive an exactly-zero side-branch residual, preventing
        the same R observation from being conditioned once as a clean token and again as
        a spatial-memory projection.
        """
        self._require_ptr_mode("denoise_ptr")
        wan = self._wan_model()
        batch_size, _, num_frames, height, width = latent_noisy.shape
        patch_t, patch_h, patch_w = self.patch_size
        post_patch_frames = num_frames // patch_t
        post_patch_height = height // patch_h
        post_patch_width = width // patch_w
        spatial_tokens = post_patch_height * post_patch_width
        reference_tokens = (int(num_reference_frames) // patch_t) * spatial_tokens
        total_tokens = post_patch_frames * spatial_tokens
        if timestep.ndim != 2 or tuple(timestep.shape) != (batch_size, total_tokens):
            raise ValueError(
                "denoise_ptr timestep must be per Wan patch token: "
                f"actual={tuple(timestep.shape)}, expected={(batch_size, total_tokens)}"
            )

        rotary_emb = self._ptr_rotary_emb(latent_noisy, int(num_reference_frames))
        side_rotary_emb = self._slice_rotary_tokens(rotary_emb, reference_tokens)
        hidden_states = wan.patch_embedding(latent_noisy).flatten(2).transpose(1, 2)
        control_tokens, control_grid = self.side_branch.prepare_control_tokens(
            cond_volume.to(self.dtype), wan.patch_embedding
        )
        expected_control_grid = (
            post_patch_frames - int(num_reference_frames) // patch_t,
            post_patch_height,
            post_patch_width,
        )
        if control_grid != expected_control_grid:
            raise ValueError(
                f"PTR conditioning token grid {control_grid} differs from non-reference "
                f"[T,P] grid {expected_control_grid}"
            )
        control_embedding = self.side_branch.project_control_tokens(control_tokens)
        initial_control_tokens = hidden_states[:, reference_tokens:]
        if initial_control_tokens.shape != control_embedding.shape:
            raise ValueError(
                f"main/control token shapes differ: {tuple(initial_control_tokens.shape)} vs "
                f"{tuple(control_embedding.shape)}"
            )
        control_hidden = initial_control_tokens + control_embedding

        timestep_input = timestep.flatten()
        temb, timestep_proj, side_encoder_hidden_states, image_hidden_states = (
            wan.condition_embedder(
                timestep_input,
                encoder_hidden_states,
                None,
                timestep_seq_len=timestep.shape[1],
            )
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        if image_hidden_states is not None:
            side_encoder_hidden_states = torch.cat(
                [image_hidden_states, side_encoder_hidden_states], dim=1
            )
        side_timestep_proj = timestep_proj[:, reference_tokens:]

        control_by_layer = {
            layer: control_index
            for control_index, layer in enumerate(self.control_layers)
        }
        for layer_index, block in enumerate(wan.blocks):
            control_index = control_by_layer.get(layer_index)
            residual = None
            if control_index is not None:
                control_hidden, residual_non_reference = self.side_branch(
                    control_index,
                    control_hidden,
                    side_encoder_hidden_states,
                    side_timestep_proj,
                    side_rotary_emb,
                )
                if reference_tokens:
                    residual = torch.cat(
                        [
                            torch.zeros_like(hidden_states[:, :reference_tokens]),
                            residual_non_reference,
                        ],
                        dim=1,
                    )
                else:
                    residual = residual_non_reference
            if torch.is_grad_enabled() and wan.gradient_checkpointing:
                hidden_states = wan._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    side_encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    side_encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                )
            if residual is not None:
                hidden_states = hidden_states + residual.to(hidden_states.dtype)

        shift, scale = (
            wan.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_states.device)
        scale = scale.squeeze(2).to(hidden_states.device)
        hidden_states = (
            wan.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).to(hidden_states.dtype)
        hidden_states = wan.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_frames,
            post_patch_height,
            post_patch_width,
            patch_t,
            patch_h,
            patch_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def clear_training_residual_cache(self) -> None:
        """Compatibility no-op: chained residuals are computed in the block loop."""

    def _patch_token_timesteps(
        self,
        timestep: torch.Tensor,
        latent_shape,
        clean_prefix_len: int,
    ) -> torch.Tensor:
        """Expand shared or per-frame timesteps to temporal-major Wan patch tokens."""
        batch, _, frames, height, width = latent_shape
        patch_t, patch_h, patch_w = self.patch_size
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError(
                "latent shape must be divisible by Wan patch size: "
                f"latent={(frames, height, width)}, patch={self.patch_size}"
            )
        if timestep.ndim == 1:
            if timestep.shape[0] != batch:
                raise ValueError(
                    f"shared timestep batch differs: {timestep.shape[0]} != {batch}"
                )
            frame_timestep = timestep[:, None].expand(batch, frames).clone()
        elif timestep.ndim == 2:
            if tuple(timestep.shape) != (batch, frames):
                raise ValueError(
                    "per-frame timestep shape differs: "
                    f"actual={tuple(timestep.shape)}, expected={(batch, frames)}"
                )
            frame_timestep = timestep.clone()
        else:
            raise ValueError(
                f"timestep must be shaped [B] or [B,T], got {tuple(timestep.shape)}"
            )
        frame_timestep[:, :clean_prefix_len] = 0
        # Wan patch embedding flattens [T,H,W] in temporal-major order. A temporal
        # patch uses the timestep of its first latent frame, matching the existing
        # clean-prefix mask behavior for patch_t > 1.
        patch_time = frame_timestep[:, ::patch_t]
        spatial_tokens = (height // patch_h) * (width // patch_w)
        return patch_time.repeat_interleave(spatial_tokens, dim=1)

    @staticmethod
    def _validate_native_chunk_tensors(latent: torch.Tensor, cond_volume: torch.Tensor, source: str) -> None:
        if latent.ndim != 5 or cond_volume.ndim != 5:
            raise ValueError(
                f"{source} expects latent and conditioning shaped [B,C,T,H,W], got "
                f"{tuple(latent.shape)} and {tuple(cond_volume.shape)}"
            )
        validate_wan_chunk_contract(
            latent.shape[2], WAN_VAE_TEMPORAL_STRIDE, source=source
        )
        if cond_volume.shape[0] != latent.shape[0] or cond_volume.shape[2:] != latent.shape[2:]:
            raise ValueError(
                f"{source} latent/conditioning batch-time-spatial shapes differ: "
                f"latent={tuple(latent.shape)}, conditioning={tuple(cond_volume.shape)}"
            )

    # ----------------------------------------------------------- flow matching
    def forward(
        self,
        latent,
        encoder_hidden_states,
        cond_volume,
        sigma_min: float = 1e-4,
        prefix_override: Optional[torch.Tensor] = None,
        ptr_preceding_latent: Optional[torch.Tensor] = None,
        ptr_reference_latent: Optional[torch.Tensor] = None,
        ptr_preceding_count=None,
        ptr_reference_count=None,
        camera_motion_weight: Optional[torch.Tensor] = None,
        counterfactual_cond_volume: Optional[torch.Tensor] = None,
    ):
        """DDP-safe training entry point returning the flow-matching loss.

        Added after the 2026-07-10 DDP audit. Training must call the wrapped module as
        ``backbone(...)`` so DistributedDataParallel can install its gradient-reduction
        hooks. The previous direct ``unwrap_model(...).flow_matching_loss(...)`` call
        bypassed DDP forward and let each rank update independently.
        """
        if getattr(self, "temporal_mode", NATIVE_TEMPORAL_MODE) == PTR_TEMPORAL_MODE:
            return self.flow_matching_loss_ptr(
                latent,
                encoder_hidden_states,
                cond_volume,
                ptr_preceding_latent=ptr_preceding_latent,
                ptr_reference_latent=ptr_reference_latent,
                ptr_preceding_count=ptr_preceding_count,
                ptr_reference_count=ptr_reference_count,
                sigma_min=sigma_min,
                prefix_override=prefix_override,
                camera_motion_weight=camera_motion_weight,
                counterfactual_cond_volume=counterfactual_cond_volume,
            )
        if any(
            value is not None
            for value in (
                ptr_preceding_latent,
                ptr_reference_latent,
                ptr_preceding_count,
                ptr_reference_count,
            )
        ):
            raise ValueError(
                "PTR context was supplied to the native temporal path; set "
                f"temporal_mode={PTR_TEMPORAL_MODE!r} explicitly"
            )
        camera_kwargs = {}
        if camera_motion_weight is not None:
            camera_kwargs["camera_motion_weight"] = camera_motion_weight
        if counterfactual_cond_volume is not None:
            camera_kwargs["counterfactual_cond_volume"] = counterfactual_cond_volume
        return self.flow_matching_loss(
            latent,
            encoder_hidden_states,
            cond_volume,
            sigma_min=sigma_min,
            prefix_override=prefix_override,
            **camera_kwargs,
        )

    def flow_matching_loss(
        self,
        latent,
        encoder_hidden_states,
        cond_volume,
        sigma_min: float = 1e-4,
        prefix_override: Optional[torch.Tensor] = None,
        ptr_preceding_latent: Optional[torch.Tensor] = None,
        ptr_reference_latent: Optional[torch.Tensor] = None,
        ptr_preceding_count=None,
        ptr_reference_count=None,
        camera_motion_weight: Optional[torch.Tensor] = None,
        counterfactual_cond_volume: Optional[torch.Tensor] = None,
    ):
        """Rectified-flow / flow-matching loss used by Wan.

        x_t = (1 - sigma) * x1 + sigma * x0 with x1 the clean latent and x0 ~ N(0, I); the model
        predicts the velocity v = x0 - x1; loss = ||v_pred - (x0 - x1)||^2.
        """
        if getattr(self, "temporal_mode", NATIVE_TEMPORAL_MODE) == PTR_TEMPORAL_MODE:
            return self.flow_matching_loss_ptr(
                latent,
                encoder_hidden_states,
                cond_volume,
                ptr_preceding_latent=ptr_preceding_latent,
                ptr_reference_latent=ptr_reference_latent,
                ptr_preceding_count=ptr_preceding_count,
                ptr_reference_count=ptr_reference_count,
                sigma_min=sigma_min,
                prefix_override=prefix_override,
                camera_motion_weight=camera_motion_weight,
                counterfactual_cond_volume=counterfactual_cond_volume,
            )
        if any(
            value is not None
            for value in (
                ptr_preceding_latent,
                ptr_reference_latent,
                ptr_preceding_count,
                ptr_reference_count,
            )
        ):
            raise ValueError(
                "PTR context was supplied to native flow_matching_loss; set "
                f"temporal_mode={PTR_TEMPORAL_MODE!r} explicitly"
            )
        self._validate_native_chunk_tensors(latent, cond_volume, "flow-matching training chunk")
        B = latent.shape[0]
        x1 = latent
        x0 = torch.randn_like(x1)
        clean_prefix_len = int(getattr(self, "clean_prefix_len", 0))
        per_target = getattr(self, "timestep_sampling", "uniform") == "uniform_per_target"
        t_shape = (B, x1.shape[2]) if per_target else (B,)
        t = self._sample_flow_t(t_shape, device=x1.device, sigma_min=sigma_min)
        shift = float(self.scheduler.config.flow_shift)
        sigma = shift * t / (1 + (shift - 1) * t)
        if per_target:
            sigma = sigma.clone()
            sigma[:, :clean_prefix_len] = 0
            sigma_b = sigma[:, None, :, None, None]
        else:
            sigma_b = sigma.view(B, *([1] * (x1.dim() - 1)))
        x_t = (1 - sigma_b) * x1 + sigma_b * x0
        target = x0 - x1
        if per_target:
            # Mirage conditions each target position with its continuous shifted sigma.
            timestep = sigma * self.scheduler.config.num_train_timesteps
        else:
            # Preserve the historical shared-timestep contract for old configurations.
            timestep = (
                sigma * self.scheduler.config.num_train_timesteps
            ).round().long().clamp(0, self.scheduler.config.num_train_timesteps - 1)
        if clean_prefix_len:
            x_t = x_t.clone()
            # Scheduled sampling (P2): continue from a backbone-generated prefix instead of
            # the GT clean prefix so training sees the same slightly-off appearance the
            # rollout feeds back. Loss still targets the GT generated latents only.
            if prefix_override is not None:
                prefix = prefix_override.to(x_t)[:, :, :clean_prefix_len]
            else:
                prefix = x1[:, :, :clean_prefix_len]
            x_t[:, :, :clean_prefix_len] = prefix
        if clean_prefix_len or timestep.ndim == 2:
            timestep = self._patch_token_timesteps(timestep, x1.shape, clean_prefix_len)
        v_pred = self.denoise(x_t.to(self.dtype), timestep, encoder_hidden_states, cond_volume)
        squared_error = (v_pred.float() - target.float()) ** 2
        if clean_prefix_len:
            squared_error = squared_error[:, :, clean_prefix_len:]
        negative_squared_error = None
        if getattr(self, "counterfactual_loss_weight", 0.0) > 0.0:
            self._validate_counterfactual_condition(cond_volume, counterfactual_cond_volume)
            negative_prediction = self.denoise(
                x_t.to(self.dtype),
                timestep,
                encoder_hidden_states,
                counterfactual_cond_volume,
            )
            negative_squared_error = (
                negative_prediction.float() - target.float()
            ).square()
            if clean_prefix_len:
                negative_squared_error = negative_squared_error[:, :, clean_prefix_len:]
        motion_weight = self._prepare_camera_motion_weight(
            camera_motion_weight,
            squared_error,
            clean_prefix_len=clean_prefix_len,
        )
        loss, objective_diagnostics = self._camera_objective(
            squared_error,
            motion_weight,
            negative_squared_error,
        )
        self._record_flow_diagnostics(squared_error, sigma, cond_volume, clean_prefix_len)
        self.last_flow_diagnostics.update(objective_diagnostics)
        return loss

    def flow_matching_loss_ptr(
        self,
        latent,
        encoder_hidden_states,
        cond_volume,
        ptr_preceding_latent: Optional[torch.Tensor],
        ptr_reference_latent: Optional[torch.Tensor],
        ptr_preceding_count=None,
        ptr_reference_count=None,
        sigma_min: float = 1e-4,
        prefix_override: Optional[torch.Tensor] = None,
        camera_motion_weight: Optional[torch.Tensor] = None,
        counterfactual_cond_volume: Optional[torch.Tensor] = None,
    ):
        """Mirage-style PTR objective with clean ``R/P/T0`` and loss only on ``T1..T8``."""
        reference, target, preceding, _ = self._prepare_ptr_inputs(
            latent,
            cond_volume,
            ptr_preceding_latent,
            ptr_reference_latent,
            ptr_preceding_count,
            ptr_reference_count,
            source="PTR flow-matching",
        )
        batch = target.shape[0]
        target_frames = target.shape[2]
        if target_frames != PTR_TARGET_LATENT_FRAMES:
            raise ValueError(
                f"PTR target must contain {PTR_TARGET_LATENT_FRAMES} latent frames, "
                f"got {target_frames}"
            )

        noise = torch.randn_like(target)
        per_target = getattr(self, "timestep_sampling", "uniform") == "uniform_per_target"
        sample_shape = (batch, target_frames) if per_target else (batch,)
        sampled_t = self._sample_flow_t(
            sample_shape, device=target.device, sigma_min=sigma_min
        )
        shift = float(self.scheduler.config.flow_shift)
        sigma = shift * sampled_t / (1 + (shift - 1) * sampled_t)
        if sigma.ndim == 1:
            sigma = sigma[:, None].expand(batch, target_frames).clone()
        else:
            sigma = sigma.clone()
        sigma[:, 0] = 0
        sigma_b = sigma[:, None, :, None, None]
        noisy_target = (1 - sigma_b) * target + sigma_b * noise
        if prefix_override is not None:
            if (
                prefix_override.ndim != 5
                or prefix_override.shape[0] != batch
                or prefix_override.shape[1] != target.shape[1]
                or prefix_override.shape[2] < 1
                or prefix_override.shape[3:] != target.shape[3:]
            ):
                raise ValueError(
                    "PTR prefix_override must provide at least T0 with matching "
                    f"[B,C,H,W], got {tuple(prefix_override.shape)}"
                )
            noisy_target = noisy_target.clone()
            noisy_target[:, :, :1] = prefix_override.to(noisy_target)[:, :, :1]
        else:
            noisy_target[:, :, :1] = target[:, :, :1]

        combined_noisy = torch.cat([reference, noisy_target, preceding], dim=2)
        zeros_reference = sigma.new_zeros((batch, reference.shape[2]))
        zeros_preceding = sigma.new_zeros((batch, preceding.shape[2]))
        frame_timestep = torch.cat(
            [
                zeros_reference,
                sigma * self.scheduler.config.num_train_timesteps,
                zeros_preceding,
            ],
            dim=1,
        )
        token_timestep = self._patch_token_timesteps(
            frame_timestep, combined_noisy.shape, clean_prefix_len=0
        )
        flow_pred = self.denoise_ptr(
            combined_noisy.to(self.dtype),
            token_timestep,
            encoder_hidden_states,
            cond_volume,
            num_reference_frames=reference.shape[2],
        )
        target_start = reference.shape[2]
        flow_pred_target = flow_pred[
            :, :, target_start + 1 : target_start + target_frames
        ]
        flow_target = (noise - target)[:, :, 1:]
        squared_error = (flow_pred_target.float() - flow_target.float()) ** 2
        negative_squared_error = None
        if getattr(self, "counterfactual_loss_weight", 0.0) > 0.0:
            self._validate_counterfactual_condition(cond_volume, counterfactual_cond_volume)
            negative_flow_pred = self.denoise_ptr(
                combined_noisy.to(self.dtype),
                token_timestep,
                encoder_hidden_states,
                counterfactual_cond_volume,
                num_reference_frames=reference.shape[2],
            )
            negative_flow_pred_target = negative_flow_pred[
                :, :, target_start + 1 : target_start + target_frames
            ]
            negative_squared_error = (
                negative_flow_pred_target.float() - flow_target.float()
            ).square()
        motion_weight = self._prepare_camera_motion_weight(
            camera_motion_weight,
            squared_error,
            clean_prefix_len=1,
        )
        loss, objective_diagnostics = self._camera_objective(
            squared_error,
            motion_weight,
            negative_squared_error,
        )
        self._record_flow_diagnostics(
            squared_error,
            sigma,
            cond_volume[:, :, :target_frames],
            clean_prefix_len=1,
        )
        self.last_flow_diagnostics.update(objective_diagnostics)
        return loss

    def _validate_counterfactual_condition(
        self,
        cond_volume: torch.Tensor,
        counterfactual_cond_volume: Optional[torch.Tensor],
    ) -> None:
        if counterfactual_cond_volume is None:
            raise ValueError(
                "counterfactual camera loss is enabled but the batch has no "
                "counterfactual_cond_volume"
            )
        if counterfactual_cond_volume.shape != cond_volume.shape:
            raise ValueError(
                "counterfactual conditioning must match the positive conditioning: "
                f"positive={tuple(cond_volume.shape)}, "
                f"counterfactual={tuple(counterfactual_cond_volume.shape)}"
            )

    def _prepare_camera_motion_weight(
        self,
        camera_motion_weight: Optional[torch.Tensor],
        squared_error: torch.Tensor,
        *,
        clean_prefix_len: int,
    ) -> Optional[torch.Tensor]:
        if camera_motion_weight is None:
            if getattr(self, "motion_loss_alpha", 0.0) > 0.0:
                raise ValueError(
                    "motion-sensitive flow loss is enabled but the batch has no "
                    "camera_motion_weight"
                )
            return None
        if camera_motion_weight.ndim != 4:
            raise ValueError(
                "camera_motion_weight must be [B,T,H,W], got "
                f"{tuple(camera_motion_weight.shape)}"
            )
        weight = camera_motion_weight[:, int(clean_prefix_len) :]
        expected = (
            squared_error.shape[0],
            squared_error.shape[2],
            squared_error.shape[3],
            squared_error.shape[4],
        )
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"camera motion map differs from supervised flow grid: "
                f"actual={tuple(weight.shape)}, expected={expected}"
            )
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("camera motion weights must be finite and non-negative")
        return weight.to(device=squared_error.device, dtype=torch.float32)

    def _camera_objective(
        self,
        positive_squared_error: torch.Tensor,
        motion_weight: Optional[torch.Tensor],
        negative_squared_error: Optional[torch.Tensor],
    ):
        per_cell_positive = positive_squared_error.mean(dim=1)
        if motion_weight is None:
            weight = torch.ones_like(per_cell_positive)
        else:
            weight = 1.0 + getattr(self, "motion_loss_alpha", 0.0) * motion_weight
        denominator = weight.flatten(1).sum(dim=1).clamp_min(1.0)
        positive_error = (
            per_cell_positive * weight
        ).flatten(1).sum(dim=1) / denominator
        flow_loss = positive_error.mean()

        if negative_squared_error is None:
            ranking_loss = flow_loss.new_zeros(())
            negative_error = positive_error.detach().new_full(
                positive_error.shape, float("nan")
            )
        else:
            if negative_squared_error.shape != positive_squared_error.shape:
                raise ValueError("positive and counterfactual flow errors must have equal shape")
            per_cell_negative = negative_squared_error.mean(dim=1)
            negative_error = (
                per_cell_negative * weight
            ).flatten(1).sum(dim=1) / denominator
            ranking_loss = torch.relu(
                getattr(self, "counterfactual_loss_margin", 0.01)
                + positive_error
                - negative_error
            ).mean()
        total = flow_loss + getattr(self, "counterfactual_loss_weight", 0.0) * ranking_loss
        diagnostics = {
            "base_flow_loss": positive_squared_error.mean().detach(),
            "motion_weighted_flow_loss": flow_loss.detach(),
            "counterfactual_positive_error": positive_error.mean().detach(),
            "counterfactual_negative_error": negative_error.mean().detach(),
            "counterfactual_ranking_loss": ranking_loss.detach(),
            "camera_objective_total": total.detach(),
        }
        return total, diagnostics

    def _sample_flow_t(self, shape, device, sigma_min: float) -> torch.Tensor:
        """Sample the flow-matching interpolation position t in (sigma_min, 1].

        ``uniform`` reproduces the earlier runs. ``logit_normal`` draws t = sigmoid(n),
        n ~ N(mean, std), concentrating samples in the mid-noise band that flow-matching
        video models (SD3 / Wan) learn from most efficiently.
        """
        if isinstance(shape, int):
            shape = (shape,)
        if getattr(self, "timestep_sampling", "uniform") == "logit_normal":
            noise = torch.randn(shape, device=device, dtype=torch.float32)
            t = torch.sigmoid(self.logit_normal_mean + self.logit_normal_std * noise)
            return t.clamp(sigma_min, 1.0)
        return torch.rand(shape, device=device, dtype=torch.float32).clamp(sigma_min, 1.0)

    @torch.no_grad()
    def _record_flow_diagnostics(self, squared_error, sigma, cond_volume, clean_prefix_len):
        """Stash per-microbatch sigma / covered-vs-uncovered loss for the trainer logger."""
        per_cell = squared_error.mean(dim=1)                     # (B, T', h, w)
        if sigma.ndim == 2:
            noisy_sigma = sigma[:, clean_prefix_len:]
            position_loss = per_cell.flatten(2).mean(dim=2)      # (B, T')
            if position_loss.shape != noisy_sigma.shape:
                raise ValueError(
                    "per-target diagnostic shape differs: "
                    f"sigma={tuple(noisy_sigma.shape)}, loss={tuple(position_loss.shape)}"
                )
            diagnostic_sigma = noisy_sigma
            diagnostic_loss = position_loss
            granularity = "noisy_position"
        else:
            diagnostic_sigma = sigma
            diagnostic_loss = per_cell.flatten(1).mean(dim=1)    # (B,)
            granularity = "sample"
        try:
            covered = conditioning_coverage_mask(
                cond_volume, self.top_k, self.latent_channels, self.composite_cond
            )
            if clean_prefix_len:
                covered = covered[:, clean_prefix_len:]
            covered = covered.to(per_cell.device)
            cov_sum = (per_cell * covered).sum()
            cov_cnt = covered.sum().to(per_cell.dtype)
            unc = ~covered
            unc_sum = (per_cell * unc).sum()
            unc_cnt = unc.sum().to(per_cell.dtype)
        except Exception:
            cov_sum = per_cell.new_zeros(())
            cov_cnt = per_cell.new_zeros(())
            unc_sum = per_cell.new_zeros(())
            unc_cnt = per_cell.new_zeros(())
        self.last_flow_diagnostics = {
            "sigma": diagnostic_sigma.detach(),
            "loss": diagnostic_loss.detach(),
            "granularity": granularity,
            "covered_sum": cov_sum.detach(),
            "covered_count": cov_cnt.detach(),
            "uncovered_sum": unc_sum.detach(),
            "uncovered_count": unc_cnt.detach(),
        }

    # ----------------------------------------------------------------- vae io
    @torch.no_grad()
    def encode(self, frames):
        """frames: (B, C=3, T, H, W) in [-1, 1] -> latent (B, Cz, t, h, w)."""
        # 2026-07-14 latent-contract fix: all VAE crossings use the canonical
        # posterior-mode + channel-statistics conversion shared with precompute.
        frames = frames.to(device=self.device, dtype=self.vae.dtype).contiguous()
        return encode_wan_latents(self.vae, frames)

    @torch.no_grad()
    def decode(self, latent):
        # Generated latents live in Wan transformer space; restore raw VAE
        # channel statistics before decoding pixels.
        return decode_wan_latents(self.vae, latent)

    def reset_decode_stream(self) -> None:
        """Start a new scene while keeping VAE causal state between its chunks."""
        self._stream_decoder.reset()

    @torch.no_grad()
    def decode_stream(self, latent):
        """Decode only new latent frames using the current scene's causal VAE cache."""
        return self._stream_decoder.decode(latent)

    # ----------------------------------------------------------- chunk sampling
    @torch.no_grad()
    def sample_chunk(self, encoder_hidden_states, cond_volume, shape, num_steps: int = 40,
                     guidance_scale: float = 1.0, neg_encoder_hidden_states=None,
                     ref_latent: Optional[torch.Tensor] = None, ref_len: int = 0, generator=None):
        """Sample one latent chunk conditioned on memory (cond_volume) via the scheduler.

        shape: (B, Cz, T, h, w). Returns the clean latent estimate. idea-1 uses UniPC 40
        steps with CFG disabled (guidance_scale=1.0); CFG is only applied when a negative
        embedding is supplied and guidance_scale>1.

        TI2V / autoregressive conditioning: if ``ref_latent``
        (B, Cz, ref_len, h, w) is given, the first ``ref_len`` temporal latents are held
        fixed and their Wan patch tokens use timestep zero throughout sampling.
        """
        if len(shape) != 5:
            raise ValueError(f"sample_chunk shape must be [B,C,T,H,W], got {shape}")
        validate_wan_chunk_contract(
            shape[2], WAN_VAE_TEMPORAL_STRIDE, source="inference sampling chunk"
        )
        if cond_volume.ndim != 5 or cond_volume.shape[0] != shape[0] or cond_volume.shape[2:] != shape[2:]:
            raise ValueError(
                f"sample_chunk conditioning must match [B,T,H,W]={shape[0], shape[2], shape[3], shape[4]}, "
                f"got {tuple(cond_volume.shape)}"
            )
        self.scheduler.set_timesteps(num_steps, device=self.device)
        latent = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)
        ref_timestep_mask = None
        if ref_latent is not None and ref_len > 0:
            latent[:, :, :ref_len] = ref_latent.to(latent)
            ref_timestep_mask = True
        use_cfg = bool(guidance_scale and guidance_scale > 1.0 and neg_encoder_hidden_states is not None)
        v_last = None
        for t in self.scheduler.timesteps:
            if ref_timestep_mask is None:
                tb = t.expand(shape[0]).to(self.device)
            else:
                tb = self._patch_token_timesteps(
                    t.expand(shape[0]).to(self.device),
                    shape,
                    ref_len,
                )
            v = self.denoise(latent, tb, encoder_hidden_states, cond_volume)
            if use_cfg:
                v_uncond = self.denoise(latent, tb, neg_encoder_hidden_states, cond_volume)
                v = v_uncond + guidance_scale * (v - v_uncond)
            v_last = v
            latent = self.scheduler.step(v, t, latent, return_dict=False)[0]
            if ref_latent is not None and ref_len > 0:
                latent[:, :, :ref_len] = ref_latent.to(latent)
        # Generation-quality proxy (Sec 3.2): per-cell L2 of the final-step velocity update.
        # A large remaining update at the lowest noise level == a less-settled, less
        # confident token. Shape (T, h, w) over the chunk's latent frames.
        if v_last is not None:
            self.last_generation_residual = torch.linalg.vector_norm(
                v_last.float(), dim=1)[0].detach()
        else:
            self.last_generation_residual = None
        return latent

    @torch.no_grad()
    def sample_ptr(
        self,
        encoder_hidden_states,
        cond_volume,
        shape,
        target_prefix_latent: torch.Tensor,
        ptr_preceding_latent: Optional[torch.Tensor] = None,
        ptr_reference_latent: Optional[torch.Tensor] = None,
        ptr_preceding_count=None,
        ptr_reference_count=None,
        num_steps: int = 40,
        guidance_scale: float = 1.0,
        neg_encoder_hidden_states=None,
        generator=None,
    ):
        """Sample PTR target ``T`` while holding clean ``R/P/T0`` fixed.

        The transformer sees ``[R,T,P]`` on every denoising step, but the scheduler only
        updates the nine-frame target tensor. The return value is therefore always T9;
        clean reference and preceding context never leak into the rollout output.
        """
        self._require_ptr_mode("sample_ptr")
        if len(shape) != 5:
            raise ValueError(f"sample_ptr shape must be [B,C,T,H,W], got {shape}")
        validate_wan_chunk_contract(
            shape[2], WAN_VAE_TEMPORAL_STRIDE, source="PTR inference target"
        )
        if shape[2] != PTR_TARGET_LATENT_FRAMES:
            raise ValueError(
                f"sample_ptr target must contain {PTR_TARGET_LATENT_FRAMES} latent "
                f"frames, got {shape[2]}"
            )
        if (
            target_prefix_latent.ndim != 5
            or target_prefix_latent.shape[0] != shape[0]
            or target_prefix_latent.shape[1] != shape[1]
            or target_prefix_latent.shape[2] < 1
            or target_prefix_latent.shape[3:] != shape[3:]
        ):
            raise ValueError(
                "target_prefix_latent must contain T0 with matching [B,C,H,W]: "
                f"target shape={shape}, prefix={tuple(target_prefix_latent.shape)}"
            )

        target = torch.randn(
            shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        target_prefix = target_prefix_latent.to(target)[:, :, :1]
        target[:, :, :1] = target_prefix
        reference, _, preceding, _ = self._prepare_ptr_inputs(
            target,
            cond_volume,
            ptr_preceding_latent,
            ptr_reference_latent,
            ptr_preceding_count,
            ptr_reference_count,
            source="PTR inference",
        )

        self.scheduler.set_timesteps(num_steps, device=self.device)
        use_cfg = bool(
            guidance_scale
            and guidance_scale > 1.0
            and neg_encoder_hidden_states is not None
        )
        target_velocity_last = None
        for scheduler_timestep in self.scheduler.timesteps:
            combined = torch.cat([reference.to(target), target, preceding.to(target)], dim=2)
            frame_timestep = torch.zeros(
                (shape[0], combined.shape[2]),
                device=self.device,
                dtype=scheduler_timestep.dtype,
            )
            target_start = reference.shape[2]
            frame_timestep[:, target_start + 1 : target_start + shape[2]] = scheduler_timestep
            token_timestep = self._patch_token_timesteps(
                frame_timestep, combined.shape, clean_prefix_len=0
            )
            velocity = self.denoise_ptr(
                combined,
                token_timestep,
                encoder_hidden_states,
                cond_volume,
                num_reference_frames=reference.shape[2],
            )
            if use_cfg:
                velocity_uncond = self.denoise_ptr(
                    combined,
                    token_timestep,
                    neg_encoder_hidden_states,
                    cond_volume,
                    num_reference_frames=reference.shape[2],
                )
                velocity = velocity_uncond + guidance_scale * (
                    velocity - velocity_uncond
                )
            target_velocity = velocity[
                :, :, target_start : target_start + shape[2]
            ]
            target_velocity_last = target_velocity
            target = self.scheduler.step(
                target_velocity,
                scheduler_timestep,
                target,
                return_dict=False,
            )[0]
            target[:, :, :1] = target_prefix

        if target_velocity_last is not None:
            self.last_generation_residual = torch.linalg.vector_norm(
                target_velocity_last.float(), dim=1
            )[0].detach()
        else:
            self.last_generation_residual = None
        return target

    # ---------------------------------------------------------------- utilities
    def enable_gradient_checkpointing(self):
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        self.side_branch.enable_gradient_checkpointing()

    def load_lora(
        self,
        path: str,
        latent_stats_path: str,
        expected_model_contract=None,
        expected_camera_objective_contract=None,
    ):
        """Load LoRA adapter weights saved by ``transformer.save_pretrained`` (Stage 2 resume)."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        if expected_model_contract is None:
            expected_model_contract = wan_checkpoint_model_contract(self)
        validation_kwargs = {"expected_model_contract": expected_model_contract}
        if expected_camera_objective_contract is not None:
            validation_kwargs["expected_camera_objective_contract"] = (
                expected_camera_objective_contract
            )
        checkpoint_metadata = validate_lora_checkpoint(
            path,
            self.vae,
            latent_stats_path,
            **validation_kwargs,
        )
        weights_file = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(weights_file):
            state = load_file(weights_file)
        else:
            state = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu")
        set_peft_model_state_dict(self.transformer, state)
        return checkpoint_metadata

    def freeze_side_branch(self) -> None:
        """Keep the Stage-1 conditioning branch fixed during LoRA training."""
        self.side_branch.requires_grad_(False)
        self.side_branch.eval()
        self._side_branch_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_side_branch_frozen", False):
            self.side_branch.eval()
        return self

    def trainable_parameters(self):
        params = [p for p in self.side_branch.parameters() if p.requires_grad]
        params += [p for p in self.transformer.parameters() if p.requires_grad]
        return params
