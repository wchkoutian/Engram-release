"""Wan/VACE-style side branch for Engram memory conditioning."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ControlNetSideBranch(nn.Module):
    """Chained Wan/VACE side blocks with explicit initialization provenance.

    Legacy runs copy Wan blocks and patch weights.  V10 uses newly constructed Wan blocks
    plus independent feature projections, while retaining the Engram Top-K/metadata/ray
    input representation.  In both modes only the residual projections into the frozen
    Wan trunk are zero initialized.
    """

    architecture = "wan_chained_vace_topk_patch_v4"
    confidence_channel_transform = "precision_over_one_plus_precision_v1"

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hidden: int,
        source_blocks: Sequence[nn.Module],
        source_patch_embedding: nn.Module,
        top_k: int = 1,
        composite_cond: bool = False,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        camera_ray_channels: int = 0,
        initialization: str = "wan_copy",
        block_init_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__()
        if in_channels < latent_channels:
            raise ValueError(
                f"conditioning channels {in_channels} cannot contain "
                f"{latent_channels} Wan latent channels"
            )
        if not source_blocks:
            raise ValueError("the VACE side branch requires at least one Wan source block")

        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        self.composite_cond = bool(composite_cond)
        self.camera_ray_channels = int(camera_ray_channels)
        if self.camera_ray_channels < 0:
            raise ValueError(f"camera_ray_channels must be >= 0, got {self.camera_ray_channels}")
        self.hidden = int(hidden)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.initialization = str(initialization)
        if self.initialization not in ("wan_copy", "scratch"):
            raise ValueError(
                "side-branch initialization must be 'wan_copy' or 'scratch', got "
                f"{self.initialization!r}"
            )
        self.gradient_checkpointing = False
        self.architecture = "wan_chained_vace_topk_patch_v4"

        base_channels = (
            self.latent_channels + 1
            if self.composite_cond
            else self.top_k * (self.latent_channels + 3)
        )
        expected_channels = base_channels + self.camera_ray_channels
        if self.in_channels != expected_channels:
            raise ValueError(
                "conditioning layout differs from the side-branch contract: "
                f"actual={self.in_channels}, expected={expected_channels}, "
                f"top_k={self.top_k}, composite={self.composite_cond}, "
                f"camera_ray_channels={self.camera_ray_channels}"
            )

        if self.initialization == "scratch":
            if block_init_kwargs is None:
                raise ValueError("scratch side-branch initialization requires block_init_kwargs")
            block_type = type(source_blocks[0])
            if any(type(block) is not block_type for block in source_blocks):
                raise ValueError("all Wan source blocks must share one concrete class")
            self.blocks = nn.ModuleList(
                [block_type(**dict(block_init_kwargs)) for _ in source_blocks]
            )
            self.architecture = (
                f"wan_chained_vace{len(source_blocks)}_scratch_topk_patch_v5"
            )
        else:
            self.blocks = nn.ModuleList([copy.deepcopy(block) for block in source_blocks])
        for parameter in self.blocks.parameters():
            parameter.requires_grad_(True)

        self.before_proj = nn.Linear(self.hidden, self.hidden)
        self.zero_out = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in self.blocks]
        )
        for projection in self.zero_out:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        additional_feature_slots = 0 if self.composite_cond else self.top_k - 1
        if self.initialization == "scratch":
            self.primary_feature_patch = nn.Conv3d(
                self.latent_channels,
                self.hidden,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.additional_feature_patch = nn.ModuleList(
                [
                    nn.Conv3d(
                        self.latent_channels,
                        self.hidden,
                        kernel_size=self.patch_size,
                        stride=self.patch_size,
                    )
                    for _ in range(additional_feature_slots)
                ]
            )
        else:
            self.additional_feature_patch = nn.ModuleList(
                [copy.deepcopy(source_patch_embedding) for _ in range(additional_feature_slots)]
            )
        for parameter in self.additional_feature_patch.parameters():
            parameter.requires_grad_(True)

        metadata_channels = 1 if self.composite_cond else self.top_k * 3
        self.metadata_patch = nn.Conv3d(
            metadata_channels,
            self.hidden,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        nn.init.xavier_uniform_(self.metadata_patch.weight)
        nn.init.zeros_(self.metadata_patch.bias)

        # Explicit camera (Plucker) patch embedding: a trainable input projection (not a
        # zero-init residual) so camera control reaches the frozen trunk from step one.
        if self.camera_ray_channels:
            self.ray_patch = nn.Conv3d(
                self.camera_ray_channels,
                self.hidden,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            nn.init.xavier_uniform_(self.ray_patch.weight)
            nn.init.zeros_(self.ray_patch.bias)
            self.architecture += "_camray"

    def _split_conditioning(
        self, cond_volume: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        rays = None
        base = cond_volume
        if self.camera_ray_channels:
            rays = cond_volume[:, -self.camera_ray_channels :]
            base = cond_volume[:, : -self.camera_ray_channels]

        if self.composite_cond:
            return (
                [base[:, : self.latent_channels]],
                base[:, self.latent_channels : self.latent_channels + 1],
                rays,
            )

        features = []
        metadata = []
        stride = self.latent_channels + 3
        for slot in range(self.top_k):
            start = slot * stride
            features.append(base[:, start : start + self.latent_channels])
            metadata.append(
                base[
                    :,
                    start + self.latent_channels : start + self.latent_channels + 3,
                ]
            )
        return features, torch.cat(metadata, dim=1), rays

    def prepare_control_tokens(
        self,
        cond_volume: torch.Tensor,
        shared_patch_embedding: nn.Module,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Patchify top-K latent features and their metadata on Wan's token grid."""
        if cond_volume.ndim != 5 or cond_volume.shape[1] != self.in_channels:
            raise ValueError(
                f"conditioning must have shape [B,{self.in_channels},T,H,W], got "
                f"{tuple(cond_volume.shape)}"
            )

        features, metadata, rays = self._split_conditioning(cond_volume)
        primary_patch = (
            self.primary_feature_patch
            if self.initialization == "scratch"
            else shared_patch_embedding
        )
        feature_tokens = [primary_patch(features[0])]
        feature_tokens.extend(
            projection(feature)
            for projection, feature in zip(
                self.additional_feature_patch, features[1:]
            )
        )
        primary = torch.stack(feature_tokens, dim=0).sum(dim=0)
        primary = primary + self.metadata_patch(metadata)
        if self.camera_ray_channels:
            primary = primary + self.ray_patch(rays)
        _, _, t, h, w = primary.shape
        return primary.flatten(2).transpose(1, 2), (t, h, w)

    def _run_block(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            def forward(value):
                return block(value, encoder_hidden_states, timestep_proj, rotary_emb)

            return checkpoint(forward, hidden_states, use_reentrant=False)
        return block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    def project_control_tokens(self, control_tokens: torch.Tensor) -> torch.Tensor:
        """Project the memory tokens before they enter the continuous control stream."""
        return self.before_proj(control_tokens)

    def forward(
        self,
        control_index: int,
        control_hidden: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the continuous control state and emit its residual hint."""
        control_index = int(control_index)
        if control_index < 0 or control_index >= len(self.blocks):
            raise IndexError(
                f"control block index {control_index} is outside [0,{len(self.blocks)})"
            )

        control_hidden = self._run_block(
            self.blocks[control_index],
            control_hidden,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
        )
        return control_hidden, self.zero_out[control_index](control_hidden)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
