"""Contract metadata for Engram side-branch and LoRA checkpoints."""
from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, Optional

import torch

from ..geometry.projection import (
    LEGACY_CAMERA_GEOMETRY_CONTRACT,
    validate_camera_geometry_contract,
)
from ..memory.policy import MemoryFusionPolicy
from .wan_latents import (
    validate_wan_latent_contract,
    wan_latent_contract,
    wan_temporal_chunk_contract,
)


CHECKPOINT_CONTRACT_VERSION = 6
WAN_FLOW_CONTRACT_NAME = "wan_flow_prediction_v1"
LORA_METADATA_NAME = "engram_checkpoint_meta.json"
CALIBRATOR_CONTRACT_VERSION = 1
LEGACY_TEXT_MAX_SEQUENCE_LENGTH = 226
NATIVE_TEMPORAL_MODE = "native_chunk_v1"
PTR_TEMPORAL_MODE = "ptr_clean_tokens_v1"
PTR_REFERENCE_POLICY = "all_non_pt_top4"
PTR_SELECTION_ALGORITHM = "latent_anchor_voxel_iou_non_pt_top4_v1"
PTR_MAX_REFERENCES = 4
_PTR_LEGACY_REFERENCE_POLICY = "causal_single_best"
_PTR_LEGACY_SELECTION_ALGORITHM = "latent_anchor_voxel_iou_causal_single_best_v1"
_TEMPORAL_MODES = {NATIVE_TEMPORAL_MODE, PTR_TEMPORAL_MODE}
_MODEL_RESOURCE_FINGERPRINT_CACHE: Dict[Any, str] = {}


def side_branch_metadata_path(checkpoint_path: str) -> str:
    stem, _ = os.path.splitext(checkpoint_path)
    return stem + ".meta.json"


def lora_metadata_path(lora_path: str) -> str:
    return os.path.join(lora_path, LORA_METADATA_NAME)


def checkpoint_text_max_sequence_length(metadata: Optional[Dict[str, Any]]) -> int:
    """Return the training text length, preserving the legacy Wan default."""
    if not isinstance(metadata, dict):
        return LEGACY_TEXT_MAX_SEQUENCE_LENGTH
    contract = metadata.get("conditioning_contract")
    if not isinstance(contract, dict) or "text_max_sequence_length" not in contract:
        return LEGACY_TEXT_MAX_SEQUENCE_LENGTH
    value = int(contract["text_max_sequence_length"])
    if value <= 0:
        raise ValueError(f"checkpoint text_max_sequence_length must be positive, got {value}")
    return value


def checkpoint_temporal_mode(metadata: Optional[Dict[str, Any]]) -> str:
    """Return the clean-token layout, preserving native semantics for older runs."""
    contract = metadata.get("conditioning_contract") if isinstance(metadata, dict) else None
    value = (
        contract.get("temporal_mode", NATIVE_TEMPORAL_MODE)
        if isinstance(contract, dict)
        else NATIVE_TEMPORAL_MODE
    )
    value = str(value)
    if value not in _TEMPORAL_MODES:
        raise ValueError(f"checkpoint has unsupported temporal_mode {value!r}")
    return value


def validate_checkpoint_temporal_mode(
    metadata: Optional[Dict[str, Any]], expected: str, artifact: str
) -> str:
    """Reject same-shape weights trained with a different temporal token contract."""
    expected = str(expected)
    if expected not in _TEMPORAL_MODES:
        raise ValueError(f"unsupported evaluation temporal_mode {expected!r}")
    actual = checkpoint_temporal_mode(metadata)
    if actual != expected:
        raise ValueError(
            f"{artifact} temporal mode differs: actual={actual!r}, expected={expected!r}"
        )
    return actual


def checkpoint_ptr_runtime_contract(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract and validate the inference-relevant fields of ``ptr_clean_tokens_v1``."""
    if checkpoint_temporal_mode(metadata) != PTR_TEMPORAL_MODE:
        return {}
    conditioning = metadata.get("conditioning_contract")
    if not isinstance(conditioning, dict):
        raise ValueError("PTR checkpoint has no conditioning_contract")
    index = conditioning.get("ptr_index_contract")
    reference = conditioning.get("ptr_reference_contract")
    if not isinstance(index, dict) or not isinstance(reference, dict):
        raise ValueError("PTR checkpoint is missing its index/reference contract")

    common_index = {
        "preceding_indices": [7, 8],
        "target_indices": list(range(9, 18)),
        "reference_optional": True,
        "memory_source_indices": [9],
        "side_condition_projection_order": ["target", "preceding"],
        "reference_in_spatial_memory": False,
        "reference_in_side_condition": False,
    }
    max_references = int(index.get("max_references", -1))
    if max_references == 1:
        expected_index = {
            **common_index,
            "reference_candidates": list(range(7)),
            "max_references": 1,
        }
        expected_policy = _PTR_LEGACY_REFERENCE_POLICY
        expected_algorithm = _PTR_LEGACY_SELECTION_ALGORITHM
    elif max_references == PTR_MAX_REFERENCES:
        expected_index = {
            **common_index,
            "reference_candidates": [*range(7), *range(18, 21)],
            "max_references": PTR_MAX_REFERENCES,
        }
        expected_policy = PTR_REFERENCE_POLICY
        expected_algorithm = PTR_SELECTION_ALGORITHM
    else:
        raise ValueError(f"unsupported PTR max_references {max_references}")
    for key, expected in expected_index.items():
        if index.get(key) != expected:
            raise ValueError(
                f"PTR checkpoint index contract differs for {key}: "
                f"actual={index.get(key)!r}, expected={expected!r}"
            )
    if reference.get("policy") != expected_policy:
        raise ValueError(f"unsupported PTR reference policy {reference.get('policy')!r}")
    if reference.get("selection_algorithm") != expected_algorithm:
        raise ValueError(
            "unsupported PTR reference selection algorithm "
            f"{reference.get('selection_algorithm')!r}"
        )
    iou_threshold = float(reference.get("iou_threshold", -1.0))
    voxel_size = float(reference.get("voxel_size", -1.0))
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"PTR checkpoint has invalid IoU threshold {iou_threshold}")
    if voxel_size <= 0.0:
        raise ValueError(f"PTR checkpoint has invalid voxel size {voxel_size}")
    manifest_sha256 = conditioning.get("ptr_manifest_sha256")
    contract_sha256 = conditioning.get("ptr_contract_sha256")
    for name, value in (
        ("ptr_manifest_sha256", manifest_sha256),
        ("ptr_contract_sha256", contract_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"PTR checkpoint has no valid {name}")
    return {
        "preceding_frames": len(expected_index["preceding_indices"]),
        "max_references": int(expected_index["max_references"]),
        "reference_policy": expected_policy,
        "iou_threshold": iou_threshold,
        "voxel_size": voxel_size,
        "manifest_sha256": manifest_sha256,
        "contract_sha256": contract_sha256,
    }


def checkpoint_camera_geometry_contract(
    metadata: Optional[Dict[str, Any]],
) -> str:
    """Return the camera semantics, treating pre-field checkpoints as legacy absolute rays."""
    contract = metadata.get("conditioning_contract") if isinstance(metadata, dict) else None
    value = (
        contract.get("camera_geometry_contract", LEGACY_CAMERA_GEOMETRY_CONTRACT)
        if isinstance(contract, dict)
        else LEGACY_CAMERA_GEOMETRY_CONTRACT
    )
    return validate_camera_geometry_contract(value)


def validate_checkpoint_camera_geometry_contract(
    metadata: Optional[Dict[str, Any]],
    expected: str,
    artifact: str,
) -> str:
    """Reject a same-shape checkpoint whose six ray channels use different semantics."""
    expected = validate_camera_geometry_contract(expected)
    actual = checkpoint_camera_geometry_contract(metadata)
    if actual != expected:
        raise ValueError(
            f"{artifact} camera geometry contract differs: actual={actual!r}, "
            f"expected={expected!r}"
        )
    return actual


def wan_checkpoint_model_contract(backbone) -> Dict[str, Any]:
    """Fingerprint the architecture and conditioning layout used by a checkpoint."""
    def canonical(value):
        if isinstance(value, dict):
            result = {}
            for key, item in sorted(value.items()):
                # Diffusers materializes this set-like bookkeeping field as a list.
                # Its process-dependent order must not change checkpoint identity.
                if key == "_use_default_values" and isinstance(item, (list, tuple)):
                    item = sorted(item)
                result[str(key)] = canonical(item)
            return result
        if isinstance(value, (set, frozenset)):
            return sorted(canonical(item) for item in value)
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def config_sha(config) -> str:
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        return hashlib.sha256(
            json.dumps(
                canonical(config_dict), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def resource_manifest_sha(model_id: Optional[str]) -> str:
        if not model_id or not os.path.isdir(model_id):
            raise ValueError(
                "checkpoint provenance requires model_id to be a local model directory; "
                f"a mutable Hugging Face repository ID is not sufficient: {model_id!r}"
            )
        model_root = os.path.realpath(model_id)
        entries = []
        candidates = []
        small_content_suffixes = (
            ".json",
            ".txt",
            ".model",
            ".tiktoken",
            ".yaml",
            ".yml",
        )
        weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
        weight_candidates = 0
        for root, _, names in os.walk(model_root):
            for name in sorted(names):
                path = os.path.join(root, name)
                relative = os.path.relpath(path, model_root).replace(os.sep, "/")
                if name.endswith(small_content_suffixes):
                    candidates.append((relative, path))
                elif name.endswith(weight_suffixes):
                    candidates.append((relative, path))
                    weight_candidates += 1
        if not weight_candidates:
            raise ValueError(
                f"local model directory contains no supported weight shards: {model_root}"
            )
        candidate_stats = []
        for relative, path in sorted(candidates):
            stat = os.stat(path)
            candidate_stats.append(
                (
                    relative,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_dev,
                    stat.st_ino,
                )
            )
        cache_key = (
            model_root,
            tuple(candidate_stats),
        )
        if cache_key in _MODEL_RESOURCE_FINGERPRINT_CACHE:
            return _MODEL_RESOURCE_FINGERPRINT_CACHE[cache_key]
        for relative, path in sorted(candidates):
            entries.append((relative, _sha256_file(path), os.path.getsize(path)))
        fingerprint = hashlib.sha256(
            json.dumps(entries, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _MODEL_RESOURCE_FINGERPRINT_CACHE[cache_key] = fingerprint
        return fingerprint

    peft_configs = getattr(backbone.transformer, "peft_config", {}) or {}
    lora_contract = None
    if peft_configs:
        peft_config = next(iter(peft_configs.values()))
        targets = getattr(peft_config, "target_modules", []) or []
        canonical_targets = (
            [str(targets)]
            if isinstance(targets, str)
            else sorted(str(value) for value in targets)
        )
        peft_dict = peft_config.to_dict() if hasattr(peft_config, "to_dict") else vars(peft_config)
        lora_contract = {
            "rank": int(peft_config.r),
            "alpha": int(peft_config.lora_alpha),
            "target_modules": canonical_targets,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    canonical(peft_dict), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
        }
    return {
        "transformer_config_sha256": config_sha(backbone.transformer.config),
        "vae_config_sha256": config_sha(backbone.vae.config),
        "scheduler_config_sha256": config_sha(backbone.scheduler.config),
        "model_resource_manifest_sha256": resource_manifest_sha(
            getattr(backbone, "model_id", None)
        ),
        "temporal_chunk": wan_temporal_chunk_contract(),
        "side_branch": {
            "architecture": str(backbone.side_branch.architecture),
            "confidence_channel_transform": str(
                backbone.side_branch.confidence_channel_transform
            ),
            "top_k": int(backbone.top_k),
            "composite_cond": bool(backbone.composite_cond),
            "camera_cond": bool(getattr(backbone, "camera_cond", False)),
            "camera_ray_channels": int(
                getattr(backbone.side_branch, "camera_ray_channels", 0)
            ),
            "control_layers": [int(value) for value in backbone.control_layers],
            "in_channels": int(backbone.side_branch.in_channels),
            "latent_channels": int(backbone.side_branch.latent_channels),
            "initialization": str(
                getattr(backbone.side_branch, "initialization", "wan_copy")
            ),
        },
        "lora": lora_contract,
    }


def checkpoint_memory_fusion_policy(
    metadata: Optional[Dict[str, Any]]
) -> MemoryFusionPolicy:
    """Fusion policy a checkpoint was trained under.

    Metadata written before this contract existed carries no ``memory_contract`` entry,
    which unambiguously means the original unbounded fusion.
    """
    memory_contract = (metadata or {}).get("memory_contract")
    if memory_contract is None:
        return MemoryFusionPolicy()
    if not isinstance(memory_contract, dict):
        raise ValueError(f"memory_contract must be a mapping, got {type(memory_contract)}")
    return MemoryFusionPolicy.from_metadata(memory_contract.get("fusion"))


def validate_checkpoint_memory_fusion_policy(
    metadata: Optional[Dict[str, Any]],
    active: MemoryFusionPolicy,
    artifact: str,
) -> MemoryFusionPolicy:
    """Reject a checkpoint trained to read conditioning fused under a different policy.

    Shapes are identical either way, so nothing else would catch this: the side branch
    would simply be reading confidence and geometry channels whose distribution it never
    saw during training.
    """
    trained = checkpoint_memory_fusion_policy(metadata)
    active.validate_against(trained, artifact)
    return trained


def build_memory_contract(
    fusion_policy: Optional[MemoryFusionPolicy] = None,
    augmentation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record how the training memory was fused and how it was perturbed.

    ``fusion`` is enforced at load time. ``augmentation`` is provenance rather than a
    constraint: it does not change the eval code path, but it is what tells you the depth
    error magnitude the model was trained to tolerate, which is how the inference-side
    alignment and write-gate tolerances should be chosen.
    """
    policy = fusion_policy or MemoryFusionPolicy()
    return {"fusion": policy.to_metadata(), "augmentation": augmentation}


def _training_objective_contract(
    stage: int,
    clean_prefix_len: Optional[int] = None,
    camera_objective_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if clean_prefix_len is None:
        clean_prefix_len = 1 if int(stage) >= 2 else 0
    clean_prefix_len = int(clean_prefix_len)
    objective = {
        "clean_prefix_len": clean_prefix_len,
        "loss_on_generated_only": bool(clean_prefix_len),
    }
    if camera_objective_contract is not None:
        objective["camera_objective"] = dict(camera_objective_contract)
    return objective


def _valid_training_objective(objective: Any, stage: int) -> bool:
    """Structural validity of a training objective for a stage.

    Stage 1 accepts the legacy all-noisy objective or the Mirage-aligned one-frame clean
    prefix. Stage 2 uses a clean continuation prefix of length 1 or 2.
    """
    if not isinstance(objective, dict):
        return False
    base_keys = {"clean_prefix_len", "loss_on_generated_only"}
    if set(objective.keys()) not in (base_keys, base_keys | {"camera_objective"}):
        return False
    clean_prefix_len = objective["clean_prefix_len"]
    if not isinstance(clean_prefix_len, int) or isinstance(clean_prefix_len, bool):
        return False
    if bool(objective["loss_on_generated_only"]) != bool(clean_prefix_len):
        return False
    camera_objective = objective.get("camera_objective")
    if camera_objective is not None:
        expected_keys = {
            "motion_weighting",
            "motion_loss_alpha",
            "motion_loss_quantile",
            "counterfactual_condition",
            "counterfactual_loss_weight",
            "counterfactual_loss_margin",
            "counterfactual_shared_flow_sample",
        }
        if not isinstance(camera_objective, dict) or set(camera_objective) != expected_keys:
            return False
        if camera_objective["motion_weighting"] != (
            "visible_static_prefix_parallax_quantile_v1"
        ):
            return False
        if camera_objective["counterfactual_condition"] != "reverse_relative_pose_v1":
            return False
        if camera_objective["counterfactual_shared_flow_sample"] is not True:
            return False
        try:
            alpha = float(camera_objective["motion_loss_alpha"])
            quantile = float(camera_objective["motion_loss_quantile"])
            weight = float(camera_objective["counterfactual_loss_weight"])
            margin = float(camera_objective["counterfactual_loss_margin"])
        except (TypeError, ValueError):
            return False
        if alpha < 0.0 or not 0.0 < quantile <= 1.0 or weight < 0.0 or margin < 0.0:
            return False
    stage = int(stage)
    if stage <= 1:
        return clean_prefix_len in (0, 1)
    if stage == 2:
        return clean_prefix_len in (1, 2)
    return False


def build_checkpoint_metadata(
    vae,
    stage: int,
    step: int,
    model_contract: Optional[Dict[str, Any]] = None,
    clean_prefix_len: Optional[int] = None,
    conditioning_contract: Optional[Dict[str, Any]] = None,
    training_runtime: Optional[Dict[str, Any]] = None,
    memory_contract: Optional[Dict[str, Any]] = None,
    camera_objective_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_CONTRACT_VERSION,
        "stage": int(stage),
        "step": int(step),
        "flow_contract": WAN_FLOW_CONTRACT_NAME,
        "latent_contract": wan_latent_contract(vae),
        "training_objective": _training_objective_contract(
            stage, clean_prefix_len, camera_objective_contract
        ),
        "model_contract": model_contract,
        "conditioning_contract": conditioning_contract,
        "memory_contract": memory_contract or build_memory_contract(),
        "training_runtime": training_runtime,
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lora_weights_path(lora_path: str) -> str:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = os.path.join(lora_path, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"no LoRA adapter weights found in {lora_path}")


def _lora_config_path(lora_path: str) -> str:
    path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no LoRA adapter config found in {lora_path}")
    return path


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(partial, path)


def write_checkpoint_metadata(
    side_branch_path: str,
    vae,
    stage: int,
    step: int,
    latent_stats_path: str,
    model_contract: Dict[str, Any],
    lora_path: Optional[str] = None,
    clean_prefix_len: Optional[int] = None,
    conditioning_contract: Optional[Dict[str, Any]] = None,
    training_runtime: Optional[Dict[str, Any]] = None,
    memory_contract: Optional[Dict[str, Any]] = None,
    camera_objective_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = build_checkpoint_metadata(
        vae,
        stage,
        step,
        model_contract=model_contract,
        clean_prefix_len=clean_prefix_len,
        conditioning_contract=conditioning_contract,
        training_runtime=training_runtime,
        memory_contract=memory_contract,
        camera_objective_contract=camera_objective_contract,
    )
    metadata["side_branch_sha256"] = _sha256_file(side_branch_path)
    metadata["latent_stats_sha256"] = _sha256_file(latent_stats_path)
    metadata["lora_sha256"] = _sha256_file(_lora_weights_path(lora_path)) if lora_path else None
    metadata["lora_config_sha256"] = (
        _sha256_file(_lora_config_path(lora_path)) if lora_path else None
    )
    _atomic_write_json(side_branch_metadata_path(side_branch_path), metadata)
    if lora_path is not None:
        _atomic_write_json(lora_metadata_path(lora_path), metadata)
    return metadata


def _read_metadata(path: str, artifact: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ValueError(
            f"{artifact} has no Engram checkpoint metadata ({path}); it is treated as a legacy "
            "flow/latent checkpoint and cannot be resumed or evaluated"
        )
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata


def validate_checkpoint_metadata(
    metadata: Dict[str, Any],
    expected_latent_contract: Dict[str, Any],
    artifact: str,
    expected_stage: Optional[int] = None,
    expected_model_contract: Optional[Dict[str, Any]] = None,
    expected_conditioning_contract: Optional[Dict[str, Any]] = None,
    expected_camera_objective_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if metadata.get("format_version") != CHECKPOINT_CONTRACT_VERSION:
        raise ValueError(
            f"{artifact} uses checkpoint contract version {metadata.get('format_version')!r}, "
            f"expected {CHECKPOINT_CONTRACT_VERSION}"
        )
    if metadata.get("flow_contract") != WAN_FLOW_CONTRACT_NAME:
        raise ValueError(
            f"{artifact} uses flow contract {metadata.get('flow_contract')!r}, "
            f"expected {WAN_FLOW_CONTRACT_NAME!r}"
        )
    validate_wan_latent_contract(
        metadata.get("latent_contract"), expected_latent_contract, source=artifact
    )
    stage = int(metadata.get("stage", -1))
    if not _valid_training_objective(metadata.get("training_objective"), stage):
        raise ValueError(
            f"{artifact} training objective differs: "
            f"actual={metadata.get('training_objective')!r}, expected a valid "
            f"clean-prefix objective for stage {stage} "
            f"(stage1 clean_prefix_len in {{0,1}}, stage2 clean_prefix_len in {{1,2}})"
        )
    if expected_stage is not None and int(metadata.get("stage", -1)) != int(expected_stage):
        raise ValueError(
            f"{artifact} was saved for stage {metadata.get('stage')!r}, expected stage {expected_stage}"
        )
    if expected_camera_objective_contract is not None:
        actual_camera_objective = metadata["training_objective"].get("camera_objective")
        if actual_camera_objective != expected_camera_objective_contract:
            raise ValueError(
                f"{artifact} camera objective differs: "
                f"actual={actual_camera_objective!r}, "
                f"expected={expected_camera_objective_contract!r}"
            )
    stats_hash = metadata.get("latent_stats_sha256")
    if not isinstance(stats_hash, str) or len(stats_hash) != 64:
        raise ValueError(f"{artifact} has no valid latent-stats SHA-256")
    actual_model_contract = metadata.get("model_contract")
    if not isinstance(actual_model_contract, dict):
        raise ValueError(f"{artifact} has no model architecture contract")
    if expected_model_contract is None:
        raise ValueError(f"{artifact} validation requires the active model architecture contract")
    for key in (
        "transformer_config_sha256",
        "vae_config_sha256",
        "scheduler_config_sha256",
        "model_resource_manifest_sha256",
        "temporal_chunk",
        "side_branch",
    ):
        if actual_model_contract.get(key) != expected_model_contract.get(key):
            raise ValueError(
                f"{artifact} model architecture differs for {key}: "
                f"actual={actual_model_contract.get(key)!r}, "
                f"expected={expected_model_contract.get(key)!r}"
            )
    if stage == 1 and actual_model_contract.get("lora") is not None:
        raise ValueError(f"{artifact} Stage-1 checkpoint unexpectedly records a LoRA contract")
    if stage == 2 and actual_model_contract.get("lora") != expected_model_contract.get("lora"):
        raise ValueError(
            f"{artifact} LoRA architecture does not match the active model: "
            f"actual={actual_model_contract.get('lora')!r}, "
            f"expected={expected_model_contract.get('lora')!r}"
        )
    if (
        expected_conditioning_contract is not None
        and metadata.get("conditioning_contract") != expected_conditioning_contract
    ):
        raise ValueError(
            f"{artifact} conditioning contract differs: "
            f"actual={metadata.get('conditioning_contract')!r}, "
            f"expected={expected_conditioning_contract!r}"
        )
    return dict(metadata)


def load_side_branch_checkpoint(
    module,
    checkpoint_path: str,
    vae,
    expected_stage: Optional[int] = None,
    latent_stats_path: Optional[str] = None,
    expected_model_contract: Optional[Dict[str, Any]] = None,
    expected_conditioning_contract: Optional[Dict[str, Any]] = None,
    expected_camera_objective_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expected = wan_latent_contract(vae)
    metadata = _read_metadata(side_branch_metadata_path(checkpoint_path), checkpoint_path)
    validate_checkpoint_metadata(
        metadata,
        expected,
        checkpoint_path,
        expected_stage=expected_stage,
        expected_model_contract=expected_model_contract,
        expected_conditioning_contract=expected_conditioning_contract,
        expected_camera_objective_contract=expected_camera_objective_contract,
    )
    expected_hash = metadata.get("side_branch_sha256")
    actual_hash = _sha256_file(checkpoint_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"{checkpoint_path} SHA-256 does not match its metadata: "
            f"actual={actual_hash}, expected={expected_hash!r}"
        )
    if latent_stats_path is None:
        raise ValueError(f"{checkpoint_path} requires the latent_stats_path used for training")
    actual_stats_hash = _sha256_file(latent_stats_path)
    if metadata["latent_stats_sha256"] != actual_stats_hash:
        raise ValueError(
            f"{checkpoint_path} latent-stats SHA-256 does not match {latent_stats_path}: "
            f"actual={actual_stats_hash}, expected={metadata['latent_stats_sha256']!r}"
        )
    module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return metadata


def validate_lora_checkpoint(
    lora_path: str,
    vae,
    latent_stats_path: Optional[str] = None,
    expected_model_contract: Optional[Dict[str, Any]] = None,
    expected_camera_objective_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expected = wan_latent_contract(vae)
    metadata = _read_metadata(lora_metadata_path(lora_path), lora_path)
    validate_checkpoint_metadata(
        metadata,
        expected,
        lora_path,
        expected_stage=2,
        expected_model_contract=expected_model_contract,
        expected_camera_objective_contract=expected_camera_objective_contract,
    )
    actual_hash = _sha256_file(_lora_weights_path(lora_path))
    expected_hash = metadata.get("lora_sha256")
    if expected_hash != actual_hash:
        raise ValueError(
            f"{lora_path} SHA-256 does not match its metadata: "
            f"actual={actual_hash}, expected={expected_hash!r}"
        )
    actual_config_hash = _sha256_file(_lora_config_path(lora_path))
    expected_config_hash = metadata.get("lora_config_sha256")
    if expected_config_hash != actual_config_hash:
        raise ValueError(
            f"{lora_path} adapter config SHA-256 does not match its metadata: "
            f"actual={actual_config_hash}, expected={expected_config_hash!r}"
        )
    if latent_stats_path is None:
        raise ValueError(f"{lora_path} requires the latent_stats_path used for training")
    actual_stats_hash = _sha256_file(latent_stats_path)
    if metadata["latent_stats_sha256"] != actual_stats_hash:
        raise ValueError(
            f"{lora_path} latent-stats SHA-256 does not match {latent_stats_path}: "
            f"actual={actual_stats_hash}, expected={metadata['latent_stats_sha256']!r}"
        )
    return metadata


def validate_checkpoint_pair(
    side_branch_metadata: Dict[str, Any] | None,
    lora_metadata: Dict[str, Any],
    artifact: str,
) -> None:
    """Require a Stage-2 side branch and LoRA saved by the same checkpoint event."""
    if side_branch_metadata is None:
        raise ValueError(f"{artifact} requires a matching side-branch checkpoint")
    if not side_branch_metadata.get("lora_sha256") or not lora_metadata.get("lora_sha256"):
        raise ValueError(f"{artifact} Stage-2 checkpoint metadata has no LoRA SHA-256")
    if (
        not side_branch_metadata.get("lora_config_sha256")
        or not lora_metadata.get("lora_config_sha256")
    ):
        raise ValueError(f"{artifact} Stage-2 checkpoint metadata has no LoRA config SHA-256")
    for key in (
        "format_version",
        "stage",
        "step",
        "flow_contract",
        "latent_contract",
        "training_objective",
        "conditioning_contract",
        "training_runtime",
        "latent_stats_sha256",
        "model_contract",
        "side_branch_sha256",
        "lora_sha256",
        "lora_config_sha256",
    ):
        if side_branch_metadata.get(key) != lora_metadata.get(key):
            raise ValueError(
                f"side branch and LoRA metadata differ for {key}: "
                f"side={side_branch_metadata.get(key)!r}, lora={lora_metadata.get(key)!r}"
            )
    if int(side_branch_metadata.get("stage", -1)) != 2:
        raise ValueError(f"{artifact} LoRA must be paired with a Stage-2 side branch")


def validate_checkpoint_bundle(
    side_branch_metadata: Dict[str, Any] | None,
    lora_metadata: Dict[str, Any] | None,
    artifact: str,
) -> None:
    """Validate the complete checkpoint selection before resume or evaluation.

    A Stage-1 side branch is a standalone artifact and is the valid starting
    point for Stage 2. A Stage-2 side branch is only valid together with the
    LoRA adapter written by the same save event.
    """
    if side_branch_metadata is None:
        if lora_metadata is not None:
            raise ValueError(f"{artifact} LoRA requires a matching side-branch checkpoint")
        return

    stage = int(side_branch_metadata.get("stage", -1))
    if stage == 1:
        if lora_metadata is not None:
            raise ValueError(f"{artifact} cannot pair a LoRA with a Stage-1 side branch")
        return
    if stage == 2:
        if lora_metadata is None:
            raise ValueError(f"{artifact} Stage-2 side branch requires its matching LoRA")
        validate_checkpoint_pair(side_branch_metadata, lora_metadata, artifact)
        return
    raise ValueError(f"{artifact} uses unsupported checkpoint stage {stage}")


def build_calibrator_metadata(
    side_branch_metadata: Dict[str, Any] | None,
    lora_metadata: Dict[str, Any] | None,
    confidence_weights: Dict[str, float],
    hit_threshold: float,
    method: str,
    rollout_config: Dict[str, Any],
    fit_dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind a fitted calibrator to the exact checkpoint and proxy definition."""
    validate_checkpoint_bundle(side_branch_metadata, lora_metadata, "calibrator")
    if side_branch_metadata is None:
        raise ValueError("calibrator requires a side-branch checkpoint")
    checkpoint_keys = (
        "stage",
        "step",
        "flow_contract",
        "latent_contract",
        "training_objective",
        "conditioning_contract",
        "training_runtime",
        "latent_stats_sha256",
        "model_contract",
        "side_branch_sha256",
        "lora_sha256",
        "lora_config_sha256",
    )
    if method not in ("isotonic", "temperature"):
        raise ValueError(f"unsupported calibrator method {method!r}")
    metadata = {
        "format_version": CALIBRATOR_CONTRACT_VERSION,
        "method": method,
        "checkpoint": {key: side_branch_metadata.get(key) for key in checkpoint_keys},
        "confidence_weights": {
            str(key): float(value) for key, value in sorted(confidence_weights.items())
        },
        "hit_threshold": float(hit_threshold),
        "rollout": dict(rollout_config),
    }
    if fit_dataset_id is not None:
        metadata["fit_dataset_id"] = str(fit_dataset_id)
    return metadata
