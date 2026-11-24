"""Shared contracts for Strict20K ViPE geometry confidence calibration."""

from __future__ import annotations

from .calibration import RELIABILITY_CLAMP_EPS


STRICT20K_CONTEXT_ANCHOR_INDICES = tuple(range(9))
STRICT20K_CONFIDENCE_CONTRACT = {
    "name": "vipe_anchor_reprojection_v1",
    "scale": 0.05,
    "occlusion_tolerance": 0.03,
    "calibration": "none",
}
STRICT20K_VIDEO_GROUP_CONTRACT = {
    "name": "strict20k_underlying_video_v1",
    "official_re10k_frames": "clip_id",
    "modelscope_re10k_mp4": "archive_member component 1",
}
STRICT20K_SCORE_CONTRACT = {
    "name": "vipe_near_reprojection_plus_graze_v1",
    "near_offsets": [-1, 1],
    "anchor_indices": list(STRICT20K_CONTEXT_ANCHOR_INDICES),
    "confidence": STRICT20K_CONFIDENCE_CONTRACT,
}
STRICT20K_LABEL_CONTRACT = {
    "name": "vipe_far_reprojection_hit_v1",
    "far_offsets": [-2, 2],
    "relative_depth_threshold": 0.05,
    "occlusion_tolerance": 0.03,
    "anchor_indices": list(STRICT20K_CONTEXT_ANCHOR_INDICES),
}
STRICT20K_PRECISION_LINK_CONTRACT = {
    "name": "reliability_odds_v1",
    "reliability_clamp_eps": RELIABILITY_CLAMP_EPS,
    "nominal_minimum_precision": RELIABILITY_CLAMP_EPS
    / (1.0 - RELIABILITY_CLAMP_EPS),
    "nominal_maximum_precision": (1.0 - RELIABILITY_CLAMP_EPS)
    / RELIABILITY_CLAMP_EPS,
}


def strict20k_geometry_calibrator_expected_metadata(
    train_manifest_sha256: str, confidence_weights: dict
) -> dict:
    if not isinstance(train_manifest_sha256, str) or len(train_manifest_sha256) != 64:
        raise ValueError("Strict20K training manifest requires a SHA-256 digest")
    return {
        "format_version": 1,
        "method": "isotonic",
        "purpose": "strict20k_vipe_geometry_training",
        "score_contract": STRICT20K_SCORE_CONTRACT,
        "label_contract": STRICT20K_LABEL_CONTRACT,
        "video_group_contract": STRICT20K_VIDEO_GROUP_CONTRACT,
        "confidence_weights": {
            str(key): float(value) for key, value in sorted(confidence_weights.items())
        },
        "precision_link": STRICT20K_PRECISION_LINK_CONTRACT,
        "train_manifest_sha256": train_manifest_sha256,
        "quality_gate": {"passed": True},
    }
