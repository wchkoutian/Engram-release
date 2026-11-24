#!/usr/bin/env python
"""Encode, merge, and validate a RealEstate10K training LMDB."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from engram.data.preprocessing import read_scene_manifest as _read_manifest
RGB_FRAMES = 81
ANCHOR_INDICES = tuple(range(0, RGB_FRAMES, 4))
ANCHOR_FRAMES = len(ANCHOR_INDICES)
LATENT_HEIGHT = 44
LATENT_WIDTH = 80
STORE_NAME = "precomputed_train_wan_model_v1.lmdb"
STATS_NAME = "latent_stats_train_wan_model_v1.pt"
DATASET_CONTRACT = "engram_vipe_21anchor_v1"
DATA_CONTRACT_KEY = "__engram_dataset_contract__"
CONFIDENCE_CONTRACT = {
    "name": "vipe_anchor_reprojection_v1", "scale": 0.05,
    "occlusion_tolerance": 0.03, "calibration": "none",
}
CAPTION_CONTRACT = {"name": "optional_caption_v1", "file": "caption.txt", "required": False}


def load_runtime():
    """Keep argument parsing and manifest checks available without GPU packages."""
    global cv2, np, torch, PrecomputedStore, DATASET_ID_KEY, LATENT_CONTRACT_KEY
    global backproject_depth_map, latent_intrinsics, project_to_latent_grid
    global encode_wan_latents, wan_latent_contract
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'source'))
    import cv2
    import numpy as np
    import torch
    from engram.data.realestate10k import (
        PrecomputedStore, DATASET_ID_KEY, LATENT_CONTRACT_KEY,
    )
    from engram.geometry.projection import (
        backproject_depth_map, latent_intrinsics, project_to_latent_grid,
    )
    from engram.models.wan_latents import encode_wan_latents, wan_latent_contract


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()




def _data_contract(manifest):
    return {
        'name': DATASET_CONTRACT,
        'source_manifest_sha256': sha256_file(manifest),
        'confidence': CONFIDENCE_CONTRACT, 'caption': CAPTION_CONTRACT,
    }


def _dataset_id(manifest, rows, latent_contract):
    payload = {
        'data_contract': _data_contract(manifest),
        'sample_ids': [row['sample_id'] for row in rows],
        'latent_contract': latent_contract,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _caption(sample_dir):
    path = sample_dir / CAPTION_CONTRACT['file']
    return ' '.join(path.read_text(encoding='utf-8').split()) if path.is_file() else ''


def _load_vae(model_id, device):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(
        model_id, subfolder='vae', torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval()



def _load_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) != RGB_FRAMES:
        raise ValueError(f"{path} has {len(frames)} decoded frames, expected {RGB_FRAMES}")
    if any(frame.shape != (704, 1280, 3) for frame in frames):
        raise ValueError(f"{path} must contain only 1280x704 RGB frames")
    return frames


def _encode_video(vae, frames: list[np.ndarray], device: str) -> torch.Tensor:
    video = torch.from_numpy(np.stack(frames, axis=0))
    video = video.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0).unsqueeze(0)
    latent = encode_wan_latents(vae, video.to(device=device))[0].float().cpu()
    expected = (48, ANCHOR_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    if tuple(latent.shape) != expected:
        raise ValueError(f"continuous VAE output is {tuple(latent.shape)}, expected {expected}")
    return latent


def _load_npz_series(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    with np.load(path) as data:
        values = np.asarray(data["data"])
        indices = np.asarray(data["inds"])
    if tuple(values.shape) != shape:
        raise ValueError(f"{path} has shape {values.shape}, expected {shape}")
    if not np.array_equal(indices, np.arange(RGB_FRAMES)):
        raise ValueError(f"{path} indices are not 0..{RGB_FRAMES - 1}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values.astype(np.float32, copy=False)


def _decode_exr(payload: bytes) -> np.ndarray:
    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(io.BytesIO(payload))
    try:
        window = exr.header()["dataWindow"]
        width = int(window.max.x - window.min.x + 1)
        height = int(window.max.y - window.min.y + 1)
        raw = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
    finally:
        exr.close()
    return np.frombuffer(raw, dtype=np.float32).reshape(height, width)


def _load_anchor_depths(path: Path) -> np.ndarray:
    depths = []
    with zipfile.ZipFile(path) as archive:
        for index in ANCHOR_INDICES:
            depth = _decode_exr(archive.read(f"{index:05d}.exr"))
            if depth.shape != (704, 1280):
                raise ValueError(f"{path}:{index:05d}.exr has shape {depth.shape}")
            depth = cv2.resize(depth, (LATENT_WIDTH, LATENT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            depths.append(depth.astype(np.float32, copy=False))
    return np.stack(depths, axis=0)


def _pool_static_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        dynamic = np.any(mask != 0, axis=-1)
    else:
        dynamic = mask != 0
    if dynamic.shape != (704, 1280):
        raise ValueError(f"mask has shape {dynamic.shape}, expected (704, 1280)")
    dynamic = dynamic.reshape(LATENT_HEIGHT, 16, LATENT_WIDTH, 16).any(axis=(1, 3))
    return ~dynamic


def _load_anchor_static_masks(path: Path) -> np.ndarray:
    masks = []
    with zipfile.ZipFile(path) as archive:
        for index in ANCHOR_INDICES:
            payload = np.frombuffer(archive.read(f"{index:05d}.png"), dtype=np.uint8)
            mask = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"cannot decode {path}:{index:05d}.png")
            masks.append(_pool_static_mask(mask))
    return np.stack(masks, axis=0)


def _camera_geometry(c2w: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c2w = c2w[list(ANCHOR_INDICES)]
    intrinsics = intrinsics[list(ANCHOR_INDICES)]
    extrinsics = np.linalg.inv(c2w).astype(np.float32)

    K_pixel = np.zeros((ANCHOR_FRAMES, 3, 3), dtype=np.float32)
    K_pixel[:, 0, 0] = intrinsics[:, 0]
    K_pixel[:, 1, 1] = intrinsics[:, 1]
    K_pixel[:, 0, 2] = intrinsics[:, 2]
    K_pixel[:, 1, 2] = intrinsics[:, 3]
    K_pixel[:, 2, 2] = 1.0
    K_l = latent_intrinsics(
        torch.from_numpy(K_pixel), 704, 1280, LATENT_HEIGHT, LATENT_WIDTH
    ).numpy()
    return K_l.astype(np.float32), extrinsics


def _latent_geometry(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c2w = _load_npz_series(sample_dir / "pose.npz", (RGB_FRAMES, 4, 4))
    intrinsics = _load_npz_series(sample_dir / "intrinsics.npz", (RGB_FRAMES, 4))
    K_l, extrinsics = _camera_geometry(c2w, intrinsics)

    depths = _load_anchor_depths(sample_dir / "depth.zip")
    static = _load_anchor_static_masks(sample_dir / "mask.zip")
    valid_depth = np.isfinite(depths) & (depths > 0)
    static &= valid_depth
    return depths, static, K_l, extrinsics


def _residual_to_neighbor(
    source: int,
    target: int,
    depths: torch.Tensor,
    static: torch.Tensor,
    K_l: torch.Tensor,
    E: torch.Tensor,
    occlusion_tolerance: float,
) -> torch.Tensor:
    points, uu, vv = backproject_depth_map(
        depths[source], K_l[source], E[source], valid=static[source]
    )
    residual = torch.full_like(depths[source], float("nan"))
    if points.numel() == 0:
        return residual
    target_u, target_v, projected_z, in_front, in_bounds = project_to_latent_grid(
        points, E[target], K_l[target], LATENT_HEIGHT, LATENT_WIDTH
    )
    safe_u = target_u.clamp(0, LATENT_WIDTH - 1)
    safe_v = target_v.clamp(0, LATENT_HEIGHT - 1)
    observed_z = depths[target, safe_v, safe_u]
    valid = in_front & in_bounds
    valid &= static[target, safe_v, safe_u]
    valid &= torch.isfinite(observed_z) & (observed_z > 0)
    valid &= projected_z <= observed_z * (1.0 + occlusion_tolerance)
    relative = 2.0 * (projected_z - observed_z).abs()
    relative /= projected_z.abs() + observed_z.abs() + 1e-6
    residual[vv[valid], uu[valid]] = relative[valid]
    return residual


def _reprojection_reliability(
    depths_np: np.ndarray,
    static_np: np.ndarray,
    K_l_np: np.ndarray,
    E_np: np.ndarray,
    scale: float = 0.05,
    occlusion_tolerance: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    depths = torch.from_numpy(depths_np).float()
    # `static &= supported` below is intentionally local to this computation.
    # Clone because torch.from_numpy shares storage with a boolean NumPy input.
    static = torch.from_numpy(static_np).bool().clone()
    K_l = torch.from_numpy(K_l_np).float()
    E = torch.from_numpy(E_np).float()
    reliability = torch.zeros_like(depths)
    supported = torch.zeros_like(static)
    for index in range(ANCHOR_FRAMES):
        residuals = []
        if index > 0:
            residuals.append(
                _residual_to_neighbor(
                    index, index - 1, depths, static, K_l, E, occlusion_tolerance
                )
            )
        if index + 1 < ANCHOR_FRAMES:
            residuals.append(
                _residual_to_neighbor(
                    index, index + 1, depths, static, K_l, E, occlusion_tolerance
                )
            )
        stacked = torch.stack(residuals, dim=0)
        finite = torch.isfinite(stacked)
        count = finite.sum(dim=0)
        mean_residual = torch.where(
            count > 0,
            torch.nan_to_num(stacked, nan=0.0).sum(dim=0) / count.clamp_min(1),
            torch.zeros_like(depths[index]),
        )
        supported[index] = count > 0
        reliability[index] = torch.where(
            supported[index], torch.exp(-mean_residual / scale), torch.zeros_like(mean_residual)
        )
    reliability = reliability.clamp(0.0, 1.0 - 1e-4)
    static &= supported
    return reliability.numpy().astype(np.float32), static.numpy()


def _welford_update(state: tuple[int, torch.Tensor | None, torch.Tensor | None], latent: torch.Tensor):
    count, mean, M2 = state
    values = latent.permute(1, 2, 3, 0).reshape(-1, latent.shape[0]).double()
    batch_count = values.shape[0]
    batch_mean = values.mean(dim=0)
    batch_M2 = values.var(dim=0, unbiased=False) * batch_count
    if mean is None:
        return batch_count, batch_mean, batch_M2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_M2 = M2 + batch_M2 + delta.square() * (count * batch_count / total)
    return total, merged_mean, merged_M2


def _merge_welford(states: list[tuple[int, torch.Tensor, torch.Tensor]]):
    merged = (0, None, None)
    for count, mean, M2 in states:
        if merged[1] is None:
            merged = (count, mean.double(), M2.double())
            continue
        total_count, total_mean, total_M2 = merged
        combined = total_count + count
        delta = mean.double() - total_mean
        total_mean = total_mean + delta * (count / combined)
        total_M2 = total_M2 + M2.double() + delta.square() * (total_count * count / combined)
        merged = (combined, total_mean, total_M2)
    return merged


def _put_records(store: PrecomputedStore, records: list[tuple[str, dict]]) -> None:
    with store.env.begin(write=True) as transaction:
        for key, value in records:
            sample_id, anchor = key.split(':')
            _validate_record(value, sample_id, int(anchor))
            transaction.put(
                key.encode("utf-8"), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            )


def _shard_paths(output_dir: Path, shard_id: int) -> tuple[Path, Path]:
    return (
        output_dir / f"precomputed_train_wan_model_v1.shard_{shard_id:02d}.lmdb",
        output_dir / f"latent_stats_train_wan_model_v1.shard_{shard_id:02d}.pt",
    )


def encode(args) -> None:
    manifest = Path(args.manifest)
    sample_root = Path(args.sample_root)
    output_dir = Path(args.output_dir)
    rows = _read_manifest(manifest, args.max_samples)
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_id]
    if not rows:
        raise ValueError(f"shard {args.shard_id} has no samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_lmdb, shard_stats = _shard_paths(output_dir, args.shard_id)
    partial_lmdb = Path(str(shard_lmdb) + ".partial")
    partial_stats = Path(str(shard_stats) + ".partial")
    for path in (shard_lmdb, shard_stats, partial_lmdb, partial_stats):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    vae = _load_vae(args.model_id, args.device)
    latent_contract = wan_latent_contract(vae)
    all_rows = _read_manifest(manifest, args.max_samples)
    dataset_id = _dataset_id(manifest, all_rows, latent_contract)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

    store = PrecomputedStore(str(partial_lmdb), readonly=False, map_size=args.map_size_gib << 30)
    clips_meta = []
    stat_state = (0, None, None)
    try:
        store.put_latent_contract(latent_contract)
        store.put_dataset_id(dataset_id)
        store.put(
            DATA_CONTRACT_KEY,
            {
                "name": DATASET_CONTRACT,
                "source_manifest_sha256": manifest_sha256,
                "confidence": CONFIDENCE_CONTRACT,
                "caption": CAPTION_CONTRACT,
            },
        )
        pending = []
        for item_index, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            sample_dir = sample_root / sample_id
            if row.get('clip_sha256') and sha256_file(sample_dir / 'clip.mp4') != row['clip_sha256'].lower():
                raise ValueError(f'Source video differs from manifest: {sample_id}')
            caption = _caption(sample_dir)
            frames = _load_video(sample_dir / "clip.mp4")
            with torch.no_grad():
                latent = _encode_video(vae, frames, args.device)
            if not torch.isfinite(latent).all():
                raise ValueError(f'Non-finite latent for {sample_id}')
            depths, static, K_l, E = _latent_geometry(sample_dir)
            confidence, static = _reprojection_reliability(depths, static, K_l, E)
            for anchor, rgb_index in enumerate(ANCHOR_INDICES):
                pending.append(
                    (
                        f"{sample_id}:{anchor}",
                        {
                            "latent": latent[:, anchor].numpy().astype(np.float32),
                            "source_rgb_index": int(rgb_index),
                            "depth": depths[anchor],
                            "conf": confidence[anchor],
                            "static_mask": static[anchor],
                            "K_l": K_l[anchor],
                            "E": E[anchor],
                            "caption": caption,
                        },
                    )
                )
            if len(pending) >= 84:
                _put_records(store, pending)
                pending.clear()
            stat_state = _welford_update(stat_state, latent)
            clips_meta.append((sample_id, ANCHOR_FRAMES))
            print(
                f"[shard {args.shard_id}] {item_index}/{len(rows)} {sample_id} "
                f"anchors={ANCHOR_FRAMES}",
                flush=True,
            )
        if pending:
            _put_records(store, pending)
        store.put_meta(clips_meta)
    finally:
        store.close()

    stat_count, stat_mean, stat_M2 = stat_state
    torch.save(
        {
            "count": int(stat_count),
            "mean": stat_mean.float(),
            "M2": stat_M2,
            "latent_contract": latent_contract,
            "dataset_id": dataset_id,
            "dataset_contract": DATASET_CONTRACT,
            "source_manifest_sha256": manifest_sha256,
            "confidence": CONFIDENCE_CONTRACT,
            "caption": CAPTION_CONTRACT,
        },
        partial_stats,
    )
    os.replace(partial_lmdb, shard_lmdb)
    os.replace(partial_stats, shard_stats)
    print(f"shard_complete={args.shard_id} samples={len(clips_meta)}")


def merge(args) -> None:
    import lmdb

    output_dir = Path(args.output_dir)
    final_lmdb = output_dir / STORE_NAME
    final_stats = output_dir / STATS_NAME
    partial_lmdb = Path(str(final_lmdb) + ".partial")
    partial_stats = Path(str(final_stats) + ".partial")
    for path in (final_lmdb, final_stats, partial_lmdb, partial_stats):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    source_stores = []
    shard_stats = []
    all_meta = []
    dataset_id = None
    latent_contract = None
    data_contract = None
    try:
        manifest_rows = _read_manifest(Path(args.manifest), args.max_samples)
        manifest_ids = [row['sample_id'] for row in manifest_rows]
        for shard_id in range(args.num_shards):
            lmdb_path, stats_path = _shard_paths(output_dir, shard_id)
            store = PrecomputedStore(str(lmdb_path), readonly=True)
            source_stores.append(store)
            current_id = store.require_dataset_id()
            current_contract = store.require_latent_contract()
            current_data_contract = store.get(DATA_CONTRACT_KEY)
            if dataset_id is None:
                dataset_id = current_id
                latent_contract = current_contract
                data_contract = current_data_contract
            if (
                current_id != dataset_id
                or current_contract != latent_contract
                or current_data_contract != data_contract
            ):
                raise ValueError(f"shard {shard_id} contract or dataset ID differs")
            if current_data_contract != _data_contract(args.manifest):
                raise ValueError('Shard source manifest differs from requested manifest')
            if _dataset_id(Path(args.manifest), manifest_rows, latent_contract) != dataset_id:
                raise ValueError('Shard dataset ID differs from requested selection')
            all_meta.extend(store.get_meta())
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            if (
                stats["dataset_id"] != dataset_id
                or stats["latent_contract"] != latent_contract
                or stats["source_manifest_sha256"]
                != data_contract["source_manifest_sha256"]
                or stats["confidence"] != CONFIDENCE_CONTRACT
                or stats["caption"] != CAPTION_CONTRACT
            ):
                raise ValueError(f"shard {shard_id} latent stats contract differs")
            shard_stats.append(stats)

        meta_by_id = {clip_id: int(anchors) for clip_id, anchors in all_meta}
        if len(meta_by_id) != len(all_meta):
            raise ValueError("shard metadata contains duplicate clip IDs")
        if set(meta_by_id) != set(manifest_ids):
            raise ValueError("shard metadata IDs differ from the final manifest")
        all_meta = [(sample_id, meta_by_id[sample_id]) for sample_id in manifest_ids]
        if any(anchors != ANCHOR_FRAMES for _, anchors in all_meta):
            raise ValueError("shard metadata contains a clip without 21 anchors")

        env = lmdb.open(
            str(partial_lmdb),
            map_size=args.map_size_gib << 30,
            subdir=True,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
        )
        transaction = env.begin(write=True)
        written = 0
        try:
            for store in source_stores:
                with store.env.begin(write=False) as source_txn:
                    for key, value in source_txn.cursor():
                        if key.startswith(b"__"):
                            continue
                        parts = key.decode('utf-8').split(':')
                        if (len(parts) != 2 or parts[0] not in meta_by_id or not parts[1].isdigit()
                                or not 0 <= int(parts[1]) < ANCHOR_FRAMES
                                or key != f'{parts[0]}:{int(parts[1])}'.encode()):
                            raise ValueError(f'Unexpected LMDB record key: {key!r}')
                        _validate_record(pickle.loads(value), parts[0], int(parts[1]))
                        if not transaction.put(key, value, overwrite=False):
                            raise ValueError(f"duplicate LMDB key {key.decode('utf-8')}")
                        written += 1
                        if written % 512 == 0:
                            transaction.commit()
                            transaction = env.begin(write=True)
            for key, value in (
                (b"__meta__", all_meta),
                (LATENT_CONTRACT_KEY.encode("utf-8"), latent_contract),
                (DATASET_ID_KEY.encode("utf-8"), {"dataset_id": dataset_id}),
                (DATA_CONTRACT_KEY.encode("utf-8"), data_contract),
            ):
                transaction.put(key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            transaction.commit()
            transaction = None
        finally:
            if transaction is not None:
                transaction.abort()
            env.sync()
            env.close()

        states = [(int(item["count"]), item["mean"], item["M2"]) for item in shard_stats]
        count, mean, M2 = _merge_welford(states)
        expected_count = len(all_meta) * ANCHOR_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
        if count != expected_count:
            raise ValueError(f"merged stats count is {count}, expected {expected_count}")
        torch.save(
            {
                "mean": mean.float(),
                "var": (M2 / max(count, 1)).float(),
                "count": int(count),
                "latent_contract": latent_contract,
                "dataset_id": dataset_id,
                "dataset_contract": DATASET_CONTRACT,
                "source_manifest_sha256": data_contract["source_manifest_sha256"],
                "confidence": CONFIDENCE_CONTRACT,
                "caption": CAPTION_CONTRACT,
            },
            partial_stats,
        )
        expected_records = len(all_meta) * ANCHOR_FRAMES
        if written != expected_records:
            raise ValueError(f"merged {written} records, expected {expected_records}")
        os.replace(partial_lmdb, final_lmdb)
        os.replace(partial_stats, final_stats)
        print(f"merge_complete=1 samples={len(all_meta)} records={written}")
    finally:
        for store in source_stores:
            store.close()



def _validate_record(record, sample_id, anchor):
    expected = {
        'latent': (48, LATENT_HEIGHT, LATENT_WIDTH),
        'depth': (LATENT_HEIGHT, LATENT_WIDTH), 'conf': (LATENT_HEIGHT, LATENT_WIDTH),
        'static_mask': (LATENT_HEIGHT, LATENT_WIDTH), 'K_l': (3, 3), 'E': (4, 4),
    }
    for name, shape in expected.items():
        value = np.asarray(record[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'{sample_id}:{anchor} invalid {name}')
    mask = np.asarray(record['static_mask'])
    confidence = np.asarray(record['conf'])
    if mask.dtype != np.bool_ or not np.array_equal(confidence > 0, mask):
        raise ValueError(f'{sample_id}:{anchor} confidence support differs from static mask')
    if (confidence < 0).any() or (confidence >= 1).any():
        raise ValueError(f'{sample_id}:{anchor} confidence outside [0, 1)')
    if (np.asarray(record['depth'])[mask] <= 0).any():
        raise ValueError(f'{sample_id}:{anchor} invalid supported depth')
    if int(record.get('source_rgb_index', -1)) != ANCHOR_INDICES[anchor]:
        raise ValueError(f'{sample_id}:{anchor} RGB/latent anchor misalignment')
    if not isinstance(record.get('caption', ''), str):
        raise ValueError(f'{sample_id}:{anchor} invalid caption')


def validate(args):
    rows = _read_manifest(Path(args.manifest), args.max_samples)
    output_dir = Path(args.output_dir)
    store = PrecomputedStore(str(output_dir / STORE_NAME), readonly=True)
    try:
        meta = [(str(sample), int(count)) for sample, count in store.get_meta()]
        expected_meta = [(row['sample_id'], ANCHOR_FRAMES) for row in rows]
        if meta != expected_meta:
            raise ValueError('LMDB scene order or anchor counts differ from manifest')
        if store.env.stat()['entries'] != len(rows) * ANCHOR_FRAMES + 4:
            raise ValueError('LMDB has missing or unexpected records')
        latent_contract = store.require_latent_contract()
        dataset_id = store.require_dataset_id()
        if store.get(DATA_CONTRACT_KEY) != _data_contract(args.manifest):
            raise ValueError('LMDB source manifest or data contract differs')
        if dataset_id != _dataset_id(Path(args.manifest), rows, latent_contract):
            raise ValueError('LMDB dataset ID differs from manifest and latent contract')
        # Check every record; recompute geometry for an evenly spaced source subset.
        for row in rows:
            for anchor in range(ANCHOR_FRAMES):
                _validate_record(store.get(f"{row['sample_id']}:{anchor}"), row['sample_id'], anchor)
        audit_indices = sorted(set(np.linspace(0, len(rows) - 1, min(args.audit_scenes, len(rows))).round().astype(int).tolist()))
        for index in audit_indices:
            row = rows[index]
            sample_id = row['sample_id']
            sample_dir = Path(args.sample_root) / sample_id
            if row.get('clip_sha256') and sha256_file(sample_dir / 'clip.mp4') != row['clip_sha256'].lower():
                raise ValueError(f'Source video differs from manifest: {sample_id}')
            depths, static, cameras, poses = _latent_geometry(sample_dir)
            confidence, static = _reprojection_reliability(depths, static, cameras, poses)
            caption = _caption(sample_dir)
            for anchor in range(ANCHOR_FRAMES):
                record = store.get(f'{sample_id}:{anchor}')
                for name, expected in (('depth', depths[anchor]), ('K_l', cameras[anchor]),
                                       ('E', poses[anchor]), ('conf', confidence[anchor])):
                    if not np.allclose(record[name], expected, rtol=1e-5, atol=1e-6):
                        raise ValueError(f'{sample_id}:{anchor} {name} differs from source geometry')
                if not np.array_equal(record['static_mask'], static[anchor]):
                    raise ValueError(f'{sample_id}:{anchor} static mask differs from source geometry')
                if record.get('caption', '') != caption:
                    raise ValueError(f'{sample_id}:{anchor} caption differs from source')
        stats = torch.load(output_dir / STATS_NAME, map_location='cpu', weights_only=False)
        if (stats.get('dataset_id') != dataset_id or stats.get('latent_contract') != latent_contract
                or stats.get('dataset_contract') != DATASET_CONTRACT
                or stats.get('source_manifest_sha256') != sha256_file(args.manifest)
                or stats.get('confidence') != CONFIDENCE_CONTRACT
                or stats.get('caption') != CAPTION_CONTRACT):
            raise ValueError('LMDB and latent statistics are not a matching pair')
        expected_count = len(rows) * ANCHOR_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
        if int(stats.get('count', -1)) != expected_count:
            raise ValueError('Latent statistics have an incorrect sample count')
        for name in ('mean', 'var'):
            value = stats[name]
            if tuple(value.shape) != (48,) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid latent statistics: {name}')
        if not torch.all(stats['var'] > 0):
            raise ValueError('Latent variance must be positive')
        report = {'pass': True, 'scenes': len(rows), 'anchors_per_scene': ANCHOR_FRAMES,
                  'source_scenes_checked': len(audit_indices), 'dataset_id': dataset_id,
                  'latent_stats_sha256': sha256_file(output_dir / STATS_NAME)}
        print(json.dumps(report, sort_keys=True))
        return report
    finally:
        store.close()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, function in (('encode', encode), ('merge', merge), ('validate', validate)):
        child = commands.add_parser(command)
        child.add_argument('--manifest', type=Path, default=ROOT / 'configs/re10k_train256_scenes.json')
        child.add_argument('--output-dir', type=Path, required=True)
        child.add_argument('--max-samples', type=int, default=0)
        if command != 'validate':
            child.add_argument('--num-shards', type=int, default=1)
            child.add_argument('--map-size-gib', type=int, default=1024)
        if command == 'encode':
            child.add_argument('--sample-root', type=Path, required=True)
            child.add_argument('--model-id', required=True)
            child.add_argument('--shard-id', type=int, default=0)
            child.add_argument('--device', default='cuda')
        if command == 'validate':
            child.add_argument('--sample-root', type=Path, required=True)
            child.add_argument('--audit-scenes', type=int, default=8)
        child.set_defaults(func=function)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if getattr(args, 'num_shards', 1) <= 0 or getattr(args, 'map_size_gib', 1) <= 0:
        parser.error('num-shards and map-size-gib must be positive')
    if hasattr(args, 'shard_id') and not 0 <= args.shard_id < args.num_shards:
        parser.error('shard-id must be in [0, num-shards)')
    if getattr(args, 'audit_scenes', 1) <= 0:
        parser.error('audit-scenes must be positive')
    rows = _read_manifest(args.manifest, args.max_samples)
    if getattr(args, 'num_shards', 1) > len(rows):
        parser.error('num-shards exceeds the number of selected scenes')
    if hasattr(args, 'sample_root') and not args.sample_root.is_dir():
        parser.error('sample-root must be an existing directory')
    load_runtime()
    args.func(args)


if __name__ == '__main__':
    main()

