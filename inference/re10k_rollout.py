"""RE10K E11 rollout extracted from the archived final experiment runner.
Scale calibration, real-anchor veto, inversion handoff, and chunk order retained.
"""
import json
import time
from pathlib import Path
from engram.common import digest
from re10k_protocol import FRAME_COUNTS, IDENTIFIER, chunk_indices

GEN_SEED = 20917

def read_composition(memory, pose, intrinsics, h=44, w=80):
    """Front-most slot provenance per latent cell for one target view (same projection as memory.read)."""
    import torch
    from engram.memory.readout import frustum_filter
    store = memory.memory.store
    _, positions, _, _ = store.gather_alive()
    real = store.alive_origin_real()
    pose = pose.to(memory.device).float()
    intrinsics = intrinsics.to(memory.device).float()
    keep, cu, cv, z = frustum_filter(positions, pose, intrinsics, h, w)
    cells = h * w
    if not keep.any():
        return {'hole': 1.0, 'front_generated': 0.0, 'real_visible': 0.0}
    cell = (cv[keep] * w + cu[keep]).long()
    zk = z[keep].float()
    rk = real[keep]
    zmin = torch.full((cells,), float('inf'), device=zk.device).scatter_reduce(0, cell, zk, reduce='amin')
    front = zk <= zmin[cell]
    covered = torch.isfinite(zmin)
    front_gen = torch.zeros(cells, dtype=torch.bool, device=zk.device)
    front_gen[cell[front & ~rk]] = True
    real_vis = torch.zeros(cells, dtype=torch.bool, device=zk.device)
    real_vis[cell[rk]] = True
    n = max(int(covered.sum()), 1)
    return {'hole': float(1 - covered.float().mean()), 'front_generated': float((front_gen & covered).sum()) / n, 'real_visible': float(real_vis.float().mean())}

def run_case(root, case, manifest, model, estimator):
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from engram.core import ProtectedMemory, conditioning, decoded_rgb, seed_all, write_json
    from engram.models.wan_latents import decode_wan_latents
    bundle = torch.load(root / 'data' / case['name'] / 'inference.pt', map_location='cpu', weights_only=False)
    if digest(root / 'data' / case['name'] / 'inference.pt') != case['input_sha256']:
        raise ValueError('Prepared input changed')
    text = bundle['text'].to('cuda', dtype=torch.bfloat16)
    for protocol in ['nvs', 'closed_loop']:
        output = root / 'evaluation' / case['name'] / protocol
        if (output / 'COMPLETE.json').exists():
            raise FileExistsError(f'Output already completed: {output}; select a new root or another case')
        output.mkdir(parents=True, exist_ok=True)
        poses = bundle['poses'][protocol]
        cal = json.loads((Path(__file__).resolve().parents[1] / 'configs/scale_calibration.json').read_text())
        scale = float(np.exp(cal['a'] + cal['b'] * np.log(bundle['seed_depth_metric_median'])))
        poses = poses.clone()
        poses[:, :3, 3] *= scale
        bundle['poses'] = {**bundle['poses'], protocol: poses}
        print(json.dumps({'variant': 'e11_invhandoff', 'case': case['name'], 'protocol': protocol, 'translation_scale': scale}), flush=True)
        memory = ProtectedMemory(bundle, 'cpu', writeback=True)
        original = memory.fingerprint(True)
        prefix = bundle['seed_latent'].to('cuda', dtype=torch.bfloat16)
        seed_all(GEN_SEED)
        generator = torch.Generator(device='cuda').manual_seed(GEN_SEED)
        chunks, log = ([], [])
        for start in range(0, FRAME_COUNTS[protocol] - 1, 32):
            t0 = time.monotonic()
            indices, valid_count = chunk_indices(protocol, start)
            chunk_poses = poses[indices[::4]]
            cameras = bundle['intrinsics_latent'][0].repeat(9, 1, 1)
            reads = [read_composition(memory, pose, cameras[0]) for pose in chunk_poses]
            condition = conditioning(chunk_poses, cameras, 44, 80, memory).to('cuda', dtype=torch.bfloat16)
            latent = model.sample_chunk(text, condition, (1, 48, 9, 44, 80), num_steps=40, guidance_scale=1.0, ref_latent=prefix, ref_len=1, generator=generator)
            if not torch.isfinite(latent).all():
                raise FloatingPointError('Nonfinite generation')
            frames = decoded_rgb(decode_wan_latents(model.vae, latent))
            if start == 0:
                frames[0] = bundle['seed_rgb']
            chunks.append(frames[:valid_count] if start == 0 else frames[1:valid_count])
            wrote = 'generated'
            gate = None
            from vtw import gated_write_chunk
            gate = gated_write_chunk(memory, estimator, bundle, frames, latent, chunk_poses, start, valid_count, mode='real_only')['gate']
            if memory.fingerprint(True) != original:
                raise RuntimeError('Real seed changed')
            prefix_source = None
            if start + 32 < FRAME_COUNTS[protocol] - 1:
                from inv_handoff import invert_last_frame
                prefix = invert_last_frame(model.vae, frames[-1])
                prefix_source = 'generated_last_inverted'
            log.append({'start': start, 'valid_count': valid_count, 'write': wrote, 'next_prefix': prefix_source, 'read_per_latent': reads, 'gate': gate, 'memory_after': memory.stats(), 'seconds': time.monotonic() - t0})
            write_json(output / 'CHUNK_LOG.json', log)
            print(json.dumps({'variant': 'e11_invhandoff', 'case': case['name'], 'protocol': protocol, 'chunk_start': start, 'write': wrote, 'prefix': prefix_source}), flush=True)
        video = np.concatenate(chunks)
        if video.shape != (FRAME_COUNTS[protocol], 704, 1280, 3):
            raise RuntimeError('Wrong output shape')
        np.savez_compressed(output / 'frames.npz', rgb=video)
        imageio.mimwrite(output / 'engram_raw.mp4', list(video), fps=24, quality=8)
        write_json(output / 'COMPLETE.json', {'identifier': IDENTIFIER, 'case': case['name'], 'protocol': protocol, 'frames': len(video), 'checkpoint_sha256': manifest['checkpoint_sha256'], 'input_sha256': case['input_sha256'], 'recipe': 'e11_invhandoff', 'GT_history_read': False, 'generation_seed': GEN_SEED, 'steps': 40, 'cfg': 1.0, 'immutable_real_unchanged': True})
