"""Inversion handoff (2026-10-25): next-chunk prefix latent from the decoded last frame WITHOUT an extra VAE round trip.
Diagnosis: each chunk handoff re-encodes the decoded last frame, adding one decode(encode()) per chunk; sharpness of generated
chunks (.87/.80/.75 of the input) matches 1/2/3 VAE round trips of the input (.86/.79/.73).  Here the prefix is
z* = argmin_z |D(z) - x|_1 (Adam lr .02, 50 steps, init E(x)), x = decoded last frame, so D(z*) reproduces x (55 dB vs 45 dB)
and detail no longer compounds (input test: .862/.861 after 2/3 handoffs vs .786/.732).  Pure handoff fidelity: no GT, no
pixels pasted into the output; the model still generates every frame."""
import os

import torch

STEPS, LR = 50, .02
# 10-25 rev2: the backward pass through the decoder needs ~20 GiB extra (untiled; tiled needs 28 GiB and is 3x slower) and the
# runners cap the process at .27-.30 of the GPU, so inversion runs untiled and raises the cap to INV_MEM_FRACTION (default .47);
# the runners' free-memory gates are raised by the same amount, so no co-tenant memory is taken.
MEM_FRACTION = float(os.environ.get('ENGRAM_MEMORY_FRACTION', os.environ.get('INV_MEM_FRACTION', '.47')))


def invert_last_frame(vae, frame, steps=STEPS, lr=LR):
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
    tiled = getattr(vae, 'use_tiling', False)
    vae.disable_tiling()
    try:
        z = _invert(vae, frame, steps, lr)
    finally:
        if tiled:
            vae.enable_tiling()
        torch.cuda.empty_cache()
    return z


def _invert(vae, frame, steps, lr):
    from engram.common import rgb_tensor
    from engram.models.wan_latents import denormalize_wan_latents, encode_wan_latents
    with torch.enable_grad():
        target = rgb_tensor(frame[None]).to('cuda', dtype=torch.float32)
        with torch.no_grad():
            z = encode_wan_latents(vae, target).float()
        z = z.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            rec = vae.decode(denormalize_wan_latents(vae, z)).sample
            loss = (rec.float() - target).abs().mean()
            loss.backward()
            opt.step()
    return z.detach().to(torch.bfloat16)
