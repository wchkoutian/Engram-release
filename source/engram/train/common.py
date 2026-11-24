import random
from pathlib import Path
import numpy as np
import torch
from ..core import tensor_digest
from ..common import digest


def training_source_hashes():
    root = Path(__file__).resolve().parents[3]
    files = [*(root / 'scripts').glob('*.py'), *(root / 'source/engram').rglob('*.py')]
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(files)}


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


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
    noise_hash = tensor_digest([noise])
    return (errors * mask).sum() / mask.sum().clamp_min(1), float(sigma.mean().detach()), noise_hash
