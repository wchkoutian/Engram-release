import argparse
import collections
import json
import time
from pathlib import Path
import sys

import os

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.core import append_json, build_backbone, digest, recipe_contract, save_tensor, seed_all, write_json
from engram.common import tensor_digest
from engram.train.common import capture_rng, flow_loss, training_source_hashes
from engram.data.windows import epoch_order
from engram.core import MODEL_ROOT
from engram.data.contracts import SCENE_COUNT, WINDOW_COUNT, data_contract, require_data_contract, validate_training_manifest, validate_training_window


CAPTION_DROPOUT = 0.1
CAPTION_SEED = 52021
# V13: per-microstep choice between the real-history window and its self-generated-history twin.
SELFGEN_PROBABILITY = 0.0  # V14: self-gen twins disabled (V13 lost on Local-10 2026-10-15)
SELFGEN_SEED = 61021
LATENT_SHAPE = (1, 48, 9, 44, 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--resume', action='store_true')  # V14: exact continuation from runs/resume_latest.pt
    args = parser.parse_args()
    # Data parallel over microsteps: each process sees one GPU (CUDA_VISIBLE_DEVICES) and owns microsteps k with
    # k % world == rank. Every process replays the full CUDA RNG sequence so noise/sigma equal the 1-GPU run.
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(0)
    if args.accumulation % world:
        raise ValueError('accumulation must be divisible by WORLD_SIZE')
    main_rank = rank == 0
    root = args.root
    output = root / 'runs'
    output.mkdir(exist_ok=True)
    if world > 1:
        dist.barrier()
    if not args.resume and (main_rank and (output / 'metrics.jsonl').exists() or any(output.glob('*.pt'))):
        raise RuntimeError('Use a fresh output root for direct training')
    manifest = json.loads((root / 'manifest.json').read_text())
    ready = json.loads((root / 'DATA_READY.json').read_text())
    require_data_contract(ready)
    validate_training_manifest(manifest)
    if ready['manifest_sha256'] != digest(root / 'manifest.json'):
        raise RuntimeError('Training manifest hash mismatch')
    for record in manifest['records']:
        if digest(root / record['file']) != record['sha256'] or ('selfgen_file' in record and digest(root / record['selfgen_file']) != record['selfgen_sha256']):
            raise RuntimeError('Training cache hash mismatch')
    config = {'version': 'V11', 'arm': 'G', 'profile': 'reference8', 'recipe': recipe_contract(), 'data_contract': data_contract(), 'manifest_sha256': digest(root / 'manifest.json'), 'initialization': 'direct_wan_copy_no_engram_checkpoint', 'pretrained_model': MODEL_ROOT, 'pretrained_config_sha256': digest(Path(MODEL_ROOT) / 'transformer/config.json'), 'initialization_seed': 3407, 'engram_checkpoint_loaded': False, 'optimizer': 'AdamW', 'learning_rate': 5e-6, 'betas': [.9, .999], 'weight_decay': .01, 'scheduler': 'constant', 'gradient_clip': 1., 'noise_seed': 44111, 'steps': args.steps, 'gradient_accumulation': args.accumulation, 'effective_batch': args.accumulation, 'scene_count': SCENE_COUNT, 'window_count': WINDOW_COUNT, 'initial_optimizer': 'fresh_no_single_scene_optimizer_state', 'text_condition': 'per_window_vlm_caption_v1', 'caption_dropout': CAPTION_DROPOUT, 'caption_dropout_seed': CAPTION_SEED, 'history_mix': {'selfgen_probability': SELFGEN_PROBABILITY, 'seed': SELFGEN_SEED, 'selfgen_checkpoint_sha256': manifest.get('selfgen_checkpoint_sha256'), 'selfgen_prefix': 'generated frame at window start'}, 'caption_text_sha256': digest(root / 'caption_text.pt'), 'captions_sha256': digest(root / 'captions.json'), 'source_hashes': training_source_hashes()}
    config_path = output / 'config.json'
    if args.resume:
        saved = json.loads(config_path.read_text())
        skip = ('steps', 'resumed', 'source_hashes')  # train.py itself gains --resume
        if {k: v for k, v in saved.items() if k not in skip} != {k: v for k, v in config.items() if k not in skip}:
            raise RuntimeError('Resume config differs beyond steps')
        config['resumed'] = saved.get('resumed', []) + [{'from_steps': saved['steps'], 'to_steps': args.steps, 'previous_source_hashes': saved['source_hashes']}]
    elif main_rank and config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing training config differs')
    if main_rank:
        write_json(config_path, config)
    seed_all(3407)
    model = build_backbone('reference8', training=True, memory_fraction=.40)
    assert model.side_branch.initialization == 'wan_copy'
    for branch, layer in zip(model.side_branch.blocks, model.control_layers):
        reference = model.transformer.blocks[layer].state_dict()
        for name, tensor in branch.state_dict().items():
            if not torch.equal(tensor, reference[name].to(dtype=torch.bfloat16).to(dtype=tensor.dtype)):
                raise RuntimeError(f'Wan-copy initialization mismatch: layer{layer}/{name}')
    for projection in model.side_branch.zero_out:
        if any(torch.count_nonzero(parameter).item() for parameter in projection.parameters()):
            raise RuntimeError('Residual projection is not zero initialized')
    main_rank and write_json(output / 'INITIALIZATION_AUDIT.json', {'pass': True, 'source': MODEL_ROOT, 'wan_copy_blocks_equal': list(model.control_layers), 'copy_dtype_path': 'source_to_bfloat16_then_float32_existing_V11_behavior', 'zero_residuals': True, 'engram_checkpoint_loaded': False, 'initialization_seed': 3407})
    parameters = list(model.side_branch.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=5e-6, betas=(.9, .999), weight_decay=.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    bundle = torch.load(root / 'caption_text.pt', map_location='cpu', weights_only=False)
    if set(bundle['captions']) != {record['file'] for record in manifest['records']}:
        raise RuntimeError('Caption bundle does not cover the training windows exactly')
    captions = {key: value.to('cuda', dtype=torch.bfloat16) for key, value in bundle['captions'].items()}
    empty_text = bundle['empty'].to('cuda', dtype=torch.bfloat16)
    # Separate CPU generator: the noise/sigma RNG stream stays identical to the empty-text A3 run.
    caption_rng = torch.Generator().manual_seed(CAPTION_SEED)
    selfgen_rng = torch.Generator().manual_seed(SELFGEN_SEED)
    selfgen_count = 0
    dropped = 0
    latest = output / 'resume_latest.pt'
    start, consumed = 0, 0
    exposures = collections.Counter()
    if main_rank and latest.exists() and not args.resume:
        raise RuntimeError('Direct-training entrypoint refuses existing checkpoints')
    seed_all(44111)
    if args.resume:
        import random
        import numpy as np
        resume = torch.load(latest, map_location='cpu', weights_only=False)
        model.side_branch.load_state_dict(resume['side_branch'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        start, consumed = int(resume['step']), int(resume['consumed_windows'])
        exposures = collections.Counter(resume['scene_exposures'])
        assert start < args.steps and consumed == start * args.accumulation
        rng = resume['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
        for _ in range(consumed):
            dropped += bool(torch.rand((), generator=caption_rng) < CAPTION_DROPOUT)
            selfgen_count += bool(torch.rand((), generator=selfgen_rng) < SELFGEN_PROBABILITY)
        del resume
        main_rank and print(json.dumps({'event': 'RESUMED', 'step': start, 'consumed': consumed, 'target': args.steps}), flush=True)
    audit = {'world_size': world, 'trainable': sum(parameter.numel() for parameter in parameters), 'backbone_frozen': not any(parameter.requires_grad for parameter in model.transformer.parameters()), 'optimizer_steps': args.steps, 'microbatch': 1, 'gradient_accumulation': args.accumulation}
    assert audit['backbone_frozen']
    if main_rank:
        write_json(output / 'parameter_audit.json', audit)
        write_json(output / 'status.json', {'phase': 'training_initialized', 'step': start, 'target_step': args.steps})
    main_rank and print(json.dumps({'event': 'TRAIN_START', 'step': start, **audit}), flush=True)
    cache = collections.OrderedDict()
    order_epoch, order = None, None
    for step in range(start + 1, args.steps + 1):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        batch = []
        local = torch.zeros((args.accumulation, 2), dtype=torch.float64, device='cuda')
        for microstep in range(args.accumulation):
            epoch, position = divmod(consumed, len(manifest['records']))
            if order_epoch != epoch:
                order = epoch_order(len(manifest['records']), epoch)
                order_epoch = epoch
            record = manifest['records'][order[position]]
            drop = bool(torch.rand((), generator=caption_rng) < CAPTION_DROPOUT)
            dropped += drop
            selfgen = bool(torch.rand((), generator=selfgen_rng) < SELFGEN_PROBABILITY)
            selfgen_count += selfgen
            key = record['selfgen_file'] if selfgen else record['file']
            if microstep % world == rank:
                if key not in cache:
                    cache[key] = torch.load(root / key, map_location='cpu', weights_only=False)
                    validate_training_window(cache[key], record)
                    if len(cache) > 32:
                        cache.popitem(last=False)
                cache.move_to_end(key)
                item = cache[key]
                text = empty_text if drop else captions[record['file']]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, sigma, noise_hash = flow_loss(model, item['latent'].to('cuda'), text, item['condition'].to('cuda'), item['valid_mask'].to('cuda'))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {step}')
                (loss / args.accumulation).backward()
                local[microstep, 0], local[microstep, 1] = float(loss.detach()), sigma
            else:
                # Replay flow_loss RNG consumption (randn_like on the bf16 latent, then rand) without compute.
                noise = torch.randn(LATENT_SHAPE, dtype=torch.bfloat16, device='cuda')
                torch.rand((1,), device='cuda')
                noise_hash = tensor_digest([noise]) if main_rank else None
            exposures[record['sample_id']] += 1
            batch.append({'sample_id': record['sample_id'], 'window': record['window'], 'noise_sha256': noise_hash, 'caption_dropped': drop, 'selfgen': selfgen})
            consumed += 1
        if world > 1:
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        losses, sigmas = local[:, 0].tolist(), local[:, 1].tolist()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        assert not any(parameter.grad is not None for parameter in model.transformer.parameters())
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        row = {'step': step, 'loss': sum(losses) / len(losses), 'sigma_mean': sum(sigmas) / len(sigmas), 'gradient_norm': float(gradient), 'seconds': time.monotonic() - tick, 'consumed_windows': consumed, 'scenes_seen': len(exposures), 'batch': batch, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
        if not main_rank:
            continue
        append_json(output / 'metrics.jsonl', row)
        write_json(output / 'status.json', {'phase': 'training', 'step': step, 'target_step': args.steps, 'loss': row['loss'], 'scenes_seen': len(exposures), 'consumed_windows': consumed, 'complete': False})
        if step <= 5 or step % 25 == 0:
            print(json.dumps(row), flush=True)
        if step % 250 == 0 or step == args.steps:
            state = {'step': step, 'profile': 'reference8', 'side_branch': model.side_branch.state_dict(), 'config': config}
            if step % 500 == 0 or step == args.steps:
                save_tensor(output / f'step_{step:04d}.pt', state)
            save_tensor(latest, {**state, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': capture_rng(), 'consumed_windows': consumed, 'scene_exposures': dict(exposures)})
            write_json(output / 'CHECKPOINT_STATUS.json', {'step': step, 'resume': str(latest), 'scene_exposures': dict(exposures)})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if not main_rank:
        return
    checkpoint = output / f'step_{args.steps:04d}.pt'
    write_json(output / 'TRAIN_COMPLETE.json', {'step': args.steps, 'checkpoint_sha256': digest(checkpoint), 'resume_sha256': digest(latest), 'scene_exposures': dict(exposures), 'caption_dropped_microbatches': dropped, 'selfgen_microbatches': selfgen_count, 'visual_acceptance': False})
    write_json(output / 'status.json', {'phase': 'complete', 'step': args.steps, 'complete': True, 'quality_acceptance': False})


if __name__ == '__main__':
    main()
