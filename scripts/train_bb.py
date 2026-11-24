"""V22 stage 2 (2026-10-20, user-approved: full backbone fine-tuning, no LoRA).
Stage 1 = V14 step_4000 side branch (loaded, frozen, bf16 as at inference).  Stage 2 trains ALL Wan transformer parameters
directly on the V14 data / order / batch / caption dropout / noise RNG stream (same microstep data-parallel replay as
scripts/train.py).  Memory: backbone params + grads stay bf16 on every rank; the fp32 master copy and AdamW state are
sharded by parameter across ranks (ZeRO-1 style): grads all-reduced, global norm clipped, each rank updates the parameters
it owns, then broadcasts the new bf16 values.  LR 5e-6 constant after 100 linear warm-up steps, AdamW betas (.9,.999),
weight decay 0 (pretrained weights), grad clip 1.0.  Resume: runs/resume_bb_latest/{common.pt, shard_<rank>.pt} every 250 steps.
Checkpoints step_NNNN.pt = {'side_branch', 'backbone_blocks' (full transformer state, bf16), 'config'} (loadable by
the release RE10K and WorldScore inference entrypoints)."""
import argparse
import collections
import json
import os
import time
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from engram.core import append_json, build_backbone, digest, recipe_contract, save_tensor, seed_all, write_json
from engram.common import tensor_digest
from engram.train.common import capture_rng, flow_loss, training_source_hashes
from engram.data.windows import epoch_order
from engram.core import MODEL_ROOT
from engram.data.contracts import data_contract, require_data_contract, validate_training_manifest, validate_training_window

CAPTION_DROPOUT = 0.1
CAPTION_SEED = 52021
LATENT_SHAPE = (1, 48, 9, 44, 80)
LR, WARMUP, WD = 5e-6, 100, 0.
STAGE1 = Path(os.environ['ENGRAM_STAGE1_CHECKPOINT'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(0)
    assert args.accumulation % world == 0
    main_rank = rank == 0
    root = args.root
    output = root / 'runs'
    output.mkdir(exist_ok=True)
    resume_dir = output / 'resume_bb_latest'
    if world > 1:
        dist.barrier()
    if not args.resume and (any(output.glob('step_*.pt')) or (output / 'metrics.jsonl').exists()):
        raise RuntimeError('Use a fresh output root or --resume')
    manifest = json.loads((root / 'manifest.json').read_text())
    ready = json.loads((root / 'DATA_READY.json').read_text())
    require_data_contract(ready)
    validate_training_manifest(manifest)
    assert ready['manifest_sha256'] == digest(root / 'manifest.json')
    stage1_sha = digest(STAGE1)
    config = {'version': 'V22-stage2-bbfull', 'recipe': recipe_contract(), 'data_contract': data_contract(), 'manifest_sha256': digest(root / 'manifest.json'),
              'pretrained_model': MODEL_ROOT, 'stage1_checkpoint': str(STAGE1), 'stage1_sha256': stage1_sha, 'side_branch': 'frozen bf16 from stage 1',
              'backbone': 'all transformer parameters trainable, bf16 params/grads, fp32 master + AdamW sharded across ranks', 'lora': False,
              'optimizer': 'AdamW', 'learning_rate': LR, 'warmup_steps': WARMUP, 'betas': [.9, .999], 'weight_decay': WD, 'gradient_clip': 1.,
              'noise_seed': 44111, 'steps': args.steps, 'gradient_accumulation': args.accumulation, 'caption_dropout': CAPTION_DROPOUT,
              'caption_dropout_seed': CAPTION_SEED, 'caption_text_sha256': digest(root / 'caption_text.pt'), 'world_size': world,
              'source_hashes': training_source_hashes()}
    cfg_path = output / 'config.json'
    if main_rank:
        if args.resume:
            saved = json.loads(cfg_path.read_text())
            skip = ('steps', 'world_size', 'resumed')
            assert {k: v for k, v in saved.items() if k not in skip} == {k: v for k, v in config.items() if k not in skip}, 'resume config differs'
            config['resumed'] = saved.get('resumed', []) + [{'world_size': world}]
        write_json(cfg_path, config)
    seed_all(3407)
    model = build_backbone('reference8', training=True, memory_fraction=.62)
    s1 = torch.load(STAGE1, map_location='cpu', weights_only=False)
    model.side_branch.load_state_dict(s1['side_branch'], strict=True)
    del s1
    model.side_branch.to(torch.bfloat16)
    model.side_branch.requires_grad_(False)
    T = model.transformer
    T.requires_grad_(True)
    names = [n for n, q in T.named_parameters()]
    params = [q for _, q in T.named_parameters()]
    dtypes = collections.Counter(str(q.dtype) for q in params)  # Wan keeps a few small tensors (e.g. norms/tables) in fp32
    # ownership: greedy size balance, deterministic
    owner, load = [], [0] * world
    for i in sorted(range(len(params)), key=lambda i: -params[i].numel()):
        r = min(range(world), key=lambda r: load[r])
        load[r] += params[i].numel()
        owner.append((i, r))
    owner = dict(owner)
    mine = [i for i in range(len(params)) if owner[i] == rank]
    master = [params[i].detach().float().clone().requires_grad_(False) for i in mine]
    optimizer = torch.optim.AdamW(master, lr=LR, betas=(.9, .999), weight_decay=WD, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / WARMUP))
    bundle = torch.load(root / 'caption_text.pt', map_location='cpu', weights_only=False)
    assert set(bundle['captions']) == {r['file'] for r in manifest['records']}
    captions = bundle['captions']
    empty_text = bundle['empty'].to('cuda', dtype=torch.bfloat16)
    caption_rng = torch.Generator().manual_seed(CAPTION_SEED)
    start, consumed, dropped = 0, 0, 0
    exposures = collections.Counter()
    seed_all(44111)
    if args.resume:
        import random
        import numpy as np
        common = torch.load(resume_dir / 'common.pt', map_location='cpu', weights_only=False)
        assert common['world'] == world and common['names'] == names
        T.load_state_dict(common['backbone'], strict=True)
        shard = torch.load(resume_dir / f'shard_{rank}.pt', map_location='cpu', weights_only=False)
        assert shard['mine'] == mine
        for m, v in zip(master, shard['master']):
            m.copy_(v)
        optimizer.load_state_dict(shard['optimizer'])
        scheduler.load_state_dict(shard['scheduler'])
        start, consumed = int(common['step']), int(common['consumed'])
        exposures = collections.Counter(common['exposures'])
        rng = common['rng']
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
        for _ in range(consumed):
            dropped += bool(torch.rand((), generator=caption_rng) < CAPTION_DROPOUT)
        del common, shard
        main_rank and print(json.dumps({'event': 'RESUMED', 'step': start}), flush=True)
    audit = {'world_size': world, 'backbone_trainable': sum(q.numel() for q in params), 'owned_params': sum(params[i].numel() for i in mine),
             'side_branch_trainable': sum(q.numel() for q in model.side_branch.parameters() if q.requires_grad), 'lora': False, 'param_dtypes': dict(dtypes)}
    assert audit['side_branch_trainable'] == 0
    main_rank and write_json(output / 'parameter_audit.json', audit)
    print(json.dumps({'event': 'TRAIN_START', 'rank': rank, 'step': start, **audit}), flush=True)
    cache = collections.OrderedDict()
    order_epoch, order = None, None
    for step in range(start + 1, args.steps + 1):
        tick = time.monotonic()
        for q in params:
            q.grad = None
        batch = []
        local = torch.zeros((args.accumulation, 2), dtype=torch.float64, device='cuda')
        for micro in range(args.accumulation):
            epoch, position = divmod(consumed, len(manifest['records']))
            if order_epoch != epoch:
                order, order_epoch = epoch_order(len(manifest['records']), epoch), epoch
            record = manifest['records'][order[position]]
            drop = bool(torch.rand((), generator=caption_rng) < CAPTION_DROPOUT)
            dropped += drop
            key = record['file']
            if micro % world == rank:
                if key not in cache:
                    cache[key] = torch.load(root / key, map_location='cpu', weights_only=False)
                    validate_training_window(cache[key], record)
                    if len(cache) > 32:
                        cache.popitem(last=False)
                item = cache[key]
                text = empty_text if drop else captions[key].to('cuda', dtype=torch.bfloat16)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, sigma, noise_hash = flow_loss(model, item['latent'].to('cuda'), text, item['condition'].to('cuda'), item['valid_mask'].to('cuda'))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {step}')
                (loss / args.accumulation).backward()
                local[micro, 0], local[micro, 1] = float(loss.detach()), sigma
            else:
                noise = torch.randn(LATENT_SHAPE, dtype=torch.bfloat16, device='cuda')
                torch.rand((1,), device='cuda')
                noise_hash = tensor_digest([noise]) if main_rank else None
            exposures[record['sample_id']] += 1
            batch.append({'sample_id': record['sample_id'], 'window': record['window'], 'noise_sha256': noise_hash, 'caption_dropped': drop})
            consumed += 1
        for q in params:
            if q.grad is None:
                q.grad = torch.zeros_like(q)
        if world > 1:
            for q in params:
                dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        norm_sq = torch.zeros((), device='cuda', dtype=torch.float32)
        for q in params:
            norm_sq += q.grad.float().pow(2).sum()
        gnorm = float(norm_sq.sqrt())
        if not (gnorm == gnorm and gnorm < float('inf')):
            raise FloatingPointError(f'Nonfinite grad norm at {step}')
        coef = min(1., 1. / (gnorm + 1e-6))
        for m, i in zip(master, mine):
            m.grad = params[i].grad.float() * coef
        optimizer.step()
        scheduler.step()
        for m, i in zip(master, mine):
            m.grad = None
            params[i].data.copy_(m.data.to(params[i].dtype))
        for q in params:
            q.grad = None
        if world > 1:
            for i, q in enumerate(params):
                dist.broadcast(q.data, src=owner[i])
        torch.cuda.synchronize()
        losses, sigmas = local[:, 0].tolist(), local[:, 1].tolist()
        row = {'step': step, 'loss': sum(losses) / len(losses), 'sigma_mean': sum(sigmas) / len(sigmas), 'gradient_norm': gnorm,
               'lr': scheduler.get_last_lr()[0], 'seconds': time.monotonic() - tick, 'consumed_windows': consumed, 'batch': batch,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
        if main_rank:
            append_json(output / 'metrics.jsonl', row)
            write_json(output / 'status.json', {'phase': 'training', 'step': step, 'target_step': args.steps, 'loss': row['loss']})
            if step <= 5 or step % 25 == 0:
                print(json.dumps({k: v for k, v in row.items() if k != 'batch'}), flush=True)
        if step % 250 == 0 or step == args.steps:
            tmp = output / 'resume_bb_tmp'
            if main_rank:
                tmp.mkdir(exist_ok=True)
            if world > 1:
                dist.barrier()
            save_tensor(tmp / f'shard_{rank}.pt', {'mine': mine, 'master': [m.detach().cpu() for m in master], 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})
            if main_rank:
                bstate = {k: v.detach().cpu() for k, v in T.state_dict().items()}
                save_tensor(tmp / 'common.pt', {'step': step, 'consumed': consumed, 'exposures': dict(exposures), 'rng': capture_rng(), 'world': world, 'names': names, 'backbone': bstate})
                if step % 1000 == 0 or step == args.steps:
                    save_tensor(output / f'step_{step:04d}.pt', {'step': step, 'profile': 'reference8', 'side_branch': model.side_branch.state_dict(), 'backbone_blocks': bstate, 'config': config})
                del bstate
            if world > 1:
                dist.barrier()
            if main_rank:
                import shutil
                if resume_dir.exists():
                    shutil.rmtree(resume_dir)
                tmp.rename(resume_dir)
                write_json(output / 'CHECKPOINT_STATUS.json', {'step': step, 'resume': str(resume_dir)})
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if main_rank:
        ck = output / f'step_{args.steps:04d}.pt'
        write_json(output / 'TRAIN_COMPLETE.json', {'step': args.steps, 'checkpoint_sha256': digest(ck), 'stage1_sha256': stage1_sha, 'caption_dropped_microbatches': dropped})


if __name__ == '__main__':
    main()
