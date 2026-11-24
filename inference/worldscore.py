"""WorldScore generation with text scheduling, real-anchor veto and inversion.

The caller supplies toolkit-prepared input images and camera_data.json files.
Benchmark selection and scoring are separate from this generation adapter.
"""
import json
import os
from pathlib import Path
import shutil
import time
from engram.common import digest as sha, write_json as write
from vtw import gated_write_chunk
from inv_handoff import invert_last_frame

CODE = Path(__file__).resolve().parents[1]
IDENTIFIER = 'engram-worldscore-release-v1'

def event(**obj):
    print(json.dumps(obj), flush=True)

def initialize():
    import re10k
    return re10k, ASSETS

def read_manifest():
    m = json.loads((ROOT / 'MANIFEST.json').read_text())
    if m['identifier'] != IDENTIFIER or m['frozen_core_manifest_sha256'] != sha(CODE / 'PACKAGE_MANIFEST.json'):
        raise ValueError('Prepared WorldScore inputs or release version differ')
    return m

def configure(args, assets):
    global ROOT, OUTPUT, ASSETS
    ROOT, ASSETS = args.root, assets
    OUTPUT = ROOT / 'model_outputs/Engram/worldscore_output'

def prepare(args, assets):
    configure(args, assets)
    from input_validation import validate_worldscore_inputs
    source = args.manifest.resolve()
    records = validate_worldscore_inputs(source)
    if ROOT.exists() and any(ROOT.iterdir()):
        raise FileExistsError('Use a new WorldScore output root')
    cases = []
    for row in records:
        folder = row['directory']
        name = row['name']
        out = OUTPUT / name
        out.mkdir(parents=True)
        for file in ('input_image.png', 'camera_data.json', 'image_data.json'):
            shutil.copy2(folder / file, out / file)
        cases.append({'name': name, 'relative_output': name, 'seed': 42,
                      'positive_prompt': row['positive_prompt'],
                      'positive_prompt_2': row['positive_prompt_2'],
                      'metadata_sha256': {file: sha(out / file) for file in ('input_image.png', 'camera_data.json', 'image_data.json')}})
    write(ROOT / 'MANIFEST.json', {'identifier': IDENTIFIER, 'cases': cases,
          'frames': 81, 'resolution': [1280, 704], 'fps': 16, 'steps': 40,
          'guidance_scale': 1., 'checkpoint_sha256': assets['checkpoint_sha256'],
          'frozen_core_manifest_sha256': sha(CODE / 'PACKAGE_MANIFEST.json'),
          'input_manifest_sha256': sha(source), 'input_manifest': str(source),
          'scope': 'User-provided WorldScore cases; no automatic paper-subset claim'})
    text_conditions()
    geometry()

def infer(args, assets):
    import fcntl
    configure(args, assets)
    # One writer per shard. Keep the shard layout fixed when resuming.
    shard = os.environ.get('ENGRAM_SHARD', '0/1')
    from input_validation import parse_shard
    k, n = parse_shard(shard)
    with (ROOT / f'generate_{k}_of_{n}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generate()

def text_conditions():
    m = read_manifest()
    local10, assets = initialize()
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers import WanPipeline
    from engram.core import save_tensor, setup_cuda
    setup_cuda(.27)
    destination = ROOT / 'data/text.pt'
    if destination.exists():
        info = json.loads((ROOT / 'TEXT_COMPLETE.json').read_text())
        assert sha(destination) == info['sha256']
        event(text='existing verified')
        return
    old = torch.load(assets['text_bundle'], map_location='cpu', weights_only=False)['text']
    length = old.shape[1]
    tokenizer = AutoTokenizer.from_pretrained(assets['wan_model'], subfolder='tokenizer', local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(assets['wan_model'], subfolder='text_encoder', torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
    pipe = WanPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None, transformer=None, scheduler=None)
    embeddings = {}
    with torch.no_grad():
        for prompt in ['', *[prompt for case in m['cases'] for prompt in (case['positive_prompt'], case['positive_prompt_2'])]]:
            encoded, _ = pipe.encode_prompt(prompt=[prompt], device='cuda', num_videos_per_prompt=1, do_classifier_free_guidance=False, max_sequence_length=length)
            assert tuple(encoded.shape) == tuple(old.shape) and torch.isfinite(encoded).all()
            embeddings[prompt] = encoded.cpu()
            event(text_encoded=len(embeddings), total=2 * len(m['cases']) + 1)
    assert all(not torch.equal(embeddings[''], embeddings[c['positive_prompt']]) for c in m['cases'])
    save_tensor(destination, embeddings)
    write(ROOT / 'TEXT_COMPLETE.json', {'sha256': sha(destination), 'length': length, 'shape': list(old.shape), 'dtype': 'bfloat16', 'model': assets['wan_model'], 'encoder': 'WanPipeline.encode_prompt with pretrained UMT5 and tokenizer; positive and empty negative encoded together', 'source_empty_embedding_max_abs_difference': float((embeddings[''].float() - old.float()).abs().max())})


def geometry():
    m = read_manifest()
    local10, assets = initialize()
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan
    from engram.core import build_depth_estimator, rgb_tensor, save_tensor, setup_cuda
    from engram.geometry.projection import blender_c2w_to_relative_opencv_w2c
    from engram.models.wan_latents import encode_wan_latents, wan_latent_contract
    state, training_sha = local10.checkpoint(assets)
    assert assets['checkpoint_sha256'] == m['checkpoint_sha256']
    del state
    setup_cuda(.27)
    estimator = build_depth_estimator()
    vae = AutoencoderKLWan.from_pretrained(assets['wan_model'], subfolder='vae', torch_dtype=torch.float32, local_files_only=True).to('cuda').eval()
    vae.enable_tiling()
    vae.enable_slicing()
    with torch.no_grad():
        for case in m['cases']:
            folder = ROOT / 'data' / case['name']
            done = folder / 'PREPARED.json'
            if done.exists():
                assert sha(folder / 'inference.pt') == json.loads(done.read_text())['sha256']
                continue
            out = OUTPUT / case['relative_output']
            for name, h in case['metadata_sha256'].items():
                assert sha(out / name) == h
            rgb = np.asarray(Image.open(out / 'input_image.png').convert('RGB')).copy()
            if rgb.shape != (704, 1280, 3):
                raise ValueError('WorldScore input must be toolkit-cropped to 1280x704')
            camera = torch.tensor([[500., 0., 640.], [0., 500., 352.], [0., 0., 1.]])
            depth = estimator.depth_for_views([rgb], [camera], [torch.eye(4)], 44, 80)[0]['depth'].cpu()
            valid = torch.isfinite(depth) & (depth > 0)
            # Outdoor benchmark images can contain large masked sky regions.
            # V11 itself uses all finite positive depth; no 50%-area rule exists.
            assert valid.any(), f'No finite positive first-frame depth: {case["name"]}'
            event(depth_case=case['name'], valid_fraction=float(valid.float().mean()))
            unit = float(depth[valid].median())
            cam_data = json.loads((out / 'camera_data.json').read_text())
            poses = blender_c2w_to_relative_opencv_w2c(torch.tensor(cam_data['cameras_interp'], dtype=torch.float32))
            if poses.shape != (81, 4, 4) or not torch.isfinite(poses).all():
                raise ValueError('Expected 81 finite camera matrices')
            poses[:, :3, 3] /= unit
            assert torch.allclose(poses[0], torch.eye(4), atol=1e-5)
            latent = encode_wan_latents(vae, rgb_tensor(rgb[None]).to('cuda')).cpu()
            kl = camera.clone()
            kl[:2] /= 16
            bundle = {'scene': case['name'], 'height': 704, 'width': 1280, 'seed_rgb': rgb, 'seed_latent': latent, 'seed_depth': depth / unit, 'seed_depth_metric_median': unit, 'intrinsics': camera[None], 'intrinsics_latent': kl[None], 'poses': {'nvs': poses}, 'latent_contract': wan_latent_contract(vae), 'geometry_calibration': 'official Blender c2w converted to first-view-relative OpenCV w2c; poses and MapAnything depth divided by same first-frame median; no fitted scale', 'case': case['name']}
            save_tensor(folder / 'inference.pt', bundle)
            write(done, {'sha256': sha(folder / 'inference.pt'), 'unit': unit, 'training_manifest_sha256': training_sha, 'depth_valid_fraction': float(valid.float().mean())})
            event(prepared=case['name'])


def generate():
    m = read_manifest()
    local10, assets = initialize()
    import numpy as np
    import torch
    from PIL import Image
    from engram.core import ProtectedMemory, build_depth_estimator, conditioning, decoded_rgb, seed_all, save_video
    from engram.models.wan_latents import decode_wan_latents
    state, training_sha = local10.checkpoint(assets)
    assert assets['checkpoint_sha256'] == m['checkpoint_sha256']
    text_meta = json.loads((ROOT / 'TEXT_COMPLETE.json').read_text())
    assert sha(ROOT / 'data/text.pt') == text_meta['sha256']
    texts = torch.load(ROOT / 'data/text.pt', map_location='cpu', weights_only=False)
    from model_loader import load_model
    model = load_model(state)
    del state
    estimator = build_depth_estimator()
    with torch.no_grad():
        shard = os.environ.get('ENGRAM_SHARD')  # 'k/n': generate cases[k::n] only (parallel GPUs)
        selected = m['cases'] if shard is None else m['cases'][int(shard.split('/')[0])::int(shard.split('/')[1])]
        for case in selected:
            out = OUTPUT / case['relative_output']
            if (out / 'GENERATION_COMPLETE.json').exists():
                verify_case(case, m)
                event(skipped_complete=case['name'])
                continue
            started = time.time()
            folder = ROOT / 'data' / case['name']
            prepared = json.loads((folder / 'PREPARED.json').read_text())
            assert sha(folder / 'inference.pt') == prepared['sha256']
            assert prepared['training_manifest_sha256'] == training_sha
            bundle = torch.load(folder / 'inference.pt', map_location='cpu', weights_only=False)
            poses = bundle['poses']['nvs']
            memory = ProtectedMemory(bundle, 'cpu', writeback=True)
            original = memory.fingerprint(True)
            prefix = bundle['seed_latent'].to('cuda', dtype=torch.bfloat16)
            seed_all(case['seed'])
            generator = torch.Generator(device='cuda').manual_seed(case['seed'])
            positive = texts[case['positive_prompt']].to('cuda', dtype=torch.bfloat16)
            positive2 = texts[case['positive_prompt_2']].to('cuda', dtype=torch.bfloat16)
            chunks = []
            for start in [0, 32, 64]:
                indices = torch.arange(start, start + 33).clamp(max=80)
                valid_count = min(33, 81 - start)
                chunk_poses = poses[indices[::4]]
                cameras = bundle['intrinsics_latent'][0].repeat(9, 1, 1)
                condition = conditioning(chunk_poses, cameras, 44, 80, memory).to('cuda', dtype=torch.bfloat16)
                latent = model.sample_chunk(positive if start == 0 else positive2, condition, (1, 48, 9, 44, 80), num_steps=40, guidance_scale=1., ref_latent=prefix, ref_len=1, generator=generator)
                assert torch.isfinite(latent).all(), 'Nonfinite generated latent'
                frames = decoded_rgb(decode_wan_latents(model.vae, latent))
                if start == 0:
                    frames[0] = bundle['seed_rgb']
                chunks.append(frames[:valid_count] if start == 0 else frames[1:valid_count])
                gated_write_chunk(memory, estimator, bundle, frames, latent, chunk_poses, start, valid_count, mode='real_only')
                assert memory.fingerprint(True) == original, 'Real first-frame memory changed'
                if start < 64:
                    prefix = invert_last_frame(model.vae, frames[-1])
                progress = {'case': case['name'], 'chunk_start': start, 'valid_count': valid_count, 'elapsed_seconds': time.time() - started, 'memory': memory.stats()}
                write(ROOT / 'PROGRESS.json', progress)
                event(**progress)
            video = np.concatenate(chunks)
            assert video.shape == (81, 704, 1280, 3) and np.array_equal(video[0], bundle['seed_rgb'])
            frame_dir = out / 'frames'
            frame_dir.mkdir(exist_ok=True)
            hashes = {}
            for index, frame in enumerate(video):
                file = frame_dir / f'{index:03d}.png'
                Image.fromarray(frame).save(file)
                hashes[str(file.relative_to(out))] = sha(file)
            save_video(out / 'videos/output.mp4', video, fps=16)
            hashes['videos/output.mp4'] = sha(out / 'videos/output.mp4')
            write(out / 'GENERATION_COMPLETE.json', {'identifier': IDENTIFIER, 'case': case['name'], 'checkpoint_sha256': m['checkpoint_sha256'], 'input_sha256': prepared['sha256'], 'text_sha256': text_meta['sha256'], 'seed': case['seed'], 'frames': 81, 'fps': 16, 'steps': 40, 'cfg': 1., 'cfg_negative_supplied': False, 'only_real_image': str(out / 'input_image.png'), 'immutable_real_unchanged': True, 'inversion_handoff': True, 'elapsed_seconds': time.time() - started, 'output_sha256': hashes})
            verify_case(case, m)
            event(completed=case['name'], elapsed_seconds=time.time() - started)
            del memory, video, chunks, bundle
    if os.environ.get('ENGRAM_SHARD') is not None:
        return
    write(ROOT / 'GENERATION_COMPLETE.json', {'identifier': IDENTIFIER, 'videos': len(m['cases']), 'checkpoint_sha256': m['checkpoint_sha256']})


def verify_case(case, manifest):
    from PIL import Image
    out = OUTPUT / case['relative_output']
    done = json.loads((out / 'GENERATION_COMPLETE.json').read_text())
    assert done['checkpoint_sha256'] == manifest['checkpoint_sha256']
    assert len(list((out / 'frames').glob('*.png'))) == 81
    for name, h in {**case['metadata_sha256'], **done['output_sha256']}.items():
        assert sha(out / name) == h, f'Changed output {case["name"]}/{name}'
    for frame in sorted((out / 'frames').glob('*.png')):
        with Image.open(frame) as image:
            image.load()
            assert image.size == (1280, 704) and image.mode == 'RGB'
