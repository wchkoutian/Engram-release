"""Generate per-window captions and Wan text embeddings from prepared training latents."""
import hashlib, json, time
from pathlib import Path
import torch
INSTRUCTION = ("These three images are frames from one short video of a real place. "
               "Describe the place in one English sentence of 20 to 40 words: the kind of space, its main objects, "
               "their colors and materials, and the lighting. Describe only what is visible. "
               "Do not mention the camera, movement, frames, images, or video.")

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(**k): print(json.dumps({"time": time.strftime("%H:%M:%S"), **k}), flush=True)


def frames_phase():
    from diffusers import AutoencoderKLWan
    from engram.models.wan_latents import decode_wan_latents
    from PIL import Image
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to("cuda").eval()
    vae.enable_tiling(); vae.enable_slicing()
    for i, r in enumerate(records):
        stem = r["file"].replace("/", "_").replace(".pt", "")
        if all((FR / f"{stem}_{j}.jpg").exists() for j in range(3)): continue
        item = torch.load(SRC / r["file"], map_location="cpu", weights_only=False)
        assert hashlib.sha256((SRC / r["file"]).read_bytes()).hexdigest() == r["sha256"]
        mask = item["valid_mask"].tolist(); last = max(k for k, v in enumerate(mask) if v)
        with torch.no_grad():
            video = decode_wan_latents(vae, item["latent"][:, :, : last + 1].to("cuda"))  # [1,3,T,H,W] in [-1,1]
        T = video.shape[2]
        for j, t in enumerate([0, T // 2, T - 1]):
            x = ((video[0, :, t].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(x).resize((640, 352)).save(FR / f"{stem}_{j}.jpg", quality=92)
        if i % 24 == 0: log(phase="frames", done=i + 1, total=len(records))
    del vae; torch.cuda.empty_cache()


def caption_phase():
    out = ROOT / "captions.json"
    caps = json.loads(out.read_text()) if out.exists() else {}
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    proc = AutoProcessor.from_pretrained(VLM, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    for i, r in enumerate(records):
        if r["file"] in caps: continue
        stem = r["file"].replace("/", "_").replace(".pt", "")
        imgs = [Image.open(FR / f"{stem}_{j}.jpg").convert("RGB") for j in range(3)]
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        cap = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        caps[r["file"]] = cap
        out.write_text(json.dumps(caps, indent=1, ensure_ascii=False))
        if i % 24 == 0: log(phase="caption", done=i + 1, total=len(records), example=cap)
    del model; torch.cuda.empty_cache()


def encode_phase():
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers import WanPipeline
    caps = json.loads((ROOT / "captions.json").read_text())
    assert set(caps) == {r["file"] for r in records} and all(caps.values())
    old = torch.load(EMPTY, map_location="cpu", weights_only=False)["text"]; length = old.shape[1]
    tok = AutoTokenizer.from_pretrained(WAN, subfolder="tokenizer", local_files_only=True)
    enc = UMT5EncoderModel.from_pretrained(WAN, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    pipe = WanPipeline(tokenizer=tok, text_encoder=enc, vae=None, transformer=None, scheduler=None)
    emb = {}
    with torch.no_grad():
        e0, _ = pipe.encode_prompt(prompt=[""], device="cuda", num_videos_per_prompt=1, do_classifier_free_guidance=False, max_sequence_length=length)
        for k, c in caps.items():
            e, _ = pipe.encode_prompt(prompt=[c], device="cuda", num_videos_per_prompt=1, do_classifier_free_guidance=False, max_sequence_length=length)
            assert tuple(e.shape) == tuple(old.shape) and torch.isfinite(e).all()
            emb[k] = e.cpu().to(torch.bfloat16)
    torch.save({"captions": emb, "empty": e0.cpu().to(torch.bfloat16)}, ROOT / "caption_text.pt")
    (ROOT / "CAPTIONS_READY.json").write_text(json.dumps({"windows": len(emb), "captions_sha256": sha(ROOT / "captions.json"), "caption_text_sha256": sha(ROOT / "caption_text.pt"), "vlm": VLM, "instruction": INSTRUCTION, "decoding": "greedy, max_new_tokens=80", "frames": "Wan VAE decode of window target latent; first/middle/last valid frame at 640x352", "text_encoder": WAN + "/text_encoder (UMT5) via WanPipeline.encode_prompt", "length": length, "empty_matches_bundle_max_abs": float((e0.cpu().float() - old.float()).abs().max())}, indent=2))
    log(phase="encode", done=len(emb))
def run(args, assets):
    global SRC, ROOT, WAN, VLM, EMPTY, manifest, records, FR
    SRC = ROOT = args.root
    WAN, VLM, EMPTY = assets['wan_model'], assets['caption_model'], assets['text_bundle']
    manifest = json.loads((SRC / 'manifest.json').read_text())
    records = manifest['records']
    FR = ROOT / 'caption_frames'
    FR.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    for phase in args.phases.split(','):
        {'frames': frames_phase, 'caption': caption_phase, 'encode': encode_phase}[phase]()
