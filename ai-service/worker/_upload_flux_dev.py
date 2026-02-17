"""Upload FLUX.1-dev (standard text-to-image) to Modal Volume.

The standard FLUX.1-dev base model is required by FluxControlNetPipeline
(ControlNet Depth) and FluxPuLIDPipeline (PuLID face identity).
FLUX.1-Fill-dev has a different transformer architecture and cannot be used.

Components: scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, transformer
Transformer alone is ~24GB.
"""
import modal

app = modal.App("stylelens-upload-flux-dev")
model_volume = modal.Volume.from_name("stylelens-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6", "transformers>=4.48", "diffusers>=0.32",
        "safetensors>=0.5", "accelerate>=1.3", "huggingface_hub>=0.28",
    )
)


@app.function(
    image=upload_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    timeout=1800,
    memory=32768,
)
def download_flux_dev():
    """Download FLUX.1-dev pipeline to Volume."""
    import os, time
    from huggingface_hub import login, snapshot_download

    t0 = time.time()

    # Login with HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print(f"HF login OK (token: {hf_token[:10]}...)")

    target_dir = "/models/flux_dev"
    os.makedirs(target_dir, exist_ok=True)

    print("Downloading FLUX.1-dev to Volume...")
    print("This includes: scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, transformer")
    print("(transformer is ~24GB - the standard text-to-image transformer)")

    # Download the full pipeline including transformer
    # This is the base FLUX.1-dev model, NOT the Fill-dev variant
    local_dir = snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_dir=target_dir,
        ignore_patterns=[
            "*.bin",  # Skip any old format binaries
            ".gitattributes",
        ],
    )

    print(f"Downloaded to: {local_dir}")

    # List what was downloaded
    total_size = 0
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            fpath = os.path.join(root, f)
            sz = os.path.getsize(fpath)
            total_size += sz
            rel = os.path.relpath(fpath, target_dir)
            print(f"  {rel}: {sz/1e6:.1f}MB")

    print(f"\nTotal: {total_size/1e9:.2f}GB")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Show model_index.json
    import json
    idx_path = os.path.join(target_dir, "model_index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        print(f"\nmodel_index.json: {json.dumps(idx, indent=2)}")

    # Commit volume changes
    model_volume = modal.Volume.from_name("stylelens-models")
    model_volume.commit()
    print("\nVolume committed!")

    return {"status": "OK", "total_gb": round(total_size / 1e9, 2), "elapsed_sec": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    import json
    result = download_flux_dev.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
