"""Upload FLUX.1-Fill-dev TRANSFORMER to Modal Volume.

Downloads ONLY the transformer directory (~24GB) from
black-forest-labs/FLUX.1-Fill-dev to /models/flux_fill_dev/transformer/
on the Modal Volume.

This complements _upload_flux_fill.py which skipped the transformer.
Together they provide the full FLUX.1-Fill-dev pipeline on the Volume
so FluxFillPipeline.from_pretrained can load without a transformer override.
"""
import modal

app = modal.App("stylelens-upload-flux-transformer")
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
    timeout=3600,
    memory=32768,
)
def download_flux_transformer():
    """Download ONLY the FLUX.1-Fill-dev transformer to Volume."""
    import os, time
    from huggingface_hub import login, snapshot_download

    t0 = time.time()

    # Login with HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print(f"HF login OK (token: {hf_token[:10]}...)")

    target_dir = "/models/flux_fill_dev"
    os.makedirs(target_dir, exist_ok=True)

    print("Downloading FLUX.1-Fill-dev TRANSFORMER to Volume...")
    print("This is ~24GB - may take 5-10 minutes.")

    # Download ONLY the transformer directory
    local_dir = snapshot_download(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        local_dir=target_dir,
        allow_patterns=["transformer/*"],
    )

    print(f"Downloaded to: {local_dir}")

    # List what was downloaded
    total_size = 0
    transformer_dir = os.path.join(target_dir, "transformer")
    if os.path.isdir(transformer_dir):
        for root, dirs, files in os.walk(transformer_dir):
            for f in files:
                fpath = os.path.join(root, f)
                sz = os.path.getsize(fpath)
                total_size += sz
                rel = os.path.relpath(fpath, target_dir)
                print(f"  {rel}: {sz/1e6:.1f}MB")
    else:
        print("WARNING: transformer directory not found after download!")

    print()
    print(f"Transformer total: {total_size/1e9:.2f}GB")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Commit volume changes
    model_volume = modal.Volume.from_name("stylelens-models")
    model_volume.commit()
    print()
    print("Volume committed!")

    return {
        "status": "OK",
        "total_gb": round(total_size / 1e9, 2),
        "elapsed_sec": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main():
    import json
    result = download_flux_transformer.remote()
    print()
    print(f"Result: {json.dumps(result, indent=2)}")
