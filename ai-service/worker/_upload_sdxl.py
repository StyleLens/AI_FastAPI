"""Upload SDXL base model to Modal Volume.

stabilityai/stable-diffusion-xl-base-1.0
License: CreativeML Open RAIL++-M (commercial use allowed)
Size: ~7GB (fp16 variant)
"""
import modal

app = modal.App("stylelens-upload-sdxl")
model_volume = modal.Volume.from_name("stylelens-models", create_if_missing=True)

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6", "transformers>=4.48", "diffusers>=0.32",
        "safetensors>=0.5", "accelerate>=1.3", "huggingface_hub>=0.28",
    )
)


@app.function(
    image=upload_image,
    volumes={"/models": model_volume},
    timeout=1800,
    memory=32768,
)
def download_sdxl():
    """Download SDXL base to Volume."""
    import os, time
    from huggingface_hub import snapshot_download

    t0 = time.time()

    target_dir = "/models/sdxl_base"

    # Check if already exists
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Target directory {target_dir} already exists, checking contents...")
        existing_size = 0
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                existing_size += os.path.getsize(os.path.join(root, f))
        if existing_size > 3e9:
            print(f"Already downloaded ({existing_size/1e9:.2f}GB). Skipping.")
            return {"status": "ALREADY_EXISTS", "total_gb": round(existing_size / 1e9, 2)}

    os.makedirs(target_dir, exist_ok=True)

    print("Downloading stabilityai/stable-diffusion-xl-base-1.0 to Volume...")
    print("License: CreativeML Open RAIL++-M (commercial use allowed)")
    print("Downloading fp16 safetensors variant (~7GB).")

    local_dir = snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=target_dir,
        ignore_patterns=[
            "*.bin",
            "*.onnx",
            "*.onnx_data",
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

    # Commit volume changes
    model_volume = modal.Volume.from_name("stylelens-models")
    model_volume.commit()
    print("\nVolume committed!")

    return {"status": "OK", "total_gb": round(total_size / 1e9, 2), "elapsed_sec": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    import json
    result = download_sdxl.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
