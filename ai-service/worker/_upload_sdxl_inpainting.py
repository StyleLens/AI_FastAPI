"""Upload SDXL Inpainting model to Modal Volume.

diffusers/stable-diffusion-xl-1.0-inpainting-0.1
License: CreativeML Open RAIL++-M (commercial use allowed)
Size: ~7.5GB (fp16 safetensors)

Required for Face Refiner: 2-pass inpainting to regenerate face region at high resolution.
"""
import modal

app = modal.App("stylelens-upload-sdxl-inpainting")
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
def download_sdxl_inpainting():
    """Download SDXL Inpainting 0.1 to Volume."""
    import os, time
    from huggingface_hub import snapshot_download

    t0 = time.time()
    target_dir = "/models/sdxl_inpainting"

    if os.path.exists(target_dir) and os.listdir(target_dir):
        existing_size = 0
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                existing_size += os.path.getsize(os.path.join(root, f))
        if existing_size > 3e9:
            print(f"Already downloaded ({existing_size/1e9:.2f}GB). Skipping.")
            return {"status": "ALREADY_EXISTS", "total_gb": round(existing_size / 1e9, 2)}

    os.makedirs(target_dir, exist_ok=True)

    print("Downloading diffusers/stable-diffusion-xl-1.0-inpainting-0.1 ...")
    print("License: CreativeML Open RAIL++-M (commercial OK)")

    local_dir = snapshot_download(
        repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        local_dir=target_dir,
        ignore_patterns=[
            "*.bin",
            "*.onnx",
            "*.onnx_data",
            ".gitattributes",
        ],
    )

    print(f"Downloaded to: {local_dir}")

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

    model_volume = modal.Volume.from_name("stylelens-models")
    model_volume.commit()
    print("Volume committed!")

    return {"status": "OK", "total_gb": round(total_size / 1e9, 2), "elapsed_sec": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    import json
    result = download_sdxl_inpainting.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
