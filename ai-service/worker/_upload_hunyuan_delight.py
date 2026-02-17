"""Upload Hunyuan3D delight + paint models to Modal Volume.

Downloads:
  - hunyuan3d-delight-v2-0 (relighting model for paint pipeline)
  - hunyuan3d-paint-v2-0 (texture painting pipeline)

These are stored under /models/hunyuan3d/ in the stylelens-models Volume.
"""
import modal

app = modal.App("stylelens-upload-hunyuan-delight")
model_volume = modal.Volume.from_name("stylelens-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.28",
    )
)


@app.function(
    image=upload_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    timeout=3600,
    memory=32768,
)
def download_hunyuan_delight():
    """Download Hunyuan3D delight + paint models to Volume."""
    import os
    import time
    from huggingface_hub import login, snapshot_download

    t0 = time.time()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print(f"HF login OK (token: {hf_token[:10]}...)")

    base_dir = "/models/hunyuan3d"
    os.makedirs(base_dir, exist_ok=True)

    # 1. Download delight model
    delight_dir = os.path.join(base_dir, "hunyuan3d-delight-v2-0")
    print(f"\n[1/2] Downloading hunyuan3d-delight-v2-0...")
    snapshot_download(
        repo_id="tencent/Hunyuan3D-2",
        local_dir=base_dir,
        allow_patterns=[
            "hunyuan3d-delight-v2-0/**",
        ],
    )
    print(f"  Downloaded to: {delight_dir}")

    # 2. Download paint model
    paint_dir = os.path.join(base_dir, "hunyuan3d-paint-v2-0")
    print(f"\n[2/2] Downloading hunyuan3d-paint-v2-0...")
    snapshot_download(
        repo_id="tencent/Hunyuan3D-2",
        local_dir=base_dir,
        allow_patterns=[
            "hunyuan3d-paint-v2-0/**",
        ],
    )
    print(f"  Downloaded to: {paint_dir}")

    # Summary
    total_size = 0
    for subdir in ["hunyuan3d-delight-v2-0", "hunyuan3d-paint-v2-0"]:
        dir_path = os.path.join(base_dir, subdir)
        if os.path.exists(dir_path):
            dir_size = 0
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    fpath = os.path.join(root, f)
                    sz = os.path.getsize(fpath)
                    dir_size += sz
                    rel = os.path.relpath(fpath, base_dir)
                    print(f"  {rel}: {sz/1e6:.1f}MB")
            total_size += dir_size
            print(f"  {subdir}: {dir_size/1e9:.2f}GB")
        else:
            print(f"  WARNING: {subdir} not found after download!")

    print(f"\nTotal: {total_size/1e9:.2f}GB")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    model_volume = modal.Volume.from_name("stylelens-models")
    model_volume.commit()
    print("Volume committed!")

    return {
        "status": "OK",
        "total_gb": round(total_size / 1e9, 2),
        "elapsed_sec": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main():
    import json
    result = download_hunyuan_delight.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
