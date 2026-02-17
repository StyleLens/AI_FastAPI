"""Upload PuLID-FLUX and ControlNet Depth models to Modal Volume.

Models:
1. PuLID-FLUX v0.9.1 (~1.3GB) - face identity preservation
2. Flux.1-dev-Controlnet-Depth (~3.6GB) - depth-based pose control
3. antelopev2 face analysis model (~360MB) - InsightFace for PuLID
"""
import modal

app = modal.App("stylelens-upload-pulid-controlnet")
model_volume = modal.Volume.from_name("stylelens-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.28",
    )
    .apt_install("wget", "unzip")
)


@app.function(
    image=upload_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    timeout=1800,
    memory=16384,
)
def download_pulid_controlnet():
    """Download PuLID-FLUX, ControlNet Depth, and InsightFace models."""
    import os
    import time
    from huggingface_hub import login, hf_hub_download, snapshot_download

    t0 = time.time()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("HF login OK")

    results = {}

    # 1. PuLID-FLUX v0.9.1
    print("=" * 60)
    print("1/3: Downloading PuLID-FLUX v0.9.1...")
    pulid_dir = "/models/pulid_flux"
    os.makedirs(pulid_dir, exist_ok=True)
    pulid_path = hf_hub_download(
        repo_id="guozinan/PuLID",
        filename="pulid_flux_v0.9.1.safetensors",
        local_dir=pulid_dir,
    )
    sz = os.path.getsize(pulid_path)
    print(f"PuLID-FLUX: {pulid_path} ({sz / 1e6:.1f}MB)")
    results["pulid_flux_mb"] = round(sz / 1e6, 1)

    # Also download the EVA CLIP model needed by PuLID
    print("Downloading EVA02-CLIP-L-14-336 for PuLID...")
    eva_clip_dir = "/models/pulid_flux/eva_clip"
    os.makedirs(eva_clip_dir, exist_ok=True)
    eva_clip_path = snapshot_download(
        repo_id="QuanSun/EVA-CLIP",
        allow_patterns=["EVA02_CLIP_L_336_psz14_s6B.pt"],
        local_dir=eva_clip_dir,
    )
    print(f"EVA-CLIP downloaded to: {eva_clip_path}")

    # 2. ControlNet Depth for FLUX
    print("=" * 60)
    print("2/3: Downloading Flux.1-dev-Controlnet-Depth...")
    cn_dir = "/models/flux_controlnet_depth"
    os.makedirs(cn_dir, exist_ok=True)
    cn_path = snapshot_download(
        repo_id="jasperai/Flux.1-dev-Controlnet-Depth",
        local_dir=cn_dir,
    )
    cn_size = 0
    for root, dirs, files in os.walk(cn_dir):
        for f in files:
            cn_size += os.path.getsize(os.path.join(root, f))
    print(f"ControlNet Depth: {cn_path} ({cn_size / 1e9:.2f}GB)")
    results["controlnet_depth_gb"] = round(cn_size / 1e9, 2)

    # 3. InsightFace antelopev2 (face analysis for PuLID)
    print("=" * 60)
    print("3/3: Downloading InsightFace antelopev2...")
    face_dir = "/models/insightface/models/antelopev2"
    os.makedirs(face_dir, exist_ok=True)

    face_models = [
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]
    face_size = 0
    for fname in face_models:
        try:
            fpath = hf_hub_download(
                repo_id="DIAMONIK7777/antelopev2",
                filename=fname,
                local_dir=face_dir,
            )
            sz = os.path.getsize(fpath)
            face_size += sz
            print(f"  {fname}: {sz / 1e6:.1f}MB")
        except Exception as e:
            print(f"  {fname}: FAILED - {e}")

    results["insightface_mb"] = round(face_size / 1e6, 1)

    print("=" * 60)
    print(f"Elapsed: {time.time() - t0:.1f}s")

    # Commit volume
    vol = modal.Volume.from_name("stylelens-models")
    vol.commit()
    print("Volume committed!")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    return results


@app.local_entrypoint()
def main():
    import json
    result = download_pulid_controlnet.remote()
    print(f"Result: {json.dumps(result, indent=2)}")
