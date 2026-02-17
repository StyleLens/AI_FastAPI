"""Download FASHN VTON DWPose weights to Modal Volume.

FASHN VTON v1.5 requires DWPose models for pose estimation:
- yolox_l.onnx (YOLOX-L for person detection)
- dw-ll_ucoco_384.onnx (DWPose for pose estimation)

These are downloaded using the FASHN VTON package's download script.
"""
import modal

app = modal.App("stylelens-upload-fashn-dwpose")
model_volume = modal.Volume.from_name("stylelens-models", create_if_missing=True)

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.6", "huggingface_hub>=0.28", "safetensors>=0.5",
        "onnxruntime", "requests",
    )
    .pip_install("git+https://github.com/fashn-AI/fashn-vton-1.5.git")
)


@app.function(
    image=upload_image,
    volumes={"/models": model_volume},
    timeout=600,
    memory=8192,
)
def download_dwpose():
    """Download DWPose weights for FASHN VTON."""
    import os, time, subprocess

    t0 = time.time()
    weights_dir = "/models/fashn_vton"

    # Check if already downloaded
    yolox = os.path.join(weights_dir, "dwpose", "yolox_l.onnx")
    dwpose = os.path.join(weights_dir, "dwpose", "dw-ll_ucoco_384.onnx")

    if os.path.exists(yolox) and os.path.exists(dwpose):
        y_sz = os.path.getsize(yolox)
        d_sz = os.path.getsize(dwpose)
        if y_sz > 1e6 and d_sz > 1e6:
            print(f"DWPose weights already exist:")
            print(f"  yolox_l.onnx: {y_sz/1e6:.1f}MB")
            print(f"  dw-ll_ucoco_384.onnx: {d_sz/1e6:.1f}MB")
            return {"status": "ALREADY_EXISTS"}

    # Try using the FASHN VTON download script
    print("Downloading DWPose weights using fashn_vton package...")
    try:
        result = subprocess.run(
            ["python", "-m", "fashn_vton.download_weights", "--weights-dir", weights_dir],
            capture_output=True, text=True, timeout=300,
        )
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"fashn_vton download script failed: {e}")

    # If the script doesn't exist, try the scripts/download_weights.py entrypoint
    if not os.path.exists(yolox):
        print("Trying alternative download method...")
        try:
            # Find the package location
            import fashn_vton
            pkg_dir = os.path.dirname(fashn_vton.__file__)
            scripts_dir = os.path.join(os.path.dirname(pkg_dir), "scripts")
            dl_script = os.path.join(scripts_dir, "download_weights.py")
            if os.path.exists(dl_script):
                result = subprocess.run(
                    ["python", dl_script, "--weights-dir", weights_dir],
                    capture_output=True, text=True, timeout=300,
                )
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
        except Exception as e:
            print(f"Alternative download failed: {e}")

    # Manual download as fallback
    if not os.path.exists(yolox):
        print("Manual download of DWPose weights...")
        import requests

        os.makedirs(os.path.join(weights_dir, "dwpose"), exist_ok=True)

        # Standard DWPose model URLs (commonly used across projects)
        urls = {
            "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
            "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        }

        for fname, url in urls.items():
            out_path = os.path.join(weights_dir, "dwpose", fname)
            print(f"  Downloading {fname} from {url}...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            sz = os.path.getsize(out_path)
            print(f"  {fname}: {sz/1e6:.1f}MB")

    # Verify
    total_size = 0
    for fname in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
        fpath = os.path.join(weights_dir, "dwpose", fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            total_size += sz
            print(f"  OK: {fname} ({sz/1e6:.1f}MB)")
        else:
            print(f"  MISSING: {fname}")

    # Commit
    model_volume = modal.Volume.from_name("stylelens-models")
    model_volume.commit()
    print("\nVolume committed!")

    return {
        "status": "OK",
        "total_mb": round(total_size / 1e6, 1),
        "elapsed_sec": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main():
    import json
    result = download_dwpose.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
