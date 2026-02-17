"""Upload inswapper_128.onnx face-swap model to Modal Volume.

Model:
- inswapper_128.onnx (~500MB) - face swapping model for InsightFace
- Source: https://huggingface.co/ezioruan/inswapper_128.onnx
- Destination: /models/insightface/inswapper_128.onnx
"""
import modal

app = modal.App("stylelens-upload-inswapper")
model_volume = modal.Volume.from_name("stylelens-models", create_if_missing=True)

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.28")
)


@app.function(
    image=upload_image,
    volumes={"/models": model_volume},
    timeout=600,
    memory=4096,
)
def download_inswapper():
    """Download inswapper_128.onnx from HuggingFace."""
    import os
    import time
    from huggingface_hub import hf_hub_download

    t0 = time.time()

    dest_dir = "/models/insightface"
    os.makedirs(dest_dir, exist_ok=True)

    print("Downloading inswapper_128.onnx from HuggingFace...")
    fpath = hf_hub_download(
        repo_id="ezioruan/inswapper_128.onnx",
        filename="inswapper_128.onnx",
        local_dir=dest_dir,
    )
    sz = os.path.getsize(fpath)
    print(f"Downloaded: {fpath} ({sz / 1e6:.1f}MB)")

    # Verify file exists at expected path
    expected = "/models/insightface/inswapper_128.onnx"
    if os.path.exists(expected):
        print(f"Verified: {expected} exists ({os.path.getsize(expected) / 1e6:.1f}MB)")
    else:
        print(f"WARNING: Expected path {expected} not found, actual: {fpath}")

    # List directory contents
    print(f"\nContents of {dest_dir}:")
    for item in sorted(os.listdir(dest_dir)):
        full = os.path.join(dest_dir, item)
        if os.path.isfile(full):
            print(f"  {item}: {os.path.getsize(full) / 1e6:.1f}MB")
        else:
            print(f"  {item}/ (dir)")

    # Commit volume
    vol = modal.Volume.from_name("stylelens-models")
    vol.commit()
    print("\nVolume committed!")

    elapsed = round(time.time() - t0, 1)
    print(f"Elapsed: {elapsed}s")

    return {
        "file": expected,
        "size_mb": round(sz / 1e6, 1),
        "elapsed_sec": elapsed,
    }


@app.local_entrypoint()
def main():
    import json
    result = download_inswapper.remote()
    print(f"\nResult: {json.dumps(result, indent=2)}")
