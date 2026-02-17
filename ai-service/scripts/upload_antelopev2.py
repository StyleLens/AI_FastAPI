#!/usr/bin/env python3
"""
Download antelopev2 model pack and upload to Modal Volume.

InsightFace antelopev2 requires these ONNX files:
  - scrfd_10g_bnkps.onnx  (face detection, ~16MB)
  - glintr100.onnx         (face recognition/embedding, ~260MB)
  - 1k3d68.onnx            (3D landmark, ~144MB)
  - 2d106det.onnx          (2D landmark, ~5MB)
  - genderage.onnx         (gender/age estimation, ~1MB)

These must be at: /models/insightface/models/antelopev2/
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import modal

app = modal.App("upload-antelopev2")
vol = modal.Volume.from_name("stylelens-models", create_if_missing=True)

# Use a minimal image with download tools
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "unzip")
    .pip_install("insightface>=0.7", "onnxruntime")
)


@app.function(
    image=image,
    volumes={"/models": vol},
    timeout=600,
    memory=8192,
)
def upload_models():
    import os
    import subprocess
    import glob

    target_dir = "/models/insightface/models/antelopev2"
    os.makedirs(target_dir, exist_ok=True)

    # Check if already uploaded
    existing = glob.glob(f"{target_dir}/*.onnx")
    print(f"Existing models: {len(existing)}")
    for f in existing:
        print(f"  {f} ({os.path.getsize(f)/1024/1024:.1f}MB)")

    if len(existing) >= 5:
        print("All 5 antelopev2 models already present!")
        return "Already uploaded"

    # Method 1: Try insightface built-in download
    print("\nTrying insightface auto-download...")
    try:
        import insightface
        from insightface.utils import storage

        # Force download to our target directory
        # insightface stores models at root/models/name/
        os.makedirs("/models/insightface/models", exist_ok=True)

        # Download antelopev2 pack
        storage.download_onnx_model(
            model_name='antelopev2',
            root='/models/insightface',
        )
        print("Downloaded via insightface!")
    except Exception as e:
        print(f"insightface download failed: {e}")

        # Method 2: Direct download from GitHub/HuggingFace
        print("\nTrying direct download from HuggingFace...")
        try:
            # antelopev2 model pack URL (from insightface release)
            url = "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip"
            zip_path = "/tmp/antelopev2.zip"

            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", zip_path, url],
                check=True, timeout=300,
            )
            print(f"Downloaded: {os.path.getsize(zip_path)/1024/1024:.1f}MB")

            # Extract
            subprocess.run(
                ["unzip", "-o", zip_path, "-d", "/tmp/antelopev2_extracted"],
                check=True,
            )

            # Find and copy ONNX files
            for onnx in glob.glob("/tmp/antelopev2_extracted/**/*.onnx", recursive=True):
                fname = os.path.basename(onnx)
                dest = f"{target_dir}/{fname}"
                subprocess.run(["cp", onnx, dest], check=True)
                print(f"  Copied: {fname} ({os.path.getsize(dest)/1024/1024:.1f}MB)")

        except Exception as e2:
            print(f"HuggingFace download also failed: {e2}")

            # Method 3: Try another source
            print("\nTrying alternative download...")
            try:
                # Alternative: download individual files
                base_url = "https://huggingface.co/deepinsight/inswapper/resolve/main"
                files = [
                    "scrfd_10g_bnkps.onnx",
                    "glintr100.onnx",
                    "1k3d68.onnx",
                    "2d106det.onnx",
                    "genderage.onnx",
                ]
                for fname in files:
                    dest = f"{target_dir}/{fname}"
                    if os.path.exists(dest):
                        print(f"  Already exists: {fname}")
                        continue
                    url = f"{base_url}/{fname}"
                    try:
                        subprocess.run(
                            ["wget", "-q", "-O", dest, url],
                            check=True, timeout=120,
                        )
                        print(f"  Downloaded: {fname} ({os.path.getsize(dest)/1024/1024:.1f}MB)")
                    except Exception:
                        print(f"  Failed: {fname} from {url}")
            except Exception as e3:
                print(f"Alternative download failed: {e3}")

    # Final check
    print("\n=== Final model check ===")
    final_files = glob.glob(f"{target_dir}/*.onnx")
    total_size = 0
    for f in sorted(final_files):
        size = os.path.getsize(f) / 1024 / 1024
        total_size += size
        print(f"  {os.path.basename(f)}: {size:.1f}MB")
    print(f"Total: {len(final_files)} files, {total_size:.1f}MB")

    # Also check what else is in insightface dir
    print("\n=== Full /models/insightface/ listing ===")
    for root, dirs, files in os.walk("/models/insightface"):
        for f in files:
            fp = os.path.join(root, f)
            print(f"  {fp} ({os.path.getsize(fp)/1024/1024:.1f}MB)")

    # Commit volume changes
    vol = modal.Volume.from_name("stylelens-models")
    vol.commit()

    return f"Done: {len(final_files)} ONNX models"


if __name__ == "__main__":
    with app.run():
        result = upload_models.remote()
        print(f"\n{result}")
