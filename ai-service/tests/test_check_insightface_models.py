#!/usr/bin/env python3
"""Check InsightFace model files on Modal Volume."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from worker.modal_app import app as modal_app

import modal

if __name__ == "__main__":
    # Create a simple check function
    worker_image = modal.Image.debian_slim(python_version="3.12").pip_install("insightface>=0.7", "onnxruntime-gpu")
    vol = modal.Volume.from_name("stylelens-models", create_if_missing=True)

    app2 = modal.App("check-insightface")

    @app2.function(
        image=worker_image,
        volumes={"/models": vol},
        timeout=60,
    )
    def check_models():
        import os
        import glob

        results = []

        # Check all paths under /models/insightface/
        for pattern in [
            "/models/insightface/**/*",
            "/models/pulid_flux/**/*",
        ]:
            for f in sorted(glob.glob(pattern, recursive=True)):
                if os.path.isfile(f):
                    size_mb = os.path.getsize(f) / 1024 / 1024
                    results.append(f"  {f} ({size_mb:.1f}MB)")

        # Also check root models dir
        results.append("\n/models/ top-level:")
        for item in sorted(os.listdir("/models/")):
            item_path = f"/models/{item}"
            if os.path.isdir(item_path):
                results.append(f"  [DIR] {item}")
            else:
                size_mb = os.path.getsize(item_path) / 1024 / 1024
                results.append(f"  [FILE] {item} ({size_mb:.1f}MB)")

        return "\n".join(results)

    with app2.run():
        output = check_models.remote()
        print(output)
