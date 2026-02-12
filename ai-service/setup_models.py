"""
StyleLens V6 — Model Download Script
Interactive script to download all 7 SOTA models to correct paths.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODELS = {
    "yolo26": {
        "name": "YOLO26-L (Person Detection)",
        "source": "Ultralytics/YOLO26",
        "path": MODEL_DIR / "yolo26",
        "size": "~50MB",
        "method": "ultralytics",
        "details": "pip install ultralytics>=8.4, then YOLO downloads on first use",
    },
    "sam3": {
        "name": "SAM 3 (Concept-Aware Segmentation)",
        "source": "facebook/sam3",
        "path": MODEL_DIR / "sam3",
        "size": "~2.4GB",
        "method": "huggingface",
        "hf_repo": "facebook/sam3",
    },
    "sam3d_body": {
        "name": "SAM 3D Body DINOv3 (Single-Image 3D Body)",
        "source": "facebook/sam-3d-body-dinov3",
        "path": MODEL_DIR / "sam3d_body",
        "size": "~1.2GB",
        "method": "huggingface",
        "hf_repo": "facebook/sam-3d-body-dinov3",
    },
    "fashn_parser": {
        "name": "FASHN Parser (18-Class Body Parsing)",
        "source": "fashn-ai/fashn-human-parser",
        "path": MODEL_DIR / "fashn_parser",
        "size": "~250MB",
        "method": "huggingface",
        "hf_repo": "fashn-ai/fashn-human-parser",
    },
    "catvton_flux": {
        "name": "CatVTON-FLUX (Virtual Try-On)",
        "source": "xiaozaa/catvton-flux-alpha",
        "path": MODEL_DIR / "catvton_flux",
        "size": "~1.5GB",
        "method": "huggingface",
        "hf_repo": "xiaozaa/catvton-flux-alpha",
        "sub_downloads": {
            "catvton/flux-lora": "LoRA weights",
            "catvton/mix-48k-1024/attention": "Mix attention weights",
            "catvton/DensePose": "DensePose model",
            "catvton/SCHP": "SCHP parsing model",
        },
    },
    "flux_gguf": {
        "name": "FLUX.1-dev GGUF Q8 (Base Diffusion)",
        "source": "city96/FLUX.1-dev-gguf",
        "path": MODEL_DIR / "flux_gguf",
        "size": "~12GB",
        "method": "huggingface",
        "hf_repo": "city96/FLUX.1-dev-gguf",
        "filename": "flux1-dev-Q8_0.gguf",
    },
    "hunyuan3d": {
        "name": "Hunyuan3D 2.0 (Multi-View 3D)",
        "source": "tencent/Hunyuan3D-2",
        "path": MODEL_DIR / "hunyuan3d",
        "size": "~8GB",
        "method": "huggingface",
        "hf_repo": "tencent/Hunyuan3D-2",
    },
}


def download_hf_model(repo_id: str, local_dir: Path, filename: str | None = None):
    """Download from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("  Install huggingface_hub: pip install huggingface_hub")
        return False

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        if filename:
            print(f"  Downloading {filename} from {repo_id}...")
            hf_hub_download(
                repo_id=repo_id, filename=filename,
                local_dir=str(local_dir),
            )
        else:
            print(f"  Downloading full repo {repo_id}...")
            snapshot_download(
                repo_id=repo_id, local_dir=str(local_dir),
            )
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def check_model(key: str, info: dict) -> bool:
    """Check if model is already downloaded."""
    path = info["path"]
    if not path.exists():
        return False
    if "filename" in info:
        return (path / info["filename"]).exists()
    # Check for any model files
    exts = ("*.safetensors", "*.pt", "*.gguf", "*.bin", "*.pth")
    return any(path.glob(ext) for ext in exts)


def main():
    print("=" * 60)
    print("  StyleLens V6 — Model Setup Script")
    print("=" * 60)
    print(f"\n  Model directory: {MODEL_DIR}\n")

    # Check status
    total_size = 0
    to_download = []

    for key, info in MODELS.items():
        installed = check_model(key, info)
        status = "INSTALLED" if installed else "MISSING"
        icon = "+" if installed else "-"
        print(f"  [{icon}] {info['name']}")
        print(f"      Source: {info['source']} | Size: {info['size']} | {status}")

        if not installed:
            to_download.append((key, info))

    print(f"\n  {len(MODELS) - len(to_download)}/{len(MODELS)} models installed")

    if not to_download:
        print("  All models are already downloaded!")
        return

    print(f"\n  {len(to_download)} models need downloading:")
    for key, info in to_download:
        print(f"    - {info['name']} ({info['size']})")

    answer = input("\n  Download missing models? [y/N]: ").strip().lower()
    if answer != "y":
        print("  Skipped.")
        return

    # Download each
    for key, info in to_download:
        print(f"\n  Downloading: {info['name']}...")

        if info["method"] == "huggingface":
            success = download_hf_model(
                info.get("hf_repo", info["source"]),
                info["path"],
                info.get("filename"),
            )
            if success:
                print(f"  {info['name']} downloaded successfully!")
            else:
                print(f"  {info['name']} download FAILED")

        elif info["method"] == "ultralytics":
            print("  YOLO26 will download automatically on first use.")
            info["path"].mkdir(parents=True, exist_ok=True)
            print("  Created directory — model downloads at runtime.")

        # Handle sub-downloads
        if "sub_downloads" in info:
            for sub_path, desc in info["sub_downloads"].items():
                full_path = MODEL_DIR / sub_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    print(f"  Created sub-directory: {sub_path} ({desc})")

    print("\n  Setup complete!")
    print("  Run: python main.py")


if __name__ == "__main__":
    main()
