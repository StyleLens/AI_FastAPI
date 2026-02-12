"""
StyleLens V6 — Modal Volume에 모델 가중치 업로드.

로컬의 model/ 디렉토리를 Modal Volume 'stylelens-models'의 /models/ 경로로 업로드.
한 번만 실행하면 되며, 이후 GPU 함수 실행 시 자동 마운트됨.

사용법:
  # 전체 업로드 (최초 1회)
  python scripts/upload_models_to_volume.py

  # 특정 모델만 업로드
  python scripts/upload_models_to_volume.py --models sam3 fashn_parser yolo26

  # 볼륨 내용 확인
  python scripts/upload_models_to_volume.py --list

  # 또는 CLI로 직접:
  modal volume ls stylelens-models
  modal volume put stylelens-models ./model/sam3 /models/sam3
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
VOLUME_NAME = "stylelens-models"

# GPU 워커에서 사용하는 모델 목록 (Phase별)
ALL_MODELS = {
    # Phase 1: Body Reconstruction
    "sam3d_body": {"desc": "SAM 3D Body (single-image → 3D mesh)", "size": "~2.6GB"},
    # Phase 2: Segmentation & Parsing
    "sam3":        {"desc": "SAM 3 (concept-aware segmentation)", "size": "~3.2GB"},
    "fashn_parser": {"desc": "FASHN (18-class fashion parsing)", "size": "~209MB"},
    # Phase 3: Virtual Try-On
    "catvton":     {"desc": "CatVTON (DensePose, SCHP, LoRA, attention)", "size": "~1.3GB"},
    "catvton_flux": {"desc": "CatVTON-FLUX pipeline weights", "size": "~22GB"},
    "flux_gguf":   {"desc": "FLUX.1-dev GGUF Q8 base model", "size": "~12GB"},
    # Phase 4: 3D Generation
    "hunyuan3d":   {"desc": "Hunyuan3D 2.0 (shape + paint)", "size": "~10GB"},
    # Optional
    "insightface": {"desc": "InsightFace buffalo_l (face identity)", "size": "~325MB"},
    "yolo26":      {"desc": "YOLOv26-L (person detection)", "size": "~51MB"},
}


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and print output."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if check and result.returncode != 0:
        print(f"  [ERROR] Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def list_volume():
    """볼륨 내용 확인."""
    print(f"\n=== Modal Volume '{VOLUME_NAME}' 내용 ===\n")
    run_cmd(["modal", "volume", "ls", VOLUME_NAME], check=False)


def upload_model(model_name: str):
    """단일 모델 디렉토리를 볼륨에 업로드."""
    local_path = MODEL_DIR / model_name
    if not local_path.exists():
        print(f"  [SKIP] {model_name}: 로컬에 없음 ({local_path})")
        return False

    remote_path = f"/models/{model_name}"
    info = ALL_MODELS.get(model_name, {})
    desc = info.get("desc", "Unknown")
    size = info.get("size", "?")

    print(f"\n📦 Uploading: {model_name} ({size})")
    print(f"   {desc}")
    print(f"   {local_path} → volume:{remote_path}")

    run_cmd([
        "modal", "volume", "put", VOLUME_NAME,
        str(local_path), remote_path,
    ])
    print(f"   [OK] {model_name} uploaded!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Modal Volume에 모델 가중치 업로드",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="업로드할 모델 이름 (예: sam3 yolo26). 생략하면 전체 업로드.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="볼륨 내용만 확인 (업로드 안 함).",
    )
    args = parser.parse_args()

    if args.list:
        list_volume()
        return

    # 볼륨 생성 확인 (create_if_missing은 코드에서 하지만 CLI로도 확인)
    print(f"=== Modal Volume '{VOLUME_NAME}' 모델 업로드 ===")
    print(f"로컬 모델 디렉토리: {MODEL_DIR}")

    # 업로드 대상 결정
    targets = args.models if args.models else list(ALL_MODELS.keys())

    uploaded = 0
    skipped = 0
    for name in targets:
        if upload_model(name):
            uploaded += 1
        else:
            skipped += 1

    print(f"\n=== 완료: {uploaded} 업로드, {skipped} 스킵 ===")

    # 결과 확인
    print("\n볼륨 최종 상태:")
    list_volume()


if __name__ == "__main__":
    main()
