#!/usr/bin/env python3
"""
Test Quality Try-On V7: 전체 사용자 이미지 종합 품질 테스트

목적:
- 9개 사용자 이미지 모두에 대해 CatVTON-FLUX 가상 피팅 실행
- adaptive mask 전략 적용 (fitting.py의 _generate_adaptive_mask 로직 재현)
- 한 번의 배치 처리로 비용 절감
- 결과를 3×3 그리드로 시각화 (Original → Result 비교)

출력:
- output/quality_tryon_v7/comparison_v7_grid.png (3×3 그리드)
- output/quality_tryon_v7/result_{i}_{filename}.png (개별 결과 9개)
- output/quality_tryon_v7/mask_{i}_{filename}.png (마스크 9개)
"""

import base64
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent

# Modal app import (after path setup)
from worker.modal_app import (
    app as modal_app,
    run_catvton_batch,
    run_light_models,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
OUTPUT_DIR = PROJECT_ROOT / "ai-service" / "output" / "quality_tryon_v7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU 비용 (H200 $5.40/hr)
H200_COST_PER_SEC = 5.40 / 3600


def load_image_base64(path: Path) -> str:
    """이미지를 base64 인코딩."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def base64_to_pil(b64_str: str) -> Image.Image:
    """base64 문자열을 PIL 이미지로 변환."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def pil_to_base64(img: Image.Image) -> str:
    """PIL 이미지를 base64 문자열로 변환."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_adaptive_mask(parse_map: np.ndarray, category: str = "top") -> np.ndarray:
    """
    Adaptive mask generation (fitting.py의 _generate_adaptive_mask 로직 재현).

    Strategy:
        1. Generate base FASHN mask with enhanced dilation
        2. Check coverage ratio (mask pixels / image pixels)
        3. If coverage >= 15%, use base mask
        4. If coverage < 15%, switch to Upper Body Rect strategy:
           - Find bounding box of clothes + arms
           - Extend 30% upward (shoulders/collarbone)
           - Extend 10% left/right
           - Apply Gaussian smoothing for natural edges

    Args:
        parse_map: FASHN parse map (HxW uint8, class IDs)
        category: "top", "bottom", "dress", etc.

    Returns:
        Binary mask (HxW uint8, 0/255)
    """
    # Step 1: Generate base mask
    mask = np.zeros(parse_map.shape, dtype=np.uint8)

    if category in ("top", "upper", "outerwear"):
        mask[parse_map == 4] = 255   # upper_clothes
        mask[parse_map == 14] = 255  # left_arm
        mask[parse_map == 15] = 255  # right_arm
    elif category in ("bottom", "lower"):
        mask[parse_map == 6] = 255   # pants
        mask[parse_map == 5] = 255   # skirt
    else:  # dress
        mask[parse_map == 7] = 255   # dress
        mask[parse_map == 4] = 255   # upper_clothes

    # Enhanced dilation
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Check coverage
    coverage = (mask > 0).sum() / mask.size

    if coverage >= 0.15:
        print(f"    Using base FASHN mask (coverage {coverage*100:.1f}%)")
        return mask

    # Step 3: Switch to Upper Body Rect
    print(f"    Switching to Upper Body Rect (base coverage {coverage*100:.1f}% < 15%)")

    if category in ("top", "upper", "outerwear"):
        clothes_mask = (parse_map == 4).astype(np.uint8) * 255
        arms_mask = ((parse_map == 14) | (parse_map == 15)).astype(np.uint8) * 255
    elif category in ("bottom", "lower"):
        clothes_mask = ((parse_map == 6) | (parse_map == 5)).astype(np.uint8) * 255
        arms_mask = ((parse_map == 12) | (parse_map == 13)).astype(np.uint8) * 255
    else:  # dress
        clothes_mask = ((parse_map == 7) | (parse_map == 4)).astype(np.uint8) * 255
        arms_mask = ((parse_map == 14) | (parse_map == 15) |
                     (parse_map == 12) | (parse_map == 13)).astype(np.uint8) * 255

    combined = np.maximum(clothes_mask, arms_mask)

    ys, xs = np.where(combined > 0)
    if len(ys) == 0:
        print("    No clothes/arms detected, using full mask fallback")
        return np.ones(parse_map.shape, dtype=np.uint8) * 255

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Extend upward 30% (shoulders/collarbone region)
    height = y_max - y_min
    y_min = max(0, y_min - int(height * 0.3))

    # Extend left/right 10%
    width = x_max - x_min
    x_min = max(0, x_min - int(width * 0.1))
    x_max = min(parse_map.shape[1] - 1, x_max + int(width * 0.1))

    # Create rectangular mask
    rect_mask = np.zeros(parse_map.shape, dtype=np.uint8)
    rect_mask[y_min:y_max, x_min:x_max] = 255

    # Smooth edges for natural blending
    rect_mask = cv2.GaussianBlur(rect_mask, (21, 21), 0)
    _, rect_mask = cv2.threshold(rect_mask, 127, 255, cv2.THRESH_BINARY)

    print(f"    Upper Body Rect: bbox=[{y_min}:{y_max}, {x_min}:{x_max}], "
          f"coverage {(rect_mask > 0).sum() / rect_mask.size * 100:.1f}%")

    return rect_mask


def create_grid_comparison(
    originals: list[Image.Image],
    results: list[Image.Image],
    filenames: list[str],
) -> Image.Image:
    """
    3×3 그리드 생성: 각 셀은 Original(좌) → Result(우)

    Args:
        originals: 9개 원본 이미지
        results: 9개 결과 이미지
        filenames: 9개 파일명 (라벨용)

    Returns:
        그리드 이미지
    """
    # 각 셀의 크기 (Original + Result 나란히)
    cell_h = 400
    cell_w = cell_h * 2 + 20  # Original + gap + Result
    label_h = 30

    # 3×3 그리드
    grid_w = cell_w * 3 + 40  # 좌우 여백
    grid_h = (cell_h + label_h) * 3 + 40  # 상하 여백

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # 기본 폰트 사용 (시스템 폰트)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    for i in range(9):
        row = i // 3
        col = i % 3

        x_base = 20 + col * cell_w
        y_base = 20 + row * (cell_h + label_h)

        # Original 이미지
        orig = originals[i]
        aspect = orig.width / orig.height
        orig_w = int(cell_h * aspect * 0.45)  # 셀의 45% 사용
        orig_resized = orig.resize((orig_w, cell_h), Image.Resampling.LANCZOS)
        canvas.paste(orig_resized, (x_base, y_base + label_h))

        # Result 이미지
        res = results[i]
        res_aspect = res.width / res.height
        res_w = int(cell_h * res_aspect * 0.45)
        res_resized = res.resize((res_w, cell_h), Image.Resampling.LANCZOS)
        canvas.paste(res_resized, (x_base + orig_w + 10, y_base + label_h))

        # 라벨
        label = f"{i}: {filenames[i]}"
        draw.text((x_base + 5, y_base + 5), label, fill=(0, 0, 0), font=font)

    return canvas


def main():
    print("=" * 80)
    print("Quality Try-On V7: 전체 사용자 이미지 종합 품질 테스트")
    print("=" * 80)

    # 1. 이미지 로드
    print("\n[1/6] 이미지 로드 중...")

    # 9개 사용자 이미지
    user_img_paths = sorted(USER_IMG_DIR.glob("*.jpg"))
    if len(user_img_paths) != 9:
        print(f"  WARNING: Expected 9 user images, found {len(user_img_paths)}")

    user_filenames = [p.name for p in user_img_paths]
    user_b64_list = [load_image_base64(p) for p in user_img_paths]

    print(f"  ✓ Loaded {len(user_img_paths)} user images")
    for i, fname in enumerate(user_filenames):
        print(f"    [{i}] {fname}")

    # 가먼트 이미지 (wear_imgs[3]: 블라우스)
    wear_imgs = sorted(WEAR_DIR.glob("*.png"))
    garment_img_path = wear_imgs[3]
    garment_b64 = load_image_base64(garment_img_path)
    print(f"  ✓ Garment: {garment_img_path.name}")

    # 2. Modal GPU 작업: SAM3 + FASHN (한 블록에서 순차 실행)
    print("\n[2/6] SAM3 segmentation + FASHN parsing (Modal GPU)...")
    t0 = time.time()

    with modal_app.run():
        # SAM3: 가먼트 segmentation
        print("  - SAM3 segmentation (가먼트)...")
        garment_seg_result = run_light_models.remote(
            task="segment_sam3",
            image_b64=garment_b64,
        )
        garment_seg_b64 = garment_seg_result["segmented_b64"]

        # FASHN: 9개 사용자 이미지 parsing (순차)
        print("  - FASHN parsing (9개 사용자 이미지)...")
        fashn_results = []
        for i, user_b64 in enumerate(user_b64_list):
            print(f"    [{i}] {user_filenames[i]}...")
            result = run_light_models.remote(
                task="parse_fashn",
                image_b64=user_b64,
            )
            fashn_results.append(result)

    t_preprocessing = time.time() - t0
    print(f"  ✓ SAM3 + FASHN 완료 ({t_preprocessing:.2f}s)")

    # 3. Adaptive mask 생성 (9개)
    print("\n[3/6] Adaptive mask 생성 중...")

    mask_b64_list = []
    mask_images = []

    for i, fashn_result in enumerate(fashn_results):
        print(f"  [{i}] {user_filenames[i]}")

        # Parse map을 numpy array로 변환
        parsemap_b64 = fashn_result["parsemap_b64"]
        parsemap_pil = base64_to_pil(parsemap_b64).convert("L")
        parsemap_np = np.array(parsemap_pil)

        # Adaptive mask 생성
        mask_np = generate_adaptive_mask(parsemap_np, category="top")

        # 마스크 저장
        mask_pil = Image.fromarray(mask_np)
        mask_images.append(mask_pil)
        mask_path = OUTPUT_DIR / f"mask_{i}_{user_filenames[i].replace('.jpg', '.png')}"
        mask_pil.save(mask_path)

        # Base64 변환
        mask_b64 = pil_to_base64(mask_pil)
        mask_b64_list.append(mask_b64)

    print(f"  ✓ 9개 마스크 생성 완료")

    # 4. CatVTON 배치 처리 (9개 동시)
    print("\n[4/6] CatVTON 배치 처리 (9개 동시)...")
    print("  - guidance=30.0, num_steps=30")

    t0 = time.time()

    with modal_app.run():
        catvton_result = run_catvton_batch.remote(
            persons_b64=user_b64_list,           # list of 9 base64 strings
            clothing_b64=garment_seg_b64,        # single string
            masks_b64=mask_b64_list,             # list of 9 base64 strings
            num_steps=30,
            guidance=30.0,
        )

    t_catvton = time.time() - t0
    print(f"  ✓ CatVTON 완료 ({t_catvton:.2f}s)")

    # 결과 추출
    results_b64 = catvton_result["results_b64"]
    num_angles = catvton_result["num_angles"]
    print(f"  ✓ Received {len(results_b64)} results (num_angles={num_angles})")

    # 5. 개별 결과 저장
    print("\n[5/6] 개별 결과 저장 중...")

    original_pils = [base64_to_pil(b64) for b64 in user_b64_list]
    result_pils = [base64_to_pil(b64) for b64 in results_b64]

    for i, result_pil in enumerate(result_pils):
        result_path = OUTPUT_DIR / f"result_{i}_{user_filenames[i].replace('.jpg', '.png')}"
        result_pil.save(result_path)

    print(f"  ✓ 9개 결과 저장 완료")

    # 6. 3×3 그리드 비교 이미지 생성
    print("\n[6/6] 3×3 그리드 비교 이미지 생성 중...")

    grid = create_grid_comparison(
        originals=original_pils,
        results=result_pils,
        filenames=user_filenames,
    )

    grid_path = OUTPUT_DIR / "comparison_v7_grid.png"
    grid.save(grid_path)
    print(f"  ✓ 그리드 저장 완료: {grid_path}")

    # 7. 비용 계산
    print("\n" + "=" * 80)
    print("비용 계산 (H200: $5.40/hr)")
    print("=" * 80)

    total_time = t_preprocessing + t_catvton
    total_cost = total_time * H200_COST_PER_SEC

    print(f"  SAM3 + FASHN (x9):  {t_preprocessing:6.2f}s  →  ${t_preprocessing * H200_COST_PER_SEC:.4f}")
    print(f"  CatVTON (x9 batch): {t_catvton:6.2f}s  →  ${t_catvton * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 40}")
    print(f"  총 GPU 시간:        {total_time:6.2f}s  →  ${total_cost:.4f}")
    print()

    # 8. 결과 안내
    print("=" * 80)
    print("테스트 완료")
    print("=" * 80)
    print(f"\n전체 비교 그리드: {grid_path}")
    print(f"\n개별 결과 (9개):")
    for i, fname in enumerate(user_filenames):
        result_fname = f"result_{i}_{fname.replace('.jpg', '.png')}"
        print(f"  [{i}] {result_fname}")
    print(f"\n마스크 (9개):")
    for i, fname in enumerate(user_filenames):
        mask_fname = f"mask_{i}_{fname.replace('.jpg', '.png')}"
        print(f"  [{i}] {mask_fname}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
