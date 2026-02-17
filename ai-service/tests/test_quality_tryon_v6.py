#!/usr/bin/env python3
"""
Test Quality Try-On V6: 탱크탑 마스크 전략 비교 테스트

문제:
- user_imgs[6] 탱크탑은 피부 밀착 + 측면 각도 → CatVTON 실패
- FASHN은 upper_clothes(4), arms(14/15) 정확히 감지했으나 마스크 영역 부족

해결:
- 3가지 마스크 전략 비교 테스트
  A) expanded_skin: upper_clothes + arms + 목/쇄골 확장
  B) upper_body_rect: 상체 bounding box + 어깨/쇄골 확장
  C) full_torso: 얼굴 제외 상체 전체

출력:
- output/quality_tryon_v6/comparison_v6.png (5장 비교)
- mask_a.png, mask_b.png, mask_c.png
"""

import base64
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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
OUTPUT_DIR = PROJECT_ROOT / "ai-service" / "output" / "quality_tryon_v6"
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


def make_mask_expanded(parse_map: np.ndarray) -> np.ndarray:
    """
    마스크 전략 A: expanded_skin
    - upper_clothes(4) + arms(14/15) + 목/쇄골 영역까지 확장
    """
    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    mask[parse_map == 4] = 255   # upper_clothes
    mask[parse_map == 14] = 255  # left_arm
    mask[parse_map == 15] = 255  # right_arm

    # 얼굴(11) 영역의 아래 50%를 포함하여 목/쇄골 커버
    face_mask = (parse_map == 11).astype(np.uint8) * 255
    h = face_mask.shape[0]
    face_lower = face_mask.copy()
    face_lower[:h//2, :] = 0  # 윗부분 제거 (눈/이마)
    mask = np.maximum(mask, face_lower)

    # 넉넉한 dilation
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def make_mask_rect(parse_map: np.ndarray) -> np.ndarray:
    """
    마스크 전략 B: upper_body_rect
    - 상의 bounding box + 위로 확장하여 어깨/쇄골 커버
    """
    clothes_mask = (parse_map == 4).astype(np.uint8) * 255
    arms_mask = ((parse_map == 14) | (parse_map == 15)).astype(np.uint8) * 255
    combined = np.maximum(clothes_mask, arms_mask)

    # Bounding box
    ys, xs = np.where(combined > 0)
    if len(ys) == 0:
        return np.ones(parse_map.shape, dtype=np.uint8) * 255

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # 위로 확장 (어깨/쇄골): y_min에서 높이의 30% 위로
    height = y_max - y_min
    y_min = max(0, y_min - int(height * 0.3))
    # 좌우도 10% 확장
    width = x_max - x_min
    x_min = max(0, x_min - int(width * 0.1))
    x_max = min(parse_map.shape[1] - 1, x_max + int(width * 0.1))

    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def make_mask_full_torso(parse_map: np.ndarray) -> np.ndarray:
    """
    마스크 전략 C: full_torso
    - 얼굴/머리 제외 상체 전체 (가장 공격적)
    """
    mask = np.ones(parse_map.shape, dtype=np.uint8) * 255

    # 보존할 영역: 배경(0), 머리카락(2), 얼굴 윗부분(11의 상단 60%)
    mask[parse_map == 0] = 0    # background
    mask[parse_map == 2] = 0    # hair

    # 얼굴 상단 60% 보존 (눈/이마)
    face_mask = (parse_map == 11).astype(np.uint8)
    h = face_mask.shape[0]
    ys = np.where(face_mask.any(axis=1))[0]
    if len(ys) > 0:
        face_top = ys[0]
        face_bottom = ys[-1]
        face_h = face_bottom - face_top
        preserve_end = face_top + int(face_h * 0.6)
        # 보존: 눈/이마 영역
        for y in range(face_top, preserve_end):
            mask[y, face_mask[y] > 0] = 0

    # 하의 보존 (pants=6, skirt=5)
    mask[parse_map == 6] = 0
    mask[parse_map == 5] = 0

    # Smooth
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def create_comparison_image(
    original: Image.Image,
    garment: Image.Image,
    result_a: Image.Image,
    result_b: Image.Image,
    result_c: Image.Image,
    labels: list[str],
) -> Image.Image:
    """5장의 이미지를 가로로 나란히 배치 (라벨 포함)."""
    images = [original, garment, result_a, result_b, result_c]

    # 모든 이미지를 동일 크기로 리사이즈
    target_h = 512
    resized = []
    for img in images:
        aspect = img.width / img.height
        new_w = int(target_h * aspect)
        resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))

    # 라벨 높이
    label_h = 40
    total_h = target_h + label_h
    total_w = sum(img.width for img in resized) + 20  # 여백

    # 캔버스 생성
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # 이미지 배치
    x_offset = 10
    for img, label in zip(resized, labels):
        # 이미지
        canvas.paste(img, (x_offset, label_h))

        # 라벨 (OpenCV로 텍스트 추가)
        cv_canvas = np.array(canvas)
        cv2.putText(
            cv_canvas,
            label,
            (x_offset + 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        canvas = Image.fromarray(cv_canvas)

        x_offset += img.width + 10

    return canvas


def main():
    print("=" * 80)
    print("Quality Try-On V6: 탱크탑 마스크 전략 비교 테스트")
    print("=" * 80)

    # 1. 이미지 로드
    print("\n[1/6] 이미지 로드 중...")

    # User images (탱크탑만: idx 6)
    user_imgs = sorted(USER_IMG_DIR.glob("*.jpg"))
    tanktop_img_path = user_imgs[6]
    print(f"  - Tanktop: {tanktop_img_path.name}")

    # Garment image (close-up frontal blouse: wear_imgs[3])
    wear_imgs = sorted(WEAR_DIR.glob("*.png"))
    garment_img_path = wear_imgs[3]
    print(f"  - Garment: {garment_img_path.name}")

    tanktop_b64 = load_image_base64(tanktop_img_path)
    garment_b64 = load_image_base64(garment_img_path)

    # 2. SAM3 segmentation (가먼트)
    print("\n[2/6] SAM3 segmentation (가먼트)...")
    t0 = time.time()

    with modal_app.run():
        garment_seg_result = run_light_models.remote(
            task="segment_sam3",
            image_b64=garment_b64,
        )

    t_sam3 = time.time() - t0
    garment_seg_b64 = garment_seg_result["segmented_b64"]
    print(f"  ✓ SAM3 완료 ({t_sam3:.2f}s)")

    # 3. FASHN parsing (탱크탑)
    print("\n[3/6] FASHN parsing (탱크탑)...")
    t0 = time.time()

    with modal_app.run():
        fashn_result = run_light_models.remote(
            task="parse_fashn",
            image_b64=tanktop_b64,
        )

    t_fashn = time.time() - t0
    parsemap_b64 = fashn_result["parsemap_b64"]
    print(f"  ✓ FASHN 완료 ({t_fashn:.2f}s)")

    # Parse map을 numpy array로 변환 (grayscale PNG base64)
    parsemap_pil = base64_to_pil(parsemap_b64).convert("L")
    parsemap_np = np.array(parsemap_pil)

    # 4. 3가지 마스크 생성
    print("\n[4/6] 3가지 마스크 생성 중...")

    mask_a = make_mask_expanded(parsemap_np)
    mask_b = make_mask_rect(parsemap_np)
    mask_c = make_mask_full_torso(parsemap_np)

    # 마스크 저장
    Image.fromarray(mask_a).save(OUTPUT_DIR / "mask_a.png")
    Image.fromarray(mask_b).save(OUTPUT_DIR / "mask_b.png")
    Image.fromarray(mask_c).save(OUTPUT_DIR / "mask_c.png")
    print("  ✓ 마스크 저장 완료 (mask_a/b/c.png)")

    # 마스크를 base64로 변환
    mask_a_b64 = pil_to_base64(Image.fromarray(mask_a))
    mask_b_b64 = pil_to_base64(Image.fromarray(mask_b))
    mask_c_b64 = pil_to_base64(Image.fromarray(mask_c))

    # 5. CatVTON 실행 (3가지 마스크 배치 처리)
    print("\n[5/6] CatVTON 실행 (3가지 마스크 배치)...")
    print("  - guidance=30.0, num_steps=30")

    t0 = time.time()

    with modal_app.run():
        catvton_result = run_catvton_batch.remote(
            persons_b64=[tanktop_b64, tanktop_b64, tanktop_b64],
            clothing_b64=garment_seg_b64,  # 단일 string (list 아님!)
            masks_b64=[mask_a_b64, mask_b_b64, mask_c_b64],
            num_steps=30,
            guidance=30.0,
        )

    t_catvton = time.time() - t0
    print(f"  ✓ CatVTON 완료 ({t_catvton:.2f}s)")

    # 결과 추출 (results_b64는 list of base64 strings)
    results_b64 = catvton_result["results_b64"]
    result_a_b64 = results_b64[0]
    result_b_b64 = results_b64[1]
    result_c_b64 = results_b64[2]

    # 6. 비교 이미지 생성
    print("\n[6/6] 비교 이미지 생성 중...")

    original_pil = base64_to_pil(tanktop_b64)
    garment_pil = base64_to_pil(garment_b64)
    result_a_pil = base64_to_pil(result_a_b64)
    result_b_pil = base64_to_pil(result_b_b64)
    result_c_pil = base64_to_pil(result_c_b64)

    comparison = create_comparison_image(
        original_pil,
        garment_pil,
        result_a_pil,
        result_b_pil,
        result_c_pil,
        labels=[
            "Original (Tanktop)",
            "Garment (Blouse)",
            "Mask A: Expanded Skin",
            "Mask B: Upper Body Rect",
            "Mask C: Full Torso",
        ],
    )

    comparison_path = OUTPUT_DIR / "comparison_v6.png"
    comparison.save(comparison_path)
    print(f"  ✓ 저장 완료: {comparison_path}")

    # 개별 결과도 저장
    result_a_pil.save(OUTPUT_DIR / "result_a_expanded_skin.png")
    result_b_pil.save(OUTPUT_DIR / "result_b_upper_body_rect.png")
    result_c_pil.save(OUTPUT_DIR / "result_c_full_torso.png")
    print(f"  ✓ 개별 결과 저장 완료")

    # 7. 비용 계산
    print("\n" + "=" * 80)
    print("비용 계산 (H200: $5.40/hr)")
    print("=" * 80)

    total_time = t_sam3 + t_fashn + t_catvton
    total_cost = total_time * H200_COST_PER_SEC

    print(f"  SAM3 (가먼트):       {t_sam3:6.2f}s  →  ${t_sam3 * H200_COST_PER_SEC:.4f}")
    print(f"  FASHN (탱크탑):      {t_fashn:6.2f}s  →  ${t_fashn * H200_COST_PER_SEC:.4f}")
    print(f"  CatVTON (x3 마스크): {t_catvton:6.2f}s  →  ${t_catvton * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 40}")
    print(f"  총 GPU 시간:         {total_time:6.2f}s  →  ${total_cost:.4f}")
    print()

    # 8. 마스크 전략 비교 안내
    print("=" * 80)
    print("마스크 전략 비교 결과")
    print("=" * 80)
    print("\n각 마스크를 확인하여 탱크탑을 가장 잘 교체한 전략을 선택하세요:")
    print()
    print(f"  A) Expanded Skin:      {OUTPUT_DIR / 'result_a_expanded_skin.png'}")
    print(f"  B) Upper Body Rect:    {OUTPUT_DIR / 'result_b_upper_body_rect.png'}")
    print(f"  C) Full Torso:         {OUTPUT_DIR / 'result_c_full_torso.png'}")
    print()
    print(f"  전체 비교:             {comparison_path}")
    print()
    print("마스크 A: upper_clothes + arms + 목/쇄골 확장 (보수적)")
    print("마스크 B: 상체 bounding box + 어깨/쇄골 확장 (중간)")
    print("마스크 C: 얼굴 제외 상체 전체 (공격적)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
