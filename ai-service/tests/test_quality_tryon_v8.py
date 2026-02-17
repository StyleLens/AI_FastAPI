#!/usr/bin/env python3
"""
Test Quality Try-On V8: 가상 모델 기반 360° 피팅 테스트

새 파이프라인 개념 (원본 사진에 합성하지 않음!):
1. 원본 사진 → 체형/외모 분석 자료로만 사용
2. SAM 3D Body → 사용자와 동일한 가상 모델 (3D 메시) 생성
3. 8각도 렌더링 → 가상 모델 이미지 8장 (기존 E2E 출력물 재사용)
4. CatVTON-FLUX × 8 → 가상 모델에 옷 피팅
5. Hunyuan3D → 3D GLB

출력:
- output/quality_tryon_v8/comparison_v8_8angles.png (2행 × 4열 그리드)
- output/quality_tryon_v8/fitted_0.png ~ fitted_315.png (개별 결과)
- output/quality_tryon_v8/phase4_model_v8.glb (3D 모델)
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
    run_hunyuan3d,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
E2E_OUTPUT = ROOT / "output" / "e2e_test"
OUTPUT_DIR = ROOT / "output" / "quality_tryon_v8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 8각도 렌더링 경로
RENDER_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

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


def pil_to_base64(img: Image.Image, quality: int = 95) -> str:
    """PIL 이미지를 base64 문자열로 변환 (JPEG)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_bytes(b64_str: str) -> bytes:
    """base64 문자열을 bytes로 변환."""
    return base64.b64decode(b64_str)


def make_full_mask(width: int, height: int) -> str:
    """
    Full mask 생성 (메시 렌더는 전체 재생성).
    White (255) = inpaint region.
    Returns base64 PNG string.
    """
    mask = np.ones((height, width), dtype=np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_8angle_grid(
    results: list[Image.Image],
    angles: list[int],
) -> Image.Image:
    """
    8각도 결과를 2행 × 4열 그리드로 배치 (라벨 포함).

    Args:
        results: 8개의 결과 이미지 (PIL)
        angles: 8개의 각도 리스트 [0, 45, 90, 135, 180, 225, 270, 315]

    Returns:
        그리드 이미지 (PIL)
    """
    # 모든 이미지를 동일 크기로 리사이즈
    target_h = 400
    resized = []
    for img in results:
        aspect = img.width / img.height
        new_w = int(target_h * aspect)
        resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))

    # 그리드 레이아웃 (2행 × 4열)
    cols = 4
    rows = 2
    label_h = 30
    cell_h = target_h + label_h

    # 각 셀의 최대 너비 계산
    max_w = max(img.width for img in resized)
    cell_w = max_w + 20  # 여백

    # 캔버스 생성
    canvas_w = cell_w * cols
    canvas_h = cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 시스템 폰트 사용
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    # 이미지 배치
    for idx, (img, angle) in enumerate(zip(resized, angles)):
        row = idx // cols
        col = idx % cols

        # 셀 중앙 정렬
        x_offset = col * cell_w + (cell_w - img.width) // 2
        y_offset = row * cell_h + label_h

        # 이미지 붙이기
        canvas.paste(img, (x_offset, y_offset))

        # 라벨 추가
        label = f"{angle}°"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = col * cell_w + (cell_w - text_w) // 2
        text_y = row * cell_h + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return canvas


def main():
    print("=" * 80)
    print("Quality Try-On V8: 가상 모델 기반 360° 피팅 테스트")
    print("=" * 80)

    # Phase 1: 메시 렌더 이미지 로드 (기존 E2E 출력물 재사용)
    print("\n[Phase 1] 메시 렌더 이미지 로드 중...")

    render_paths = {}
    for angle in RENDER_ANGLES:
        p = E2E_OUTPUT / f"phase1_render_{angle}.jpg"
        if not p.exists():
            print(f"  ✗ 메시 렌더 이미지 없음: {p}")
            print("  → E2E 테스트를 먼저 실행하여 메시 렌더를 생성하세요.")
            return
        render_paths[angle] = p

    print(f"  ✓ 8각도 메시 렌더 이미지 로드 완료")

    # 렌더 이미지를 base64로 변환
    render_b64s = []
    for angle in RENDER_ANGLES:
        b64 = load_image_base64(render_paths[angle])
        render_b64s.append(b64)

    # 이미지 해상도 확인 (첫 번째 렌더)
    first_img = base64_to_pil(render_b64s[0])
    img_w, img_h = first_img.size
    print(f"  - 메시 렌더 해상도: {img_w}×{img_h}")

    # Phase 2: 가먼트 준비 (SAM3 segmentation)
    print("\n[Phase 2] 가먼트 준비 중...")

    # Garment image (wear_imgs[3] — 블라우스)
    wear_imgs = sorted(WEAR_DIR.glob("*.png"))
    garment_img_path = wear_imgs[3]
    print(f"  - 가먼트: {garment_img_path.name}")

    garment_b64 = load_image_base64(garment_img_path)

    # SAM3 segmentation
    print("  - SAM3 segmentation 실행 중...")
    t0 = time.time()

    with modal_app.run():
        garment_seg_result = run_light_models.remote(
            task="segment_sam3",
            image_b64=garment_b64,
        )

    t_sam3 = time.time() - t0

    if "error" in garment_seg_result:
        print(f"  ✗ SAM3 실패: {garment_seg_result['error']}")
        return

    garment_seg_b64 = garment_seg_result["segmented_b64"]
    print(f"  ✓ SAM3 완료 ({t_sam3:.2f}s)")

    # Phase 3: CatVTON-FLUX 8각도 배치 처리
    print("\n[Phase 3] CatVTON-FLUX 8각도 배치 처리 중...")
    print("  - guidance=30.0, num_steps=30")

    # Full masks for all 8 angles (메시 렌더는 전체 재생성)
    masks_b64 = [make_full_mask(img_w, img_h) for _ in RENDER_ANGLES]

    t0 = time.time()

    with modal_app.run():
        catvton_result = run_catvton_batch.remote(
            persons_b64=render_b64s,
            clothing_b64=garment_seg_b64,
            masks_b64=masks_b64,
            num_steps=30,
            guidance=30.0,
        )

    t_catvton = time.time() - t0

    if "error" in catvton_result:
        print(f"  ✗ CatVTON 실패: {catvton_result['error']}")
        return

    print(f"  ✓ CatVTON 완료 ({t_catvton:.2f}s)")

    # 결과 추출
    results_b64 = catvton_result["results_b64"]
    print(f"  - 생성된 각도: {catvton_result['num_angles']}")

    # Phase 4: Hunyuan3D → GLB
    print("\n[Phase 4] Hunyuan3D 3D 모델 생성 중...")
    print("  - shape_steps=50, paint_steps=20, texture_res=4096")

    # 정면(0°) 결과를 front image로 사용
    # 측면(45°, 315°)을 reference로 사용
    front_b64 = results_b64[0]  # 0°
    ref_45_b64 = results_b64[1]  # 45°
    ref_315_b64 = results_b64[7]  # 315°

    t0 = time.time()

    with modal_app.run():
        hy3d_result = run_hunyuan3d.remote(
            front_image_b64=front_b64,
            reference_images_b64=[ref_45_b64, ref_315_b64],
            shape_steps=50,
            paint_steps=20,
            texture_res=4096,
        )

    t_hy3d = time.time() - t0

    if "error" in hy3d_result:
        print(f"  ✗ Hunyuan3D 실패: {hy3d_result['error']}")
    else:
        print(f"  ✓ Hunyuan3D 완료 ({t_hy3d:.2f}s)")
        print(f"  - Vertices: {hy3d_result['num_vertices']}")
        print(f"  - Faces: {hy3d_result['num_faces']}")
        print(f"  - Textured: {hy3d_result['textured']}")

        # GLB 저장
        glb_bytes = base64_to_bytes(hy3d_result["glb_bytes_b64"])
        glb_path = OUTPUT_DIR / "phase4_model_v8.glb"
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        print(f"  ✓ GLB 저장: {glb_path}")

    # Phase 5: 결과 시각화
    print("\n[Phase 5] 결과 시각화 중...")

    # 개별 결과 저장
    result_pils = []
    for angle, result_b64 in zip(RENDER_ANGLES, results_b64):
        result_pil = base64_to_pil(result_b64)
        result_pils.append(result_pil)

        # 개별 파일 저장
        output_path = OUTPUT_DIR / f"fitted_{angle}.png"
        result_pil.save(output_path)
        print(f"  ✓ 저장: fitted_{angle}.png")

    # 8각도 그리드 생성
    grid = create_8angle_grid(result_pils, RENDER_ANGLES)
    grid_path = OUTPUT_DIR / "comparison_v8_8angles.png"
    grid.save(grid_path)
    print(f"  ✓ 그리드 저장: {grid_path}")

    # 비용 계산
    print("\n" + "=" * 80)
    print("비용 계산 (H200: $5.40/hr)")
    print("=" * 80)

    total_time = t_sam3 + t_catvton
    if "error" not in hy3d_result:
        total_time += t_hy3d

    total_cost = total_time * H200_COST_PER_SEC

    print(f"  SAM3 (가먼트):        {t_sam3:6.2f}s  →  ${t_sam3 * H200_COST_PER_SEC:.4f}")
    print(f"  CatVTON (8각도):      {t_catvton:6.2f}s  →  ${t_catvton * H200_COST_PER_SEC:.4f}")

    if "error" not in hy3d_result:
        print(f"  Hunyuan3D (3D GLB):   {t_hy3d:6.2f}s  →  ${t_hy3d * H200_COST_PER_SEC:.4f}")

    print(f"  {'─' * 40}")
    print(f"  총 GPU 시간:          {total_time:6.2f}s  →  ${total_cost:.4f}")
    print()

    # 최종 요약
    print("=" * 80)
    print("테스트 완료")
    print("=" * 80)
    print()
    print(f"  8각도 그리드:  {grid_path}")
    print(f"  개별 결과:     {OUTPUT_DIR}/fitted_{{0,45,90,...}}.png")

    if "error" not in hy3d_result:
        print(f"  3D 모델:       {OUTPUT_DIR}/phase4_model_v8.glb")

    print()
    print("가상 모델 기반 360° 피팅 파이프라인 검증 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
