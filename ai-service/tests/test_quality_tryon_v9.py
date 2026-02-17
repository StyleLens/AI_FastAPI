#!/usr/bin/env python3
"""
Test Quality Try-On V9: 리얼 가상 모델 기반 360° 피팅 + 3D

새 파이프라인 (Phase 1.5 추가):
1. Phase 1: SAM 3D Body → 8각도 메시 렌더 (기존 E2E 재사용)
2. Phase 1.5: run_mesh_to_realistic → 메시 렌더 + 사용자 사진 → 리얼 가상 모델 8장
3. Phase 2: SAM3 segmentation → 의류 분리
4. Phase 3: CatVTON-FLUX × 8 → 리얼 가상 모델에 옷 피팅
5. Phase 4: Hunyuan3D → 정면 결과 → 3D GLB

출력:
- output/quality_tryon_v9/realistic_{0..315}.png (Phase 1.5 결과)
- output/quality_tryon_v9/fitted_{0..315}.png (Phase 3 결과)
- output/quality_tryon_v9/comparison_v9_part1.png (3행 × 4열: 메시/리얼/피팅, 0-135°)
- output/quality_tryon_v9/comparison_v9_part2.png (3행 × 4열: 메시/리얼/피팅, 180-315°)
- output/quality_tryon_v9/model_v9.glb (3D 모델)
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
    run_mesh_to_realistic,
    run_catvton_batch,
    run_light_models,
    run_hunyuan3d,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
E2E_OUTPUT = ROOT / "output" / "e2e_test"
OUTPUT_DIR = ROOT / "output" / "quality_tryon_v9"
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
    Full mask 생성 (리얼 가상 모델은 전체 재생성).
    White (255) = inpaint region.
    Returns base64 PNG string.
    """
    mask = np.ones((height, width), dtype=np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_comparison_grid(
    mesh_renders: list[Image.Image],
    realistic_renders: list[Image.Image],
    fitted_results: list[Image.Image],
    angles: list[int],
    title: str = "V9 Pipeline Comparison",
) -> Image.Image:
    """
    3행 × 4열 비교 그리드 생성.

    Args:
        mesh_renders: 메시 렌더 이미지 4장 (1행)
        realistic_renders: 리얼 가상 모델 이미지 4장 (2행)
        fitted_results: 피팅 결과 이미지 4장 (3행)
        angles: 4개의 각도 리스트
        title: 그리드 제목

    Returns:
        그리드 이미지 (PIL)
    """
    # 모든 이미지를 동일 크기로 리사이즈
    target_h = 350
    resized_mesh = []
    resized_realistic = []
    resized_fitted = []

    for img in mesh_renders:
        aspect = img.width / img.height
        new_w = int(target_h * aspect)
        resized_mesh.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))

    for img in realistic_renders:
        aspect = img.width / img.height
        new_w = int(target_h * aspect)
        resized_realistic.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))

    for img in fitted_results:
        aspect = img.width / img.height
        new_w = int(target_h * aspect)
        resized_fitted.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))

    # 그리드 레이아웃 (3행 × 4열)
    cols = 4
    rows = 3
    label_h = 30
    row_label_w = 120  # 행 라벨 너비 (Raw Mesh / Realistic / Fitted)
    cell_h = target_h + label_h

    # 각 셀의 최대 너비 계산
    max_w = max(
        max(img.width for img in resized_mesh),
        max(img.width for img in resized_realistic),
        max(img.width for img in resized_fitted),
    )
    cell_w = max_w + 20  # 여백

    # 캔버스 생성 (제목 + 행 라벨 공간 추가)
    title_h = 50
    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 시스템 폰트 사용
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 제목 추가
    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_w = bbox[2] - bbox[0]
    text_x = (canvas_w - text_w) // 2
    draw.text((text_x, 15), title, fill=(0, 0, 0), font=font_title)

    # 행 라벨
    row_labels = ["Raw Mesh", "Realistic", "Fitted"]

    # 이미지 배치
    all_rows = [resized_mesh, resized_realistic, resized_fitted]

    for row_idx, (row_images, row_label) in enumerate(zip(all_rows, row_labels)):
        # 행 라벨 추가
        label_y = title_h + row_idx * cell_h + cell_h // 2
        draw.text((10, label_y), row_label, fill=(0, 0, 0), font=font_label)

        for col_idx, (img, angle) in enumerate(zip(row_images, angles)):
            # 셀 중앙 정렬
            x_offset = row_label_w + col_idx * cell_w + (cell_w - img.width) // 2
            y_offset = title_h + row_idx * cell_h + label_h

            # 이미지 붙이기
            canvas.paste(img, (x_offset, y_offset))

            # 각도 라벨 추가 (첫 행에만)
            if row_idx == 0:
                angle_label = f"{angle}°"
                bbox = draw.textbbox((0, 0), angle_label, font=font_small)
                text_w = bbox[2] - bbox[0]
                text_x = row_label_w + col_idx * cell_w + (cell_w - text_w) // 2
                text_y = title_h + row_idx * cell_h + 5
                draw.text((text_x, text_y), angle_label, fill=(0, 0, 0), font=font_small)

    return canvas


def main():
    print("=" * 80)
    print("Quality Try-On V9: 리얼 가상 모델 기반 360° 피팅 + 3D")
    print("=" * 80)

    # Phase 0: 사용자 이미지 로드
    print("\n[Phase 0] 사용자 이미지 로드 중...")

    user_imgs = sorted(USER_IMG_DIR.glob("*.*"))
    if not user_imgs:
        print("  ✗ 사용자 이미지 없음")
        return

    user_img_path = user_imgs[0]  # 첫 번째 사용자 이미지
    print(f"  - 사용자 이미지: {user_img_path.name}")
    user_b64 = load_image_base64(user_img_path)

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

    # Phase 1.5: Mesh → Realistic Virtual Model
    print("\n[Phase 1.5] 메시 → 리얼 가상 모델 변환 중...")
    print("  - 메시 렌더 8장 + 사용자 사진 → 리얼 가상 모델 8장")
    print("  - guidance=30.0, num_steps=28")

    t0 = time.time()

    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64=user_b64,
            angles=RENDER_ANGLES,
            num_steps=28,
            guidance=30.0,
        )

    t_realistic = time.time() - t0

    if "error" in realistic_result:
        print(f"  ✗ Mesh → Realistic 실패: {realistic_result['error']}")
        return

    realistic_b64s = realistic_result["realistic_renders_b64"]
    print(f"  ✓ Mesh → Realistic 완료 ({t_realistic:.2f}s)")
    print(f"  - 생성된 리얼 가상 모델: {len(realistic_b64s)}장")

    # 개별 리얼 렌더 저장
    for angle, r_b64 in zip(RENDER_ANGLES, realistic_b64s):
        img = base64_to_pil(r_b64)
        img.save(OUTPUT_DIR / f"realistic_{angle}.png")

    print(f"  ✓ 리얼 렌더 저장 완료")

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

    # Phase 3: CatVTON-FLUX 8각도 배치 처리 (리얼 가상 모델에 피팅)
    print("\n[Phase 3] CatVTON-FLUX 8각도 배치 처리 중...")
    print("  - 리얼 가상 모델 8장 + 가먼트 → 피팅 결과 8장")
    print("  - guidance=30.0, num_steps=30")

    # Full masks for all 8 angles (리얼 가상 모델도 전체 재생성)
    masks_b64 = [make_full_mask(img_w, img_h) for _ in RENDER_ANGLES]

    t0 = time.time()

    with modal_app.run():
        catvton_result = run_catvton_batch.remote(
            persons_b64=realistic_b64s,
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
    print(f"  - 생성된 피팅 결과: {catvton_result['num_angles']}장")

    # 개별 피팅 결과 저장
    for angle, result_b64 in zip(RENDER_ANGLES, results_b64):
        result_pil = base64_to_pil(result_b64)
        result_pil.save(OUTPUT_DIR / f"fitted_{angle}.png")

    print(f"  ✓ 피팅 결과 저장 완료")

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

    has_3d = "error" not in hy3d_result

    if has_3d:
        print(f"  ✓ Hunyuan3D 완료 ({t_hy3d:.2f}s)")
        print(f"  - Vertices: {hy3d_result['num_vertices']}")
        print(f"  - Faces: {hy3d_result['num_faces']}")
        print(f"  - Textured: {hy3d_result['textured']}")

        # GLB 저장
        glb_bytes = base64_to_bytes(hy3d_result["glb_bytes_b64"])
        glb_path = OUTPUT_DIR / "model_v9.glb"
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)
        print(f"  ✓ GLB 저장: {glb_path} ({len(glb_bytes) / 1e6:.1f}MB)")
    else:
        print(f"  ✗ Hunyuan3D 실패: {hy3d_result['error']}")

    # Phase 5: 결과 시각화
    print("\n[Phase 5] 결과 시각화 중...")

    # PIL 이미지로 변환
    mesh_pils = [base64_to_pil(b64) for b64 in render_b64s]
    realistic_pils = [base64_to_pil(b64) for b64 in realistic_b64s]
    fitted_pils = [base64_to_pil(b64) for b64 in results_b64]

    # 3행 비교 그리드 생성 (Part 1: 0-135°)
    grid1 = create_comparison_grid(
        mesh_renders=mesh_pils[:4],
        realistic_renders=realistic_pils[:4],
        fitted_results=fitted_pils[:4],
        angles=RENDER_ANGLES[:4],
        title="V9 Pipeline Comparison (0-135°)",
    )
    grid1_path = OUTPUT_DIR / "comparison_v9_part1.png"
    grid1.save(grid1_path)
    print(f"  ✓ Part 1 그리드 저장: {grid1_path}")

    # 3행 비교 그리드 생성 (Part 2: 180-315°)
    grid2 = create_comparison_grid(
        mesh_renders=mesh_pils[4:],
        realistic_renders=realistic_pils[4:],
        fitted_results=fitted_pils[4:],
        angles=RENDER_ANGLES[4:],
        title="V9 Pipeline Comparison (180-315°)",
    )
    grid2_path = OUTPUT_DIR / "comparison_v9_part2.png"
    grid2.save(grid2_path)
    print(f"  ✓ Part 2 그리드 저장: {grid2_path}")

    # 비용 계산
    print("\n" + "=" * 80)
    print("비용 계산 (H200: $5.40/hr)")
    print("=" * 80)

    total_time = t_realistic + t_sam3 + t_catvton
    if has_3d:
        total_time += t_hy3d

    total_cost = total_time * H200_COST_PER_SEC

    print(f"  Mesh → Realistic (8각도): {t_realistic:6.2f}s  →  ${t_realistic * H200_COST_PER_SEC:.4f}")
    print(f"  SAM3 (가먼트):            {t_sam3:6.2f}s  →  ${t_sam3 * H200_COST_PER_SEC:.4f}")
    print(f"  CatVTON (8각도):          {t_catvton:6.2f}s  →  ${t_catvton * H200_COST_PER_SEC:.4f}")

    if has_3d:
        print(f"  Hunyuan3D (3D GLB):       {t_hy3d:6.2f}s  →  ${t_hy3d * H200_COST_PER_SEC:.4f}")

    print(f"  {'─' * 40}")
    print(f"  총 GPU 시간:              {total_time:6.2f}s  →  ${total_cost:.4f}")
    print()

    # 최종 요약
    print("=" * 80)
    print("테스트 완료")
    print("=" * 80)
    print()
    print(f"  리얼 렌더:     {OUTPUT_DIR}/realistic_{{0,45,90,...}}.png")
    print(f"  피팅 결과:     {OUTPUT_DIR}/fitted_{{0,45,90,...}}.png")
    print(f"  비교 그리드 1: {grid1_path}")
    print(f"  비교 그리드 2: {grid2_path}")

    if has_3d:
        print(f"  3D 모델:       {OUTPUT_DIR}/model_v9.glb")

    print()
    print("리얼 가상 모델 기반 360° 피팅 파이프라인 검증 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
