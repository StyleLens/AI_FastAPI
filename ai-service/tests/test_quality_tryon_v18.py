#!/usr/bin/env python3
"""
Test Quality Try-On V18: Commercial Stack (FLUX.1-schnell + XLabs ControlNet + FASHN VTON)

CHANGES from v13 (commercial stack transition):
- Phase 1.5: FLUX.1-dev -> FLUX.1-schnell (Apache 2.0, CFG-distilled, 8 steps)
  + XLabs ControlNet Depth diffusers (Apache 2.0)
- Phase 2B: REMOVED (FASHN VTON is maskless -- no FASHN Parser needed)
- Phase 3: CatVTON-FLUX -> FASHN VTON v1.5 (Apache 2.0, maskless, 972M params)
- All models are now 100% Apache 2.0 / MIT licensed

Pipeline:
1. Phase 0: Load test images + mesh renders from E2E output
2. Phase 1.5: Mesh -> Realistic (schnell + XLabs ControlNet Depth)
3. Phase 2A: Garment segmentation (SAM3, reuse if available)
4. Phase 3: FASHN VTON v1.5 8-angle batch (maskless)
5. Comparison grids (3 rows: mesh, realistic, fitted)

Output:
- output/quality_tryon_v18/mesh_{0..315}.png
- output/quality_tryon_v18/realistic_{0..315}.png
- output/quality_tryon_v18/fitted_{0..315}.png
- output/quality_tryon_v18/comparison_v18_part1.png
- output/quality_tryon_v18/comparison_v18_part2.png
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -- Path setup ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # ai-service/
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent                             # ai-server/

from worker.modal_app import (                         # noqa: E402
    app as modal_app,
    run_light_models,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
FACE_DIR = IMG_DATA / "Face"
E2E_OUTPUT = ROOT / "output" / "e2e_test"
OUTPUT_DIR = ROOT / "output" / "quality_tryon_v18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RENDER_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# GPU cost: H200 $5.40/hr
H200_COST_PER_SEC = 5.40 / 3600  # ~$0.0015/sec


# -- Utility functions --------------------------------------------------------

def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def base64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def save_b64_image(b64_str: str, path: str):
    raw = base64.b64decode(b64_str)
    with open(path, "wb") as f:
        f.write(raw)


def create_comparison_grid(
    rows_data: list[list[Image.Image]],
    row_labels: list[str],
    angles: list[int],
    title: str = "V18 Pipeline Comparison",
) -> Image.Image:
    """Create N-row x M-col comparison grid."""
    target_h = 300

    def resize_row(images):
        resized = []
        for img in images:
            aspect = img.width / img.height
            new_w = int(target_h * aspect)
            resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))
        return resized

    resized_rows = [resize_row(row) for row in rows_data]

    cols = len(angles)
    rows = len(rows_data)
    label_h = 30
    row_label_w = 140
    title_h = 50

    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w = max_w + 20
    cell_h = target_h + label_h

    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - text_w) // 2, 15), title, fill=(0, 0, 0), font=font_title)

    for row_idx, (row_images, row_label) in enumerate(zip(resized_rows, row_labels)):
        label_y = title_h + row_idx * cell_h + cell_h // 2
        draw.text((10, label_y), row_label, fill=(0, 0, 0), font=font_label)

        for col_idx, (img, angle) in enumerate(zip(row_images, angles)):
            x_off = row_label_w + col_idx * cell_w + (cell_w - img.width) // 2
            y_off = title_h + row_idx * cell_h + label_h
            canvas.paste(img, (x_off, y_off))

            if row_idx == 0:
                angle_text = f"{angle} deg"
                bbox = draw.textbbox((0, 0), angle_text, font=font_small)
                tw = bbox[2] - bbox[0]
                tx = row_label_w + col_idx * cell_w + (cell_w - tw) // 2
                ty = title_h + row_idx * cell_h + 5
                draw.text((tx, ty), angle_text, fill=(0, 0, 0), font=font_small)

    return canvas


# -- Main test ----------------------------------------------------------------

def main():
    print("=" * 80)
    print("Quality Try-On V18: Commercial Stack (schnell + XLabs CN + FASHN VTON)")
    print("=" * 80)

    # ---- Phase 0: Load test images ------------------------------------------
    print("\n[Phase 0] Loading test images...")

    user_imgs = sorted(
        list(USER_IMG_DIR.glob("*.jpg")) + list(USER_IMG_DIR.glob("*.png"))
    )
    if not user_imgs:
        print("  ERROR: No user images found in IMG_Data/User_IMG/")
        return
    user_img_path = user_imgs[0]
    print(f"  User image : {user_img_path.name}")
    user_b64 = load_image_base64(user_img_path)

    wear_imgs = sorted(
        list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png"))
    )
    if not wear_imgs:
        print("  ERROR: No wear images found in IMG_Data/wear/")
        return
    wear_img_path = wear_imgs[0]
    print(f"  Wear image : {wear_img_path.name}")
    wear_b64 = load_image_base64(wear_img_path)

    face_path = FACE_DIR / "Face.png"
    if not face_path.exists():
        print(f"  ERROR: Face image not found at {face_path}")
        return
    print(f"  Face image : {face_path.name}")
    face_b64 = load_image_base64(face_path)
    save_b64_image(face_b64, str(OUTPUT_DIR / "face_reference.png"))

    # ---- Load mesh renders --------------------------------------------------
    print("\n[Phase 0] Loading mesh renders from E2E output...")

    render_b64s = []
    for angle in RENDER_ANGLES:
        p = E2E_OUTPUT / f"phase1_render_{angle}.jpg"
        if not p.exists():
            print(f"  ERROR: Mesh render not found: {p}")
            return
        render_b64s.append(load_image_base64(p))

    print(f"  Loaded {len(render_b64s)} mesh renders (8 angles)")
    first_mesh = base64_to_pil(render_b64s[0])
    mesh_w, mesh_h = first_mesh.size
    print(f"  Mesh render resolution: {mesh_w} x {mesh_h}")

    for angle, b64 in zip(RENDER_ANGLES, render_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"mesh_{angle}.png"))

    # ---- Garment segmentation (try reuse) -----------------------------------
    existing_seg = E2E_OUTPUT / "phase2_garment_seg.png"
    reuse_seg = existing_seg.exists()
    if reuse_seg:
        print(f"\n  Reusing existing garment segmentation: {existing_seg}")
        garment_seg_b64 = load_image_base64(existing_seg)
    else:
        garment_seg_b64 = None

    # ---- GPU Phases ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Phase 1.5 + 2A + 3] Running GPU phases in single Modal session")

    with modal_app.run():
        # ---- Phase 1.5: ControlNet Depth -> Realistic (schnell) ---------------
        print("\n  [Phase 1.5] Mesh -> Realistic (schnell + XLabs ControlNet Depth)")
        print(f"    guidance = 0.0, num_steps = 8 (schnell CFG-distilled)")
        print(f"    prompt: wearing plain GRAY t-shirt + jeans")

        t_phase15_start = time.time()

        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64=user_b64,
            face_image_b64=face_b64,
            angles=RENDER_ANGLES,
            num_steps=8,
            guidance=0.0,
            controlnet_conditioning_scale=0.7,
        )

        t_phase15 = time.time() - t_phase15_start

        if "error" in realistic_result:
            print(f"    ERROR: {realistic_result['error']}")
            return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        print(f"    Generated {len(realistic_b64s)} realistic renders in {t_phase15:.1f}s")
        print(f"    Cost: ${t_phase15 * H200_COST_PER_SEC:.4f}")

        first_realistic = base64_to_pil(realistic_b64s[0])
        r_w, r_h = first_realistic.size
        print(f"    Realistic render resolution: {r_w} x {r_h}")

        for angle, b64 in zip(RENDER_ANGLES, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle}.png"))

        # ---- Phase 2A: Garment segmentation -----------------------------------
        if not reuse_seg:
            print("\n  [Phase 2A] Garment segmentation (SAM3)")
            t_phase2a_start = time.time()
            garment_seg_result = run_light_models.remote(
                task="segment_sam3",
                image_b64=wear_b64,
            )
            t_phase2a = time.time() - t_phase2a_start
            if "error" in garment_seg_result:
                print(f"    ERROR: {garment_seg_result['error']}")
                return
            garment_seg_b64 = garment_seg_result["segmented_b64"]
            print(f"    SAM3 segmentation done in {t_phase2a:.1f}s")
        else:
            t_phase2a = 0.0
            print(f"\n  [Phase 2A] Garment segmentation: REUSED from E2E output")

        save_b64_image(garment_seg_b64, str(OUTPUT_DIR / "garment_segmented.png"))

        # ---- Phase 3: FASHN VTON v1.5 (maskless) -----------------------------
        print("\n  [Phase 3] FASHN VTON v1.5 8-angle batch (maskless)")
        print(f"    guidance_scale = 1.5, num_timesteps = 30")
        print(f"    category = tops, segmentation_free = True")

        t_phase3_start = time.time()

        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=realistic_b64s,
            clothing_b64=garment_seg_b64,
            category="tops",
            garment_photo_type="flat-lay",
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
        )

        t_phase3 = time.time() - t_phase3_start

        if "error" in fashn_result:
            print(f"    ERROR: {fashn_result['error']}")
            return

        fitted_b64s = fashn_result["results_b64"]
        print(f"    FASHN VTON generated {len(fitted_b64s)} fitted images in {t_phase3:.1f}s")
        print(f"    Cost: ${t_phase3 * H200_COST_PER_SEC:.4f}")

    # Save fitted results
    first_fitted = base64_to_pil(fitted_b64s[0])
    f_w, f_h = first_fitted.size
    print(f"  Fitted render resolution: {f_w} x {f_h}")

    for angle, b64 in zip(RENDER_ANGLES, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle}.png"))
    print("  Fitted results saved")

    # ---- Phase 4: Comparison grids ------------------------------------------
    print("\n" + "=" * 80)
    print("[Phase 4] Creating comparison grids...")

    mesh_pils = [base64_to_pil(b64) for b64 in render_b64s]
    realistic_pils = [base64_to_pil(b64) for b64 in realistic_b64s]
    fitted_pils = [base64_to_pil(b64) for b64 in fitted_b64s]

    # Part 1: angles 0-135
    grid1 = create_comparison_grid(
        rows_data=[mesh_pils[:4], realistic_pils[:4], fitted_pils[:4]],
        row_labels=["Raw Mesh", "CN Depth (schnell)", "FASHN VTON"],
        angles=RENDER_ANGLES[:4],
        title="V18 Commercial Stack (schnell + FASHN VTON) (0-135 deg)",
    )
    grid1_path = OUTPUT_DIR / "comparison_v18_part1.png"
    grid1.save(grid1_path)
    print(f"  Grid Part 1 saved: {grid1_path}")

    # Part 2: angles 180-315
    grid2 = create_comparison_grid(
        rows_data=[mesh_pils[4:], realistic_pils[4:], fitted_pils[4:]],
        row_labels=["Raw Mesh", "CN Depth (schnell)", "FASHN VTON"],
        angles=RENDER_ANGLES[4:],
        title="V18 Commercial Stack (schnell + FASHN VTON) (180-315 deg)",
    )
    grid2_path = OUTPUT_DIR / "comparison_v18_part2.png"
    grid2.save(grid2_path)
    print(f"  Grid Part 2 saved: {grid2_path}")

    # ---- Cost summary -------------------------------------------------------
    total_gpu_time = t_phase15 + t_phase2a + t_phase3
    total_cost = total_gpu_time * H200_COST_PER_SEC

    print("\n" + "=" * 80)
    print("Cost Summary (H200: $5.40/hr)")
    print("=" * 80)
    print(f"  Phase 1.5  (schnell + XLabs CN, 8 angles):  {t_phase15:7.1f}s  ${t_phase15 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 2A   (SAM3 garment seg):               {t_phase2a:7.1f}s  ${t_phase2a * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 3    (FASHN VTON, 8 angles):           {t_phase3:7.1f}s  ${t_phase3 * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 55}")
    print(f"  Total GPU time:                              {total_gpu_time:7.1f}s  ${total_cost:.4f}")

    # ---- Final summary ------------------------------------------------------
    output_files = list(OUTPUT_DIR.iterdir())

    print("\n" + "=" * 80)
    print("V18 QUALITY TEST COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total files: {len(output_files)}")
    print()
    print(f"  Mesh renders:      mesh_{{0,45,...,315}}.png")
    print(f"  Realistic renders: realistic_{{0,45,...,315}}.png")
    print(f"  Fitted results:    fitted_{{0,45,...,315}}.png")
    print(f"  Comparison Part 1: {grid1_path}")
    print(f"  Comparison Part 2: {grid2_path}")

    print("\n  Opening results for visual inspection...")
    os.system(f"open '{grid1_path}'")
    os.system(f"open '{grid2_path}'")

    for angle in [0, 90, 180]:
        p = OUTPUT_DIR / f"fitted_{angle}.png"
        if p.exists():
            os.system(f"open '{p}'")

    print()
    print("V18 Commercial Stack (schnell + FASHN VTON) pipeline validation complete.")
    print("=" * 80)

    return {
        "total_gpu_time": total_gpu_time,
        "total_cost": total_cost,
        "phase_times": {
            "phase15_controlnet_depth_schnell": t_phase15,
            "phase2a_sam3_segment": t_phase2a,
            "phase3_fashn_vton_batch": t_phase3,
        },
        "num_outputs": len(output_files),
        "resolutions": {
            "mesh": f"{mesh_w}x{mesh_h}",
            "realistic": f"{r_w}x{r_h}",
            "fitted": f"{f_w}x{f_h}",
        },
    }


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\nResult JSON:\n{json.dumps(result, indent=2)}")
