#!/usr/bin/env python3
"""
Test Quality Try-On V10: ControlNet Depth + CatVTON Pipeline

Changes from v9:
- Phase 1.5: ControlNet Depth replaces FLUX Fill inpainting
  - guidance=3.5 (depth-conditioned generation, not inpainting)
  - Output is single images (768x1024), not concatenated (1536x1024)
- Face.png added as face reference for identity preservation (future PuLID)
- Better pose preservation through depth conditioning from mesh renders
- Hunyuan3D skipped to save GPU cost (focus on realistic render + fitting quality)

Pipeline:
1. Phase 0: Light models (SAM 3D Body reconstruction + mesh renders)
2. Phase 1.5: Mesh -> Realistic (ControlNet Depth, face_image_b64 param)
3. Phase 2: Garment segmentation (SAM3)
4. Phase 3: CatVTON-FLUX 8-angle batch (guidance=30.0, steps=30)

Output:
- output/quality_tryon_v10/mesh_{0..315}.png          (Phase 0 mesh renders)
- output/quality_tryon_v10/realistic_{0..315}.png      (Phase 1.5 results)
- output/quality_tryon_v10/fitted_{0..315}.png         (Phase 3 results)
- output/quality_tryon_v10/face_reference.png          (Face image copy)
- output/quality_tryon_v10/comparison_v10_part1.png    (3-row grid: 0-135 deg)
- output/quality_tryon_v10/comparison_v10_part2.png    (3-row grid: 180-315 deg)
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
    run_catvton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
FACE_DIR = IMG_DATA / "Face"
E2E_OUTPUT = ROOT / "output" / "e2e_test"
OUTPUT_DIR = ROOT / "output" / "quality_tryon_v10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RENDER_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# GPU cost: H200 $5.40/hr
H200_COST_PER_SEC = 5.40 / 3600  # ~$0.0015/sec


# -- Utility functions --------------------------------------------------------

def load_image_base64(path: Path) -> str:
    """Load an image file and return its base64 encoding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def base64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> str:
    """Encode a PIL Image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def save_b64_image(b64_str: str, path: str):
    """Decode a base64 image and write it to disk."""
    raw = base64.b64decode(b64_str)
    with open(path, "wb") as f:
        f.write(raw)


def make_full_mask(width: int, height: int) -> str:
    """
    Create a full white mask (all inpaint region).
    White (255) = inpaint, Black (0) = preserve.
    Returns base64-encoded PNG.
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
    title: str = "V10 Pipeline Comparison",
) -> Image.Image:
    """
    Create a 3-row x N-col comparison grid.

    Row 1: Raw mesh renders
    Row 2: ControlNet Depth realistic renders
    Row 3: CatVTON fitted results

    Args:
        mesh_renders: Mesh render PIL images (one per angle).
        realistic_renders: Realistic render PIL images (one per angle).
        fitted_results: Fitted result PIL images (one per angle).
        angles: Angle labels (degrees).
        title: Title text drawn at top of grid.

    Returns:
        Grid PIL Image.
    """
    target_h = 350

    def resize_row(images):
        resized = []
        for img in images:
            aspect = img.width / img.height
            new_w = int(target_h * aspect)
            resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))
        return resized

    resized_mesh = resize_row(mesh_renders)
    resized_realistic = resize_row(realistic_renders)
    resized_fitted = resize_row(fitted_results)

    cols = len(angles)
    rows = 3
    label_h = 30
    row_label_w = 140
    title_h = 50

    max_w = max(
        max(img.width for img in resized_mesh),
        max(img.width for img in resized_realistic),
        max(img.width for img in resized_fitted),
    )
    cell_w = max_w + 20
    cell_h = target_h + label_h

    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - text_w) // 2, 15), title, fill=(0, 0, 0), font=font_title)

    row_labels = ["Raw Mesh", "CN Depth", "CatVTON"]
    all_rows = [resized_mesh, resized_realistic, resized_fitted]

    for row_idx, (row_images, row_label) in enumerate(zip(all_rows, row_labels)):
        label_y = title_h + row_idx * cell_h + cell_h // 2
        draw.text((10, label_y), row_label, fill=(0, 0, 0), font=font_label)

        for col_idx, (img, angle) in enumerate(zip(row_images, angles)):
            x_off = row_label_w + col_idx * cell_w + (cell_w - img.width) // 2
            y_off = title_h + row_idx * cell_h + label_h
            canvas.paste(img, (x_off, y_off))

            # Angle label on first row only
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
    print("Quality Try-On V10: ControlNet Depth + CatVTON Pipeline")
    print("=" * 80)

    # ---- Phase 0: Load test images ------------------------------------------
    print("\n[Phase 0] Loading test images...")

    # User image (first from User_IMG)
    user_imgs = sorted(
        list(USER_IMG_DIR.glob("*.jpg")) + list(USER_IMG_DIR.glob("*.png"))
    )
    if not user_imgs:
        print("  ERROR: No user images found in IMG_Data/User_IMG/")
        return
    user_img_path = user_imgs[0]
    print(f"  User image : {user_img_path.name}")
    user_b64 = load_image_base64(user_img_path)

    # Wear image (first from wear)
    wear_imgs = sorted(
        list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png"))
    )
    if not wear_imgs:
        print("  ERROR: No wear images found in IMG_Data/wear/")
        return
    wear_img_path = wear_imgs[0]
    print(f"  Wear image : {wear_img_path.name}")
    wear_b64 = load_image_base64(wear_img_path)

    # Face image
    face_path = FACE_DIR / "Face.png"
    if not face_path.exists():
        print(f"  ERROR: Face image not found at {face_path}")
        return
    print(f"  Face image : {face_path.name}")
    face_b64 = load_image_base64(face_path)

    # Save face reference copy
    save_b64_image(face_b64, str(OUTPUT_DIR / "face_reference.png"))

    # ---- Load mesh renders from E2E output ----------------------------------
    print("\n[Phase 0] Loading mesh renders from E2E output...")

    render_b64s = []
    for angle in RENDER_ANGLES:
        p = E2E_OUTPUT / f"phase1_render_{angle}.jpg"
        if not p.exists():
            print(f"  ERROR: Mesh render not found: {p}")
            print("  Run the E2E test first to generate mesh renders.")
            return
        render_b64s.append(load_image_base64(p))

    print(f"  Loaded {len(render_b64s)} mesh renders (8 angles)")

    # Check resolution of first mesh render
    first_mesh = base64_to_pil(render_b64s[0])
    mesh_w, mesh_h = first_mesh.size
    print(f"  Mesh render resolution: {mesh_w} x {mesh_h}")

    # Save mesh renders to v10 output for comparison
    for angle, b64 in zip(RENDER_ANGLES, render_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"mesh_{angle}.png"))

    # ---- Try to reuse existing garment segmentation --------------------------
    existing_seg = E2E_OUTPUT / "phase2_garment_seg.png"
    reuse_seg = existing_seg.exists()
    if reuse_seg:
        print(f"\n  Reusing existing garment segmentation: {existing_seg}")
        garment_seg_b64 = load_image_base64(existing_seg)
    else:
        print("  No existing garment segmentation found, will run SAM3")
        garment_seg_b64 = None

    # ---- GPU Phases (all in one app.run() to avoid multiple cold starts) ----
    print("\n" + "=" * 80)
    print("[Phase 1.5 + 2 + 3] Running GPU phases in single Modal session")

    with modal_app.run():
        # ---- Phase 1.5: Mesh -> Realistic (ControlNet Depth) ----------------
        print("\n  [Phase 1.5] Mesh -> Realistic (ControlNet Depth)")
        print(f"    guidance = 3.5, num_steps = 28")
        print(f"    face_image_b64 provided: yes ({len(face_b64)} chars)")
        print("    Output: single images (768x1024), NOT concatenated")

        t_phase15_start = time.time()

        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64=user_b64,
            face_image_b64=face_b64,
            angles=RENDER_ANGLES,
            num_steps=28,
            guidance=3.5,
        )

        t_phase15 = time.time() - t_phase15_start

        if "error" in realistic_result:
            print(f"    ERROR: {realistic_result['error']}")
            return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        print(f"    Generated {len(realistic_b64s)} realistic renders in {t_phase15:.1f}s")
        print(f"    Cost: ${t_phase15 * H200_COST_PER_SEC:.4f}")

        # Check output resolution
        first_realistic = base64_to_pil(realistic_b64s[0])
        r_w, r_h = first_realistic.size
        print(f"    Realistic render resolution: {r_w} x {r_h}")

        # Save realistic renders
        for angle, b64 in zip(RENDER_ANGLES, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle}.png"))
        print("    Realistic renders saved")

        # ---- Phase 2: Garment segmentation (SAM3 if needed) -----------------
        if not reuse_seg:
            print("\n  [Phase 2] Garment segmentation (SAM3)")
            t_phase2_start = time.time()

            garment_seg_result = run_light_models.remote(
                task="segment_sam3",
                image_b64=wear_b64,
            )
            t_phase2 = time.time() - t_phase2_start

            if "error" in garment_seg_result:
                print(f"    ERROR: {garment_seg_result['error']}")
                return

            garment_seg_b64 = garment_seg_result["segmented_b64"]
            print(f"    SAM3 segmentation done in {t_phase2:.1f}s")
            print(f"    Cost: ${t_phase2 * H200_COST_PER_SEC:.4f}")
        else:
            t_phase2 = 0.0
            print(f"\n  [Phase 2] Garment segmentation: REUSED from E2E output")

        # Save segmented garment
        save_b64_image(garment_seg_b64, str(OUTPUT_DIR / "garment_segmented.png"))

        # ---- Phase 3: CatVTON-FLUX 8-angle batch ----------------------------
        print("\n  [Phase 3] CatVTON-FLUX 8-angle batch")
        print(f"    guidance = 30.0, num_steps = 30")
        print(f"    Persons: {len(realistic_b64s)} realistic renders")

        # Create full masks for all 8 angles
        target_w, target_h = 768, 1024
        masks_b64 = [make_full_mask(target_w, target_h) for _ in RENDER_ANGLES]
        print(f"    Masks: {len(masks_b64)} full masks ({target_w}x{target_h})")

        t_phase3_start = time.time()

        catvton_result = run_catvton_batch.remote(
            persons_b64=realistic_b64s,
            clothing_b64=garment_seg_b64,
            masks_b64=masks_b64,
            num_steps=30,
            guidance=30.0,
        )

        t_phase3 = time.time() - t_phase3_start

        if "error" in catvton_result:
            print(f"    ERROR: {catvton_result['error']}")
            return

        fitted_b64s = catvton_result["results_b64"]
        print(f"    CatVTON generated {len(fitted_b64s)} fitted images in {t_phase3:.1f}s")
        print(f"    Cost: ${t_phase3 * H200_COST_PER_SEC:.4f}")

    # Check fitted output resolution
    first_fitted = base64_to_pil(fitted_b64s[0])
    f_w, f_h = first_fitted.size
    print(f"  Fitted render resolution: {f_w} x {f_h}")

    # Save fitted results
    for angle, b64 in zip(RENDER_ANGLES, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle}.png"))
    print("  Fitted results saved")

    # ---- Phase 4: Comparison grids ------------------------------------------
    print("\n" + "=" * 80)
    print("[Phase 4] Creating comparison grids...")

    mesh_pils = [base64_to_pil(b64) for b64 in render_b64s]
    realistic_pils = [base64_to_pil(b64) for b64 in realistic_b64s]
    fitted_pils = [base64_to_pil(b64) for b64 in fitted_b64s]

    # Part 1: angles 0-135 deg
    grid1 = create_comparison_grid(
        mesh_renders=mesh_pils[:4],
        realistic_renders=realistic_pils[:4],
        fitted_results=fitted_pils[:4],
        angles=RENDER_ANGLES[:4],
        title="V10 ControlNet Depth + CatVTON (0-135 deg)",
    )
    grid1_path = OUTPUT_DIR / "comparison_v10_part1.png"
    grid1.save(grid1_path)
    print(f"  Grid Part 1 saved: {grid1_path}")

    # Part 2: angles 180-315 deg
    grid2 = create_comparison_grid(
        mesh_renders=mesh_pils[4:],
        realistic_renders=realistic_pils[4:],
        fitted_results=fitted_pils[4:],
        angles=RENDER_ANGLES[4:],
        title="V10 ControlNet Depth + CatVTON (180-315 deg)",
    )
    grid2_path = OUTPUT_DIR / "comparison_v10_part2.png"
    grid2.save(grid2_path)
    print(f"  Grid Part 2 saved: {grid2_path}")

    # ---- Cost summary -------------------------------------------------------
    total_gpu_time = t_phase15 + t_phase2 + t_phase3
    total_cost = total_gpu_time * H200_COST_PER_SEC

    print("\n" + "=" * 80)
    print("Cost Summary (H200: $5.40/hr)")
    print("=" * 80)
    print(f"  Phase 1.5 (ControlNet Depth, 8 angles): {t_phase15:7.1f}s  ${t_phase15 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 2   (SAM3 garment seg):           {t_phase2:7.1f}s  ${t_phase2 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 3   (CatVTON, 8 angles):          {t_phase3:7.1f}s  ${t_phase3 * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 50}")
    print(f"  Total GPU time:                         {total_gpu_time:7.1f}s  ${total_cost:.4f}")

    # ---- Final summary ------------------------------------------------------
    output_files = list(OUTPUT_DIR.iterdir())

    print("\n" + "=" * 80)
    print("V10 QUALITY TEST COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total files: {len(output_files)}")
    print()
    print(f"  Mesh renders:      {OUTPUT_DIR}/mesh_{{0,45,...,315}}.png")
    print(f"  Realistic renders: {OUTPUT_DIR}/realistic_{{0,45,...,315}}.png")
    print(f"  Fitted results:    {OUTPUT_DIR}/fitted_{{0,45,...,315}}.png")
    print(f"  Face reference:    {OUTPUT_DIR}/face_reference.png")
    print(f"  Comparison Part 1: {grid1_path}")
    print(f"  Comparison Part 2: {grid2_path}")

    # Open comparison grids and key renders for visual inspection
    print("\n  Opening results for visual inspection...")
    os.system(f"open '{grid1_path}'")
    os.system(f"open '{grid2_path}'")

    # Open front, side, back realistic renders for quick check
    for angle in [0, 90, 180]:
        p = OUTPUT_DIR / f"realistic_{angle}.png"
        if p.exists():
            os.system(f"open '{p}'")

    print()
    print("ControlNet Depth + CatVTON pipeline validation complete.")
    print("=" * 80)

    return {
        "total_gpu_time": total_gpu_time,
        "total_cost": total_cost,
        "phase_times": {
            "phase15_controlnet_depth": t_phase15,
            "phase2_sam3_segment": t_phase2,
            "phase3_catvton_batch": t_phase3,
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
