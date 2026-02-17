#!/usr/bin/env python3
"""
Test Quality Try-On V17D: Integrated PuLID Improvements for Maximum Face Identity

V17D: Integrated PuLID Improvements for Maximum Face Identity

CHANGES from v17b → v17d:
- v17b: id_weight=1.3, start_step=0, guidance=4.0, cn_scale=0.5
  → Face too different from reference (PuLID too weak vs ControlNet)
- v17d: INTEGRATED IMPROVEMENTS:
  A. id_weight=2.5 (nearly 2x v17b, forces PuLID face dominance)
  B. Weighted embedding average (first ref image gets 50% weight)
  C. Step-wise PuLID decay (full weight 0-70%, linear decay to 0.3x after 70%)
     → Protects body structure in late denoising steps
  D. Enhanced lighting prompts (consistent studio lighting, uniform skin tone)
  E. cn_scale=0.5 maintained (body shape protection)

Strategy: PuLID dominates face in early steps, decays in late steps
to let ControlNet finalize body structure. This prevents both
face identity loss AND body shape distortion.

Pipeline:
1. Phase 0: Load test images + mesh renders + face reference photos
2. Phase 1.5: Mesh → Realistic + Face ID (ControlNet Depth + PuLID)
3. Phase 2A: Garment segmentation (SAM3, reuse if available)
4. Phase 2B: FASHN parse → extended agnostic masks (per angle)
5. Phase 3: CatVTON-FLUX 8-angle batch with FASHN partial masks
6. Comparison grids (5 rows: mesh, realistic+PuLID, mask, fitted, face ref)

Expected improvements over v17b:
- MUCH stronger face identity preservation (id_weight=2.5)
- Weighted embedding prioritizes primary reference photo
- Step-wise decay prevents late-stage body distortion
- Enhanced lighting prompts improve consistency

Output:
- output/quality_tryon_v17d/mesh_{0..315}.png
- output/quality_tryon_v17d/realistic_{0..315}.png (WITH integrated face identity)
- output/quality_tryon_v17d/mask_{0..315}.png
- output/quality_tryon_v17d/fitted_{0..315}.png
- output/quality_tryon_v17d/comparison_v17d_part1.png
- output/quality_tryon_v17d/comparison_v17d_part2.png
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -- Path setup ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # ai-service/
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent                             # ai-server/

from worker.modal_app import (                         # noqa: E402
    app as modal_app,
    run_light_models,
    run_mesh_to_realistic_with_face,
    run_catvton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
FACE_DIR = IMG_DATA / "Face"
E2E_OUTPUT = ROOT / "output" / "e2e_test"
OUTPUT_DIR = ROOT / "output" / "quality_tryon_v17d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RENDER_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# GPU cost: H200 $5.40/hr
H200_COST_PER_SEC = 5.40 / 3600  # ~$0.0015/sec

# INTEGRATED PuLID improvements (v17d)
PULID_ID_WEIGHT = 2.5              # Much higher: PuLID dominates face (was 1.3 in v17b)
PULID_START_STEP = 0               # Same as v17b (official default)
PULID_GUIDANCE = 4.0               # Same as v17b
CONTROLNET_CONDITIONING_SCALE = 0.5  # Maintain body structure (was 0.5 in v17b)


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


def make_full_mask(width: int, height: int) -> str:
    """Fallback: full white mask (all inpaint)."""
    mask = np.ones((height, width), dtype=np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_extended_agnostic_mask(parsemap_b64: str, width: int, height: int) -> str:
    """Create extended agnostic mask from FASHN parse map.

    FASHN 18-class labels:
    0=bg, 1=hat, 2=hair, 3=sunglasses, 4=upper_clothes,
    5=skirt, 6=pants, 7=dress, 8=belt, 9=left_shoe,
    10=right_shoe, 11=face, 12=left_leg, 13=right_leg,
    14=left_arm, 15=right_arm, 16=bag, 17=scarf

    White(255) = upper_clothes(4) + left_arm(14) + right_arm(15)
               + dress(7) + belt(8)
    Black(0) = everything else → preserve
    """
    raw = base64.b64decode(parsemap_b64)
    parse_map = np.array(Image.open(io.BytesIO(raw)).convert("L"))

    mask = np.zeros_like(parse_map, dtype=np.uint8)
    mask[parse_map == 4] = 255   # upper_clothes
    mask[parse_map == 7] = 255   # dress
    mask[parse_map == 8] = 255   # belt
    mask[parse_map == 14] = 255  # left_arm
    mask[parse_map == 15] = 255  # right_arm

    mask_pil = Image.fromarray(mask, mode="L")
    mask_pil = mask_pil.resize((width, height), Image.NEAREST)

    # Dilate 2x with MaxFilter(15) for overlap
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(15))
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(15))

    # Smooth edges + threshold back
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(5))
    mask_arr = np.array(mask_pil)
    mask_arr = np.where(mask_arr > 128, 255, 0).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(mask_arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_comparison_grid(
    rows_data: list[list[Image.Image]],
    row_labels: list[str],
    angles: list[int],
    title: str = "V17D Pipeline Comparison",
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
    row_label_w = 160
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
    print("Quality Try-On V17D: INTEGRATED PuLID Improvements + CatVTON")
    print("=" * 80)
    print("\nINTEGRATED IMPROVEMENTS:")
    print(f"  id_weight: {PULID_ID_WEIGHT} (was 1.3 in v17b, nearly 2x)")
    print(f"  start_step: {PULID_START_STEP} (same as v17b)")
    print(f"  guidance: {PULID_GUIDANCE} (same as v17b)")
    print(f"  controlnet_scale: {CONTROLNET_CONDITIONING_SCALE} (same as v17b)")
    print("\nBACKEND IMPROVEMENTS:")
    print("  - Weighted embedding average (first ref gets 50% weight)")
    print("  - Step-wise PuLID decay (full weight 0-70%, linear decay after)")
    print("  - Enhanced lighting prompts (studio lighting, uniform skin tone)")
    print("\nExpected: Maximum face identity + protected body structure")

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

    wear_imgs = sorted(
        list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png"))
    )
    if not wear_imgs:
        print("  ERROR: No wear images found in IMG_Data/wear/")
        return
    wear_img_path = wear_imgs[0]
    print(f"  Wear image : {wear_img_path.name}")
    wear_b64 = load_image_base64(wear_img_path)

    # Load face reference images (Multi-Reference Identity Bank)
    print(f"\n[Phase 0] Loading face reference images from {FACE_DIR}...")
    face_paths = sorted(
        list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png"))
    )
    if not face_paths:
        print(f"  ERROR: No face images found in {FACE_DIR}")
        return

    face_b64s = []
    for fp in face_paths:
        face_b64s.append(load_image_base64(fp))
        print(f"  Face ref: {fp.name}")
        # Save face references to output
        save_b64_image(face_b64s[-1], str(OUTPUT_DIR / f"face_reference_{len(face_b64s)}.png"))

    print(f"  Loaded {len(face_b64s)} face reference images (Multi-Reference Identity Bank)")

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
    print("[Phase 1.5 + 2A + 2B + 3] Running GPU phases in single Modal session")
    print(f"  PuLID INTEGRATED: id_weight={PULID_ID_WEIGHT}, start_step={PULID_START_STEP}")
    print(f"  guidance={PULID_GUIDANCE}, controlnet_scale={CONTROLNET_CONDITIONING_SCALE}")
    print(f"  Face refs: {len(face_b64s)} images")
    print("  Backend: Weighted embeddings + step-wise decay + enhanced prompts")

    target_w, target_h = 768, 1024

    with modal_app.run():
        # ---- Phase 1.5: ControlNet Depth + PuLID → Realistic + Face ID ------
        print("\n  [Phase 1.5] Mesh → Realistic + INTEGRATED PuLID Face Identity")
        print(f"    guidance={PULID_GUIDANCE}, steps=28, id_weight={PULID_ID_WEIGHT}")
        print(f"    start_step={PULID_START_STEP}, controlnet_scale={CONTROLNET_CONDITIONING_SCALE}")
        print(f"    face refs: {len(face_b64s)} images → weighted average (first=50%)")
        print("    step-wise decay: full weight 0-70%, linear decay to 0.3x after")

        t_phase15_start = time.time()

        realistic_result = run_mesh_to_realistic_with_face.remote(
            mesh_renders_b64=render_b64s,
            face_images_b64=face_b64s,
            angles=RENDER_ANGLES,
            num_steps=28,
            guidance=PULID_GUIDANCE,
            id_weight=PULID_ID_WEIGHT,
            start_step=PULID_START_STEP,
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        )

        t_phase15 = time.time() - t_phase15_start

        if "error" in realistic_result:
            print(f"    ERROR: {realistic_result['error']}")
            return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        num_face_refs = realistic_result.get("num_face_refs", 0)
        print(f"    Generated {len(realistic_b64s)} realistic renders in {t_phase15:.1f}s")
        print(f"    Face refs used: {num_face_refs}")
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

        # ---- Phase 2B: FASHN parse → extended agnostic masks ------------------
        print("\n  [Phase 2B] FASHN parse realistic renders → extended agnostic masks")
        print("    Classes: upper_clothes(4) + dress(7) + belt(8) + arms(14,15)")
        masks_b64 = []
        t_phase2b_start = time.time()

        for idx, (realistic_b64, angle) in enumerate(zip(realistic_b64s, RENDER_ANGLES)):
            print(f"    Parsing realistic_{angle}...", end=" ")
            parse_result = run_light_models.remote(
                task="parse_fashn",
                image_b64=realistic_b64,
            )
            if "error" in parse_result:
                print(f"ERROR: {parse_result['error']} → full mask fallback")
                masks_b64.append(make_full_mask(target_w, target_h))
            else:
                parsemap_b64 = parse_result["parsemap_b64"]
                agnostic_mask = create_extended_agnostic_mask(
                    parsemap_b64, target_w, target_h
                )
                masks_b64.append(agnostic_mask)
                save_b64_image(agnostic_mask, str(OUTPUT_DIR / f"mask_{angle}.png"))
                save_b64_image(parsemap_b64, str(OUTPUT_DIR / f"parsemap_{angle}.png"))
                print("OK")

        t_phase2b = time.time() - t_phase2b_start
        print(f"    FASHN parsing done: {len(masks_b64)} masks in {t_phase2b:.1f}s")
        print(f"    Cost: ${t_phase2b * H200_COST_PER_SEC:.4f}")

        # ---- Phase 3: CatVTON-FLUX with extended masks ------------------------
        print("\n  [Phase 3] CatVTON-FLUX 8-angle batch (extended FASHN masks)")
        print(f"    guidance=30.0, steps=30")

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
    mask_pils = [base64_to_pil(b64).convert("RGB") for b64 in masks_b64]
    fitted_pils = [base64_to_pil(b64) for b64 in fitted_b64s]

    # Face reference row: show the primary face reference repeated
    face_ref_pil = base64_to_pil(face_b64s[0])
    face_ref_row = [face_ref_pil.copy() for _ in range(4)]  # repeated for grid

    # Part 1: angles 0-135
    grid1 = create_comparison_grid(
        rows_data=[
            mesh_pils[:4],
            realistic_pils[:4],
            mask_pils[:4],
            fitted_pils[:4],
            face_ref_row,
        ],
        row_labels=["Raw Mesh", "CN+PuLID", "Ext Mask", "CatVTON", "Face Ref"],
        angles=RENDER_ANGLES[:4],
        title=f"V17D Integrated (id={PULID_ID_WEIGHT}, cn={CONTROLNET_CONDITIONING_SCALE}, decay@70%) — 0-135 deg",
    )
    grid1_path = OUTPUT_DIR / "comparison_v17d_part1.png"
    grid1.save(grid1_path)
    print(f"  Grid Part 1 saved: {grid1_path}")

    # Part 2: angles 180-315
    grid2 = create_comparison_grid(
        rows_data=[
            mesh_pils[4:],
            realistic_pils[4:],
            mask_pils[4:],
            fitted_pils[4:],
            face_ref_row,
        ],
        row_labels=["Raw Mesh", "CN+PuLID", "Ext Mask", "CatVTON", "Face Ref"],
        angles=RENDER_ANGLES[4:],
        title=f"V17D Integrated (id={PULID_ID_WEIGHT}, cn={CONTROLNET_CONDITIONING_SCALE}, decay@70%) — 180-315 deg",
    )
    grid2_path = OUTPUT_DIR / "comparison_v17d_part2.png"
    grid2.save(grid2_path)
    print(f"  Grid Part 2 saved: {grid2_path}")

    # ---- Cost summary -------------------------------------------------------
    total_gpu_time = t_phase15 + t_phase2a + t_phase2b + t_phase3
    total_cost = total_gpu_time * H200_COST_PER_SEC

    print("\n" + "=" * 80)
    print("Cost Summary (H200: $5.40/hr)")
    print("=" * 80)
    print(f"  Phase 1.5  (CN Depth + PuLID, 8 angles): {t_phase15:7.1f}s  ${t_phase15 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 2A   (SAM3 garment seg):            {t_phase2a:7.1f}s  ${t_phase2a * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 2B   (FASHN parse, 8 angles):       {t_phase2b:7.1f}s  ${t_phase2b * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 3    (CatVTON, 8 angles):           {t_phase3:7.1f}s  ${t_phase3 * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 57}")
    print(f"  Total GPU time:                           {total_gpu_time:7.1f}s  ${total_cost:.4f}")

    # ---- Final summary ------------------------------------------------------
    output_files = list(OUTPUT_DIR.iterdir())

    print("\n" + "=" * 80)
    print("V17D INTEGRATED PULID TEST COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total files: {len(output_files)}")
    print("\n  INTEGRATED PARAMETERS:")
    print(f"    id_weight={PULID_ID_WEIGHT} (was 1.3 in v17b, nearly 2x)")
    print(f"    start_step={PULID_START_STEP} (same as v17b)")
    print(f"    guidance={PULID_GUIDANCE} (same as v17b)")
    print(f"    controlnet_scale={CONTROLNET_CONDITIONING_SCALE} (same as v17b)")
    print(f"  Face refs: {len(face_b64s)} images, {num_face_refs} embeddings used")
    print("  Backend: Weighted embeddings + step-wise decay + enhanced prompts")
    print()
    print(f"  Mesh renders:      mesh_{{0,45,...,315}}.png")
    print(f"  Realistic+PuLID:   realistic_{{0,45,...,315}}.png")
    print(f"  Extended masks:    mask_{{0,45,...,315}}.png")
    print(f"  Fitted results:    fitted_{{0,45,...,315}}.png")
    print(f"  Face references:   face_reference_{{1,...,{len(face_b64s)}}}.png")
    print(f"  Comparison Part 1: {grid1_path}")
    print(f"  Comparison Part 2: {grid2_path}")

    print("\n  Opening results for visual inspection...")
    os.system(f"open '{grid1_path}'")
    os.system(f"open '{grid2_path}'")

    for angle in [0, 90, 180]:
        p = OUTPUT_DIR / f"fitted_{angle}.png"
        if p.exists():
            os.system(f"open '{p}'")

    # Also open v17b for comparison
    v17b_dir = ROOT / "output" / "quality_tryon_v17b"
    v17b_realistic = v17b_dir / "realistic_0.png"
    if v17b_realistic.exists():
        os.system(f"open '{v17b_realistic}'")

    print()
    print("V17D Integrated PuLID pipeline validation complete.")
    print("Compare realistic_0.png face with face_reference_1.png for identity match.")
    print("Expected: MUCH stronger face identity vs v17b due to integrated improvements.")
    print("Compare with v17b realistic_0.png for face fidelity difference.")
    print("=" * 80)

    return {
        "total_gpu_time": total_gpu_time,
        "total_cost": total_cost,
        "phase_times": {
            "phase15_controlnet_pulid": t_phase15,
            "phase2a_sam3_segment": t_phase2a,
            "phase2b_fashn_parse": t_phase2b,
            "phase3_catvton_batch": t_phase3,
        },
        "pulid_params": {
            "id_weight": PULID_ID_WEIGHT,
            "start_step": PULID_START_STEP,
            "guidance": PULID_GUIDANCE,
            "controlnet_conditioning_scale": CONTROLNET_CONDITIONING_SCALE,
            "num_face_refs": len(face_b64s),
            "num_embeddings_used": num_face_refs,
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
