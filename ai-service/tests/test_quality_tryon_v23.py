#!/usr/bin/env python3
"""
v23 Quality Test — Face Refiner on v22 results

Reuses v22 mesh_data.npz to skip Phase 1 ($0 savings).
Applies Face Refiner (SDXL Inpainting) to:
  1. Realistic images (Phase 1.5 output)
  2. Fitted images (Phase 3 VTON output)

Pipeline: Phase 1R (CPU, free) → 1.5 (SDXL) → Face Refine → 3 (VTON) → Face Refine
Estimated cost: ~$0.20 (Phase 1.5 + VTON reuse from v22, Face Refiner ~$0.10)
"""

import base64
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent

from worker.modal_app import (
    app as modal_app,
    run_face_refiner,
)

OUTPUT_DIR = ROOT / "tests" / "v23_facerefine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V22_DIR = ROOT / "tests" / "v22_newuser"

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600

# Face is typically visible only in front-facing angles
FACE_ANGLES = [0, 22.5, 45, 315, 337.5]  # ±45° from front


def angle_label(angle: float) -> str:
    return str(int(angle)) if angle == int(angle) else str(angle)


def load_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def save_b64_image(b64_str: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def base64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def create_comparison_grid(rows_data, row_labels, angles, title=""):
    target_h = 256
    def resize_row(images):
        return [img.resize((int(target_h * img.width / img.height), target_h),
                           Image.Resampling.LANCZOS) for img in images]

    resized_rows = [resize_row(row) for row in rows_data]
    cols, rows = len(angles), len(rows_data)
    label_h, row_label_w, title_h = 28, 200, 45
    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w, cell_h = max_w + 10, target_h + label_h
    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        ft, fl, fs = (ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)
                       for s in (22, 13, 12))
    except Exception:
        ft = fl = fs = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=ft)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) // 2, 12), title,
              fill=(0, 0, 0), font=ft)

    for ri, (imgs, label) in enumerate(zip(resized_rows, row_labels)):
        draw.text((8, title_h + ri * cell_h + cell_h // 2), label,
                  fill=(0, 0, 0), font=fl)
        for ci, (img, ang) in enumerate(zip(imgs, angles)):
            x = row_label_w + ci * cell_w + (cell_w - img.width) // 2
            y = title_h + ri * cell_h + label_h
            canvas.paste(img, (x, y))
            if ri == 0:
                at = f"{angle_label(ang)}°"
                abbox = draw.textbbox((0, 0), at, font=fs)
                draw.text((row_label_w + ci * cell_w +
                           (cell_w - abbox[2] + abbox[0]) // 2,
                           title_h + 4), at, fill=(0, 0, 0), font=fs)
    return canvas


def main():
    print("=" * 80)
    print("v23 Quality Test — Face Refiner (SDXL Inpainting)")
    print(f"  Reusing v22 results from: {V22_DIR}")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Load v22 realistic + fitted images ----------------------------------
    print("\n[Phase 0] Loading v22 images...")

    realistic_b64s = []
    fitted_b64s = []
    for a in ANGLES_16:
        lbl = angle_label(a)
        r_path = V22_DIR / f"realistic_{lbl}.png"
        f_path = V22_DIR / f"fitted_{lbl}.png"
        if not r_path.exists() or not f_path.exists():
            print(f"  ERROR: v22 output missing for {lbl}°"); return
        realistic_b64s.append(load_b64(r_path))
        fitted_b64s.append(load_b64(f_path))

    print(f"  Loaded {len(realistic_b64s)} realistic + {len(fitted_b64s)} fitted images")

    # Load face reference
    face_dir = PROJECT_ROOT / "IMG_Data" / "User_New" / "User_Face"
    face_imgs = sorted(list(face_dir.glob("*.jpg")) + list(face_dir.glob("*.png")))
    face_b64 = ""
    if face_imgs:
        face_b64 = load_b64(face_imgs[0])
        print(f"  Face reference: {face_imgs[0].name}")

    # ---- Face Refine: Realistic images (GPU) ---------------------------------
    print("\n" + "=" * 80)
    print("[Phase 2A] Face Refiner on Realistic images")
    print(f"  Refining all 16 angles (face detection auto-skips non-face angles)")
    print("=" * 80)

    t_fr_start = time.time()
    with modal_app.run():
        fr_realistic = run_face_refiner.remote(
            images_b64=realistic_b64s,
            face_reference_b64=face_b64,
            face_expand_ratio=1.8,
            num_steps=25,
            guidance=7.0,
            strength=0.65,
            seed=42,
        )

        if "error" in fr_realistic:
            print(f"  ERROR: {fr_realistic['error']}"); return

        t_fr_realistic = time.time() - t_fr_start
        timings["face_refine_realistic_sec"] = round(t_fr_realistic, 2)

        face_r_detected = fr_realistic["face_detected"]
        print(f"  Face detected in: {sum(face_r_detected)}/{len(face_r_detected)} angles")
        for i, (a, det) in enumerate(zip(ANGLES_16, face_r_detected)):
            if det:
                print(f"    {angle_label(a)}°: face refined")
        print(f"  Time: {t_fr_realistic:.1f}s  Cost: ${t_fr_realistic * H200_COST_PER_SEC:.4f}")

        # Save refined realistic
        refined_realistic_b64s = fr_realistic["refined_b64"]
        for a, b64 in zip(ANGLES_16, refined_realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_refined_{angle_label(a)}.png"))

        # ---- Face Refine: Fitted images (GPU) --------------------------------
        print("\n" + "=" * 80)
        print("[Phase 2B] Face Refiner on Fitted (VTON) images")
        print("=" * 80)

        t_ff_start = time.time()
        fr_fitted = run_face_refiner.remote(
            images_b64=fitted_b64s,
            face_reference_b64=face_b64,
            face_expand_ratio=1.8,
            num_steps=25,
            guidance=7.0,
            strength=0.65,
            seed=42,
        )

        if "error" in fr_fitted:
            print(f"  ERROR: {fr_fitted['error']}"); return

        t_fr_fitted = time.time() - t_ff_start
        timings["face_refine_fitted_sec"] = round(t_fr_fitted, 2)

        face_f_detected = fr_fitted["face_detected"]
        print(f"  Face detected in: {sum(face_f_detected)}/{len(face_f_detected)} angles")
        print(f"  Time: {t_fr_fitted:.1f}s  Cost: ${t_fr_fitted * H200_COST_PER_SEC:.4f}")

    # Save refined fitted
    refined_fitted_b64s = fr_fitted["refined_b64"]
    for a, b64 in zip(ANGLES_16, refined_fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_refined_{angle_label(a)}.png"))

    # ---- Comparison grids: v22 vs v23 (face-visible angles only) -------------
    print("\n[Phase 3] Creating comparison grids...")

    # Hero comparison: face-visible angles only
    hero_angles = [0, 22.5, 45, 315, 337.5]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]

    v22_r = [base64_to_pil(realistic_b64s[i]) for i in hero_idx]
    v23_r = [base64_to_pil(refined_realistic_b64s[i]) for i in hero_idx]
    v22_f = [base64_to_pil(fitted_b64s[i]) for i in hero_idx]
    v23_f = [base64_to_pil(refined_fitted_b64s[i]) for i in hero_idx]

    hero_grid = create_comparison_grid(
        [v22_r, v23_r, v22_f, v23_f],
        ["v22 Realistic", "v23 Face Refined", "v22 Fitted", "v23 Fitted Refined"],
        hero_angles,
        "v22 vs v23 Face Refiner — Front Angles",
    )
    hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    # Full 16-angle: realistic before/after
    all_v22_r = [base64_to_pil(b) for b in realistic_b64s]
    all_v23_r = [base64_to_pil(b) for b in refined_realistic_b64s]
    all_v22_f = [base64_to_pil(b) for b in fitted_b64s]
    all_v23_f = [base64_to_pil(b) for b in refined_fitted_b64s]

    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [all_v22_r[sl], all_v23_r[sl], all_v22_f[sl], all_v23_f[sl]],
            ["v22 Realistic", "v23 Refined", "v22 Fitted", "v23 Fit Refined"],
            angles_sl,
            f"v22 vs v23 Face Refiner ({suffix}°)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # ---- Cost / Report -------------------------------------------------------
    gpu_total = t_fr_realistic + t_fr_fitted
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v23_face_refiner",
        "timestamp": timestamp,
        "changes_from_v22": [
            "SDXL Inpainting face refiner added",
            "MediaPipe Face Detection → elliptical soft mask → SDXL Inpainting",
            "Applied to both realistic and fitted images",
            "strength=0.65, guidance=7.0, steps=25",
        ],
        "parameters": {
            "face_refiner": {
                "face_expand_ratio": 1.8,
                "num_steps": 25,
                "guidance": 7.0,
                "strength": 0.65,
            },
        },
        "face_detection": {
            "realistic_detected": sum(face_r_detected),
            "realistic_total": len(face_r_detected),
            "fitted_detected": sum(face_f_detected),
            "fitted_total": len(face_f_detected),
        },
        "timings": timings,
        "cost": {
            "gpu_total_sec": round(gpu_total, 2),
            "gpu_total_usd": round(gpu_total * H200_COST_PER_SEC, 4),
        },
    }
    (OUTPUT_DIR / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_hero.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part1.png'}'")

    # Open face-refined key images
    for a in [0, 45, 315]:
        for prefix in ["realistic_refined", "fitted_refined"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v23 Face Refiner Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
