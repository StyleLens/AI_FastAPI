#!/usr/bin/env python3
"""
v33b — Face swap re-run on ALL 16 angles using v33 VTON results.
Changes from v33:
  - blend_radius: 25 → 30 (softer edges)
  - LAB color transfer: L channel 60% strength (prevents bright patches)
  - mask_bool threshold: 0.1 → 0.3 (core face region only)
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
from PIL import Image, ImageDraw, ImageFont, ExifTags

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent

from worker.modal_app import (
    app as modal_app,
    run_face_swap,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
FACE_DIR = IMG_DATA / "User_New" / "User_Face"
V22_DIR = ROOT / "tests" / "v22_newuser"
V32_DIR = ROOT / "tests" / "v32_no_facerefine"
V33_DIR = ROOT / "tests" / "v33_face_consistency"
OUTPUT_DIR = ROOT / "tests" / "v33b_face_improved"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600


def fix_exif_rotation(img):
    try:
        exif = img._getexif()
        if exif is None:
            return img
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        else:
            return img
        if orientation_key not in exif:
            return img
        o = exif[orientation_key]
        if o == 3: return img.rotate(180, expand=True)
        elif o == 6: return img.rotate(270, expand=True)
        elif o == 8: return img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def load_b64(path):
    pil = Image.open(path)
    pil = fix_exif_rotation(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return pil, base64.b64encode(buf.getvalue()).decode("ascii")


def angle_label(a):
    return str(int(a)) if a == int(a) else str(a)


def save_b64_image(b64_str, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def base64_to_pil(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def create_comparison_grid(rows_data, row_labels, angles, title=""):
    target_h = 256
    def resize_row(images):
        return [img.resize((int(target_h * img.width / img.height), target_h),
                           Image.Resampling.LANCZOS) for img in images]
    resized_rows = [resize_row(row) for row in rows_data]
    cols, rows = len(angles), len(rows_data)
    label_h, row_label_w, title_h = 28, 260, 45
    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w, cell_h = max_w + 10, target_h + label_h
    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        ft, fl, fs = (ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s) for s in (22, 12, 12))
    except Exception:
        ft = fl = fs = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=ft)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) // 2, 12), title, fill=(0, 0, 0), font=ft)
    for ri, (imgs, label) in enumerate(zip(resized_rows, row_labels)):
        draw.text((8, title_h + ri * cell_h + cell_h // 2), label, fill=(0, 0, 0), font=fl)
        for ci, (img, ang) in enumerate(zip(imgs, angles)):
            x = row_label_w + ci * cell_w + (cell_w - img.width) // 2
            y = title_h + ri * cell_h + label_h
            canvas.paste(img, (x, y))
            if ri == 0:
                at = f"{angle_label(ang)}\u00b0"
                abbox = draw.textbbox((0, 0), at, font=fs)
                draw.text((row_label_w + ci * cell_w + (cell_w - abbox[2] + abbox[0]) // 2,
                           title_h + 4), at, fill=(0, 0, 0), font=fs)
    return canvas


def main():
    print("=" * 80)
    print("v33b — Face Swap Improvement (reusing v33 VTON results)")
    print("  Changes: blend_radius=30, LAB L-channel 60%, mask threshold 0.3")
    print("=" * 80)

    # Load face reference
    face_imgs = sorted(list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png")))
    if not face_imgs:
        face_path = V22_DIR / "input_face.jpg"
        face_pil, face_b64 = load_b64(face_path)
    else:
        face_pil, face_b64 = load_b64(face_imgs[0])
    print(f"  Face ref: {face_pil.size}")
    face_pil.save(OUTPUT_DIR / "face_reference.png")

    # Load v33 VTON results (before face swap)
    print("\n[Loading] v33 VTON results (before face swap)...")
    fitted_b64s = []
    for a in ANGLES_16:
        p = V33_DIR / f"fitted_before_face_{angle_label(a)}.png"
        if not p.exists():
            print(f"  ERROR: Missing {p}")
            return None
        _, b64 = load_b64(p)
        fitted_b64s.append(b64)
    print(f"  Loaded {len(fitted_b64s)} images")

    # Run face swap with improved parameters
    print("\n[Face Swap v33b] Running with improved blending...")
    with modal_app.run():
        t_start = time.time()
        result = run_face_swap.remote(
            images_b64=fitted_b64s,
            face_reference_b64=face_b64,
            angles=ANGLES_16,
            blend_radius=30,  # Increased from 25
            face_scale=1.0,
        )
        t_swap = time.time() - t_start

    if "error" in result:
        print(f"  ERROR: {result['error'][:500]}")
        return None

    final_b64s = result["swapped_b64"]
    face_detected = result.get("face_detected", [])
    n_swapped = sum(1 for d in face_detected if d)
    print(f"  Face Swap: {n_swapped}/{len(ANGLES_16)} swapped in {t_swap:.1f}s")

    # Save results
    for a, b64 in zip(ANGLES_16, final_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(a)}.png"))

    # Also copy v33 intermediate files for comparison
    for a in ANGLES_16:
        for prefix in ["mesh", "sdxl", "realistic"]:
            src = V33_DIR / f"{prefix}_{angle_label(a)}.png"
            if src.exists():
                import shutil
                shutil.copy2(src, OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png")

    # Create comparison grids
    print("\n[Comparison] Creating grids...")
    hero_angles = [0, 45, 90, 180, 270, 315]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]

    final_imgs = [base64_to_pil(b) for b in final_b64s]
    fitted_before = [Image.open(V33_DIR / f"fitted_before_face_{angle_label(a)}.png") for a in ANGLES_16]
    face_ref_row = [face_pil.copy() for _ in hero_angles]

    # v33 vs v33b comparison
    v33_fitted = [Image.open(V33_DIR / f"fitted_{angle_label(a)}.png") for a in ANGLES_16]
    v32_fitted = []
    for a in hero_angles:
        p = V32_DIR / f"fitted_{angle_label(a)}.png"
        if p.exists():
            v32_fitted.append(Image.open(p))

    grid_rows = [face_ref_row]
    grid_labels = ["Face Reference"]

    if len(v32_fitted) == len(hero_angles):
        grid_rows.append(v32_fitted)
        grid_labels.append("v32 (no face swap)")

    grid_rows.extend([
        [fitted_before[i] for i in hero_idx],
        [v33_fitted[i] for i in hero_idx],
        [final_imgs[i] for i in hero_idx],
    ])
    grid_labels.extend([
        "VTON (before face)",
        "v33 (blend=25, L=100%)",
        "v33b (blend=30, L=60%)",
    ])

    comp_grid = create_comparison_grid(
        grid_rows, grid_labels, hero_angles,
        "v33 vs v33b — Face Swap Blending Improvement",
    )
    comp_grid.save(OUTPUT_DIR / "comparison_v33_vs_v33b.png")

    # Full 16-angle
    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [fitted_before[sl], [v33_fitted[i] for i in range(sl.start, sl.stop)],
             final_imgs[sl]],
            ["VTON", "v33 Face", "v33b Face"],
            angles_sl,
            f"v33b ({suffix}\u00b0)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # Cost (face swap only)
    cost_usd = t_swap * H200_COST_PER_SEC
    cost_krw = cost_usd * 1350
    print(f"\n  Face swap cost: {t_swap:.1f}s, ${cost_usd:.4f} ({cost_krw:.0f}\uc6d0)")

    # Save report
    report = {
        "test_name": "v33b_face_improved",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": "Face swap re-run with improved blending on v33 VTON results",
        "changes_from_v33": [
            "blend_radius: 25 \u2192 30 (softer mask edges)",
            "LAB L-channel: 100% \u2192 60% transfer strength (prevents bright patches)",
            "mask_bool threshold: 0.1 \u2192 0.3 (core face region only for color stats)",
        ],
        "face_swap_params": {
            "blend_radius": 30,
            "face_scale": 1.0,
            "L_channel_strength": 0.6,
            "mask_threshold": 0.3,
        },
        "face_swap_sec": round(t_swap, 2),
        "face_detected": face_detected,
        "angles_swapped": [a for a, d in zip(ANGLES_16, face_detected) if d],
    }
    (OUTPUT_DIR / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    # Open results
    print("\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_v33_vs_v33b.png'}'")
    for a in [0, 45, 90, 270, 315]:
        p = OUTPUT_DIR / f"fitted_{angle_label(a)}.png"
        if p.exists():
            os.system(f"open '{p}'")
    os.system(f"open '{OUTPUT_DIR / 'face_reference.png'}'")

    print(f"\n{'='*60}")
    print(f"v33b Complete — {n_swapped}/{len(ANGLES_16)} faces swapped")
    print(f"{'='*60}")
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
