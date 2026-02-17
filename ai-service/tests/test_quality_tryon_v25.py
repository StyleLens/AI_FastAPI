#!/usr/bin/env python3
"""
v25 Quality Test — Posture Correction + Ground Plane

User feedback: "거북목, 대각선으로 서있어. 일자로 서있어야 해."

Key changes from v24:
  1. sw_renderer v2: posture straightening (10° forward lean fixed)
  2. sw_renderer v2: ground plane shadow (no more floating)
  3. Re-render mesh with improved renderer (NOT reuse v22 renders)
  4. guidance=7.0 (balanced: 6.5 was too loose on prompt, 7.5 was CG)
  5. Skip Face Refiner (focus on base quality first)
  6. VTON with both model-worn and flat-lay garment types

Estimated cost: ~$0.28 (Phase 1.5 + VTON, no Phase 1 or Face Refiner)
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

from core.sw_renderer import render_mesh
from worker.modal_app import (
    app as modal_app,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

# -- Directories ---------------------------------------------------------------
IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
V22_DIR = ROOT / "tests" / "v22_newuser"
OUTPUT_DIR = ROOT / "tests" / "v25_posture"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600
MESH_RESOLUTION = 768


def fix_exif_rotation(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        if orientation == 3:
            return img.rotate(180, expand=True)
        elif orientation == 6:
            return img.rotate(270, expand=True)
        elif orientation == 8:
            return img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def load_image_fixed(path: Path) -> tuple[Image.Image, str]:
    pil = Image.open(path)
    pil = fix_exif_rotation(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return pil, b64


def angle_label(angle: float) -> str:
    return str(int(angle)) if angle == int(angle) else str(angle)


def save_b64_image(b64_str: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def base64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def load_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


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
    print("v25 Quality Test — Posture Correction + Ground Plane")
    print("  Changes: straighten posture, ground shadow, guidance=7.0")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Load v22 mesh data (reuse 3D reconstruction) -------------------------
    print("\n[Phase 0] Loading v22 mesh data...")
    mesh_data = np.load(V22_DIR / "mesh_data.npz")
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")

    # Load wear image
    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    # ---- Phase 1R: NEW renderer with posture correction -----------------------
    print("\n[Phase 1R] CPU Mesh Rendering (v2: posture + ground plane)...")
    t1r_start = time.time()
    render_b64s = []
    for angle in ANGLES_16:
        lbl = angle_label(angle)
        rendered = render_mesh(
            vertices, faces,
            angle_deg=angle,
            resolution=MESH_RESOLUTION,
            straighten=True,
            ground_plane=True,
        )
        ok, buf = cv2.imencode(".png", rendered)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        render_b64s.append(b64)

        # Save depth maps
        cv2.imwrite(str(OUTPUT_DIR / f"mesh_{lbl}.png"), rendered)

    t1r = time.time() - t1r_start
    timings["phase1r_sec"] = round(t1r, 2)
    print(f"  Rendered {len(render_b64s)} angles in {t1r:.1f}s (CPU)")

    # ---- Phase 1.5: SDXL — balanced photorealistic ----------------------------
    v25_prompt = (
        "RAW photo, a real photograph of a young Korean woman, "
        "slim petite build, long straight dark brown hair, {angle_desc}, "
        "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
        "standing upright on a flat floor with feet planted firmly on the ground, "
        "natural upright posture, straight spine, shoulders back, "
        "shot with DSLR camera, 85mm lens, "
        "clean neutral background, soft natural lighting, "
        "realistic skin texture, natural expression"
    )
    v25_negative = (
        "anime, cartoon, illustration, painting, drawing, sketch, "
        "3d render, CGI, CG, computer graphics, digital art, "
        "smooth plastic skin, airbrushed, doll-like, porcelain, "
        "floating, levitating, hovering, feet off ground, mid-air, "
        "leaning forward, hunched, slouching, bent over, turtle neck, "
        "tilted, diagonal posture, crooked stance, "
        "cropped, cut off, missing limbs, extra limbs, "
        "oversized body, muscular, tall, "
        "nsfw, nude, revealing, "
        "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    )

    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL Photorealistic (guidance=7.0, posture-corrected depth)")
    print("=" * 80)

    t15_start = time.time()
    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64="",
            angles=ANGLES_16,
            num_steps=30,
            guidance=7.0,
            controlnet_conditioning_scale=0.5,
            prompt_template=v25_prompt,
            negative_prompt_override=v25_negative,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}"); return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        t15 = time.time() - t15_start
        timings["phase15_sec"] = round(t15, 2)
        print(f"  Generated {len(realistic_b64s)} realistic in {t15:.1f}s")

        for a, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(a)}.png"))

        # ---- Phase 3: FASHN VTON on realistic images --------------------------
        print("\n[Phase 3] FASHN VTON (model garment type)")
        t3_start = time.time()
        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=realistic_b64s,
            clothing_b64=wear_b64,
            category="tops",
            garment_photo_type="model",
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
        )
        t3 = time.time() - t3_start
        timings["phase3_sec"] = round(t3, 2)

        if "error" in fashn_result:
            print(f"  ERROR: {fashn_result['error']}"); return

        fitted_b64s = fashn_result["results_b64"]
        print(f"  VTON: {len(fitted_b64s)} images in {t3:.1f}s")

    for a, b64 in zip(ANGLES_16, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(a)}.png"))

    # ---- Comparison grids: v24 vs v25 ----------------------------------------
    print("\n[Phase 4] Creating comparison grids...")

    # Load v24 for comparison (if available)
    v24_dir = ROOT / "tests" / "v24_photorealistic"
    has_v24 = (v24_dir / "realistic_refined_0.png").exists()

    v25_r = [base64_to_pil(b) for b in realistic_b64s]
    v25_f = [base64_to_pil(b) for b in fitted_b64s]

    # Also save depth map comparison
    depth_imgs = [Image.open(OUTPUT_DIR / f"mesh_{angle_label(a)}.png") for a in ANGLES_16]

    if has_v24:
        v24_r = [Image.open(v24_dir / f"realistic_refined_{angle_label(a)}.png") for a in ANGLES_16]
        v24_f = [Image.open(v24_dir / f"fitted_{angle_label(a)}.png") for a in ANGLES_16]

        hero_angles = [0, 45, 90, 180, 270, 315]
        hero_idx = [ANGLES_16.index(a) for a in hero_angles]

        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [v24_r[i] for i in hero_idx],
             [v25_r[i] for i in hero_idx],
             [v24_f[i] for i in hero_idx],
             [v25_f[i] for i in hero_idx]],
            ["v25 Depth", "v24 Realistic", "v25 Realistic",
             "v24 Fitted", "v25 Fitted"],
            hero_angles,
            "v24 (floating/lean) vs v25 (grounded/straight)",
        )
        hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

        for part, sl, angles_sl, suffix in [
            ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
            ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
        ]:
            grid = create_comparison_grid(
                [depth_imgs[sl], v24_r[sl], v25_r[sl], v25_f[sl]],
                ["v25 Depth", "v24 Realistic", "v25 Realistic", "v25 Fitted"],
                angles_sl,
                f"v24 vs v25 ({suffix}°)",
            )
            grid.save(OUTPUT_DIR / f"comparison_{part}.png")
    else:
        # Just save v25 standalone grids
        hero_angles = [0, 45, 90, 180, 270, 315]
        hero_idx = [ANGLES_16.index(a) for a in hero_angles]

        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [v25_r[i] for i in hero_idx],
             [v25_f[i] for i in hero_idx]],
            ["Depth Map", "Realistic", "Fitted"],
            hero_angles,
            "v25 — Posture Corrected + Grounded",
        )
        hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    # ---- Cost / Report -------------------------------------------------------
    gpu_total = t15 + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v25_posture_correction",
        "timestamp": timestamp,
        "user_feedback": "거북목, 대각선 자세 → 일자로 서있어야 함",
        "changes_from_v24": [
            "sw_renderer v2: posture straightening (~10° forward lean fixed)",
            "sw_renderer v2: ground plane shadow (no more floating)",
            "Re-rendered mesh with corrected renderer (not reusing v22/v24 renders)",
            "guidance=7.0 (balanced: 6.5=loose, 7.5=CG)",
            "Anti-leaning negative prompts added",
            "No Face Refiner (focus on base quality first)",
        ],
        "parameters": {
            "renderer": {"straighten": True, "ground_plane": True, "resolution": MESH_RESOLUTION},
            "phase15": {"num_steps": 30, "guidance": 7.0, "cn_scale": 0.5},
            "phase3": {
                "category": "tops",
                "garment_photo_type": "model",
                "num_timesteps": 30,
                "guidance_scale": 1.5,
            },
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
    for a in [0, 45, 90, 180, 315]:
        for prefix in ["realistic", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v25 Posture Correction Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
