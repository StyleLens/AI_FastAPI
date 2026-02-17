#!/usr/bin/env python3
"""
v24 Quality Test — Photorealistic tuning (fix anime/CG look)

User feedback: "너무 애니메이션 느낌이야. AI 실사 느낌으로 만들어야 해."

Changes from v22/v23:
  1. Reuse v22 mesh_data.npz (skip Phase 1 = save $0.07)
  2. Heavily anti-anime/CG negative prompt
  3. Photo-specific positive prompt (RAW photo, DSLR, film grain)
  4. guidance=6.5 (lower = less stylized)
  5. Face Refiner: strength=0.45, more realistic prompt
  6. VTON unchanged (already realistic)

Estimated cost: ~$0.25 (Phase 1.5 + Face Refine + VTON, no Phase 1)
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
    run_face_refiner,
)

# -- Directories ---------------------------------------------------------------
IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
V22_DIR = ROOT / "tests" / "v22_newuser"
OUTPUT_DIR = ROOT / "tests" / "v24_photorealistic"
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
    print("v24 Quality Test — Photorealistic (Anti-Anime/CG)")
    print("  User feedback: '너무 애니메이션 느낌 → AI 실사로'")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Load v22 mesh data (skip Phase 1) -----------------------------------
    print("\n[Phase 0] Loading v22 mesh data (reuse)...")
    mesh_data = np.load(V22_DIR / "mesh_data.npz")
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")

    # Load wear image
    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    # Load face reference
    face_dir = PROJECT_ROOT / "IMG_Data" / "User_New" / "User_Face"
    face_imgs = sorted(list(face_dir.glob("*.jpg")) + list(face_dir.glob("*.png")))
    face_b64 = load_b64(face_imgs[0]) if face_imgs else ""

    # ---- Phase 1R: CPU Mesh Rendering (free, reuse v22 meshes) ---------------
    print("\n[Phase 1R] CPU Mesh Rendering (reuse from v22)...")
    render_b64s = []
    for angle in ANGLES_16:
        lbl = angle_label(angle)
        mesh_path = V22_DIR / f"mesh_{lbl}.png"
        if mesh_path.exists():
            render_b64s.append(load_b64(mesh_path))
        else:
            rendered = render_mesh(vertices, faces, angle_deg=angle, resolution=MESH_RESOLUTION)
            ok, buf = cv2.imencode(".jpg", rendered)
            render_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    print(f"  Loaded {len(render_b64s)} mesh renders")

    # ---- Phase 1.5: SDXL — PHOTOREALISTIC prompt -----------------------------
    # Key changes for photorealism:
    # 1. "RAW photo" / "DSLR" / "film grain" keywords
    # 2. Lower guidance (6.5 vs 7.5) for less stylized output
    # 3. Heavy anti-anime/CG negative prompt
    v24_prompt = (
        "RAW photo, a real photograph of a young Korean woman, "
        "slim petite build, long straight dark brown hair, {angle_desc}, "
        "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
        "standing naturally on flat ground, "
        "shot with DSLR camera, 85mm lens, shallow depth of field, "
        "natural indoor lighting, subtle film grain, "
        "muted color grading, realistic skin pores and texture, "
        "no makeup or minimal makeup, candid natural expression"
    )
    v24_negative = (
        "anime, cartoon, illustration, painting, drawing, art, sketch, "
        "3d render, CGI, CG, computer graphics, digital art, "
        "smooth skin, plastic skin, airbrushed, porcelain, doll-like, "
        "oversaturated, vibrant colors, neon, "
        "floating, levitating, hovering, feet off ground, "
        "cropped, cut off, missing limbs, extra limbs, "
        "oversized body, muscular, tall, "
        "nsfw, nude, revealing, "
        "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    )

    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL Photorealistic (guidance=6.5, anti-anime prompt)")
    print("=" * 80)

    t15_start = time.time()
    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64="",
            angles=ANGLES_16,
            num_steps=30,
            guidance=6.5,  # Lower for less stylized
            controlnet_conditioning_scale=0.5,
            prompt_template=v24_prompt,
            negative_prompt_override=v24_negative,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}"); return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        t15 = time.time() - t15_start
        timings["phase15_sec"] = round(t15, 2)
        print(f"  Generated {len(realistic_b64s)} realistic in {t15:.1f}s")

        for a, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(a)}.png"))

        # ---- Face Refiner on realistic (lower strength) ----------------------
        print("\n[Phase 2A] Face Refiner (strength=0.45, realistic prompt)")
        t_fr_start = time.time()
        fr_result = run_face_refiner.remote(
            images_b64=realistic_b64s,
            face_reference_b64=face_b64,
            face_expand_ratio=1.5,  # Slightly smaller mask
            num_steps=25,
            guidance=6.0,  # Lower for realism
            strength=0.45,  # Much lower — subtle face enhancement only
            seed=42,
        )

        if "error" in fr_result:
            print(f"  Face Refiner ERROR: {fr_result['error']}")
            refined_realistic_b64s = realistic_b64s  # fallback to unrefined
        else:
            refined_realistic_b64s = fr_result["refined_b64"]
            face_detected = fr_result["face_detected"]
            print(f"  Faces: {sum(face_detected)}/{len(face_detected)} detected")

        t_fr = time.time() - t_fr_start
        timings["face_refine_sec"] = round(t_fr, 2)

        for a, b64 in zip(ANGLES_16, refined_realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_refined_{angle_label(a)}.png"))

        # ---- Phase 3: FASHN VTON on refined realistic images -----------------
        print("\n[Phase 3] FASHN VTON on face-refined images")
        t3_start = time.time()
        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=refined_realistic_b64s,
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

    # ---- Comparison grids: v22 vs v24 ----------------------------------------
    print("\n[Phase 4] Creating comparison grids...")

    # Load v22 for comparison
    v22_r = [base64_to_pil(load_b64(V22_DIR / f"realistic_{angle_label(a)}.png")) for a in ANGLES_16]
    v22_f = [base64_to_pil(load_b64(V22_DIR / f"fitted_{angle_label(a)}.png")) for a in ANGLES_16]
    v24_r = [base64_to_pil(b) for b in refined_realistic_b64s]
    v24_f = [base64_to_pil(b) for b in fitted_b64s]

    # Hero: key angles
    hero_angles = [0, 45, 90, 180, 270]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]

    hero_grid = create_comparison_grid(
        [[v22_r[i] for i in hero_idx],
         [v24_r[i] for i in hero_idx],
         [v22_f[i] for i in hero_idx],
         [v24_f[i] for i in hero_idx]],
        ["v22 Realistic", "v24 Photorealistic", "v22 Fitted", "v24 Fitted"],
        hero_angles,
        "v22 (anime) vs v24 (photorealistic)",
    )
    hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    # Full 16-angle
    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [v22_r[sl], v24_r[sl], v22_f[sl], v24_f[sl]],
            ["v22 Realistic", "v24 Photo", "v22 Fitted", "v24 Fitted"],
            angles_sl,
            f"v22 vs v24 Photorealistic ({suffix}°)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # ---- Cost / Report -------------------------------------------------------
    gpu_total = t15 + t_fr + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v24_photorealistic",
        "timestamp": timestamp,
        "user_feedback": "너무 애니메이션 느낌 → AI 실사로",
        "changes_from_v22": [
            "RAW photo / DSLR / film grain prompt keywords",
            "Heavy anti-anime/CG/cartoon negative prompt",
            "guidance=6.5 (was 7.5) for less stylized output",
            "Face Refiner strength=0.45 (was 0.65)",
            "Face Refiner expand_ratio=1.5 (was 1.8)",
            "Reused v22 mesh data (skipped Phase 1)",
        ],
        "parameters": {
            "phase15": {"num_steps": 30, "guidance": 6.5, "cn_scale": 0.5},
            "face_refiner": {"strength": 0.45, "guidance": 6.0, "expand_ratio": 1.5},
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
    for a in [0, 90, 180]:
        for prefix in ["realistic_refined", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v24 Photorealistic Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
