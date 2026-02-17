#!/usr/bin/env python3
"""
v22 Quality Test — New User Data + Body Info + Corrected Pipeline

New user: User_New (162cm, 55kg, B cup)
Changes from v21:
  1. User_New/User_IMG as input (EXIF rotation auto-fix)
  2. Body-specific prompt (slim Korean woman, 162cm build)
  3. wear images are model-worn → garment_photo_type="model"
  4. Mesh flip fix from v21 retained
  5. Best front-facing photo auto-selected

Pipeline: Phase 1 (SAM 3D) → 1R (CPU render) → 1.5 (SDXL) → 3 (FASHN VTON)
Estimated cost: ~$0.35 (full pipeline)
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
    run_light_models,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

# -- Directories ---------------------------------------------------------------
IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_NEW_DIR = IMG_DATA / "User_New"
USER_IMG_DIR = USER_NEW_DIR / "User_IMG"
USER_FACE_DIR = USER_NEW_DIR / "User_Face"
WEAR_DIR = IMG_DATA / "wear"
OUTPUT_DIR = ROOT / "tests" / "v22_newuser"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600
MESH_RESOLUTION = 768

# -- User body info ------------------------------------------------------------
USER_HEIGHT_CM = 162
USER_WEIGHT_KG = 55
USER_BODY_DESC = "slim petite"  # 162cm 55kg


# -- Utility functions ---------------------------------------------------------

def fix_exif_rotation(img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag."""
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
    """Load image with EXIF rotation fix, return (PIL Image, base64)."""
    pil = Image.open(path)
    pil = fix_exif_rotation(pil)
    # Convert to RGB if needed
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return pil, b64


def pil_to_b64(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def base64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def save_b64_image(b64_str: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


def b64_to_ndarray(data: dict) -> np.ndarray:
    raw = base64.b64decode(data["data"])
    return np.frombuffer(raw, dtype=data["dtype"]).reshape(data["shape"])


def angle_label(angle: float) -> str:
    return str(int(angle)) if angle == int(angle) else str(angle)


def select_best_front_photo(img_dir: Path) -> Path:
    """Select the best front-facing full-body photo.

    Heuristic: use the first image (usually front-facing) after sorting.
    User photos are taken sequentially: front → side → back.
    """
    imgs = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    )
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    # First photo is typically front-facing
    return imgs[0]


def create_comparison_grid(rows_data, row_labels, angles, title=""):
    target_h = 256
    def resize_row(images):
        return [img.resize((int(target_h * img.width / img.height), target_h),
                           Image.Resampling.LANCZOS) for img in images]

    resized_rows = [resize_row(row) for row in rows_data]
    cols, rows = len(angles), len(rows_data)
    label_h, row_label_w, title_h = 28, 180, 45
    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w, cell_h = max_w + 10, target_h + label_h
    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        ft, fl, fs = (ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)
                       for s in (22, 14, 12))
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
    print("v22 Quality Test — New User + Body Info + Full Pipeline")
    print(f"  User: 162cm / 55kg / B cup")
    print(f"  Data: {USER_NEW_DIR}")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Phase 0: Load images with EXIF fix ----------------------------------
    print("\n[Phase 0] Loading User_New images (EXIF rotation fix)...")

    # Select best front photo
    user_photo_path = select_best_front_photo(USER_IMG_DIR)
    user_pil, user_b64 = load_image_fixed(user_photo_path)
    print(f"  User photo: {user_photo_path.name}")
    print(f"  Original size: {user_pil.size} (after EXIF fix)")
    user_pil.save(OUTPUT_DIR / "input_user.jpg", quality=95)

    # Load face photo
    face_imgs = sorted(list(USER_FACE_DIR.glob("*.jpg")) + list(USER_FACE_DIR.glob("*.png")))
    if face_imgs:
        face_pil, face_b64 = load_image_fixed(face_imgs[0])
        face_pil.save(OUTPUT_DIR / "input_face.jpg", quality=95)
        print(f"  Face photo: {face_imgs[0].name} ({face_pil.size})")
    else:
        face_b64 = ""
        print("  No face photo found")

    # Load wear image (first one = front view of the garment)
    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    if not wear_imgs:
        print("  ERROR: No wear images"); return
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    wear_pil.save(OUTPUT_DIR / "input_wear.jpg", quality=95)
    print(f"  Wear image: {wear_imgs[0].name} ({wear_pil.size})")
    print(f"  Note: Model-worn photo → garment_photo_type='model'")

    # ---- Phase 1: SAM 3D Body (GPU) ------------------------------------------
    print("\n" + "=" * 80)
    print("[Phase 1] SAM 3D Body Reconstruction (GPU)")
    print("=" * 80)

    t1_start = time.time()
    with modal_app.run():
        mesh_result = run_light_models.remote(task="reconstruct_3d", image_b64=user_b64)
    t1 = time.time() - t1_start
    timings["phase1_sec"] = round(t1, 2)

    if "error" in mesh_result:
        print(f"  ERROR: {mesh_result['error']}"); return

    vertices = b64_to_ndarray(mesh_result["vertices"])
    faces = b64_to_ndarray(mesh_result["faces"])
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")
    print(f"  Y range: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
    print(f"  Time: {t1:.1f}s  Cost: ${t1 * H200_COST_PER_SEC:.4f}")

    # Save mesh data for potential reuse
    np.savez_compressed(
        OUTPUT_DIR / "mesh_data.npz",
        vertices=vertices, faces=faces,
    )

    # ---- Phase 1R: CPU Mesh Rendering (free) ---------------------------------
    print("\n" + "=" * 80)
    print("[Phase 1R] CPU Mesh Rendering (16 angles, auto-flip)")
    print("=" * 80)

    t_render_start = time.time()
    render_b64s = []
    for angle in ANGLES_16:
        rendered = render_mesh(vertices, faces, angle_deg=angle, resolution=MESH_RESOLUTION)
        ok, buf = cv2.imencode(".jpg", rendered)
        if not ok:
            print(f"  ERROR: encode failed at {angle}°"); return
        render_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))

    t_render = time.time() - t_render_start
    timings["phase1r_render_sec"] = round(t_render, 2)
    print(f"  Rendered {len(render_b64s)} mesh images in {t_render:.1f}s (CPU)")

    for a, b64 in zip(ANGLES_16, render_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"mesh_{angle_label(a)}.png"))

    # ---- Phase 1.5: SDXL ControlNet (GPU) with body-specific prompt ----------
    # Body-specific prompt for 162cm/55kg Korean woman
    v22_prompt = (
        "A photorealistic full-body photograph of a young Korean woman, "
        "slim petite build, long straight dark brown hair, {angle_desc}, "
        "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
        "standing naturally on flat ground with both feet on the floor, "
        "clean neutral gray studio background, soft diffused lighting, "
        "high quality, natural skin texture, sharp focus, "
        "professional fashion photography, full body from head to toe"
    )
    v22_negative = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, "
        "floating, levitating, hovering, feet off ground, jumping, "
        "cropped, cut off, missing limbs, extra limbs, extra fingers, "
        "oversized body, muscular, tall, "
        "nsfw, nude, revealing"
    )

    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL + ControlNet Depth (body-specific prompt)")
    print(f"  Body: {USER_BODY_DESC} ({USER_HEIGHT_CM}cm, {USER_WEIGHT_KG}kg)")
    print("=" * 80)

    t15_start = time.time()
    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64="",
            angles=ANGLES_16,
            num_steps=30,
            guidance=7.5,
            controlnet_conditioning_scale=0.5,
            prompt_template=v22_prompt,
            negative_prompt_override=v22_negative,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}"); return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        t15 = time.time() - t15_start
        timings["phase15_sec"] = round(t15, 2)
        print(f"  Generated {len(realistic_b64s)} realistic in {t15:.1f}s (${t15 * H200_COST_PER_SEC:.4f})")

        first = base64_to_pil(realistic_b64s[0])
        print(f"  Resolution: {first.size[0]}x{first.size[1]}")

        for a, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(a)}.png"))

        # ---- Phase 3: FASHN VTON (GPU) — model-worn garment -----------------
        print("\n" + "=" * 80)
        print("[Phase 3] FASHN VTON v1.5 (garment_photo_type='model')")
        print("=" * 80)

        t3_start = time.time()
        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=realistic_b64s,
            clothing_b64=wear_b64,
            category="tops",
            garment_photo_type="model",  # wear images are model-worn
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
        )
        t3 = time.time() - t3_start
        timings["phase3_sec"] = round(t3, 2)

        if "error" in fashn_result:
            print(f"  ERROR: {fashn_result['error']}"); return

        fitted_b64s = fashn_result["results_b64"]
        print(f"  FASHN VTON: {len(fitted_b64s)} images in {t3:.1f}s (${t3 * H200_COST_PER_SEC:.4f})")

    for a, b64 in zip(ANGLES_16, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(a)}.png"))

    # ---- Comparison grids: mesh → realistic → fitted -------------------------
    print("\n[Phase 4] Creating comparison grids...")

    v22_mesh = [base64_to_pil(b) for b in render_b64s]
    v22_r = [base64_to_pil(b) for b in realistic_b64s]
    v22_f = [base64_to_pil(b) for b in fitted_b64s]

    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [v22_mesh[sl], v22_r[sl], v22_f[sl]],
            ["Mesh (CPU)", "Realistic (SDXL)", "Fitted (VTON)"],
            angles_sl,
            f"v22 New User — 162cm/55kg ({suffix}°)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # Also save a hero grid: just the key angles (0, 45, 90, 180, 270)
    hero_angles = [0, 45, 90, 180, 270]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]
    hero_grid = create_comparison_grid(
        [[v22_mesh[i] for i in hero_idx],
         [v22_r[i] for i in hero_idx],
         [v22_f[i] for i in hero_idx]],
        ["Mesh", "Realistic", "Fitted"],
        hero_angles,
        "v22 Key Angles — 162cm/55kg",
    )
    hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    # ---- Cost / Report -------------------------------------------------------
    gpu_total = t1 + t15 + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v22_new_user",
        "timestamp": timestamp,
        "user_info": {
            "height_cm": USER_HEIGHT_CM,
            "weight_kg": USER_WEIGHT_KG,
            "body_desc": USER_BODY_DESC,
            "photo": user_photo_path.name,
            "photo_size": list(user_pil.size),
        },
        "changes_from_v21": [
            "User_New data with EXIF rotation fix",
            "Body-specific prompt (slim petite, 162cm build)",
            "garment_photo_type='model' (wear images are model-worn)",
            "Added negative: oversized, muscular, tall, nsfw",
            "Mesh data saved as .npz for reuse",
        ],
        "parameters": {
            "phase15": {"num_steps": 30, "guidance": 7.5, "cn_scale": 0.5},
            "phase3": {
                "category": "tops",
                "garment_photo_type": "model",
                "num_timesteps": 30,
                "guidance_scale": 1.5,
                "seed": 42,
            },
        },
        "timings": timings,
        "cost": {
            "gpu_total_sec": round(gpu_total, 2),
            "gpu_total_usd": round(gpu_total * H200_COST_PER_SEC, 4),
        },
        "vertex_y_range": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
    }
    (OUTPUT_DIR / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_hero.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part1.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part2.png'}'")
    for a in [0, 90, 180]:
        for prefix in ["mesh", "realistic", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v22 New User Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
