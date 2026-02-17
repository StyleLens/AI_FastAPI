#!/usr/bin/env python3
"""
v30 Quality Test — Leg Closing + Face Refiner + Photorealistic Prompts

All-in-one quality upgrade from v29:
  1. close_legs=True — A-pose legs closed to natural standing position
  2. Face Refiner v2 — MediaPipe face detection + SDXL Inpainting (photorealistic)
  3. Photorealistic prompts — anti Plastic Look (skin texture, film grain, pores)
  4. Keep: fold_arms=True, cn_scale=0.6

Pipeline: Mesh → Depth → SDXL Realistic → Face Refine → FASHN VTON

Estimated cost: ~$0.35 (Phase 1.5 + Face Refine + VTON)
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
    run_face_refiner,
    run_fashn_vton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
FACE_DIR = IMG_DATA / "User_New" / "User_Face"
V22_DIR = ROOT / "tests" / "v22_newuser"
V29_DIR = ROOT / "tests" / "v29_armsdown"
OUTPUT_DIR = ROOT / "tests" / "v30_fullrefine"
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


def create_comparison_grid(rows_data, row_labels, angles, title=""):
    target_h = 256
    def resize_row(images):
        return [img.resize((int(target_h * img.width / img.height), target_h),
                           Image.Resampling.LANCZOS) for img in images]
    resized_rows = [resize_row(row) for row in rows_data]
    cols, rows = len(angles), len(rows_data)
    label_h, row_label_w, title_h = 28, 220, 45
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
    print("v30 Quality Test — Legs Closed + Face Refiner + Photorealistic")
    print("  A-pose legs closed + MediaPipe face inpainting + skin texture prompts")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # Load mesh data
    print("\n[Phase 0] Loading mesh data...")
    mesh_data = np.load(V22_DIR / "mesh_data.npz")
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")

    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    # Load face reference (if available)
    face_b64 = ""
    face_imgs = sorted(list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png")))
    if face_imgs:
        _, face_b64 = load_image_fixed(face_imgs[0])
        print(f"  Face ref: {face_imgs[0].name}")

    # Phase 1R: Render with arms folded + legs closed
    print("\n[Phase 1R] CPU Mesh Rendering (arms-down + legs-closed)...")
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
            fold_arms=True,     # v29: rotate arms down
            close_legs=True,    # NEW v30: close A-pose legs
            body_scale=None,
        )
        ok, buf = cv2.imencode(".png", rendered)
        render_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        cv2.imwrite(str(OUTPUT_DIR / f"mesh_{lbl}.png"), rendered)

    t1r = time.time() - t1r_start
    timings["phase1r_sec"] = round(t1r, 2)
    print(f"  Rendered {len(render_b64s)} angles in {t1r:.1f}s")

    # Photorealistic prompt with skin texture emphasis (anti Plastic Look)
    v30_prompt = (
        "RAW photo, an extremely detailed real photograph of a young Korean woman, "
        "realistic skin pores, natural skin texture, subtle skin imperfections, "
        "average build with natural proportions, "
        "long straight dark brown hair, {angle_desc}, "
        "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
        "standing upright with perfect posture on a flat floor, "
        "head directly above shoulders, chin level, straight vertical neck, "
        "relaxed natural arms hanging at sides, hands beside thighs, "
        "legs together in natural standing position, "
        "shot with Fujifilm XT4, 85mm portrait lens, film grain, "
        "clean neutral studio background, soft natural lighting, "
        "realistic body proportions, natural calm expression, 8k uhd"
    )
    v30_negative = (
        "anime, cartoon, illustration, painting, drawing, sketch, "
        "3d render, CGI, CG, computer graphics, digital art, "
        "smooth plastic skin, airbrushed, doll-like, porcelain, wax figure, "
        "floating, levitating, hovering, feet off ground, mid-air, "
        "leaning forward, hunched, slouching, bent over, "
        "turtle neck, forward head posture, chin jutting out, neck craning, "
        "tilted, diagonal posture, crooked stance, "
        "T-pose, arms spread wide, arms extended outward, "
        "A-pose, legs spread apart, wide stance, "
        "hands on hips, arms akimbo, hands behind back, "
        "cropped, cut off, missing limbs, extra limbs, "
        "extremely thin, anorexic, bodybuilder, obese, "
        "nsfw, nude, revealing, "
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "oversaturated, overexposed"
    )

    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL Photorealistic (arms-down + legs-closed + cn=0.6)")
    print("=" * 80)

    t15_start = time.time()
    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64="",
            angles=ANGLES_16,
            num_steps=30,
            guidance=7.0,
            controlnet_conditioning_scale=0.6,
            prompt_template=v30_prompt,
            negative_prompt_override=v30_negative,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}"); return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        t15 = time.time() - t15_start
        timings["phase15_sec"] = round(t15, 2)
        print(f"  Generated {len(realistic_b64s)} realistic in {t15:.1f}s")

        for a, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(a)}.png"))

        # Phase 2: Face Refiner v2 (NEW in v30)
        print("\n[Phase 2] Face Refiner v2 (MediaPipe + photorealistic inpainting)")
        t2_start = time.time()
        face_result = run_face_refiner.remote(
            images_b64=realistic_b64s,
            face_reference_b64=face_b64,
            angles=ANGLES_16,
            face_expand_ratio=1.8,
            num_steps=25,
            guidance=6.0,       # Lower for natural look
            strength=0.45,      # Moderate: preserve structure, add texture
            seed=42,
        )
        t2 = time.time() - t2_start
        timings["phase2_face_sec"] = round(t2, 2)

        if "error" in face_result:
            print(f"  Face Refiner ERROR: {face_result['error']}")
            # Fall back to un-refined images
            refined_b64s = realistic_b64s
        else:
            refined_b64s = face_result["refined_b64"]
            face_flags = face_result.get("face_detected", [])
            n_refined = sum(1 for f in face_flags if f)
            print(f"  Face refined: {n_refined}/{len(refined_b64s)} in {t2:.1f}s")

        for a, b64 in zip(ANGLES_16, refined_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"refined_{angle_label(a)}.png"))

        # Phase 3: VTON (on face-refined images)
        print("\n[Phase 3] FASHN VTON (on face-refined images)")
        t3_start = time.time()
        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=refined_b64s,
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
            print(f"  VTON ERROR: {fashn_result['error']}"); return

        fitted_b64s = fashn_result["results_b64"]
        print(f"  VTON: {len(fitted_b64s)} images in {t3:.1f}s")

    for a, b64 in zip(ANGLES_16, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(a)}.png"))

    # Comparison grids
    print("\n[Phase 4] Creating comparison grids...")

    v30_r = [base64_to_pil(b) for b in realistic_b64s]
    v30_ref = [base64_to_pil(b) for b in refined_b64s]
    v30_f = [base64_to_pil(b) for b in fitted_b64s]
    depth_imgs = [Image.open(OUTPUT_DIR / f"mesh_{angle_label(a)}.png") for a in ANGLES_16]

    hero_angles = [0, 45, 90, 180, 270, 315]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]

    # Compare v29 vs v30
    has_v29 = (V29_DIR / "realistic_0.png").exists()
    if has_v29:
        v29_r = [Image.open(V29_DIR / f"realistic_{angle_label(a)}.png") for a in ANGLES_16]
        v29_f = [Image.open(V29_DIR / f"fitted_{angle_label(a)}.png") for a in ANGLES_16]

        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [v29_r[i] for i in hero_idx],
             [v30_r[i] for i in hero_idx],
             [v30_ref[i] for i in hero_idx],
             [v29_f[i] for i in hero_idx],
             [v30_f[i] for i in hero_idx]],
            ["v30 Depth (legs closed)",
             "v29 Realistic", "v30 Realistic",
             "v30 Face Refined",
             "v29 Fitted", "v30 Fitted"],
            hero_angles,
            "v29 (arms only) vs v30 (legs+face+realism)",
        )
    else:
        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [v30_r[i] for i in hero_idx],
             [v30_ref[i] for i in hero_idx],
             [v30_f[i] for i in hero_idx]],
            ["Depth Map", "Realistic", "Face Refined", "Fitted"],
            hero_angles,
            "v30 — Full Refine Pipeline",
        )
    hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [depth_imgs[sl], v30_r[sl], v30_ref[sl], v30_f[sl]],
            ["Depth", "Realistic", "Face Refined", "Fitted"],
            angles_sl,
            f"v30 ({suffix}\u00b0)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # Cost / Report
    gpu_total = t15 + t2 + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v30_full_refine",
        "timestamp": timestamp,
        "key_changes": [
            "NEW: close_legs=True (A-pose legs closed to natural standing)",
            "NEW: Face Refiner v2 (MediaPipe + photorealistic SDXL inpainting)",
            "NEW: Anti Plastic Look prompts (skin pores, film grain, Fujifilm XT4)",
            "NEW: A-pose negative prompt (legs spread apart, wide stance)",
            "KEEP: fold_arms=True, cn_scale=0.6",
        ],
        "parameters": {
            "renderer": {"straighten": True, "ground_plane": True,
                         "fold_arms": True, "close_legs": True,
                         "arm_angle": 65, "body_scale": None},
            "phase15": {"num_steps": 30, "guidance": 7.0, "cn_scale": 0.6},
            "face_refiner": {"num_steps": 25, "guidance": 6.0, "strength": 0.45,
                             "face_expand_ratio": 1.8, "detection": "MediaPipe"},
            "phase3": {"category": "tops", "garment_photo_type": "model",
                       "num_timesteps": 30, "guidance_scale": 1.5},
        },
        "timings": timings,
        "cost": {"gpu_total_sec": round(gpu_total, 2),
                 "gpu_total_usd": round(gpu_total * H200_COST_PER_SEC, 4)},
    }
    (OUTPUT_DIR / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_hero.png'}'")
    for a in [0, 45, 90, 180, 270]:
        for prefix in ["realistic", "refined", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v30 Full Refine Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
