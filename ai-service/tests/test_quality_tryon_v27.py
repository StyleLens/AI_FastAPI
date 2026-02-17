#!/usr/bin/env python3
"""
v27 Quality Test — Body Shape Analysis + Accurate Prompt

User feedback on v26: "체형이 실제 사진에서 나오는 체형과 좀 달라.
실제에 비해 너무 마른 체형이 되었어."

Changes from v26:
  1. NEW: Body shape analyzer (core/body_analyzer.py) extracts body type from mesh
  2. SDXL prompt uses mesh-derived body description instead of hardcoded "slim petite"
  3. Negative prompt excludes wrong body types based on analysis
  4. Keep turtle neck fix + ground plane from v26

Estimated cost: ~$0.28 (Phase 1.5 + VTON)
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
from core.body_analyzer import analyze_body_from_mesh
from worker.modal_app import (
    app as modal_app,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
V22_DIR = ROOT / "tests" / "v22_newuser"
V26_DIR = ROOT / "tests" / "v26_neckfix"
OUTPUT_DIR = ROOT / "tests" / "v27_bodyshape"
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
    label_h, row_label_w, title_h = 28, 200, 45
    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w, cell_h = max_w + 10, target_h + label_h
    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        ft, fl, fs = (ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s) for s in (22, 13, 12))
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


def build_body_aware_prompt(body_desc: str) -> tuple[str, str]:
    """Build SDXL prompt and negative prompt based on body analysis.

    The body_desc comes from body_analyzer.sdxl_description, e.g.:
    "average build, pear-shaped body shape, defined waist"
    """
    # Map body build to SDXL-friendly descriptions
    # The analyzer might say "full-figured" but the T-pose mesh inflates this
    # Apply a correction: mesh overestimates volume by ~1 tier
    build_corrections = {
        "very slim": "very slim petite",
        "slim": "slim",
        "average": "average",
        "slightly full": "average to slightly curvy",
        "full-figured": "average to slightly curvy",  # mesh overestimate correction
    }

    # Extract build type from description
    build_word = "average"
    for key in build_corrections:
        if key in body_desc:
            build_word = build_corrections[key]
            break

    # Extract shape info
    shape_word = ""
    if "hourglass" in body_desc:
        shape_word = "balanced proportions with defined curves"
    elif "pear" in body_desc:
        shape_word = "slightly wider hips than shoulders"
    elif "inverted triangle" in body_desc:
        shape_word = "broader shoulders tapering to narrower hips"
    elif "straight" in body_desc:
        shape_word = "balanced straight proportions"

    # Build the body portion of the prompt
    body_prompt_parts = [build_word + " body"]
    if shape_word:
        body_prompt_parts.append(shape_word)
    body_prompt = ", ".join(body_prompt_parts)

    prompt = (
        "RAW photo, a real photograph of a young Korean woman, "
        f"{body_prompt}, "
        "long straight dark brown hair, {angle_desc}, "
        "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
        "standing upright with perfect posture on a flat floor, "
        "head directly above shoulders, chin level, straight vertical neck, "
        "relaxed natural arms at sides, "
        "shot with DSLR camera, 85mm lens, "
        "clean neutral background, soft natural lighting, "
        "realistic skin texture, realistic body proportions, natural calm expression"
    )

    # Negative prompt: exclude wrong body types
    wrong_body_negatives = []
    if "slim" in build_word or "petite" in build_word:
        wrong_body_negatives.extend(["muscular", "overweight", "curvy"])
    elif "average" in build_word:
        wrong_body_negatives.extend(["extremely thin", "anorexic", "bodybuilder", "obese"])
    elif "curvy" in build_word or "full" in build_word:
        wrong_body_negatives.extend(["extremely thin", "skeletal", "anorexic"])

    negative = (
        "anime, cartoon, illustration, painting, drawing, sketch, "
        "3d render, CGI, CG, computer graphics, digital art, "
        "smooth plastic skin, airbrushed, doll-like, porcelain, "
        "floating, levitating, hovering, feet off ground, mid-air, "
        "leaning forward, hunched, slouching, bent over, "
        "turtle neck, forward head posture, chin jutting out, neck craning, "
        "tilted, diagonal posture, crooked stance, "
        "T-pose, arms spread wide, arms extended outward, "
        "cropped, cut off, missing limbs, extra limbs, "
        + ", ".join(wrong_body_negatives) + ", "
        "nsfw, nude, revealing, "
        "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    )

    return prompt, negative


def main():
    print("=" * 80)
    print("v27 Quality Test — Body Shape Analysis + Accurate Prompt")
    print("  Auto-detect body type from mesh → reflect in SDXL prompt")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # Load mesh data
    print("\n[Phase 0] Loading v22 mesh data...")
    mesh_data = np.load(V22_DIR / "mesh_data.npz")
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")

    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    # Phase 0B: Body shape analysis
    print("\n[Phase 0B] Body Shape Analysis...")
    t_ba_start = time.time()
    body_metrics = analyze_body_from_mesh(vertices, faces, resolution=MESH_RESOLUTION)
    t_ba = time.time() - t_ba_start
    timings["body_analysis_sec"] = round(t_ba, 2)

    print(f"  Build: {body_metrics.build_type}")
    print(f"  Shape: {body_metrics.shape_type}")
    print(f"  Waist: {body_metrics.waist_definition}")
    print(f"  SDXL desc: \"{body_metrics.sdxl_description}\"")
    print(f"  Analysis took {t_ba:.1f}s")

    # Build body-aware prompts
    v27_prompt, v27_negative = build_body_aware_prompt(body_metrics.sdxl_description)
    print(f"\n  Prompt body part: ...{v27_prompt[40:120]}...")

    # Phase 1R: Render depth maps
    print("\n[Phase 1R] CPU Mesh Rendering...")
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
        render_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        cv2.imwrite(str(OUTPUT_DIR / f"mesh_{lbl}.png"), rendered)

    t1r = time.time() - t1r_start
    timings["phase1r_sec"] = round(t1r, 2)
    print(f"  Rendered {len(render_b64s)} angles in {t1r:.1f}s")

    # Phase 1.5: SDXL with body-aware prompt
    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL Photorealistic (body-aware prompt)")
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
            prompt_template=v27_prompt,
            negative_prompt_override=v27_negative,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}"); return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        t15 = time.time() - t15_start
        timings["phase15_sec"] = round(t15, 2)
        print(f"  Generated {len(realistic_b64s)} realistic in {t15:.1f}s")

        for a, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(a)}.png"))

        # Phase 3: VTON
        print("\n[Phase 3] FASHN VTON")
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

    # Comparison grids: v26 vs v27
    print("\n[Phase 4] Creating comparison grids...")

    v27_r = [base64_to_pil(b) for b in realistic_b64s]
    v27_f = [base64_to_pil(b) for b in fitted_b64s]
    depth_imgs = [Image.open(OUTPUT_DIR / f"mesh_{angle_label(a)}.png") for a in ANGLES_16]

    has_v26 = (V26_DIR / "realistic_0.png").exists()
    hero_angles = [0, 45, 90, 180, 270, 315]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]

    if has_v26:
        v26_r = [Image.open(V26_DIR / f"realistic_{angle_label(a)}.png") for a in ANGLES_16]
        v26_f = [Image.open(V26_DIR / f"fitted_{angle_label(a)}.png") for a in ANGLES_16]

        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [v26_r[i] for i in hero_idx],
             [v27_r[i] for i in hero_idx],
             [v26_f[i] for i in hero_idx],
             [v27_f[i] for i in hero_idx]],
            ["v27 Depth", "v26 Realistic", "v27 Realistic",
             "v26 Fitted", "v27 Fitted"],
            hero_angles,
            "v26 (slim petite) vs v27 (body-aware prompt)",
        )
    else:
        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [v27_r[i] for i in hero_idx],
             [v27_f[i] for i in hero_idx]],
            ["Depth Map", "Realistic", "Fitted"],
            hero_angles,
            "v27 — Body Shape Aware",
        )
    hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [depth_imgs[sl], v27_r[sl], v27_f[sl]],
            ["Depth", "Realistic", "Fitted"],
            angles_sl,
            f"v27 ({suffix}\u00b0)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # Cost / Report
    gpu_total = t15 + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v27_body_shape_aware",
        "timestamp": timestamp,
        "user_feedback": "v26 체형이 실제보다 너무 마름 → mesh 체형 분석 도입",
        "changes_from_v26": [
            "NEW: Body shape analyzer (core/body_analyzer.py)",
            f"Body analysis result: {body_metrics.sdxl_description}",
            "SDXL prompt uses mesh-derived body description",
            "Negative prompt excludes wrong body types",
            "Keep turtle neck fix + ground plane from v26",
        ],
        "body_analysis": {
            "build_type": body_metrics.build_type,
            "shape_type": body_metrics.shape_type,
            "waist_definition": body_metrics.waist_definition,
            "depth_profile": body_metrics.depth_profile,
            "sdxl_description": body_metrics.sdxl_description,
            "shoulder_waist_ratio": round(body_metrics.shoulder_waist_ratio, 2),
            "hip_waist_ratio": round(body_metrics.hip_waist_ratio, 2),
            "waist_hip_ratio": round(body_metrics.waist_hip_ratio, 2),
            "volume_height_ratio": round(body_metrics.volume_height_ratio, 4),
        },
        "parameters": {
            "renderer": {"straighten": True, "ground_plane": True,
                        "overcorrect": 1.20, "z_compress": 0.15, "blend_start": 0.78},
            "phase15": {"num_steps": 30, "guidance": 7.0, "cn_scale": 0.5},
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
    for a in [0, 45, 90, 180, 315]:
        for prefix in ["realistic", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v27 Body Shape Aware Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
