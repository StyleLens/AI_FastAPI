#!/usr/bin/env python3
"""
v31 Quality Test — SDXL→FLUX.2-klein Hybrid Pipeline

The breakthrough: Use SDXL ControlNet for pose accuracy, then FLUX.2-klein-4B
img2img for photorealistic texture refinement.

Pipeline:
  Phase 1R:  CPU Mesh Rendering (arms-down + legs-closed)
  Phase 1.5A: SDXL + ControlNet Depth → 16 pose-accurate images
  Phase 1.5B: FLUX.2-klein-4B img2img → photorealistic texture upgrade (NEW)
  Phase 2:   FLUX.2-klein Face Refiner v3 → crop-refine-merge (NEW)
  Phase 3:   FASHN VTON → virtual try-on

Key changes from v30:
  - FLUX.2-klein-4B replaces SDXL as texture engine (Apache 2.0)
  - Face Refiner v3: No MediaPipe dependency (crop-based, not detection-based)
  - Depth maps encoded as PNG (not JPEG) to avoid compression artifacts
  - Anti Plastic Look at model level, not just prompt level

Budget target: ≤$0.35 (≈475원) per fitting
Estimated: ~$0.33 (SDXL 90s + FLUX 25s + Face 15s + VTON 97s = ~227s)
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
    run_flux_refine,
    run_face_refiner,
    run_fashn_vton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
FACE_DIR = IMG_DATA / "User_New" / "User_Face"
V22_DIR = ROOT / "tests" / "v22_newuser"
V29_DIR = ROOT / "tests" / "v29_armsdown"
V30_DIR = ROOT / "tests" / "v30_fullrefine"
OUTPUT_DIR = ROOT / "tests" / "v31_flux_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600  # $0.0015/sec
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
    print("v31 FLUX Hybrid Test — SDXL→FLUX.2-klein-4B Pipeline")
    print("  SDXL ControlNet (pose) → FLUX.2-klein (texture) → Face Refine → VTON")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ── Phase 0: Load data ──
    print("\n[Phase 0] Loading mesh data...")
    mesh_data = np.load(V22_DIR / "mesh_data.npz")
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")

    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    face_b64 = ""
    face_imgs = sorted(list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png")))
    if face_imgs:
        _, face_b64 = load_image_fixed(face_imgs[0])
        print(f"  Face ref: {face_imgs[0].name}")

    # ── Phase 1R: CPU Mesh Rendering ──
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
            fold_arms=True,
            close_legs=True,
            body_scale=None,
        )
        # CRITICAL: Use PNG encoding to prevent JPEG compression artifacts
        # v30 had severe grid artifacts at 90° likely due to JPEG encoding
        ok, buf = cv2.imencode(".png", rendered)
        render_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        cv2.imwrite(str(OUTPUT_DIR / f"mesh_{lbl}.png"), rendered)

    t1r = time.time() - t1r_start
    timings["phase1r_sec"] = round(t1r, 2)
    print(f"  Rendered {len(render_b64s)} angles in {t1r:.1f}s")

    # ── Prompts ──
    v31_prompt = (
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
    v31_negative = (
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

    # ── GPU Pipeline (single Modal session) ──
    print("\n" + "=" * 80)
    print("[Phase 1.5A] SDXL + ControlNet Depth (pose scaffold)")
    print("=" * 80)

    t_gpu_start = time.time()

    with modal_app.run():
        # Phase 1.5A: SDXL ControlNet → pose-accurate base images
        t15a_start = time.time()
        sdxl_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64="",
            angles=ANGLES_16,
            num_steps=30,
            guidance=6.5,  # v31: 7.0→6.5 for less artificial look
            controlnet_conditioning_scale=0.6,
            prompt_template=v31_prompt,
            negative_prompt_override=v31_negative,
        )

        if "error" in sdxl_result:
            print(f"  ERROR: {sdxl_result['error']}")
            return
        sdxl_b64s = sdxl_result["realistic_renders_b64"]
        t15a = time.time() - t15a_start
        timings["phase15a_sdxl_sec"] = round(t15a, 2)
        print(f"  SDXL: {len(sdxl_b64s)} images in {t15a:.1f}s")

        for a, b64 in zip(ANGLES_16, sdxl_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"sdxl_{angle_label(a)}.png"))

        # Phase 1.5B: FLUX.2-klein img2img → texture upgrade
        print("\n[Phase 1.5B] FLUX.2-klein-4B img2img (texture upgrade)")
        t15b_start = time.time()

        # FLUX refine prompt (emphasize texture over structure)
        flux_prompt = (
            "RAW photo, ultra realistic full-body photograph of a young Korean woman, "
            "{angle_desc}, "
            "extremely detailed realistic skin with visible pores and natural texture, "
            "natural hair with individual strands visible, "
            "wearing a plain gray t-shirt and dark blue jeans, "
            "realistic fabric texture with natural wrinkles and folds, "
            "clean studio background, soft natural lighting, "
            "shot on Fujifilm XT4, 85mm portrait lens, subtle film grain, "
            "professional fashion photography, 8k uhd, photorealistic"
        )

        flux_result = run_flux_refine.remote(
            images_b64=sdxl_b64s,
            prompt_template=flux_prompt,
            angles=ANGLES_16,
            num_steps=4,      # Distilled klein: 4 steps
            guidance=1.0,     # Distilled: guidance 1.0
            seed=42,
        )

        if "error" in flux_result:
            print(f"  FLUX ERROR: {flux_result['error']}")
            # Fall back to SDXL results
            realistic_b64s = sdxl_b64s
            t15b = time.time() - t15b_start
        else:
            realistic_b64s = flux_result["refined_b64"]
            t15b = time.time() - t15b_start
            print(f"  FLUX: {len(realistic_b64s)} images in {t15b:.1f}s")

        timings["phase15b_flux_sec"] = round(t15b, 2)

        for a, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(a)}.png"))

        # Phase 2: Face Refiner v3 (FLUX-based crop-refine-merge)
        print("\n[Phase 2] Face Refiner v3 (FLUX.2-klein crop-refine-merge)")
        t2_start = time.time()
        face_result = run_face_refiner.remote(
            images_b64=realistic_b64s,
            face_reference_b64=face_b64,
            angles=ANGLES_16,
            num_steps=4,
            guidance=1.0,
            seed=42,
        )
        t2 = time.time() - t2_start
        timings["phase2_face_sec"] = round(t2, 2)

        if "error" in face_result:
            print(f"  Face Refiner ERROR: {face_result['error']}")
            refined_b64s = realistic_b64s
        else:
            refined_b64s = face_result["refined_b64"]
            face_flags = face_result.get("face_detected", [])
            n_refined = sum(1 for f in face_flags if f)
            print(f"  Face refined: {n_refined}/{len(refined_b64s)} in {t2:.1f}s")

        for a, b64 in zip(ANGLES_16, refined_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"refined_{angle_label(a)}.png"))

        # Phase 3: FASHN VTON
        print("\n[Phase 3] FASHN VTON (on FLUX-refined images)")
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
            print(f"  VTON ERROR: {fashn_result['error']}")
            return
        fitted_b64s = fashn_result["results_b64"]
        print(f"  VTON: {len(fitted_b64s)} images in {t3:.1f}s")

    t_gpu_total = time.time() - t_gpu_start

    for a, b64 in zip(ANGLES_16, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(a)}.png"))

    # ── Comparison Grids ──
    print("\n[Phase 4] Creating comparison grids...")

    sdxl_imgs = [base64_to_pil(b) for b in sdxl_b64s]
    flux_imgs = [base64_to_pil(b) for b in realistic_b64s]
    face_imgs = [base64_to_pil(b) for b in refined_b64s]
    fitted_imgs = [base64_to_pil(b) for b in fitted_b64s]
    depth_imgs = [Image.open(OUTPUT_DIR / f"mesh_{angle_label(a)}.png") for a in ANGLES_16]

    hero_angles = [0, 45, 90, 180, 270, 315]
    hero_idx = [ANGLES_16.index(a) for a in hero_angles]

    # v30 vs v31 comparison
    has_v30 = (V30_DIR / "realistic_0.png").exists()
    if has_v30:
        v30_r = [Image.open(V30_DIR / f"realistic_{angle_label(a)}.png") for a in hero_angles
                 if (V30_DIR / f"realistic_{angle_label(a)}.png").exists()]

        if len(v30_r) == len(hero_angles):
            hero_grid = create_comparison_grid(
                [[depth_imgs[i] for i in hero_idx],
                 v30_r,
                 [sdxl_imgs[i] for i in hero_idx],
                 [flux_imgs[i] for i in hero_idx],
                 [face_imgs[i] for i in hero_idx],
                 [fitted_imgs[i] for i in hero_idx]],
                ["v31 Depth",
                 "v30 SDXL Only",
                 "v31 SDXL Scaffold",
                 "v31 FLUX Refined",
                 "v31 Face Refined",
                 "v31 Fitted"],
                hero_angles,
                "v30 (SDXL only) vs v31 (SDXL→FLUX hybrid)",
            )
        else:
            hero_grid = create_comparison_grid(
                [[depth_imgs[i] for i in hero_idx],
                 [sdxl_imgs[i] for i in hero_idx],
                 [flux_imgs[i] for i in hero_idx],
                 [face_imgs[i] for i in hero_idx],
                 [fitted_imgs[i] for i in hero_idx]],
                ["Depth", "SDXL Scaffold", "FLUX Refined", "Face Refined", "Fitted"],
                hero_angles,
                "v31 — SDXL→FLUX.2-klein Hybrid Pipeline",
            )
    else:
        hero_grid = create_comparison_grid(
            [[depth_imgs[i] for i in hero_idx],
             [sdxl_imgs[i] for i in hero_idx],
             [flux_imgs[i] for i in hero_idx],
             [face_imgs[i] for i in hero_idx],
             [fitted_imgs[i] for i in hero_idx]],
            ["Depth", "SDXL Scaffold", "FLUX Refined", "Face Refined", "Fitted"],
            hero_angles,
            "v31 — SDXL→FLUX.2-klein Hybrid Pipeline",
        )
    hero_grid.save(OUTPUT_DIR / "comparison_hero.png")

    # Full 16-angle grids (2 parts)
    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [depth_imgs[sl], sdxl_imgs[sl], flux_imgs[sl], face_imgs[sl], fitted_imgs[sl]],
            ["Depth", "SDXL", "FLUX", "Face", "Fitted"],
            angles_sl,
            f"v31 ({suffix}\u00b0)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # ── Cost Report ──
    gpu_phases = {
        "phase15a_sdxl": timings.get("phase15a_sdxl_sec", 0),
        "phase15b_flux": timings.get("phase15b_flux_sec", 0),
        "phase2_face": timings.get("phase2_face_sec", 0),
        "phase3_vton": timings.get("phase3_sec", 0),
    }
    gpu_total = sum(gpu_phases.values())
    cost_usd = gpu_total * H200_COST_PER_SEC
    cost_krw = cost_usd * 1350  # approximate KRW

    print(f"\n{'='*60}")
    print(f"  COST BREAKDOWN")
    print(f"{'='*60}")
    for phase, sec in gpu_phases.items():
        print(f"  {phase:20s}: {sec:6.1f}s  ${sec * H200_COST_PER_SEC:.4f}")
    print(f"  {'─'*40}")
    print(f"  {'Total GPU':20s}: {gpu_total:6.1f}s  ${cost_usd:.4f} (≈{cost_krw:.0f}원)")
    print(f"  {'Budget':20s}: {'PASS ✅' if cost_krw <= 500 else 'OVER ❌'} (target ≤500원)")

    report = {
        "test_name": "v31_flux_hybrid",
        "timestamp": timestamp,
        "strategy": "SDXL→FLUX.2-klein-4B Hybrid",
        "key_changes": [
            "NEW: FLUX.2-klein-4B img2img refiner (Phase 1.5B)",
            "NEW: Face Refiner v3 (FLUX.2-klein crop-refine-merge, no MediaPipe)",
            "NEW: Depth maps as PNG (not JPEG) to avoid compression artifacts",
            "NEW: CFG 7.0→6.5 for SDXL scaffold (less artificial)",
            "KEEP: fold_arms=True, close_legs=True, cn_scale=0.6",
        ],
        "parameters": {
            "renderer": {"straighten": True, "ground_plane": True,
                         "fold_arms": True, "close_legs": True,
                         "body_scale": None},
            "phase15a_sdxl": {"num_steps": 30, "guidance": 6.5, "cn_scale": 0.6},
            "phase15b_flux": {"model": "FLUX.2-klein-4B", "num_steps": 4,
                              "guidance": 1.0, "mode": "img2img in-context"},
            "phase2_face": {"model": "FLUX.2-klein-4B", "num_steps": 4,
                            "guidance": 1.0, "method": "crop-refine-merge",
                            "face_region": "top 30%"},
            "phase3": {"category": "tops", "garment_photo_type": "model",
                       "num_timesteps": 30, "guidance_scale": 1.5},
        },
        "timings": timings,
        "cost": {
            "gpu_total_sec": round(gpu_total, 2),
            "gpu_total_usd": round(cost_usd, 4),
            "gpu_total_krw": round(cost_krw, 0),
            "budget_pass": cost_krw <= 500,
        },
    }
    (OUTPUT_DIR / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    # Open key results
    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_hero.png'}'")
    for a in [0, 90, 180, 270]:
        for prefix in ["sdxl", "realistic", "refined", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists():
                os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v31 FLUX Hybrid Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2)}")
