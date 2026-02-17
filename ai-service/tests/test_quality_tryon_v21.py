#!/usr/bin/env python3
"""
v21 Quality Test — Mesh Flip Fix + v20 Parameters

ROOT CAUSE FOUND: sw_renderer was rendering mesh UPSIDE DOWN.
SAM 3D Body outputs Y-down convention (head at low Y), but renderer assumed Y-up.
Auto-detect flip has been added to sw_renderer.py.

v21 = Fixed mesh renders + v20 prompt/negative params
  - Reuses v18 raw mesh data (vertices/faces from SAM 3D Body)
  - Re-renders mesh at 16 angles WITH the flip fix (CPU, free)
  - Runs Phase 1.5 (SDXL) + Phase 3 (FASHN VTON) on fixed mesh

Expected improvement: MAJOR — correct depth maps should produce
properly oriented, proportioned human images at all angles.

Cost: ~$0.28 (Phase 1.5 + Phase 3 only)
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

from core.sw_renderer import render_mesh
from worker.modal_app import (
    app as modal_app,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
V18_DIR = ROOT / "tests" / "NewTest"
OUTPUT_DIR = ROOT / "tests" / "v21_meshfix"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600
MESH_RESOLUTION = 768


def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

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


def create_comparison_grid(rows_data, row_labels, angles, title=""):
    target_h = 256
    def resize_row(images):
        return [img.resize((int(target_h * img.width / img.height), target_h), Image.Resampling.LANCZOS) for img in images]

    resized_rows = [resize_row(row) for row in rows_data]
    cols, rows = len(angles), len(rows_data)
    label_h, row_label_w, title_h = 28, 160, 45
    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w, cell_h = max_w + 10, target_h + label_h
    canvas_w, canvas_h = row_label_w + cell_w * cols, title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        ft, fl, fs = (ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s) for s in (22, 14, 12))
    except Exception:
        ft = fl = fs = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=ft)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) // 2, 12), title, fill=(0,0,0), font=ft)

    for ri, (imgs, label) in enumerate(zip(resized_rows, row_labels)):
        draw.text((8, title_h + ri * cell_h + cell_h // 2), label, fill=(0,0,0), font=fl)
        for ci, (img, ang) in enumerate(zip(imgs, angles)):
            x = row_label_w + ci * cell_w + (cell_w - img.width) // 2
            y = title_h + ri * cell_h + label_h
            canvas.paste(img, (x, y))
            if ri == 0:
                at = f"{angle_label(ang)} deg"
                bbox = draw.textbbox((0, 0), at, font=fs)
                draw.text((row_label_w + ci * cell_w + (cell_w - bbox[2] + bbox[0]) // 2, title_h + 4), at, fill=(0,0,0), font=fs)
    return canvas


def main():
    print("=" * 80)
    print("v21 Quality Test — MESH FLIP FIX + v20 Parameters")
    print("Root cause: sw_renderer was rendering mesh upside down")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Load v18 mesh data (vertices/faces) via SAM 3D Body output --------
    # We need the raw mesh. Let's get it by re-running Phase 1 on GPU,
    # OR we can load from the test if it was cached.
    # Since v18 didn't save raw vertices, we need to reconstruct from GPU again.
    # BUT to save $, let's just re-render the v18 mesh images through the fixed renderer.

    # Actually, we can't — the mesh images are already rendered (flipped).
    # We need the raw vertices/faces to re-render correctly.
    # We MUST run Phase 1 (SAM 3D Body) again to get raw mesh data.

    print("\n[Phase 0] Loading images...")

    # User image
    USER_IMG_DIR = IMG_DATA / "User_IMG"
    user_imgs = sorted(list(USER_IMG_DIR.glob("*.jpg")) + list(USER_IMG_DIR.glob("*.png")))
    if not user_imgs:
        print("  ERROR: No user images"); return
    user_b64 = load_image_base64(user_imgs[0])
    print(f"  User: {user_imgs[0].name}")

    # Wear image
    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    if not wear_imgs:
        print("  ERROR: No wear images"); return
    wear_b64 = load_image_base64(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    # ---- Phase 1: SAM 3D Body (GPU) — need raw vertices --------------------
    print("\n" + "=" * 80)
    print("[Phase 1] SAM 3D Body Reconstruction (GPU) — for raw mesh data")
    print("=" * 80)

    from worker.modal_app import run_light_models

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

    # ---- Phase 1R: Re-render mesh with FIXED renderer (CPU) ----------------
    print("\n" + "=" * 80)
    print("[Phase 1R] Rendering mesh with FIXED sw_renderer (auto-flip)")
    print("=" * 80)

    t_render_start = time.time()
    render_b64s = []
    for angle in ANGLES_16:
        rendered = render_mesh(vertices, faces, angle_deg=angle, resolution=MESH_RESOLUTION)
        ok, buf = cv2.imencode(".jpg", rendered)
        if not ok:
            print(f"  ERROR: encode failed at {angle}"); return
        render_b64s.append(base64.b64encode(buf.tobytes()).decode("ascii"))

    t_render = time.time() - t_render_start
    timings["phase1r_render_sec"] = round(t_render, 2)
    print(f"  Rendered {len(render_b64s)} mesh images in {t_render:.1f}s (CPU)")

    # Save mesh renders for visual check
    for a, b64 in zip(ANGLES_16, render_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"mesh_{angle_label(a)}.png"))
    print("  Mesh renders saved — check if head is at TOP now")

    # ---- Phase 1.5: SDXL ControlNet (GPU) with v20 params -----------------
    v21_prompt = (
        "A photorealistic full-body photograph of a young Korean woman, "
        "long black hair, {angle_desc}, "
        "wearing a plain gray short-sleeve t-shirt and dark blue jeans, "
        "standing firmly on the ground with both feet on the floor, "
        "clean gray studio background, soft natural lighting, "
        "high quality, detailed skin texture, sharp focus, "
        "professional fashion photography"
    )
    v21_negative = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, "
        "floating, levitating, hovering, feet off ground, jumping, "
        "cropped, cut off, missing limbs, extra limbs"
    )

    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL + ControlNet Depth (v20 params, FIXED mesh)")
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
            prompt_template=v21_prompt,
            negative_prompt_override=v21_negative,
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

        # ---- Phase 3: FASHN VTON ----------------------------------------
        print("\n" + "=" * 80)
        print("[Phase 3] FASHN VTON v1.5 (maskless)")
        print("=" * 80)

        t3_start = time.time()
        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=realistic_b64s,
            clothing_b64=wear_b64,
            category="tops",
            garment_photo_type="flat-lay",
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

    # ---- Comparison grids: v18 (flipped mesh) vs v21 (fixed mesh) ----------
    print("\n[Phase 4] v18 vs v21 comparison grids...")

    v18_r = [Image.open(V18_DIR / f"realistic_{angle_label(a)}.png") for a in ANGLES_16]
    v18_f = [Image.open(V18_DIR / f"fitted_{angle_label(a)}.png") for a in ANGLES_16]
    v21_mesh = [base64_to_pil(b) for b in render_b64s]
    v21_r = [base64_to_pil(b) for b in realistic_b64s]
    v21_f = [base64_to_pil(b) for b in fitted_b64s]

    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [v21_mesh[sl], v18_r[sl], v21_r[sl], v18_f[sl], v21_f[sl]],
            ["v21 Mesh (fixed)", "v18 Realistic", "v21 Realistic", "v18 VTON", "v21 VTON"],
            angles_sl,
            f"v21 Mesh Fix: v18 vs v21 ({suffix} deg)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # ---- Cost / Report -------------------------------------------------------
    gpu_total = t1 + t15 + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v21_mesh_flip_fix",
        "timestamp": timestamp,
        "root_cause": "sw_renderer rendered mesh upside down (SAM 3D Body Y-down convention)",
        "fix": "Auto-detect + flip Y-axis in sw_renderer when head is at bottom",
        "parameters": {
            "phase15": {"num_steps": 30, "guidance": 7.5, "cn_scale": 0.5},
            "phase3": {"category": "tops", "num_timesteps": 30, "guidance_scale": 1.5, "seed": 42},
        },
        "timings": timings,
        "cost": {"gpu_total_sec": round(gpu_total, 2), "gpu_total_usd": round(gpu_total * H200_COST_PER_SEC, 4)},
        "vertex_y_range": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
    }
    (OUTPUT_DIR / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part1.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part2.png'}'")
    # Open key angles
    for a in [0, 90, 180]:
        for prefix in ["mesh", "realistic", "fitted"]:
            p = OUTPUT_DIR / f"{prefix}_{angle_label(a)}.png"
            if p.exists(): os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v21 Mesh Fix Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r: print(f"\n{json.dumps(r, indent=2)}")
