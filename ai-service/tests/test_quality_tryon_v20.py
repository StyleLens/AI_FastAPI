#!/usr/bin/env python3
"""
v20 Quality Improvement Test — Conservative Tuning from v18 Baseline

v19 FAILED: cn_scale=0.7 + guidance=9.0 + vague prompt caused severe degradation.
v20 strategy: Keep v18 parameters, only add minimal safe improvements:

Changes from v18:
  1. negative_prompt: Add "floating, levitating, hovering, cropped"
  2. prompt: Add "standing firmly on the ground" (keeps specific clothing description)
  3. Everything else stays the same as v18 (cn_scale=0.5, guidance=7.5, steps=30)

This is a CONSERVATIVE test to confirm the negative prompt fix helps
without breaking the baseline quality.

Estimated cost: ~$0.28 (Phase 1.5 + Phase 3 only, reuses v18 mesh)
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

# -- Path setup ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent

from worker.modal_app import (                         # noqa: E402
    app as modal_app,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

# -- Directories --------------------------------------------------------------
IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
V18_DIR = ROOT / "tests" / "NewTest"
OUTPUT_DIR = ROOT / "tests" / "v20_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]
H200_COST_PER_SEC = 5.40 / 3600


def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def base64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

def save_b64_image(b64_str: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))

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
    print("v20 Quality Test — Conservative Tuning (v18 base + negative prompt fix)")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # Load v18 mesh renders
    print("\n[Phase 0] Loading v18 mesh renders + wear image...")
    mesh_b64s = []
    for angle in ANGLES_16:
        mesh_path = V18_DIR / f"mesh_{angle_label(angle)}.png"
        if not mesh_path.exists():
            print(f"  ERROR: Missing {mesh_path}")
            return
        mesh_b64s.append(load_image_base64(mesh_path))
    print(f"  Loaded {len(mesh_b64s)} mesh renders")

    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    if not wear_imgs:
        print("  ERROR: No wear images"); return
    wear_b64 = load_image_base64(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name}")

    # v20 parameters — SAME as v18, only negative_prompt and minor prompt tweak
    v20_prompt = (
        "A photorealistic full-body photograph of a young Korean woman, "
        "long black hair, {angle_desc}, "
        "wearing a plain gray short-sleeve t-shirt and dark blue jeans, "
        "standing firmly on the ground with both feet on the floor, "
        "clean gray studio background, soft natural lighting, "
        "high quality, detailed skin texture, sharp focus, "
        "professional fashion photography"
    )

    v20_negative = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, "
        "floating, levitating, hovering, feet off ground, jumping, "
        "cropped, cut off, missing limbs, extra limbs"
    )

    v20_params = {
        "num_steps": 30,        # same as v18
        "guidance": 7.5,        # same as v18
        "cn_scale": 0.5,        # same as v18
    }

    print(f"\n  v20 Changes from v18:")
    print(f"    prompt: +standing firmly on ground (minor addition)")
    print(f"    negative: +floating/levitating/hovering/cropped/cut off")
    print(f"    steps/guidance/cn_scale: UNCHANGED from v18")

    # Phase 1.5
    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL + ControlNet Depth (v20: v18 params + negative fix)")
    print("=" * 80)

    t15_start = time.time()

    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=mesh_b64s,
            person_image_b64="",
            angles=ANGLES_16,
            num_steps=v20_params["num_steps"],
            guidance=v20_params["guidance"],
            controlnet_conditioning_scale=v20_params["cn_scale"],
            prompt_template=v20_prompt,
            negative_prompt_override=v20_negative,
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

        # Phase 3
        print("\n" + "=" * 80)
        print("[Phase 3] FASHN VTON v1.5 (same as v18)")
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

    # Comparison grids: v18 vs v20
    print("\n[Phase 4] v18 vs v20 comparison grids...")

    v18_r = [Image.open(V18_DIR / f"realistic_{angle_label(a)}.png") for a in ANGLES_16]
    v18_f = [Image.open(V18_DIR / f"fitted_{angle_label(a)}.png") for a in ANGLES_16]
    v20_r = [base64_to_pil(b) for b in realistic_b64s]
    v20_f = [base64_to_pil(b) for b in fitted_b64s]

    for part, sl, angles_sl, suffix in [
        ("part1", slice(0, 8), ANGLES_16[:8], "0-157.5"),
        ("part2", slice(8, 16), ANGLES_16[8:], "180-337.5"),
    ]:
        grid = create_comparison_grid(
            [v18_r[sl], v20_r[sl], v18_f[sl], v20_f[sl]],
            ["v18 Realistic", "v20 Realistic", "v18 VTON", "v20 VTON"],
            angles_sl,
            f"v18 vs v20 ({suffix} deg)",
        )
        grid.save(OUTPUT_DIR / f"comparison_{part}.png")

    # Cost
    gpu_total = t15 + t3
    print(f"\n  Total GPU: {gpu_total:.1f}s  ${gpu_total * H200_COST_PER_SEC:.4f}")

    report = {
        "test_name": "v20_conservative_tuning",
        "timestamp": timestamp,
        "strategy": "v18 base + negative_prompt fix + standing prompt",
        "changes": {
            "prompt": "added 'standing firmly on the ground with both feet on the floor'",
            "negative_prompt": "added floating/levitating/hovering/cropped/cut off",
            "cn_scale": "0.5 (unchanged)",
            "guidance": "7.5 (unchanged)",
            "steps": "30 (unchanged)",
        },
        "parameters": {"phase15": v20_params, "phase3": {"category": "tops", "num_timesteps": 30, "guidance_scale": 1.5, "seed": 42}},
        "timings": timings,
        "cost": {"gpu_total_sec": round(gpu_total, 2), "gpu_total_usd": round(gpu_total * H200_COST_PER_SEC, 4)},
    }
    (OUTPUT_DIR / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part1.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_part2.png'}'")
    for a in [0, 90, 180]:
        p = OUTPUT_DIR / f"fitted_{angle_label(a)}.png"
        if p.exists(): os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v20 Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r: print(f"\n{json.dumps(r, indent=2)}")
