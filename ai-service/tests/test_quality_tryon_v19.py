#!/usr/bin/env python3
"""
v19 Quality Improvement Test — Prompt & Parameter Optimization

Changes from v18:
  1. Prompt: Remove specific clothing description (VTON replaces it anyway)
     - "wearing casual clothing" instead of "plain gray short-sleeve t-shirt and dark blue jeans"
     - Add "standing on ground, feet touching floor" to fix floating pose
  2. cn_scale: 0.5 → 0.7 (stronger depth adherence)
  3. guidance: 7.5 → 9.0 (stronger prompt adherence)
  4. negative_prompt: Add "floating, levitating, hovering, feet off ground"
  5. steps: 30 → 35 (slightly more quality, ~17% more time)

Test runs Phase 1.5 + Phase 3 only (reuses v18 mesh data to save GPU cost).
Estimated cost: ~$0.28 (Phase 1.5 ~85s + Phase 3 ~100s at $5.40/hr)
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
ROOT = Path(__file__).resolve().parent.parent          # ai-service/
sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT.parent                             # ai-server/

from worker.modal_app import (                         # noqa: E402
    app as modal_app,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

# -- Directories --------------------------------------------------------------
IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
V18_DIR = ROOT / "tests" / "NewTest"              # Reuse v18 mesh renders
OUTPUT_DIR = ROOT / "tests" / "v19_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants ----------------------------------------------------------------
ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]

H200_COST_PER_SEC = 5.40 / 3600


# -- Utility functions --------------------------------------------------------

def load_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def base64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def save_b64_image(b64_str: str, path: str):
    raw = base64.b64decode(b64_str)
    with open(path, "wb") as f:
        f.write(raw)


def angle_label(angle: float) -> str:
    if angle == int(angle):
        return str(int(angle))
    return str(angle)


def create_comparison_grid(
    rows_data: list[list[Image.Image]],
    row_labels: list[str],
    angles: list[float],
    title: str = "",
) -> Image.Image:
    target_h = 256

    def resize_row(images):
        resized = []
        for img in images:
            aspect = img.width / img.height
            new_w = int(target_h * aspect)
            resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))
        return resized

    resized_rows = [resize_row(row) for row in rows_data]
    cols = len(angles)
    rows = len(rows_data)
    label_h, row_label_w, title_h = 28, 160, 45

    max_w = max(max(img.width for img in row) for row in resized_rows)
    cell_w = max_w + 10
    cell_h = target_h + label_h
    canvas_w = row_label_w + cell_w * cols
    canvas_h = title_h + cell_h * rows
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font_title = font_label = font_small = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - text_w) // 2, 12), title, fill=(0, 0, 0), font=font_title)

    for row_idx, (row_images, row_label) in enumerate(zip(resized_rows, row_labels)):
        label_y = title_h + row_idx * cell_h + cell_h // 2
        draw.text((8, label_y), row_label, fill=(0, 0, 0), font=font_label)
        for col_idx, (img, ang) in enumerate(zip(row_images, angles)):
            x_off = row_label_w + col_idx * cell_w + (cell_w - img.width) // 2
            y_off = title_h + row_idx * cell_h + label_h
            canvas.paste(img, (x_off, y_off))
            if row_idx == 0:
                angle_text = f"{angle_label(ang)} deg"
                bbox = draw.textbbox((0, 0), angle_text, font=font_small)
                tw = bbox[2] - bbox[0]
                tx = row_label_w + col_idx * cell_w + (cell_w - tw) // 2
                ty = title_h + row_idx * cell_h + 4
                draw.text((tx, ty), angle_text, fill=(0, 0, 0), font=font_small)

    return canvas


# -- Main test ----------------------------------------------------------------

def main():
    print("=" * 80)
    print("v19 Quality Improvement Test")
    print("Prompt + Parameter Optimization (cn_scale=0.7, guidance=9.0, steps=35)")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Load v18 mesh renders (reuse to save GPU cost) ---------------------
    print("\n[Phase 0] Loading v18 mesh renders + wear image...")

    mesh_b64s = []
    for angle in ANGLES_16:
        mesh_path = V18_DIR / f"mesh_{angle_label(angle)}.png"
        if not mesh_path.exists():
            print(f"  ERROR: Missing mesh render: {mesh_path}")
            return
        mesh_b64s.append(load_image_base64(mesh_path))
    print(f"  Loaded {len(mesh_b64s)} mesh renders from v18")

    # Load wear image
    wear_imgs = sorted(
        list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png"))
    )
    if not wear_imgs:
        print("  ERROR: No wear images found")
        return
    wear_img_path = wear_imgs[0]
    wear_b64 = load_image_base64(wear_img_path)
    print(f"  Wear image: {wear_img_path.name}")

    # Load user image (for reference)
    user_path = V18_DIR / "input_user.png"
    if user_path.exists():
        save_b64_image(load_image_base64(user_path), str(OUTPUT_DIR / "input_user.png"))

    # ---- v19 PROMPT + PARAMETERS -------------------------------------------
    # Key changes:
    # 1. No specific clothing in prompt (VTON will replace anyway)
    # 2. "standing on ground" to fix floating pose
    # 3. Stronger depth control (0.7 vs 0.5)
    # 4. Stronger guidance (9.0 vs 7.5)
    # 5. Better negative prompt

    v19_prompt_template = (
        "A photorealistic full-body photograph of a young Korean woman, "
        "long black hair, {angle_desc}, "
        "wearing casual clothing, "
        "standing naturally on the ground, feet touching the floor, "
        "clean gray studio background, soft natural lighting, "
        "high quality, detailed skin texture, sharp focus, "
        "professional fashion photography, full body visible"
    )

    v19_negative_prompt = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, "
        "floating, levitating, hovering, feet off ground, flying, "
        "cropped, cut off, missing limbs, extra limbs, "
        "bad hands, extra fingers, missing fingers"
    )

    v19_params = {
        "num_steps": 35,
        "guidance": 9.0,
        "controlnet_conditioning_scale": 0.7,
    }

    print(f"\n  v19 Parameters:")
    print(f"    num_steps: {v19_params['num_steps']} (v18: 30)")
    print(f"    guidance: {v19_params['guidance']} (v18: 7.5)")
    print(f"    cn_scale: {v19_params['controlnet_conditioning_scale']} (v18: 0.5)")
    print(f"    prompt: neutral clothing + standing on ground")
    print(f"    negative: + floating/levitating/cropped")

    # ---- Phase 1.5: SDXL + ControlNet Depth (GPU) --------------------------
    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL + ControlNet Depth -> 16 Realistic Images (v19 params)")
    print("=" * 80)

    t_phase15_start = time.time()

    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=mesh_b64s,
            person_image_b64="",  # Not used in current impl
            angles=ANGLES_16,
            num_steps=v19_params["num_steps"],
            guidance=v19_params["guidance"],
            controlnet_conditioning_scale=v19_params["controlnet_conditioning_scale"],
            prompt_template=v19_prompt_template,
            negative_prompt_override=v19_negative_prompt,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}")
            return

        realistic_b64s = realistic_result["realistic_renders_b64"]
        t_phase15 = time.time() - t_phase15_start
        timings["phase15_sdxl_controlnet_sec"] = round(t_phase15, 2)

        print(f"  Generated {len(realistic_b64s)} realistic renders in {t_phase15:.1f}s")
        print(f"  Cost: ${t_phase15 * H200_COST_PER_SEC:.4f}")

        first = base64_to_pil(realistic_b64s[0])
        print(f"  Resolution: {first.size[0]} x {first.size[1]}")

        # Save realistic renders
        for angle, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(angle)}.png"))

        # ---- Phase 3: FASHN VTON v1.5 (GPU) --------------------------------
        print("\n" + "=" * 80)
        print("[Phase 3] FASHN VTON v1.5 -> 16 Fitted Images (maskless)")
        print("=" * 80)

        t_phase3_start = time.time()

        fashn_result = run_fashn_vton_batch.remote(
            persons_b64=realistic_b64s,
            clothing_b64=wear_b64,
            category="tops",
            garment_photo_type="flat-lay",
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
        )

        t_phase3 = time.time() - t_phase3_start
        timings["phase3_fashn_vton_sec"] = round(t_phase3, 2)

        if "error" in fashn_result:
            print(f"  ERROR: {fashn_result['error']}")
            return

        fitted_b64s = fashn_result["results_b64"]
        print(f"  FASHN VTON: {len(fitted_b64s)} fitted images in {t_phase3:.1f}s")
        print(f"  Cost: ${t_phase3 * H200_COST_PER_SEC:.4f}")

    # Save fitted renders
    for angle, b64 in zip(ANGLES_16, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(angle)}.png"))

    # ---- Comparison grids ---------------------------------------------------
    print("\n[Phase 4] Creating comparison grids (v18 vs v19)...")

    # Load v18 results for comparison
    v18_realistic = []
    v18_fitted = []
    for angle in ANGLES_16:
        r_path = V18_DIR / f"realistic_{angle_label(angle)}.png"
        f_path = V18_DIR / f"fitted_{angle_label(angle)}.png"
        v18_realistic.append(Image.open(r_path))
        v18_fitted.append(Image.open(f_path))

    v19_realistic = [base64_to_pil(b64) for b64 in realistic_b64s]
    v19_fitted = [base64_to_pil(b64) for b64 in fitted_b64s]
    mesh_pils = [Image.open(V18_DIR / f"mesh_{angle_label(a)}.png") for a in ANGLES_16]

    # Grid 1: Front angles (0-157.5) — 4 rows: v18 realistic, v19 realistic, v18 fitted, v19 fitted
    grid1 = create_comparison_grid(
        rows_data=[
            v18_realistic[:8], v19_realistic[:8],
            v18_fitted[:8], v19_fitted[:8],
        ],
        row_labels=[
            "v18 Realistic", "v19 Realistic",
            "v18 VTON", "v19 VTON",
        ],
        angles=ANGLES_16[:8],
        title="v18 vs v19 Quality Comparison (0 - 157.5 deg)",
    )
    grid1.save(OUTPUT_DIR / "comparison_v18_v19_part1.png")

    # Grid 2: Back angles (180-337.5)
    grid2 = create_comparison_grid(
        rows_data=[
            v18_realistic[8:], v19_realistic[8:],
            v18_fitted[8:], v19_fitted[8:],
        ],
        row_labels=[
            "v18 Realistic", "v19 Realistic",
            "v18 VTON", "v19 VTON",
        ],
        angles=ANGLES_16[8:],
        title="v18 vs v19 Quality Comparison (180 - 337.5 deg)",
    )
    grid2.save(OUTPUT_DIR / "comparison_v18_v19_part2.png")

    # ---- Cost summary -------------------------------------------------------
    gpu_total = t_phase15 + t_phase3
    gpu_cost = gpu_total * H200_COST_PER_SEC

    print("\n" + "=" * 80)
    print("Cost Summary")
    print("=" * 80)
    print(f"  Phase 1.5 (SDXL ControlNet, v19 params): {t_phase15:7.1f}s  ${t_phase15 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 3   (FASHN VTON v1.5):              {t_phase3:7.1f}s  ${t_phase3 * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 55}")
    print(f"  Total GPU:                                {gpu_total:7.1f}s  ${gpu_cost:.4f}")

    # ---- Test report --------------------------------------------------------
    report = {
        "test_name": "v19_quality_improvement",
        "timestamp": timestamp,
        "changes_from_v18": {
            "prompt": "neutral clothing + standing on ground (vs specific clothing description)",
            "cn_scale": "0.7 (vs 0.5)",
            "guidance": "9.0 (vs 7.5)",
            "steps": "35 (vs 30)",
            "negative_prompt": "added floating/levitating/cropped terms",
        },
        "parameters": {
            "phase15": v19_params,
            "phase3": {
                "category": "tops",
                "garment_photo_type": "flat-lay",
                "num_timesteps": 30,
                "guidance_scale": 1.5,
                "seed": 42,
            },
        },
        "timings": timings,
        "cost": {
            "gpu_total_sec": round(gpu_total, 2),
            "gpu_total_usd": round(gpu_cost, 4),
        },
    }

    report_path = OUTPUT_DIR / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n  Report: {report_path}")
    print(f"  Output: {OUTPUT_DIR}")

    # Open results
    print("\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_v18_v19_part1.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_v18_v19_part2.png'}'")
    for a in [0, 90, 180]:
        p = OUTPUT_DIR / f"fitted_{angle_label(a)}.png"
        if p.exists():
            os.system(f"open '{p}'")

    print("\n" + "=" * 80)
    print("v19 Quality Improvement Test Complete")
    print("=" * 80)
    return report


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n{json.dumps(result, indent=2)}")
