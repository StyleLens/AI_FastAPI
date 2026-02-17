#!/usr/bin/env python3
"""
Test Commercial 16-Angle Pipeline: 100% Commercial-Safe Virtual Fitting

Validates the full commercial pipeline with 16 angles (22.5-degree increments).
All models are Apache 2.0 / MIT licensed.

Pipeline:
  Phase 0:   Load user image + wear image from IMG_Data/
  Phase 1:   SAM 3D Body reconstruction (GPU) -> mesh (vertices, faces)
  Phase 1R:  Render mesh at 16 angles (CPU, local sw_renderer)
  Phase 1.5: SDXL + ControlNet Depth -> 16 realistic images (GPU)
  Phase 3:   FASHN VTON v1.5 -> 16 fitted images (GPU, maskless)
  Phase 4:   Create comparison grids
  Save:      All results to tests/NewTest/

Output:
  tests/NewTest/
    mesh_{angle}.png           - 16 mesh renders
    realistic_{angle}.png      - 16 SDXL ControlNet results
    fitted_{angle}.png         - 16 FASHN VTON results
    comparison_part1.png       - angles 0-157.5 (8 cols, 3 rows)
    comparison_part2.png       - angles 180-337.5 (8 cols, 3 rows)
    input_user.png             - original user photo
    input_wear.png             - original wear image
    test_report.json           - timing and cost data

Commercial models used:
  - SDXL + ControlNet Depth (OpenRAIL++)
  - FASHN VTON v1.5 (Apache 2.0)
  - SAM 3D Body
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

from core.sw_renderer import render_mesh               # noqa: E402
from worker.modal_app import (                         # noqa: E402
    app as modal_app,
    run_light_models,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
)

# -- Directories --------------------------------------------------------------
IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
OUTPUT_DIR = ROOT / "tests" / "NewTest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants ----------------------------------------------------------------
ANGLES_16 = [
    0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
    180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5,
]

# GPU cost: H200 $5.40/hr
H200_COST_PER_SEC = 5.40 / 3600  # ~$0.0015/sec

MESH_RESOLUTION = 768


# -- Utility functions --------------------------------------------------------

def load_image_base64(path: Path) -> str:
    """Read image file and encode as base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def base64_to_pil(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def save_b64_image(b64_str: str, path: str):
    """Save base64-encoded image bytes to a file."""
    raw = base64.b64decode(b64_str)
    with open(path, "wb") as f:
        f.write(raw)


def b64_to_ndarray(data: dict) -> np.ndarray:
    """Deserialize numpy array from base64 with metadata (dtype, shape)."""
    raw = base64.b64decode(data["data"])
    arr = np.frombuffer(raw, dtype=data["dtype"])
    return arr.reshape(data["shape"])


def angle_label(angle: float) -> str:
    """Format angle for filenames: 0 -> '0', 22.5 -> '22.5'."""
    if angle == int(angle):
        return str(int(angle))
    return str(angle)


def create_comparison_grid(
    rows_data: list[list[Image.Image]],
    row_labels: list[str],
    angles: list[float],
    title: str = "16-Angle Commercial Pipeline",
) -> Image.Image:
    """
    Create N-row x M-col comparison grid.

    Args:
        rows_data: list of rows, each row is a list of PIL Images
        row_labels: label for each row (e.g. "Raw Mesh", "SDXL ControlNet")
        angles: list of angle values for column headers
        title: grid title text
    """
    target_h = 256

    def resize_row(images: list[Image.Image]) -> list[Image.Image]:
        resized = []
        for img in images:
            aspect = img.width / img.height
            new_w = int(target_h * aspect)
            resized.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))
        return resized

    resized_rows = [resize_row(row) for row in rows_data]

    cols = len(angles)
    rows = len(rows_data)
    label_h = 28
    row_label_w = 140
    title_h = 45

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
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_w = bbox[2] - bbox[0]
    draw.text(((canvas_w - text_w) // 2, 12), title, fill=(0, 0, 0), font=font_title)

    for row_idx, (row_images, row_label) in enumerate(zip(resized_rows, row_labels)):
        # Row label
        label_y = title_h + row_idx * cell_h + cell_h // 2
        draw.text((8, label_y), row_label, fill=(0, 0, 0), font=font_label)

        for col_idx, (img, ang) in enumerate(zip(row_images, angles)):
            x_off = row_label_w + col_idx * cell_w + (cell_w - img.width) // 2
            y_off = title_h + row_idx * cell_h + label_h
            canvas.paste(img, (x_off, y_off))

            # Column header (angle) on first row only
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
    print("Commercial 16-Angle Pipeline Test")
    print("100% Commercial-Safe: SDXL ControlNet + FASHN VTON v1.5")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}

    # ---- Phase 0: Load test images ------------------------------------------
    print("\n[Phase 0] Loading test images...")

    user_imgs = sorted(
        list(USER_IMG_DIR.glob("*.jpg")) + list(USER_IMG_DIR.glob("*.png"))
    )
    if not user_imgs:
        print("  ERROR: No user images found in IMG_Data/User_IMG/")
        return
    user_img_path = user_imgs[0]
    print(f"  User image : {user_img_path.name}")
    user_b64 = load_image_base64(user_img_path)

    wear_imgs = sorted(
        list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png"))
    )
    if not wear_imgs:
        print("  ERROR: No wear images found in IMG_Data/wear/")
        return
    wear_img_path = wear_imgs[0]
    print(f"  Wear image : {wear_img_path.name}")
    wear_b64 = load_image_base64(wear_img_path)

    # Save input copies
    save_b64_image(user_b64, str(OUTPUT_DIR / "input_user.png"))
    save_b64_image(wear_b64, str(OUTPUT_DIR / "input_wear.png"))

    # ---- Phase 1: SAM 3D Body Reconstruction (GPU) --------------------------
    print("\n" + "=" * 80)
    print("[Phase 1] SAM 3D Body Reconstruction (GPU)")
    print("=" * 80)

    t_phase1_start = time.time()

    with modal_app.run():
        mesh_result = run_light_models.remote(
            task="reconstruct_3d",
            image_b64=user_b64,
        )

    t_phase1 = time.time() - t_phase1_start
    timings["phase1_reconstruct_sec"] = round(t_phase1, 2)

    if "error" in mesh_result:
        print(f"  ERROR: {mesh_result['error']}")
        return

    vertices = b64_to_ndarray(mesh_result["vertices"])
    faces = b64_to_ndarray(mesh_result["faces"])
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")
    print(f"  Time: {t_phase1:.1f}s | Cost: ${t_phase1 * H200_COST_PER_SEC:.4f}")

    # ---- Phase 1-Render: Mesh rendering at 16 angles (CPU, local) -----------
    print("\n" + "=" * 80)
    print("[Phase 1-Render] Rendering mesh at 16 angles (CPU, local)")
    print("=" * 80)

    t_render_start = time.time()

    render_b64s = []
    for angle in ANGLES_16:
        rendered = render_mesh(
            vertices, faces,
            angle_deg=angle,
            resolution=MESH_RESOLUTION,
        )
        # render_mesh returns BGR numpy array; encode to JPEG base64
        ok, buf = cv2.imencode(".jpg", rendered)
        if not ok:
            print(f"  ERROR: Failed to encode mesh render at {angle} deg")
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        render_b64s.append(b64)

    t_render = time.time() - t_render_start
    timings["phase1_render_sec"] = round(t_render, 2)

    print(f"  Rendered {len(render_b64s)} mesh images at {MESH_RESOLUTION}x{MESH_RESOLUTION}")
    print(f"  Time: {t_render:.1f}s (CPU, no GPU cost)")

    # Save mesh renders
    for angle, b64 in zip(ANGLES_16, render_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"mesh_{angle_label(angle)}.png"))
    print("  Mesh renders saved")

    # ---- Phase 1.5: SDXL + ControlNet Depth (GPU) ---------------------------
    print("\n" + "=" * 80)
    print("[Phase 1.5] SDXL + ControlNet Depth -> 16 Realistic Images (GPU)")
    print("=" * 80)
    print(f"  num_steps=30, guidance=7.5, controlnet_conditioning_scale=0.5")

    t_phase15_start = time.time()

    with modal_app.run():
        realistic_result = run_mesh_to_realistic.remote(
            mesh_renders_b64=render_b64s,
            person_image_b64=user_b64,
            angles=ANGLES_16,
            num_steps=30,
            guidance=7.5,
            controlnet_conditioning_scale=0.5,
        )

        if "error" in realistic_result:
            print(f"  ERROR: {realistic_result['error']}")
            return

        realistic_b64s = realistic_result["realistic_renders_b64"]

        t_phase15 = time.time() - t_phase15_start
        timings["phase15_sdxl_controlnet_sec"] = round(t_phase15, 2)

        print(f"  Generated {len(realistic_b64s)} realistic renders in {t_phase15:.1f}s")
        print(f"  Cost: ${t_phase15 * H200_COST_PER_SEC:.4f}")

        # Check realistic resolution
        first_realistic = base64_to_pil(realistic_b64s[0])
        r_w, r_h = first_realistic.size
        print(f"  Realistic render resolution: {r_w} x {r_h}")

        # Save realistic renders
        for angle, b64 in zip(ANGLES_16, realistic_b64s):
            save_b64_image(b64, str(OUTPUT_DIR / f"realistic_{angle_label(angle)}.png"))
        print("  Realistic renders saved")

        # ---- Phase 3: FASHN VTON v1.5 (GPU) ---------------------------------
        print("\n" + "=" * 80)
        print("[Phase 3] FASHN VTON v1.5 -> 16 Fitted Images (GPU, maskless)")
        print("=" * 80)
        print(f"  category=tops, num_timesteps=30, guidance_scale=1.5")

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
        print(f"  FASHN VTON generated {len(fitted_b64s)} fitted images in {t_phase3:.1f}s")
        print(f"  Cost: ${t_phase3 * H200_COST_PER_SEC:.4f}")

    # -- Outside modal context: save results and build grids ------------------

    # Check fitted resolution
    first_fitted = base64_to_pil(fitted_b64s[0])
    f_w, f_h = first_fitted.size
    print(f"  Fitted render resolution: {f_w} x {f_h}")

    # Save fitted renders
    for angle, b64 in zip(ANGLES_16, fitted_b64s):
        save_b64_image(b64, str(OUTPUT_DIR / f"fitted_{angle_label(angle)}.png"))
    print("  Fitted results saved")

    # ---- Phase 4: Comparison grids ------------------------------------------
    print("\n" + "=" * 80)
    print("[Phase 4] Creating comparison grids")
    print("=" * 80)

    mesh_pils = [base64_to_pil(b64) for b64 in render_b64s]
    realistic_pils = [base64_to_pil(b64) for b64 in realistic_b64s]
    fitted_pils = [base64_to_pil(b64) for b64 in fitted_b64s]

    # Part 1: angles 0 - 157.5 (first 8 angles)
    grid1 = create_comparison_grid(
        rows_data=[mesh_pils[:8], realistic_pils[:8], fitted_pils[:8]],
        row_labels=["Raw Mesh", "SDXL ControlNet", "FASHN VTON"],
        angles=ANGLES_16[:8],
        title="Commercial 16-Angle Pipeline (0 - 157.5 deg)",
    )
    grid1_path = OUTPUT_DIR / "comparison_part1.png"
    grid1.save(grid1_path)
    print(f"  Grid Part 1 saved: {grid1_path}")

    # Part 2: angles 180 - 337.5 (last 8 angles)
    grid2 = create_comparison_grid(
        rows_data=[mesh_pils[8:], realistic_pils[8:], fitted_pils[8:]],
        row_labels=["Raw Mesh", "SDXL ControlNet", "FASHN VTON"],
        angles=ANGLES_16[8:],
        title="Commercial 16-Angle Pipeline (180 - 337.5 deg)",
    )
    grid2_path = OUTPUT_DIR / "comparison_part2.png"
    grid2.save(grid2_path)
    print(f"  Grid Part 2 saved: {grid2_path}")

    # ---- Cost summary -------------------------------------------------------
    gpu_total_sec = t_phase1 + t_phase15 + t_phase3
    gpu_total_usd = gpu_total_sec * H200_COST_PER_SEC

    print("\n" + "=" * 80)
    print("Cost Summary (H200: $5.40/hr)")
    print("=" * 80)
    print(f"  Phase 1    (SAM 3D Body, reconstruct):       {t_phase1:7.1f}s  ${t_phase1 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 1R   (Mesh render, 16 angles, CPU):    {t_render:7.1f}s  $0.0000")
    print(f"  Phase 1.5  (SDXL + ControlNet, 16 angles):   {t_phase15:7.1f}s  ${t_phase15 * H200_COST_PER_SEC:.4f}")
    print(f"  Phase 3    (FASHN VTON, 16 angles):           {t_phase3:7.1f}s  ${t_phase3 * H200_COST_PER_SEC:.4f}")
    print(f"  {'─' * 60}")
    print(f"  Total GPU time:                              {gpu_total_sec:7.1f}s  ${gpu_total_usd:.4f}")

    # ---- Test report JSON ---------------------------------------------------
    report = {
        "test_name": "commercial_16angle",
        "timestamp": timestamp,
        "angles": 16,
        "angle_values": ANGLES_16,
        "models": {
            "controlnet": "SDXL + ControlNet Depth (OpenRAIL++)",
            "tryon": "FASHN VTON v1.5 (Apache 2.0)",
            "mesh": "SAM 3D Body",
        },
        "timings": {
            "phase1_reconstruct_sec": timings["phase1_reconstruct_sec"],
            "phase1_render_sec": timings["phase1_render_sec"],
            "phase15_sdxl_controlnet_sec": timings["phase15_sdxl_controlnet_sec"],
            "phase3_fashn_vton_sec": timings["phase3_fashn_vton_sec"],
        },
        "cost": {
            "gpu_total_sec": round(gpu_total_sec, 2),
            "gpu_total_usd": round(gpu_total_usd, 4),
        },
        "resolutions": {
            "mesh": f"{MESH_RESOLUTION}x{MESH_RESOLUTION}",
            "realistic": f"{r_w}x{r_h}",
            "fitted": f"{f_w}x{f_h}",
        },
        "parameters": {
            "phase15": {
                "num_steps": 30,
                "guidance": 7.5,
                "controlnet_conditioning_scale": 0.5,
            },
            "phase3": {
                "category": "tops",
                "garment_photo_type": "flat-lay",
                "num_timesteps": 30,
                "guidance_scale": 1.5,
                "seed": 42,
            },
        },
        "input_images": {
            "user": user_img_path.name,
            "wear": wear_img_path.name,
        },
        "output_dir": str(OUTPUT_DIR),
    }

    report_path = OUTPUT_DIR / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Test report saved: {report_path}")

    # ---- Final summary ------------------------------------------------------
    output_files = list(OUTPUT_DIR.iterdir())

    print("\n" + "=" * 80)
    print("COMMERCIAL 16-ANGLE PIPELINE TEST COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total files: {len(output_files)}")
    print()
    print(f"  Mesh renders (16):      mesh_{{0,22.5,...,337.5}}.png")
    print(f"  Realistic renders (16): realistic_{{0,22.5,...,337.5}}.png")
    print(f"  Fitted results (16):    fitted_{{0,22.5,...,337.5}}.png")
    print(f"  Comparison Part 1:      {grid1_path.name}")
    print(f"  Comparison Part 2:      {grid2_path.name}")
    print(f"  Test report:            {report_path.name}")

    print("\n  Opening results for visual inspection...")
    os.system(f"open '{grid1_path}'")
    os.system(f"open '{grid2_path}'")

    for angle in [0, 90, 180]:
        p = OUTPUT_DIR / f"fitted_{angle_label(angle)}.png"
        if p.exists():
            os.system(f"open '{p}'")

    print()
    print("Commercial 16-angle pipeline validation complete.")
    print("=" * 80)

    return report


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\nResult JSON:\n{json.dumps(result, indent=2)}")
