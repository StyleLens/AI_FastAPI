#!/usr/bin/env python3
"""
v36 Quality Test — Cost-Optimized (No FLUX Refinement)

v35 achievements:
  - Dynamic cn_scale per bust cup (A=0.55, C=0.65, E=0.75)
  - Amplified mesh deformation (z_coeff=0.085, x_coeff=0.028)
  - Pipeline: SDXL→FLUX→VTON→FaceSwap, ~973원/test (over budget)

v36 optimization (CRITICAL for meeting 700원 budget target):
  1. **FLUX refine step REMOVED** — saves ~120s GPU time per test
     - v35: SDXL→FLUX→VTON→FaceSwap (~321s GPU, 973원)
     - v36: SDXL→VTON→FaceSwap (~201s GPU, target ~608円)
     WHY: FLUX img2img improved texture quality but added ~40s per cup.
           The bust size differentiation is primarily driven by mesh deformation
           + dynamic cn_scale, NOT by FLUX refinement.
  2. KEEP dynamic cn_scale: A=0.55, C=0.65, E=0.75 (proven effective in v35)
  3. KEEP amplified mesh deformation: z_coeff=0.085, x_coeff=0.028
  4. KEEP 4 hero angles: 0°, 90°, 180°, 270° (front/side/back/opposite)
  5. Face swap from v33 (InsightFace antelopev2)

Pipeline (4 phases, reduced from 5):
  Phase 1R:   CPU Mesh Rendering with bust_cup_scale (arms-down + legs-closed)
  Phase 1.5A: SDXL + ControlNet Depth → pose-accurate base (DYNAMIC cn_scale per cup)
  Phase 3:    FASHN VTON → virtual try-on (DIRECTLY from SDXL, no FLUX)
  Phase 4:    InsightFace Face Swap → face consistency

Expected cost savings:
  - v35: ~321s GPU × $0.0015/s = $0.48 (973원)
  - v36: ~201s GPU × $0.0015/s = $0.30 (608원) — 38% reduction

Budget target: ≤700원 (~$0.52) per test (3 cups × 4 angles = 12 images)
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
from core.body_analyzer import bust_cup_to_body_scale, bust_cup_to_sdxl_description
from worker.modal_app import (
    app as modal_app,
    run_mesh_to_realistic,
    run_fashn_vton_batch,
    run_face_swap,
)

IMG_DATA = PROJECT_ROOT / "IMG_Data"
WEAR_DIR = IMG_DATA / "wear"
FACE_DIR = IMG_DATA / "User_New" / "User_Face"
V22_DIR = ROOT / "tests" / "v22_newuser"
OUTPUT_DIR = ROOT / "tests" / "v36_no_flux"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test configuration
USER_HEIGHT = 165.0  # cm
USER_WEIGHT = 58.0   # kg
USER_GENDER = "female"

# Cup size test configurations — same dynamic cn_scale as v35
CUP_CONFIGS = {
    "A": {"cup": "A", "scale": 0.6, "cn_scale": 0.55},
    "C": {"cup": "C", "scale": 1.4, "cn_scale": 0.65},
    "E": {"cup": "E", "scale": 2.2, "cn_scale": 0.75},
}

# 4 hero angles: front, side, back, opposite side (cost optimized)
HERO_ANGLES = [0, 90, 180, 270]

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
    label_h, row_label_w, title_h = 28, 240, 45
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
                at = f"{angle_label(ang)}\u00b0" if isinstance(ang, (int, float)) else str(ang)
                abbox = draw.textbbox((0, 0), at, font=fs)
                draw.text((row_label_w + ci * cell_w + (cell_w - abbox[2] + abbox[0]) // 2,
                           title_h + 4), at, fill=(0, 0, 0), font=fs)
    return canvas


def main():
    print("=" * 80)
    print("v36 — Cost-Optimized (No FLUX Refinement)")
    print("  Testing A (cn=0.55), C (cn=0.65), E (cn=0.75)")
    print("  4 hero angles per cup, SDXL→VTON→FaceSwap (skip FLUX)")
    print("  Budget target: ≤700円 (~$0.52 per test)")
    print("=" * 80)

    timestamp = datetime.now(timezone.utc).isoformat()
    timings = {}
    cup_results = {}

    # ── Phase 0: Load data ──
    print("\n[Phase 0] Loading mesh data + face reference + garment...")
    mesh_data = np.load(V22_DIR / "mesh_data.npz")
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")

    wear_imgs = sorted(list(WEAR_DIR.glob("*.jpg")) + list(WEAR_DIR.glob("*.png")))
    wear_pil, wear_b64 = load_image_fixed(wear_imgs[0])
    print(f"  Wear: {wear_imgs[0].name} ({wear_pil.size})")

    # Load face reference
    face_imgs = sorted(list(FACE_DIR.glob("*.jpg")) + list(FACE_DIR.glob("*.png")))
    if not face_imgs:
        print("  ERROR: No face reference images found!")
        face_path = V22_DIR / "input_face.jpg"
        if face_path.exists():
            face_pil, face_b64 = load_image_fixed(face_path)
            print(f"  Fallback face: {face_path.name} ({face_pil.size})")
        else:
            print("  FATAL: No face reference available. Cannot proceed.")
            return None
    else:
        face_pil, face_b64 = load_image_fixed(face_imgs[0])
        print(f"  Face ref: {face_imgs[0].name} ({face_pil.size})")

    # Save face reference to output
    face_pil.save(OUTPUT_DIR / "face_reference.png")

    # ── Phase 1R: CPU Mesh Rendering for all cup sizes ──
    print("\n[Phase 1R] CPU Mesh Rendering with bust volume adjustment...")
    print("  (z_coeff=0.085, x_coeff=0.028 — hardcoded in sw_renderer.py)")
    t1r_start = time.time()

    all_cup_renders = {}  # {cup: {angle: b64}}

    for cup_name, config in CUP_CONFIGS.items():
        cup = config["cup"]
        cup_scale = config["scale"]
        cn_scale = config["cn_scale"]

        print(f"\n  Rendering cup {cup} (scale={cup_scale}, cn_scale={cn_scale})...")

        # Create output directory for this cup
        cup_dir = OUTPUT_DIR / f"cup_{cup}"
        cup_dir.mkdir(exist_ok=True)

        # Get body scale and description from body_analyzer
        body_scale = bust_cup_to_body_scale(cup, USER_HEIGHT, USER_WEIGHT, USER_GENDER)
        body_desc = bust_cup_to_sdxl_description(cup, USER_HEIGHT, USER_WEIGHT, USER_GENDER)
        print(f"    Body description: '{body_desc}'")

        cup_renders = {}
        for angle in HERO_ANGLES:
            lbl = angle_label(angle)
            rendered = render_mesh(
                vertices, faces,
                angle_deg=angle,
                resolution=MESH_RESOLUTION,
                straighten=True,
                ground_plane=True,
                fold_arms=True,
                close_legs=True,
                body_scale=body_scale,  # From bust_cup_to_body_scale
                bust_cup_scale=cup_scale,  # Direct bust scaling
                bust_band_factor=None,  # Auto from body_scale
                gender=USER_GENDER,
            )
            ok, buf = cv2.imencode(".png", rendered)
            render_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            cup_renders[angle] = render_b64
            cv2.imwrite(str(cup_dir / f"mesh_{lbl}.png"), rendered)

        all_cup_renders[cup] = cup_renders
        cup_results[cup] = {
            "body_description": body_desc,
            "body_scale": body_scale,
            "cn_scale": cn_scale,
        }

    t1r = time.time() - t1r_start
    timings["phase1r_sec"] = round(t1r, 2)
    print(f"\n  Rendered {len(CUP_CONFIGS)} cups x {len(HERO_ANGLES)} angles in {t1r:.1f}s")

    # ── GPU Pipeline (single Modal session) ──
    print("\n" + "=" * 80)
    print("[GPU] Starting 3-phase pipeline for all cup sizes (NO FLUX)")
    print("  KEY: Dynamic cn_scale per cup (A=0.55, C=0.65, E=0.75)")
    print("  COST OPTIMIZATION: Skip FLUX refine to meet 700円 budget")
    print("=" * 80)

    t_gpu_start = time.time()

    with modal_app.run():
        for cup_name, config in CUP_CONFIGS.items():
            cup = config["cup"]
            cn_scale = config["cn_scale"]
            cup_dir = OUTPUT_DIR / f"cup_{cup}"

            print(f"\n{'='*60}")
            print(f"  Processing cup {cup} (cn_scale={cn_scale})")
            print(f"{'='*60}")

            # Get renders and body description for this cup
            cup_renders = all_cup_renders[cup]
            render_b64s = [cup_renders[a] for a in HERO_ANGLES]
            body_desc = cup_results[cup]["body_description"]

            # Body-aware prompts with explicit bust description
            sdxl_prompt = (
                "RAW photo, an extremely detailed real photograph of a young Korean woman, "
                "realistic skin pores, natural skin texture, subtle skin imperfections, "
                f"{body_desc}, "  # Dynamic body description based on cup size
                "long straight dark brown hair, {{angle_desc}}, "
                "wearing a plain light gray crewneck t-shirt and dark blue fitted jeans, "
                "white low-top sneakers on both feet, "
                "standing upright with perfect posture on a flat gray floor, "
                "head directly above shoulders, chin level, straight vertical neck, "
                "relaxed natural arms hanging at sides, hands beside thighs, "
                "legs together in natural standing position, feet flat on ground, "
                "shot with Fujifilm XT4, 85mm portrait lens, film grain, "
                "clean neutral light gray studio background, soft natural lighting, "
                "realistic body proportions matching the depth silhouette exactly, "
                "natural calm expression, "
                "full body visible from head to toe, 8k uhd"
            )
            sdxl_negative = (
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
                "cropped, cut off, missing limbs, extra limbs, missing feet, "
                "extremely thin, anorexic, bodybuilder, obese, "
                "flat chest, " if cup in ("C", "E") else ""
                "nsfw, nude, revealing, "
                "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "oversaturated, overexposed, "
                "shorts, skirt, sandals, bare feet, high heels, boots, "
                "different outfits, multiple people, split image"
            )

            # Phase 1.5A: SDXL ControlNet — DYNAMIC cn_scale per cup
            print(f"\n[Phase 1.5A] SDXL + ControlNet Depth (cup {cup}, cn_scale={cn_scale})")
            t15a_start = time.time()
            sdxl_result = run_mesh_to_realistic.remote(
                mesh_renders_b64=render_b64s,
                person_image_b64="",
                angles=HERO_ANGLES,
                num_steps=30,
                guidance=6.5,
                controlnet_conditioning_scale=cn_scale,  # DYNAMIC per cup
                prompt_template=sdxl_prompt,
                negative_prompt_override=sdxl_negative,
            )

            if "error" in sdxl_result:
                print(f"  ERROR: {sdxl_result['error']}")
                return
            sdxl_b64s = sdxl_result["realistic_renders_b64"]
            t15a = time.time() - t15a_start
            cup_results[cup]["phase15a_sdxl_sec"] = round(t15a, 2)
            print(f"  SDXL: {len(sdxl_b64s)} images in {t15a:.1f}s")

            for a, b64 in zip(HERO_ANGLES, sdxl_b64s):
                save_b64_image(b64, str(cup_dir / f"sdxl_{angle_label(a)}.png"))

            # Phase 1.5B: FLUX — SKIPPED in v36 for cost optimization
            print(f"\n[Phase 1.5B] FLUX.2-klein-4B — SKIPPED (cost optimization)")
            print("  Going directly from SDXL to VTON")
            realistic_b64s = sdxl_b64s  # Use SDXL output directly
            cup_results[cup]["phase15b_flux_sec"] = 0  # Not used

            # Phase 3: FASHN VTON (DIRECTLY from SDXL)
            print(f"\n[Phase 3] FASHN VTON (cup {cup}, from SDXL output)")
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
            cup_results[cup]["phase3_sec"] = round(t3, 2)

            if "error" in fashn_result:
                print(f"  VTON ERROR: {fashn_result['error']}")
                return
            fitted_b64s = fashn_result["results_b64"]
            print(f"  VTON: {len(fitted_b64s)} images in {t3:.1f}s")

            for a, b64 in zip(HERO_ANGLES, fitted_b64s):
                save_b64_image(b64, str(cup_dir / f"fitted_before_face_{angle_label(a)}.png"))

            # Phase 4: Face Swap
            print(f"\n[Phase 4] InsightFace Face Swap (cup {cup})")
            t4_start = time.time()
            swap_result = run_face_swap.remote(
                images_b64=fitted_b64s,
                face_reference_b64=face_b64,
                angles=HERO_ANGLES,
                blend_radius=25,
                face_scale=1.0,
            )
            t4 = time.time() - t4_start
            cup_results[cup]["phase4_face_swap_sec"] = round(t4, 2)

            if "error" in swap_result:
                print(f"  FACE SWAP ERROR: {swap_result['error'][:500]}")
                final_b64s = fitted_b64s
            else:
                final_b64s = swap_result["swapped_b64"]
                face_detected = swap_result.get("face_detected", [])
                n_swapped = sum(1 for d in face_detected if d)
                n_skipped = len(face_detected) - n_swapped
                print(f"  Face Swap: {n_swapped} swapped, {n_skipped} skipped in {t4:.1f}s")

            for a, b64 in zip(HERO_ANGLES, final_b64s):
                save_b64_image(b64, str(cup_dir / f"fitted_{angle_label(a)}.png"))

            # Store results for comparison grids
            cup_results[cup]["sdxl_b64s"] = sdxl_b64s
            cup_results[cup]["fitted_b64s"] = fitted_b64s
            cup_results[cup]["final_b64s"] = final_b64s

    t_gpu_total = time.time() - t_gpu_start

    # ── Comparison Grids ──
    print("\n[Phase 5] Creating comparison grids...")

    # 1. Cup Size Comparison (per angle) — 3 rows (A/C/E) x 4 columns (angles)
    cup_names = list(CUP_CONFIGS.keys())
    cup_final_rows = []
    for cup in cup_names:
        finals = [base64_to_pil(b) for b in cup_results[cup]["final_b64s"]]
        cup_final_rows.append(finals)

    cup_comparison = create_comparison_grid(
        cup_final_rows,
        [f"Cup {cup} (cn={CUP_CONFIGS[cup]['cn_scale']})" for cup in cup_names],
        HERO_ANGLES,
        "v36 — Bust Cup Size Comparison (No FLUX) with Dynamic cn_scale (A vs C vs E)",
    )
    cup_comparison.save(OUTPUT_DIR / "comparison_cup_sizes.png")

    # 2. Pipeline Comparison (per cup) — 4 rows (Depth/SDXL/VTON/Final) x 4 columns
    for cup in cup_names:
        cup_dir = OUTPUT_DIR / f"cup_{cup}"
        cn_scale = CUP_CONFIGS[cup]["cn_scale"]
        depth_imgs = [Image.open(cup_dir / f"mesh_{angle_label(a)}.png") for a in HERO_ANGLES]
        sdxl_imgs = [base64_to_pil(b) for b in cup_results[cup]["sdxl_b64s"]]
        vton_imgs = [base64_to_pil(b) for b in cup_results[cup]["fitted_b64s"]]
        final_imgs = [base64_to_pil(b) for b in cup_results[cup]["final_b64s"]]

        pipeline_grid = create_comparison_grid(
            [depth_imgs, sdxl_imgs, vton_imgs, final_imgs],
            ["Depth Map", f"SDXL (cn={cn_scale})", "VTON", "Final+Face"],
            HERO_ANGLES,
            f"v36 — Cup {cup} Pipeline (No FLUX, cn_scale={cn_scale}, {cup_results[cup]['body_description']})",
        )
        pipeline_grid.save(cup_dir / "comparison_pipeline.png")

    # 3. Front View Focus (0 deg) — Compare A/C/E side-by-side
    front_idx = HERO_ANGLES.index(0)
    front_imgs = [cup_final_rows[i][front_idx] for i in range(len(cup_names))]
    front_grid = create_comparison_grid(
        [front_imgs],
        ["Front View (0 deg)"],
        [f"Cup {c} (cn={CUP_CONFIGS[c]['cn_scale']})" for c in cup_names],
        "v36 — Front View Bust Comparison (No FLUX, Dynamic cn_scale)",
    )
    front_grid.save(OUTPUT_DIR / "comparison_front_bust.png")

    # 4. Side View Focus (90 deg) — Compare A/C/E side-by-side
    side_idx = HERO_ANGLES.index(90)
    side_imgs = [cup_final_rows[i][side_idx] for i in range(len(cup_names))]
    side_grid = create_comparison_grid(
        [side_imgs],
        ["Side View (90 deg)"],
        [f"Cup {c} (cn={CUP_CONFIGS[c]['cn_scale']})" for c in cup_names],
        "v36 — Side View Bust Comparison (No FLUX, Dynamic cn_scale)",
    )
    side_grid.save(OUTPUT_DIR / "comparison_side_bust.png")

    # 5. Depth Map Comparison — Show mesh deformation differences across cups
    depth_rows = []
    for cup in cup_names:
        cup_dir = OUTPUT_DIR / f"cup_{cup}"
        depth_imgs = [Image.open(cup_dir / f"mesh_{angle_label(a)}.png") for a in HERO_ANGLES]
        depth_rows.append(depth_imgs)

    depth_grid = create_comparison_grid(
        depth_rows,
        [f"Cup {c} (scale={CUP_CONFIGS[c]['scale']})" for c in cup_names],
        HERO_ANGLES,
        "v36 — Depth Map Deformation Comparison (z_coeff=0.085, x_coeff=0.028)",
    )
    depth_grid.save(OUTPUT_DIR / "comparison_depth_maps.png")

    # ── Cost Report ──
    total_gpu = 0
    for cup in cup_names:
        cup_gpu = (
            cup_results[cup].get("phase15a_sdxl_sec", 0) +
            cup_results[cup].get("phase15b_flux_sec", 0) +
            cup_results[cup].get("phase3_sec", 0) +
            cup_results[cup].get("phase4_face_swap_sec", 0)
        )
        total_gpu += cup_gpu

    cost_usd = total_gpu * H200_COST_PER_SEC
    cost_krw = cost_usd * 1350

    print(f"\n{'='*60}")
    print(f"  COST BREAKDOWN (v36 — No FLUX)")
    print(f"{'='*60}")
    for cup in cup_names:
        cn_scale = CUP_CONFIGS[cup]["cn_scale"]
        cup_gpu = (
            cup_results[cup].get("phase15a_sdxl_sec", 0) +
            cup_results[cup].get("phase15b_flux_sec", 0) +
            cup_results[cup].get("phase3_sec", 0) +
            cup_results[cup].get("phase4_face_swap_sec", 0)
        )
        print(f"  Cup {cup:2s} (cn={cn_scale}): {cup_gpu:6.1f}s  ${cup_gpu * H200_COST_PER_SEC:.4f}")
    print(f"  {'─'*40}")
    print(f"  {'Total GPU':20s}: {total_gpu:6.1f}s  ${cost_usd:.4f} (approx {cost_krw:.0f} won)")
    print(f"  {'Budget':20s}: {'PASS' if cost_krw <= 700 else 'OVER'} (target <= 700 won)")

    # Compare with v35
    v35_estimated_gpu = 321  # From MEMORY.md
    v35_cost_krw = v35_estimated_gpu * H200_COST_PER_SEC * 1350
    savings_sec = v35_estimated_gpu - total_gpu
    savings_krw = v35_cost_krw - cost_krw
    savings_pct = (savings_sec / v35_estimated_gpu * 100) if v35_estimated_gpu > 0 else 0

    print(f"\n  {'='*40}")
    print(f"  COMPARISON vs v35 (with FLUX)")
    print(f"  {'='*40}")
    print(f"  {'v35 (FLUX)':20s}: ~{v35_estimated_gpu:3.0f}s  ~{v35_cost_krw:.0f} won")
    print(f"  {'v36 (No FLUX)':20s}: {total_gpu:6.1f}s  {cost_krw:.0f} won")
    print(f"  {'Savings':20s}: {savings_sec:6.1f}s  {savings_krw:.0f} won ({savings_pct:.1f}% reduction)")

    # ── Test Report ──
    report = {
        "test_name": "v36_no_flux",
        "timestamp": timestamp,
        "strategy": "Cost-optimized: Skip FLUX refine, keep dynamic cn_scale + mesh deformation",
        "key_changes": [
            "NEW: FLUX refinement step REMOVED (saves ~120s GPU time)",
            "NEW: Pipeline reduced from 5 phases to 4 (SDXL→VTON→FaceSwap)",
            "KEEP: Dynamic cn_scale — A=0.55, C=0.65, E=0.75 (effective from v35)",
            "KEEP: Amplified mesh deformation — z_coeff=0.085, x_coeff=0.028",
            "KEEP: 4 hero angles (0/90/180/270) for cost efficiency",
            "KEEP: InsightFace face swap from v33",
            "TARGET: Budget ≤700円 (~$0.52 per test)",
        ],
        "cost_optimization_rationale": [
            "FLUX img2img adds ~40s per cup (~120s total) but only improves texture quality",
            "Bust size differentiation is driven by mesh deformation + dynamic cn_scale",
            "FLUX does NOT affect body shape or bust size preservation through VTON",
            "38% cost reduction by removing FLUX brings test under 700円 budget",
        ],
        "user_profile": {
            "height_cm": USER_HEIGHT,
            "weight_kg": USER_WEIGHT,
            "gender": USER_GENDER,
            "bmi": round(USER_WEIGHT / ((USER_HEIGHT / 100) ** 2), 1),
        },
        "cup_configurations": {
            cup: {
                "cup_scale": config["scale"],
                "cn_scale": config["cn_scale"],
                "body_description": cup_results[cup]["body_description"],
                "body_scale": cup_results[cup]["body_scale"],
            }
            for cup, config in CUP_CONFIGS.items()
        },
        "parameters": {
            "renderer": {
                "straighten": True,
                "ground_plane": True,
                "fold_arms": True,
                "close_legs": True,
                "bust_cup_scale": "per cup config (A=0.6, C=1.4, E=2.2)",
                "bust_band_factor": None,
                "gender": USER_GENDER,
                "z_coeff": 0.085,
                "x_coeff": 0.028,
            },
            "phase15a_sdxl": {
                "num_steps": 30,
                "guidance": 6.5,
                "cn_scale": "DYNAMIC per cup (A=0.55, C=0.65, E=0.75)",
            },
            "phase15b_flux": "SKIPPED (cost optimization)",
            "phase3": {
                "category": "tops",
                "garment_photo_type": "model",
                "num_timesteps": 30,
                "guidance_scale": 1.5,
            },
            "phase4_face_swap": {
                "model": "InsightFace antelopev2",
                "blend_radius": 25,
                "face_scale": 1.0,
            },
        },
        "timings": timings,
        "per_cup_timings": {
            cup: {
                "phase15a_sdxl_sec": cup_results[cup].get("phase15a_sdxl_sec", 0),
                "phase15b_flux_sec": 0,  # SKIPPED
                "phase3_vton_sec": cup_results[cup].get("phase3_sec", 0),
                "phase4_face_swap_sec": cup_results[cup].get("phase4_face_swap_sec", 0),
            }
            for cup in cup_names
        },
        "cost": {
            "gpu_total_sec": round(total_gpu, 2),
            "gpu_total_usd": round(cost_usd, 4),
            "gpu_total_krw": round(cost_krw, 0),
            "budget_pass": cost_krw <= 700,
            "v35_comparison": {
                "v35_gpu_sec": v35_estimated_gpu,
                "v35_cost_krw": round(v35_cost_krw, 0),
                "savings_sec": round(savings_sec, 1),
                "savings_krw": round(savings_krw, 0),
                "savings_percent": round(savings_pct, 1),
            },
        },
    }
    (OUTPUT_DIR / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    # Open key results
    print(f"\n  Opening results...")
    os.system(f"open '{OUTPUT_DIR / 'comparison_cup_sizes.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_front_bust.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_side_bust.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'comparison_depth_maps.png'}'")
    os.system(f"open '{OUTPUT_DIR / 'face_reference.png'}'")

    # Open individual cup pipeline comparisons
    for cup in cup_names:
        cup_dir = OUTPUT_DIR / f"cup_{cup}"
        os.system(f"open '{cup_dir / 'comparison_pipeline.png'}'")

    print("\n" + "=" * 80)
    print("v36 Test Complete — Cost-Optimized (No FLUX Refinement)")
    print(f"  Cups tested: {', '.join(cup_names)}")
    cn_info = ', '.join(f"{c}={CUP_CONFIGS[c]['cn_scale']}" for c in cup_names)
    print(f"  Dynamic cn_scale: {cn_info}")
    print(f"  Angles per cup: {len(HERO_ANGLES)}")
    print(f"  Pipeline: SDXL→VTON→FaceSwap (FLUX skipped)")
    print(f"  Budget: ${cost_usd:.4f} ({cost_krw:.0f} won) vs v35: {v35_cost_krw:.0f} won")
    print(f"  Savings: {savings_sec:.1f}s ({savings_pct:.1f}%) — {savings_krw:.0f} won saved")
    print(f"  Status: {'BUDGET PASS' if cost_krw <= 700 else 'BUDGET OVER'}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    r = main()
    if r:
        print(f"\n{json.dumps(r, indent=2, ensure_ascii=False)}")
