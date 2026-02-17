"""
StyleLens V6 — Quality Virtual Try-On Test v5
Focus: use thinner-clothing user photos for better results.

Strategy:
  - User img 6 (170823): tank top, upper body, arms visible → best skin/body
  - User img 3 (170626): seated, thin t-shirt, clear face → frontal
  - Use the winning config from v4: wide mask (clothes+arms), guidance=30
  - Use SAM3 segmented garment (best from v3)
  - Also test with full-body wear[1] image (shows complete outfit)

Usage: cd ai-service && .venv/bin/python tests/test_quality_tryon_v5.py
"""

import base64
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG_DATA = ROOT.parent / "IMG_Data"
OUTPUT = ROOT / "output" / "quality_tryon_v5"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load: {path}")
    return img


def image_to_b64(img: np.ndarray, quality: int = 95) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def pil_to_b64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


def save_b64(b64: str, path: Path):
    pil = b64_to_pil(b64)
    pil.save(str(path))
    print(f"  Saved: {path.name} ({pil.size})")


def make_mask(parse_map: np.ndarray) -> np.ndarray:
    """Create agnostic mask — wide (clothes + arms), gentle dilation."""
    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    mask[parse_map == 4] = 255   # upper_clothes
    mask[parse_map == 14] = 255  # left_arm
    mask[parse_map == 15] = 255  # right_arm
    # No belt — keep it tight to upper body

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def run_single_tryon(modal_app, run_light_models, run_catvton_batch,
                     user_img, user_b64, garment_b64, label, output_dir):
    """Run FASHN → mask → CatVTON for one person image."""
    print(f"\n{'─' * 50}")
    print(f"  [{label}] Processing...")

    # FASHN parse
    with modal_app.run():
        t0 = time.time()
        fashn = run_light_models.remote(task="parse_fashn", image_b64=user_b64)
        fashn_time = time.time() - t0

    if "error" in fashn:
        print(f"  [{label}] FASHN ERROR: {fashn['error']}")
        return None, 0

    pm_b64 = fashn["parsemap_b64"]
    pm_raw = base64.b64decode(pm_b64)
    pm_arr = np.frombuffer(pm_raw, dtype=np.uint8)
    parse_map = cv2.imdecode(pm_arr, cv2.IMREAD_GRAYSCALE)
    print(f"  [{label}] FASHN: {fashn_time:.1f}s, classes: {np.unique(parse_map)}")

    # Mask
    mask = make_mask(parse_map)
    cv2.imwrite(str(output_dir / f"{label}_mask.png"), mask)
    mask_cov = (mask > 0).sum() / mask.size * 100
    print(f"  [{label}] Mask coverage: {mask_cov:.1f}%")

    # Save parse map vis
    pm_vis = cv2.applyColorMap((parse_map * 15).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{label}_parsemap.jpg"), pm_vis)

    mask_pil = Image.fromarray(mask, mode="L")
    mask_b64 = pil_to_b64(mask_pil, fmt="PNG")

    # CatVTON
    with modal_app.run():
        t0 = time.time()
        result = run_catvton_batch.remote(
            persons_b64=[user_b64],
            clothing_b64=garment_b64,
            masks_b64=[mask_b64],
            num_steps=30,
            guidance=30.0,
        )
        tryon_time = time.time() - t0

    if "error" in result:
        print(f"  [{label}] CatVTON ERROR: {result['error']}")
        return None, fashn_time

    results = result.get("results_b64", [])
    if results:
        save_b64(results[0], output_dir / f"{label}_result.png")
        print(f"  [{label}] CatVTON: {tryon_time:.1f}s")
        return results[0], fashn_time + tryon_time
    return None, fashn_time


def main():
    print("=" * 60)
    print("StyleLens V6 — Quality Try-On v5")
    print("  Focus: thinner clothing photos for best results")
    print("=" * 60)

    from worker.modal_app import (
        app as modal_app,
        run_light_models,
        run_catvton_batch,
    )

    user_imgs = sorted((IMG_DATA / "User_IMG").glob("*.jpg"))
    wear_imgs = sorted((IMG_DATA / "wear").glob("*.png"))

    # ── Garment preparation ─────────────────────────────────
    # Use wear[3] — close-up frontal (best from v3/v4)
    wear_path = wear_imgs[3]
    wear_img = load_image(wear_path)
    wear_b64 = image_to_b64(wear_img)
    print(f"\nGarment: {wear_path.name} ({wear_img.shape})")

    # Also try wear[1] — full body (complete outfit view)
    wear_path_full = wear_imgs[1]
    wear_img_full = load_image(wear_path_full)
    wear_full_b64 = image_to_b64(wear_img_full)
    print(f"Garment (full): {wear_path_full.name} ({wear_img_full.shape})")

    # SAM3 on close-up garment
    print("\n--- SAM3 Garment Segmentation ---")
    with modal_app.run():
        t0 = time.time()
        sam3 = run_light_models.remote(task="segment_sam3", image_b64=wear_b64)
        sam3_time = time.time() - t0

    garment_b64 = sam3.get("segmented_b64", wear_b64) if "error" not in sam3 else wear_b64
    save_b64(garment_b64, OUTPUT / "garment_sam3.png")
    print(f"  SAM3: {sam3_time:.1f}s")

    # SAM3 on full-body garment
    with modal_app.run():
        t0 = time.time()
        sam3_full = run_light_models.remote(task="segment_sam3", image_b64=wear_full_b64)
        sam3_full_time = time.time() - t0

    garment_full_b64 = sam3_full.get("segmented_b64", wear_full_b64) if "error" not in sam3_full else wear_full_b64
    save_b64(garment_full_b64, OUTPUT / "garment_full_sam3.png")
    print(f"  SAM3 full: {sam3_full_time:.1f}s")

    total_gpu = sam3_time + sam3_full_time

    # ── Test 4 user photos × best garment ────────────────────
    print("\n" + "=" * 60)
    print("Testing multiple user photos with SAM3 garment")
    print("=" * 60)

    test_cases = [
        # (label, image_index, description)
        ("tile_jacket", 2, "frontal full-body, leather jacket, tile wall"),
        ("seated_tee", 3, "seated, thin t-shirt, clear face"),
        ("tanktop", 6, "tank top, upper body, bathroom"),
        ("tile_jacket_fullwear", 2, "same person, full-body garment ref"),
    ]

    all_results = {}

    for label, idx, desc in test_cases:
        user_path = user_imgs[idx]
        user_img_cur = load_image(user_path)
        user_cur_b64 = image_to_b64(user_img_cur)
        print(f"\n  User [{label}]: {user_path.name} — {desc}")

        # Use full garment for the last test case
        g_b64 = garment_full_b64 if "fullwear" in label else garment_b64

        result_b64, gpu_time = run_single_tryon(
            modal_app, run_light_models, run_catvton_batch,
            user_img_cur, user_cur_b64, g_b64, label, OUTPUT,
        )
        total_gpu += gpu_time
        if result_b64:
            all_results[label] = (result_b64, user_img_cur)

    # ── Comparison grid ──────────────────────────────────────
    print("\n--- Comparison Grid ---")

    target_h = 1024
    def resize_to_h(img, h):
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)

    # Build comparison: for each test, show original + result side by side
    rows = []
    for label, (result_b64, user_img_cur) in all_results.items():
        person_pil = Image.fromarray(cv2.cvtColor(user_img_cur, cv2.COLOR_BGR2RGB))
        result_pil = b64_to_pil(result_b64).convert("RGB")

        person_r = resize_to_h(person_pil, target_h)
        result_r = resize_to_h(result_pil, target_h)

        pair_w = person_r.width + result_r.width + 5
        pair = Image.new("RGB", (pair_w, target_h), (255, 255, 255))
        pair.paste(person_r, (0, 0))
        pair.paste(result_r, (person_r.width + 5, 0))
        rows.append((label, pair))

    if rows:
        # Stack vertically with gaps
        max_w = max(p.width for _, p in rows)
        total_h = sum(p.height for _, p in rows) + 10 * (len(rows) - 1)
        grid = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for label, pair in rows:
            grid.paste(pair, (0, y))
            y += pair.height + 10

        grid.save(str(OUTPUT / "comparison_grid.png"))
        print(f"  Saved: comparison_grid.png ({len(rows)} pairs)")

    # Also save a single-row comparison for the best results
    garment_ref = Image.fromarray(cv2.cvtColor(wear_img, cv2.COLOR_BGR2RGB))
    garment_r = resize_to_h(garment_ref, target_h)

    panels = [("Garment", garment_r)]
    for label in ["tile_jacket", "seated_tee", "tanktop", "tile_jacket_fullwear"]:
        if label in all_results:
            pil = b64_to_pil(all_results[label][0]).convert("RGB")
            panels.append((label, resize_to_h(pil, target_h)))

    gap = 8
    total_w = sum(p[1].width for p in panels) + gap * (len(panels) - 1)
    comp = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    x = 0
    for _, img in panels:
        comp.paste(img, (x, 0))
        x += img.width + gap

    comp.save(str(OUTPUT / "comparison_results.png"))
    labels = " | ".join(p[0] for p in panels)
    print(f"  Saved: comparison_results.png ({labels})")

    # ── Summary ──────────────────────────────────────────────
    cost = (total_gpu / 3600) * 4.76
    print(f"\n{'=' * 60}")
    print(f"QUALITY TEST v5 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total GPU: {total_gpu:.0f}s (~${cost:.3f})")
    print(f"  Results: {OUTPUT}")
    print(f"  Check comparison_grid.png for before/after pairs")
    print(f"  Check comparison_results.png for all results side-by-side")


if __name__ == "__main__":
    main()
