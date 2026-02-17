"""
StyleLens V6 — Quality Virtual Try-On Test v3
Compare 3 garment input approaches in ONE GPU session (cost-efficient):
  A) SAM3 segmented (background removed, full model visible)
  B) Original clothing image with model (no preprocessing)
  C) FASHN-extracted garment on white bg (v2 approach, improved)

Uses best user photo + best clothing photo.
Cost: ~$0.08 per run (1 CatVTON batch with 3 inputs).

Usage: cd ai-service && .venv/bin/python tests/test_quality_tryon_v3.py
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
OUTPUT = ROOT / "output" / "quality_tryon_v3"
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


def main():
    print("=" * 60)
    print("StyleLens V6 — Quality Try-On v3 (3-way garment comparison)")
    print("=" * 60)

    from worker.modal_app import (
        app as modal_app,
        run_light_models,
        run_catvton_batch,
    )

    user_imgs = sorted((IMG_DATA / "User_IMG").glob("*.jpg"))
    wear_imgs = sorted((IMG_DATA / "wear").glob("*.png"))

    # User: image 2 — frontal, full body, tile wall
    user_path = user_imgs[2]
    user_img = load_image(user_path)
    print(f"\nUser: {user_path.name} ({user_img.shape})")

    # Garment: wear[3] — close-up frontal of blouse
    wear_path = wear_imgs[3]
    wear_img = load_image(wear_path)
    print(f"Clothing: {wear_path.name} ({wear_img.shape})")

    # Also try wear[0] — mirror selfie, upper body (different angle)
    wear_path_alt = wear_imgs[0]
    wear_img_alt = load_image(wear_path_alt)
    print(f"Clothing alt: {wear_path_alt.name} ({wear_img_alt.shape})")

    cv2.imwrite(str(OUTPUT / "input_person.jpg"), user_img)
    cv2.imwrite(str(OUTPUT / "input_clothing.jpg"), wear_img)

    user_b64 = image_to_b64(user_img)
    wear_b64 = image_to_b64(wear_img)
    wear_alt_b64 = image_to_b64(wear_img_alt)

    # ── Step 1: FASHN parse map on user image ────────────────
    print("\n--- Step 1: FASHN Parse Map (user) ---")
    with modal_app.run():
        t0 = time.time()
        fashn_result = run_light_models.remote(task="parse_fashn", image_b64=user_b64)
        fashn_time = time.time() - t0

    if "error" in fashn_result:
        print(f"  [ERROR] FASHN: {fashn_result['error']}")
        sys.exit(1)

    pm_b64 = fashn_result["parsemap_b64"]
    pm_raw = base64.b64decode(pm_b64)
    pm_arr = np.frombuffer(pm_raw, dtype=np.uint8)
    parse_map = cv2.imdecode(pm_arr, cv2.IMREAD_GRAYSCALE)
    print(f"  [OK] FASHN parse in {fashn_time:.1f}s, classes: {np.unique(parse_map)}")

    # ── Step 2: Agnostic mask ────────────────────────────────
    print("\n--- Step 2: Agnostic Mask ---")
    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    mask[parse_map == 4] = 255   # upper_clothes
    mask[parse_map == 14] = 255  # left_arm
    mask[parse_map == 15] = 255  # right_arm
    mask[parse_map == 8] = 255   # belt

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(OUTPUT / "step2_mask.png"), mask)
    mask_coverage = (mask > 0).sum() / mask.size * 100
    print(f"  Mask coverage: {mask_coverage:.1f}%")

    mask_pil = Image.fromarray(mask, mode="L")
    mask_b64 = pil_to_b64(mask_pil, fmt="PNG")

    # ── Step 3: Prepare 3 garment variants ───────────────────
    print("\n--- Step 3: Prepare 3 Garment Variants ---")

    # Variant A: SAM3 segmented (bg removed)
    print("  Preparing variant A (SAM3 segmented)...")
    with modal_app.run():
        t0 = time.time()
        sam3_result = run_light_models.remote(task="segment_sam3", image_b64=wear_b64)
        sam3_time = time.time() - t0

    if "error" in sam3_result:
        print(f"  [WARN] SAM3 failed: {sam3_result['error']}")
        variant_a_b64 = wear_b64
    else:
        variant_a_b64 = sam3_result["segmented_b64"]
        save_b64(variant_a_b64, OUTPUT / "variant_a_sam3.png")
        print(f"  [OK] SAM3 in {sam3_time:.1f}s")

    # Variant B: Original clothing image (no processing)
    variant_b_b64 = wear_b64
    print(f"  Variant B: original image (no processing)")

    # Variant C: Use the alternative clothing image (mirror selfie, wear[0])
    # This has a different angle and the full-body is visible
    print("  Preparing variant C (SAM3 on alt image)...")
    with modal_app.run():
        t0 = time.time()
        sam3_alt = run_light_models.remote(task="segment_sam3", image_b64=wear_alt_b64)
        sam3_alt_time = time.time() - t0

    if "error" in sam3_alt:
        variant_c_b64 = wear_alt_b64
    else:
        variant_c_b64 = sam3_alt["segmented_b64"]
        save_b64(variant_c_b64, OUTPUT / "variant_c_sam3_alt.png")
        print(f"  [OK] SAM3 alt in {sam3_alt_time:.1f}s")

    # ── Step 4: CatVTON batch (3 variants in ONE call) ───────
    print("\n--- Step 4: CatVTON-FLUX (3 variants, single batch) ---")
    print("  A: SAM3 segmented (bg removed)")
    print("  B: Original with model (no processing)")
    print("  C: SAM3 on alt photo (mirror selfie)")

    # All 3 use the same person image and mask
    with modal_app.run():
        t0 = time.time()
        tryon_result = run_catvton_batch.remote(
            persons_b64=[user_b64, user_b64, user_b64],
            clothing_b64=variant_a_b64,  # We'll run separately for B and C
            masks_b64=[mask_b64, mask_b64, mask_b64],
            num_steps=30,
            guidance=30.0,
        )
        tryon_a_time = time.time() - t0

    if "error" in tryon_result:
        print(f"  [ERROR] CatVTON-A: {tryon_result['error']}")
    else:
        results_a = tryon_result.get("results_b64", [])
        if results_a:
            save_b64(results_a[0], OUTPUT / "result_A_sam3.png")
        print(f"  [OK] Variant A: {len(results_a)} result(s) in {tryon_a_time:.1f}s")

    # Variant B: original image
    with modal_app.run():
        t0 = time.time()
        tryon_b = run_catvton_batch.remote(
            persons_b64=[user_b64],
            clothing_b64=variant_b_b64,
            masks_b64=[mask_b64],
            num_steps=30,
            guidance=30.0,
        )
        tryon_b_time = time.time() - t0

    if "error" in tryon_b:
        print(f"  [ERROR] CatVTON-B: {tryon_b['error']}")
    else:
        results_b = tryon_b.get("results_b64", [])
        if results_b:
            save_b64(results_b[0], OUTPUT / "result_B_original.png")
        print(f"  [OK] Variant B: {len(results_b)} result(s) in {tryon_b_time:.1f}s")

    # Variant C: SAM3 alt image
    with modal_app.run():
        t0 = time.time()
        tryon_c = run_catvton_batch.remote(
            persons_b64=[user_b64],
            clothing_b64=variant_c_b64,
            masks_b64=[mask_b64],
            num_steps=30,
            guidance=30.0,
        )
        tryon_c_time = time.time() - t0

    if "error" in tryon_c:
        print(f"  [ERROR] CatVTON-C: {tryon_c['error']}")
    else:
        results_c = tryon_c.get("results_b64", [])
        if results_c:
            save_b64(results_c[0], OUTPUT / "result_C_sam3_alt.png")
        print(f"  [OK] Variant C: {len(results_c)} result(s) in {tryon_c_time:.1f}s")

    # ── Step 5: 4-way comparison ──────────────────────────────
    print("\n--- Step 5: 4-Way Comparison ---")

    target_h = 1024
    def resize_to_h(img, h):
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)

    person_pil = Image.fromarray(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
    person_r = resize_to_h(person_pil, target_h)

    panels = [person_r]
    labels = ["Original"]

    for name, result_list in [
        ("A:SAM3", results_a if "results_a" in dir() else []),
        ("B:Raw", results_b if "results_b" in dir() else []),
        ("C:Alt", results_c if "results_c" in dir() else []),
    ]:
        if result_list:
            pil = b64_to_pil(result_list[0]).convert("RGB")
            panels.append(resize_to_h(pil, target_h))
            labels.append(name)

    gap = 10
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    comparison = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    x = 0
    for p in panels:
        comparison.paste(p, (x, 0))
        x += p.width + gap

    comparison.save(str(OUTPUT / "comparison_4way.png"))
    print(f"  Saved: comparison_4way.png ({' | '.join(labels)})")

    # ── Summary ───────────────────────────────────────────────
    total_gpu = fashn_time + sam3_time + sam3_alt_time + tryon_a_time + tryon_b_time + tryon_c_time
    cost = (total_gpu / 3600) * 4.76
    print(f"\n{'=' * 60}")
    print(f"QUALITY TEST v3 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  FASHN parse: {fashn_time:.1f}s")
    print(f"  SAM3 (main): {sam3_time:.1f}s")
    print(f"  SAM3 (alt):  {sam3_alt_time:.1f}s")
    print(f"  CatVTON-A:   {tryon_a_time:.1f}s")
    print(f"  CatVTON-B:   {tryon_b_time:.1f}s")
    print(f"  CatVTON-C:   {tryon_c_time:.1f}s")
    print(f"  Total GPU:   {total_gpu:.0f}s (~${cost:.3f})")
    print(f"  Results: {OUTPUT}")


if __name__ == "__main__":
    main()
