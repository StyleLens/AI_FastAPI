"""
StyleLens V6 — Quality Virtual Try-On Test v4
Focus: mask precision + parameter tuning for color/texture fidelity.

Improvements over v3:
  1. Tighter mask — only upper_clothes region (no arms, no belt)
     Arms are problematic: they cause hair/body modifications.
     CatVTON can handle sleeve changes from garment reference alone.
  2. Test 2 guidance values: 30.0 (default) vs 50.0 (stronger conditioning)
  3. Use SAM3 segmented garment (best from v3)

Usage: cd ai-service && .venv/bin/python tests/test_quality_tryon_v4.py
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
OUTPUT = ROOT / "output" / "quality_tryon_v4"
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


def make_mask(parse_map: np.ndarray, include_arms: bool = False,
              dilate_px: int = 15, dilate_iter: int = 2) -> np.ndarray:
    """Create agnostic mask from FASHN parse map.

    Args:
        parse_map: FASHN parse map (H, W) uint8
        include_arms: Whether to include arm regions in mask
        dilate_px: Dilation kernel size
        dilate_iter: Number of dilation iterations
    """
    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    mask[parse_map == 4] = 255   # upper_clothes — always include

    if include_arms:
        mask[parse_map == 14] = 255  # left_arm
        mask[parse_map == 15] = 255  # right_arm

    # Optional: include belt region
    # mask[parse_map == 8] = 255   # belt

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def main():
    print("=" * 60)
    print("StyleLens V6 — Quality Try-On v4")
    print("  Focus: mask precision + parameter tuning")
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

    user_b64 = image_to_b64(user_img)
    wear_b64 = image_to_b64(wear_img)

    # ── Step 1: FASHN parse on user ──────────────────────────
    print("\n--- Step 1: FASHN Parse Map ---")
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
    print(f"  [OK] in {fashn_time:.1f}s, classes: {np.unique(parse_map)}")

    # ── Step 2: SAM3 segment garment ─────────────────────────
    print("\n--- Step 2: SAM3 Garment Segmentation ---")
    with modal_app.run():
        t0 = time.time()
        sam3_result = run_light_models.remote(task="segment_sam3", image_b64=wear_b64)
        sam3_time = time.time() - t0

    if "error" in sam3_result:
        print(f"  [WARN] SAM3 failed, using original")
        garment_b64 = wear_b64
    else:
        garment_b64 = sam3_result["segmented_b64"]
        save_b64(garment_b64, OUTPUT / "garment_sam3.png")
        print(f"  [OK] SAM3 in {sam3_time:.1f}s")

    # ── Step 3: Generate 2 mask variants ─────────────────────
    print("\n--- Step 3: Mask Variants ---")

    # Mask A: upper_clothes ONLY (tight, no arms) — gentle dilation
    mask_a = make_mask(parse_map, include_arms=False, dilate_px=10, dilate_iter=1)
    cv2.imwrite(str(OUTPUT / "mask_a_tight.png"), mask_a)
    a_cov = (mask_a > 0).sum() / mask_a.size * 100
    print(f"  Mask A (tight, clothes only): {a_cov:.1f}% coverage")

    # Mask B: upper_clothes + arms (wider coverage) — standard dilation
    mask_b = make_mask(parse_map, include_arms=True, dilate_px=15, dilate_iter=2)
    cv2.imwrite(str(OUTPUT / "mask_b_wide.png"), mask_b)
    b_cov = (mask_b > 0).sum() / mask_b.size * 100
    print(f"  Mask B (wide, clothes+arms): {b_cov:.1f}% coverage")

    mask_a_pil = Image.fromarray(mask_a, mode="L")
    mask_b_pil = Image.fromarray(mask_b, mode="L")
    mask_a_b64 = pil_to_b64(mask_a_pil, fmt="PNG")
    mask_b_b64 = pil_to_b64(mask_b_pil, fmt="PNG")

    # ── Step 4: CatVTON 4 variants ──────────────────────────
    # Run 4 combinations: 2 masks × 2 guidance values
    print("\n--- Step 4: CatVTON-FLUX (4 variants) ---")

    configs = [
        ("tight_g30", mask_a_b64, 30.0),
        ("tight_g50", mask_a_b64, 50.0),
        ("wide_g30", mask_b_b64, 30.0),
        ("wide_g50", mask_b_b64, 50.0),
    ]

    results = {}
    total_tryon_time = 0

    for name, m_b64, guidance in configs:
        print(f"\n  Running {name} (guidance={guidance})...")
        with modal_app.run():
            t0 = time.time()
            result = run_catvton_batch.remote(
                persons_b64=[user_b64],
                clothing_b64=garment_b64,
                masks_b64=[m_b64],
                num_steps=30,
                guidance=guidance,
            )
            elapsed = time.time() - t0

        if "error" in result:
            print(f"  [ERROR] {name}: {result['error']}")
        else:
            r_list = result.get("results_b64", [])
            if r_list:
                save_b64(r_list[0], OUTPUT / f"result_{name}.png")
                results[name] = r_list[0]
            print(f"  [OK] {name}: {elapsed:.1f}s")
            total_tryon_time += elapsed

    # ── Step 5: 5-way comparison ─────────────────────────────
    print("\n--- Step 5: Comparison ---")

    target_h = 1024
    def resize_to_h(img, h):
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)

    panels = []

    # Original person
    person_pil = Image.fromarray(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
    panels.append(("Original", resize_to_h(person_pil, target_h)))

    # Reference garment
    wear_pil = Image.fromarray(cv2.cvtColor(wear_img, cv2.COLOR_BGR2RGB))
    panels.append(("Garment", resize_to_h(wear_pil, target_h)))

    # Results
    for name in ["tight_g30", "tight_g50", "wide_g30", "wide_g50"]:
        if name in results:
            pil = b64_to_pil(results[name]).convert("RGB")
            panels.append((name, resize_to_h(pil, target_h)))

    gap = 8
    total_w = sum(p[1].width for p in panels) + gap * (len(panels) - 1)
    comparison = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    x = 0
    for label, img in panels:
        comparison.paste(img, (x, 0))
        x += img.width + gap

    comparison.save(str(OUTPUT / "comparison_all.png"))
    labels = " | ".join(p[0] for p in panels)
    print(f"  Saved: comparison_all.png ({labels})")

    # ── Summary ──────────────────────────────────────────────
    total_gpu = fashn_time + sam3_time + total_tryon_time
    cost = (total_gpu / 3600) * 4.76
    print(f"\n{'=' * 60}")
    print(f"QUALITY TEST v4 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  FASHN:       {fashn_time:.1f}s")
    print(f"  SAM3:        {sam3_time:.1f}s")
    print(f"  CatVTON x4:  {total_tryon_time:.1f}s")
    print(f"  Total GPU:   {total_gpu:.0f}s (~${cost:.3f})")
    print(f"  Results: {OUTPUT}")
    print(f"\n  Best variant: check comparison_all.png")


if __name__ == "__main__":
    main()
