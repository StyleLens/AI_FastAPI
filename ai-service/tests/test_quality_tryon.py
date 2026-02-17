"""
StyleLens V6 — Quality Virtual Try-On Test
Proper CatVTON pipeline: user full-body photo + clothing + FASHN agnostic mask.

Goal: user's face/body + garment from shopping mall = realistic try-on result.

Usage: cd ai-service && .venv/bin/python tests/test_quality_tryon.py
"""

import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Setup paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG_DATA = ROOT.parent / "IMG_Data"
OUTPUT = ROOT / "output" / "quality_tryon"
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
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    import io
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


def save_b64(b64: str, path: Path):
    pil = b64_to_pil(b64)
    pil.save(str(path))
    print(f"  Saved: {path.name} ({pil.size})")


def main():
    print("=" * 60)
    print("StyleLens V6 — Quality Virtual Try-On Test")
    print("=" * 60)

    from worker.modal_app import (
        app as modal_app,
        run_light_models,
        run_catvton_batch,
    )

    # ── Select best user photo ────────────────────────────────
    # Image 3 (170542): frontal, full body visible, clean background
    # Image 8 (170823): upper body, side view, good body detail
    # Image 4 (170626): seated, upper body, clear face
    # Best for try-on: Image 3 (full frontal, wall background, clear body)
    user_imgs = sorted((IMG_DATA / "User_IMG").glob("*.jpg"))
    wear_imgs = sorted((IMG_DATA / "wear").glob("*.png"))

    # Use the frontal full-body shot (image 3 — standing, frontal, clean bg)
    user_path = user_imgs[2]  # Screenshot_20260207_170542_Instagram.jpg
    user_img = load_image(user_path)
    print(f"User photo: {user_path.name} ({user_img.shape})")

    # Use clothing image 1 (frontal view with model)
    wear_path = wear_imgs[0]
    wear_img = load_image(wear_path)
    print(f"Clothing: {wear_path.name} ({wear_img.shape})")

    # ── Step 1: FASHN parse map on user image ─────────────────
    # This gives us the body part segmentation to create agnostic mask
    print("\n--- Step 1: FASHN Parse Map (user image) ---")
    user_b64 = image_to_b64(user_img)

    with modal_app.run():
        t0 = time.time()
        fashn_result = run_light_models.remote(task="parse_fashn", image_b64=user_b64)
        fashn_time = time.time() - t0

    if "error" in fashn_result:
        print(f"  [ERROR] FASHN: {fashn_result['error']}")
        sys.exit(1)

    print(f"  [OK] FASHN parse in {fashn_time:.1f}s")

    # Decode parse map
    pm_b64 = fashn_result["parsemap_b64"]
    pm_raw = base64.b64decode(pm_b64)
    pm_arr = np.frombuffer(pm_raw, dtype=np.uint8)
    parse_map = cv2.imdecode(pm_arr, cv2.IMREAD_GRAYSCALE)
    print(f"  Parse map shape: {parse_map.shape}, unique classes: {np.unique(parse_map)}")

    # Save parse map visualization
    pm_vis = cv2.applyColorMap((parse_map * 15).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUTPUT / "step1_parsemap.jpg"), pm_vis)
    print(f"  Saved: step1_parsemap.jpg")

    # ── Step 2: Generate agnostic mask from parse map ─────────
    # FASHN classes: 0=bg, 1=hat, 2=hair, 3=sunglasses, 4=upper_clothes,
    # 5=skirt, 6=pants, 7=dress, 8=belt, 9=left_shoe, 10=right_shoe,
    # 11=face, 12=left_leg, 13=right_leg, 14=left_arm, 15=right_arm,
    # 16=bag, 17=scarf
    #
    # For blouse try-on: mask out upper_clothes (4) + arms (14,15) region
    # This tells CatVTON "replace this area with the new garment"
    print("\n--- Step 2: Agnostic Mask Generation ---")

    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    # Upper clothes area = replace
    mask[parse_map == 4] = 255   # upper_clothes
    # Arms = replace (garment changes sleeve appearance)
    mask[parse_map == 14] = 255  # left_arm
    mask[parse_map == 15] = 255  # right_arm
    # Optionally include belt area
    mask[parse_map == 8] = 255   # belt

    # Dilate mask slightly for smoother transitions
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(OUTPUT / "step2_agnostic_mask.jpg"), mask)
    mask_coverage = (mask > 0).sum() / mask.size * 100
    print(f"  Mask coverage: {mask_coverage:.1f}% of image")
    print(f"  Saved: step2_agnostic_mask.jpg")

    # ── Step 3: SAM3 segment clothing (remove background) ────
    print("\n--- Step 3: SAM3 Clothing Segmentation ---")
    wear_b64 = image_to_b64(wear_img)

    with modal_app.run():
        t0 = time.time()
        sam3_result = run_light_models.remote(task="segment_sam3", image_b64=wear_b64)
        sam3_time = time.time() - t0

    if "error" in sam3_result:
        print(f"  [WARN] SAM3 failed: {sam3_result['error']}, using raw clothing")
        segmented_b64 = wear_b64
    else:
        segmented_b64 = sam3_result["segmented_b64"]
        save_b64(segmented_b64, OUTPUT / "step3_clothing_segmented.png")
        print(f"  [OK] SAM3 segment in {sam3_time:.1f}s")

    # ── Step 4: CatVTON-FLUX Virtual Try-On ───────────────────
    print("\n--- Step 4: CatVTON-FLUX Virtual Try-On ---")

    # Encode mask as PNG (lossless for binary mask)
    mask_pil = Image.fromarray(mask, mode="L")
    mask_b64 = pil_to_b64(mask_pil, fmt="PNG")

    # Save input images for debugging
    cv2.imwrite(str(OUTPUT / "step4_input_person.jpg"), user_img)
    print(f"  Person: {user_img.shape}")
    print(f"  Clothing: segmented from SAM3")
    print(f"  Mask: {mask.shape}, coverage={mask_coverage:.1f}%")

    with modal_app.run():
        t0 = time.time()
        tryon_result = run_catvton_batch.remote(
            persons_b64=[user_b64],
            clothing_b64=segmented_b64,
            masks_b64=[mask_b64],
            num_steps=30,       # Full quality
            guidance=30.0,
        )
        tryon_time = time.time() - t0

    if "error" in tryon_result:
        print(f"  [ERROR] CatVTON: {tryon_result['error']}")
        sys.exit(1)

    results = tryon_result.get("results_b64", [])
    print(f"  [OK] CatVTON: {len(results)} result(s) in {tryon_time:.1f}s")

    for i, r_b64 in enumerate(results):
        save_b64(r_b64, OUTPUT / f"step4_tryon_result_{i}.png")

    # ── Step 5: Side-by-side comparison ───────────────────────
    print("\n--- Step 5: Comparison ---")

    if results:
        tryon_pil = b64_to_pil(results[0]).convert("RGB")
        person_pil = Image.fromarray(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
        wear_pil = Image.fromarray(cv2.cvtColor(wear_img, cv2.COLOR_BGR2RGB))

        # Resize all to same height for comparison
        target_h = 1024
        def resize_to_h(img, h):
            w = int(img.width * h / img.height)
            return img.resize((w, h), Image.LANCZOS)

        person_r = resize_to_h(person_pil, target_h)
        wear_r = resize_to_h(wear_pil, target_h)
        tryon_r = resize_to_h(tryon_pil, target_h)

        # Create comparison: original | clothing | try-on result
        total_w = person_r.width + wear_r.width + tryon_r.width + 20  # 10px gaps
        comparison = Image.new("RGB", (total_w, target_h), (255, 255, 255))
        x = 0
        comparison.paste(person_r, (x, 0)); x += person_r.width + 10
        comparison.paste(wear_r, (x, 0)); x += wear_r.width + 10
        comparison.paste(tryon_r, (x, 0))

        comparison.save(str(OUTPUT / "comparison.png"))
        print(f"  Saved: comparison.png (user | clothing | try-on)")

    # ── Summary ───────────────────────────────────────────────
    total_gpu = fashn_time + sam3_time + tryon_time
    cost = (total_gpu / 3600) * 4.76
    print(f"\n{'=' * 60}")
    print(f"QUALITY TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  FASHN parse: {fashn_time:.1f}s")
    print(f"  SAM3 segment: {sam3_time:.1f}s")
    print(f"  CatVTON try-on: {tryon_time:.1f}s")
    print(f"  Total GPU: {total_gpu:.0f}s (~${cost:.3f})")
    print(f"  Results: {OUTPUT}")


if __name__ == "__main__":
    main()
