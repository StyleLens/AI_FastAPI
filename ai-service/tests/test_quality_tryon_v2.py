"""
StyleLens V6 — Quality Virtual Try-On Test v2
Fixed: mask convention (garment=black/preserve, person=white/inpaint).
Improved: garment preprocessing, image selection, debug outputs.

Usage: cd ai-service && .venv/bin/python tests/test_quality_tryon_v2.py
"""

import base64
import io
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
OUTPUT = ROOT / "output" / "quality_tryon_v2"
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
    print("StyleLens V6 — Quality Virtual Try-On Test v2")
    print("  Fixed: mask convention (garment=black, person mask=white)")
    print("  Improved: garment image, debug outputs")
    print("=" * 60)

    from worker.modal_app import (
        app as modal_app,
        run_light_models,
        run_catvton_batch,
    )

    # ── Select images ────────────────────────────────────────
    user_imgs = sorted((IMG_DATA / "User_IMG").glob("*.jpg"))
    wear_imgs = sorted((IMG_DATA / "wear").glob("*.png"))

    # User: image index 2 — frontal, full body, tile wall background
    user_path = user_imgs[2]  # Screenshot_20260207_170542_Instagram.jpg
    user_img = load_image(user_path)
    print(f"\nUser photo: {user_path.name} ({user_img.shape})")

    # Garment: wear[3] — close-up frontal of blouse, best detail
    wear_path = wear_imgs[3]  # 스크린샷 2026-02-10 오후 2.30.22.png
    wear_img = load_image(wear_path)
    print(f"Clothing: {wear_path.name} ({wear_img.shape})")

    # Save input images
    cv2.imwrite(str(OUTPUT / "input_person.jpg"), user_img)
    cv2.imwrite(str(OUTPUT / "input_clothing.jpg"), wear_img)

    # ── Step 1: FASHN parse map on USER image ────────────────
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
    unique_classes = np.unique(parse_map)
    print(f"  Parse map shape: {parse_map.shape}, unique classes: {unique_classes}")

    # Save parse map visualization with class labels
    pm_vis = cv2.applyColorMap((parse_map * 15).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUTPUT / "step1_parsemap.jpg"), pm_vis)
    print(f"  Saved: step1_parsemap.jpg")

    # Print class coverage
    class_names = {
        0: "bg", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper_clothes",
        5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe",
        10: "right_shoe", 11: "face", 12: "left_leg", 13: "right_leg",
        14: "left_arm", 15: "right_arm", 16: "bag", 17: "scarf",
    }
    for c in unique_classes:
        pct = (parse_map == c).sum() / parse_map.size * 100
        name = class_names.get(c, f"unknown_{c}")
        print(f"    Class {c} ({name}): {pct:.1f}%")

    # ── Step 2: Generate agnostic mask ───────────────────────
    print("\n--- Step 2: Agnostic Mask Generation ---")

    mask = np.zeros(parse_map.shape, dtype=np.uint8)
    # Upper clothes area = replace (white=255=inpaint)
    mask[parse_map == 4] = 255   # upper_clothes
    # Arms = replace (garment changes sleeve appearance)
    mask[parse_map == 14] = 255  # left_arm
    mask[parse_map == 15] = 255  # right_arm
    # Belt area
    mask[parse_map == 8] = 255   # belt

    # Dilate mask for smoother transitions
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(OUTPUT / "step2_agnostic_mask.png"), mask)
    mask_coverage = (mask > 0).sum() / mask.size * 100
    print(f"  Mask coverage: {mask_coverage:.1f}% of image")
    print(f"  Saved: step2_agnostic_mask.png")

    # ── Step 3: FASHN parse on GARMENT image ─────────────────
    # Parse the clothing image to find only the garment region,
    # then use SAM3 to segment just the garment (removing model body)
    print("\n--- Step 3: Clothing Segmentation ---")

    # Step 3a: FASHN parse on clothing image to identify garment region
    wear_b64 = image_to_b64(wear_img)
    with modal_app.run():
        t0 = time.time()
        # Parse the garment image to find upper_clothes region
        fashn_wear = run_light_models.remote(task="parse_fashn", image_b64=wear_b64)
        fashn_wear_time = time.time() - t0

    if "error" in fashn_wear:
        print(f"  [WARN] FASHN on clothing failed: {fashn_wear['error']}")
        print(f"  Using SAM3 segmentation instead...")
        # Fall back to SAM3 only
        with modal_app.run():
            t0 = time.time()
            sam3_result = run_light_models.remote(task="segment_sam3", image_b64=wear_b64)
            sam3_time = time.time() - t0

        if "error" in sam3_result:
            print(f"  [ERROR] SAM3: {sam3_result['error']}")
            segmented_b64 = wear_b64
        else:
            segmented_b64 = sam3_result["segmented_b64"]
            save_b64(segmented_b64, OUTPUT / "step3_clothing_segmented.png")
            print(f"  [OK] SAM3 segment in {sam3_time:.1f}s")
    else:
        print(f"  [OK] FASHN parse on clothing in {fashn_wear_time:.1f}s")

        # Extract garment region from clothing parse map
        wear_pm_b64 = fashn_wear["parsemap_b64"]
        wear_pm_raw = base64.b64decode(wear_pm_b64)
        wear_pm_arr = np.frombuffer(wear_pm_raw, dtype=np.uint8)
        wear_parse = cv2.imdecode(wear_pm_arr, cv2.IMREAD_GRAYSCALE)

        wear_unique = np.unique(wear_parse)
        print(f"  Clothing parse classes: {wear_unique}")
        for c in wear_unique:
            pct = (wear_parse == c).sum() / wear_parse.size * 100
            name = class_names.get(c, f"unknown_{c}")
            print(f"    Class {c} ({name}): {pct:.1f}%")

        # Create garment-only mask (upper_clothes + skirt if present)
        garment_mask = np.zeros(wear_parse.shape, dtype=np.uint8)
        garment_mask[wear_parse == 4] = 255   # upper_clothes
        # Include skirt/dress if we want full outfit
        # garment_mask[wear_parse == 5] = 255   # skirt
        # garment_mask[wear_parse == 7] = 255   # dress

        # Dilate slightly to include edges
        gk = np.ones((5, 5), np.uint8)
        garment_mask = cv2.dilate(garment_mask, gk, iterations=1)

        cv2.imwrite(str(OUTPUT / "step3_garment_only_mask.png"), garment_mask)
        garment_pct = (garment_mask > 0).sum() / garment_mask.size * 100
        print(f"  Garment-only mask: {garment_pct:.1f}% of clothing image")

        # Extract garment from clothing image using mask
        # Create RGBA image with garment only on white background
        wear_rgb = cv2.cvtColor(wear_img, cv2.COLOR_BGR2RGB)
        h, w = wear_rgb.shape[:2]

        # Resize mask to match image
        if garment_mask.shape != (h, w):
            garment_mask = cv2.resize(garment_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply mask: keep garment, white background
        garment_only = np.ones_like(wear_rgb) * 255  # white bg
        garment_region = garment_mask > 127
        garment_only[garment_region] = wear_rgb[garment_region]

        garment_only_bgr = cv2.cvtColor(garment_only.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUTPUT / "step3_garment_extracted.jpg"), garment_only_bgr)
        print(f"  Saved: step3_garment_extracted.jpg")

        # Use extracted garment for CatVTON
        segmented_b64 = image_to_b64(garment_only_bgr)

        # Also save parse map visualization
        wear_pm_vis = cv2.applyColorMap((wear_parse * 15).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(OUTPUT / "step3_clothing_parsemap.jpg"), wear_pm_vis)

    # ── Step 4: CatVTON-FLUX Virtual Try-On ───────────────────
    print("\n--- Step 4: CatVTON-FLUX Virtual Try-On ---")

    # Encode mask as PNG (lossless for binary mask)
    mask_pil = Image.fromarray(mask, mode="L")
    mask_b64 = pil_to_b64(mask_pil, fmt="PNG")

    print(f"  Person: {user_img.shape}")
    print(f"  Clothing: garment-extracted")
    print(f"  Mask: {mask.shape}, coverage={mask_coverage:.1f}%")
    print(f"  Steps: 30, Guidance: 30.0")

    # Debug: simulate what CatVTON will see
    # Create the concatenated debug image
    target_w, target_h = 768, 1024
    person_pil = Image.fromarray(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
    garment_pil = b64_to_pil(segmented_b64).convert("RGB")

    person_resized = person_pil.resize((target_w, target_h), Image.LANCZOS)
    garment_resized = garment_pil.resize((target_w, target_h), Image.LANCZOS)
    mask_resized = mask_pil.resize((target_w, target_h), Image.NEAREST)

    # Debug: save what CatVTON sees as concatenated input
    debug_concat = Image.new("RGB", (target_w * 2, target_h))
    debug_concat.paste(garment_resized, (0, 0))
    debug_concat.paste(person_resized, (target_w, 0))
    debug_concat.save(str(OUTPUT / "step4_debug_concat_image.jpg"))

    # Debug: save the extended mask (should be: left=black, right=agnostic)
    debug_mask = Image.new("L", (target_w * 2, target_h), 0)  # ALL BLACK base
    debug_mask.paste(mask_resized, (target_w, 0))  # agnostic mask on right
    debug_mask.save(str(OUTPUT / "step4_debug_extended_mask.png"))
    print(f"  Saved debug: concat_image + extended_mask")

    # Verify mask: left should be all 0, right should have white regions
    mask_arr = np.array(debug_mask)
    left_mean = mask_arr[:, :target_w].mean()
    right_mean = mask_arr[:, target_w:].mean()
    print(f"  Mask check — left (garment) avg: {left_mean:.1f} (should be 0)")
    print(f"  Mask check — right (person) avg: {right_mean:.1f} (should be >0)")

    with modal_app.run():
        t0 = time.time()
        tryon_result = run_catvton_batch.remote(
            persons_b64=[user_b64],
            clothing_b64=segmented_b64,
            masks_b64=[mask_b64],
            num_steps=30,
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
        person_pil_orig = Image.fromarray(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
        wear_pil_orig = Image.fromarray(cv2.cvtColor(wear_img, cv2.COLOR_BGR2RGB))

        target_h = 1024
        def resize_to_h(img, h):
            w = int(img.width * h / img.height)
            return img.resize((w, h), Image.LANCZOS)

        person_r = resize_to_h(person_pil_orig, target_h)
        wear_r = resize_to_h(wear_pil_orig, target_h)
        tryon_r = resize_to_h(tryon_pil, target_h)

        # Create comparison: original | clothing ref | try-on result
        gap = 10
        total_w = person_r.width + wear_r.width + tryon_r.width + gap * 2
        comparison = Image.new("RGB", (total_w, target_h), (255, 255, 255))
        x = 0
        comparison.paste(person_r, (x, 0)); x += person_r.width + gap
        comparison.paste(wear_r, (x, 0)); x += wear_r.width + gap
        comparison.paste(tryon_r, (x, 0))

        comparison.save(str(OUTPUT / "comparison.png"))
        print(f"  Saved: comparison.png (user | clothing | try-on)")

    # ── Summary ───────────────────────────────────────────────
    total_gpu = fashn_time + fashn_wear_time + tryon_time
    cost = (total_gpu / 3600) * 4.76
    print(f"\n{'=' * 60}")
    print(f"QUALITY TEST v2 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  FASHN parse (user): {fashn_time:.1f}s")
    print(f"  FASHN parse (clothing): {fashn_wear_time:.1f}s")
    print(f"  CatVTON try-on: {tryon_time:.1f}s")
    print(f"  Total GPU: {total_gpu:.0f}s (~${cost:.3f})")
    print(f"  Results: {OUTPUT}")


if __name__ == "__main__":
    main()
