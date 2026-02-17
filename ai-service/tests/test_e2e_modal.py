"""
StyleLens V6 — E2E Modal GPU Integration Test
Tests the full 4-Phase pipeline via Modal H200 worker.

Cost-optimized: ~$0.15 per run (single GPU call per phase).
  Phase 1: YOLO detect + SAM 3D Body (~27s)
  Phase 2: SAM3 segment + FASHN parse (~8s)
  Phase 3: CatVTON-FLUX 1-angle (~33s)
  Phase 4: Hunyuan3D shape only (~39s)

Usage: cd ai-service && .venv/bin/python tests/test_e2e_modal.py
"""

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Setup paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG_DATA = ROOT.parent / "IMG_Data"
OUTPUT = ROOT / "output" / "e2e_modal"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def image_to_b64(img: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def save_b64_image(b64: str, path: Path):
    """Decode base64 image and save."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(str(path), img)
        print(f"  Saved: {path.name} ({img.shape})")


def main():
    print("=" * 60)
    print("StyleLens V6 — E2E Modal GPU Integration Test")
    print("=" * 60)

    # Import Modal worker
    from worker.modal_app import (
        app as modal_app,
        run_light_models,
        run_catvton_batch,
        run_hunyuan3d,
    )

    # Load test images
    user_imgs = sorted((IMG_DATA / "User_IMG").glob("*.jpg"))
    wear_imgs = sorted((IMG_DATA / "wear").glob("*.png"))

    if not user_imgs:
        print("ERROR: No user images found in IMG_Data/User_IMG/")
        sys.exit(1)
    if not wear_imgs:
        print("ERROR: No wear images found in IMG_Data/wear/")
        sys.exit(1)

    user_img = load_image(user_imgs[0])
    wear_img = load_image(wear_imgs[0])
    print(f"User image: {user_imgs[0].name} ({user_img.shape})")
    print(f"Wear image: {wear_imgs[0].name} ({wear_img.shape})")

    user_b64 = image_to_b64(user_img)
    wear_b64 = image_to_b64(wear_img)

    total_cost_sec = 0
    results = {}

    # ── Phase 1: Avatar (YOLO + SAM 3D Body) ─────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Avatar Generation")
    print("=" * 60)

    with modal_app.run():
        # Step 1a: YOLO detect
        print("\n  Step 1a: YOLO26-L person detection...")
        t0 = time.time()
        yolo_result = run_light_models.remote(task="detect_yolo", image_b64=user_b64)
        yolo_time = time.time() - t0

        if "error" in yolo_result:
            print(f"  [ERROR] YOLO: {yolo_result['error']}")
        else:
            print(f"  [OK] YOLO: {yolo_result['num_persons']} persons in {yolo_time:.1f}s")
            if yolo_result["num_persons"] > 0:
                bbox = yolo_result["detections"][0]["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                conf = yolo_result["detections"][0]["confidence"]
                print(f"  Best detection: [{x1},{y1},{x2},{y2}] conf={conf:.3f}")
                # Crop person
                h, w = user_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cropped = user_img[y1:y2, x1:x2]
                cv2.imwrite(str(OUTPUT / "phase1_yolo_crop.jpg"), cropped)
                user_b64 = image_to_b64(cropped)  # Use cropped for SAM3D
            total_cost_sec += yolo_time

        # Step 1b: SAM 3D Body reconstruction
        print("\n  Step 1b: SAM 3D Body reconstruction...")
        t0 = time.time()
        sam3d_result = run_light_models.remote(task="reconstruct_3d", image_b64=user_b64)
        sam3d_time = time.time() - t0

        if "error" in sam3d_result:
            print(f"  [ERROR] SAM3D: {sam3d_result['error']}")
        else:
            verts = sam3d_result.get("vertex_count", "?")
            print(f"  [OK] SAM 3D Body: {verts} vertices in {sam3d_time:.1f}s")
            total_cost_sec += sam3d_time

    results["phase1"] = {
        "yolo": "error" not in yolo_result,
        "sam3d": "error" not in sam3d_result,
        "yolo_time": yolo_time,
        "sam3d_time": sam3d_time,
    }

    # ── Phase 2: Wardrobe (SAM3 + FASHN) ─────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Wardrobe Analysis")
    print("=" * 60)

    with modal_app.run():
        # Step 2a: SAM3 segmentation
        print("\n  Step 2a: SAM3 segmentation...")
        t0 = time.time()
        sam3_result = run_light_models.remote(task="segment_sam3", image_b64=wear_b64)
        sam3_time = time.time() - t0

        if "error" in sam3_result:
            print(f"  [ERROR] SAM3: {sam3_result['error']}")
        else:
            print(f"  [OK] SAM3 segmentation in {sam3_time:.1f}s")
            save_b64_image(sam3_result["segmented_b64"], OUTPUT / "phase2_sam3_segmented.jpg")
            total_cost_sec += sam3_time

        # Step 2b: FASHN parsing
        print("\n  Step 2b: FASHN parsing...")
        segmented_b64 = sam3_result.get("segmented_b64", wear_b64)
        t0 = time.time()
        fashn_result = run_light_models.remote(task="parse_fashn", image_b64=segmented_b64)
        fashn_time = time.time() - t0

        if "error" in fashn_result:
            print(f"  [ERROR] FASHN: {fashn_result['error']}")
        else:
            print(f"  [OK] FASHN parsing in {fashn_time:.1f}s")
            total_cost_sec += fashn_time

    results["phase2"] = {
        "sam3": "error" not in sam3_result,
        "fashn": "error" not in fashn_result,
        "sam3_time": sam3_time,
        "fashn_time": fashn_time,
    }

    # ── Phase 3: CatVTON-FLUX (1-angle, cost-saving) ─────────
    print("\n" + "=" * 60)
    print("PHASE 3: Virtual Try-On (CatVTON-FLUX, 1-angle)")
    print("=" * 60)

    with modal_app.run():
        t0 = time.time()
        # Send 1 angle only (saves ~7x GPU time vs 8 angles)
        catvton_result = run_catvton_batch.remote(
            persons_b64=[user_b64],
            clothing_b64=wear_b64,
            masks_b64=[image_to_b64(np.ones((512, 512, 3), dtype=np.uint8) * 255)],
            num_steps=20,     # Reduced from 30 for cost
            guidance=30.0,
        )
        catvton_time = time.time() - t0

        if "error" in catvton_result:
            print(f"  [ERROR] CatVTON: {catvton_result['error']}")
        else:
            n_results = len(catvton_result.get("results_b64", []))
            print(f"  [OK] CatVTON-FLUX: {n_results} result(s) in {catvton_time:.1f}s")
            for i, r_b64 in enumerate(catvton_result.get("results_b64", [])):
                save_b64_image(r_b64, OUTPUT / f"phase3_tryon_{i}.jpg")
            total_cost_sec += catvton_time

    results["phase3"] = {
        "catvton": "error" not in catvton_result,
        "catvton_time": catvton_time,
    }

    # ── Phase 4: Hunyuan3D (shape only) ───────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: 3D Generation (Hunyuan3D, shape only)")
    print("=" * 60)

    with modal_app.run():
        t0 = time.time()
        hunyuan_result = run_hunyuan3d.remote(
            front_image_b64=user_b64,
            reference_images_b64=None,
            shape_steps=30,   # Reduced from 50 for cost
            paint_steps=0,    # Shape only (no texture)
            texture_res=1024,
        )
        hunyuan_time = time.time() - t0

        if "error" in hunyuan_result:
            print(f"  [ERROR] Hunyuan3D: {hunyuan_result['error']}")
        else:
            verts = hunyuan_result.get("vertex_count", "?")
            glb_size = len(hunyuan_result.get("glb_bytes_b64", ""))
            print(f"  [OK] Hunyuan3D: {verts} vertices in {hunyuan_time:.1f}s")
            print(f"  GLB base64 size: {glb_size} chars")
            # Save GLB
            if "glb_bytes_b64" in hunyuan_result:
                glb_bytes = base64.b64decode(hunyuan_result["glb_bytes_b64"])
                with open(OUTPUT / "phase4_model.glb", "wb") as f:
                    f.write(glb_bytes)
                print(f"  Saved: phase4_model.glb ({len(glb_bytes)} bytes)")
            total_cost_sec += hunyuan_time

    results["phase4"] = {
        "hunyuan": "error" not in hunyuan_result,
        "hunyuan_time": hunyuan_time,
    }

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)

    all_ok = True
    for phase, data in results.items():
        models_ok = all(v for k, v in data.items() if not k.endswith("_time"))
        status = "OK" if models_ok else "FAIL"
        if not models_ok:
            all_ok = False
        times = {k: f"{v:.1f}s" for k, v in data.items() if k.endswith("_time")}
        print(f"  {phase}: {status} {times}")

    # Cost estimate (H200 ~$4.76/hr)
    cost_estimate = (total_cost_sec / 3600) * 4.76
    print(f"\n  Total GPU time: {total_cost_sec:.0f}s")
    print(f"  Estimated cost: ${cost_estimate:.3f}")
    print(f"  Results saved to: {OUTPUT}")

    if all_ok:
        print("\n  ALL PHASES PASSED")
    else:
        print("\n  SOME PHASES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
