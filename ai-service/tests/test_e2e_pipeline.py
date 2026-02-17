"""
StyleLens V6 E2E Pipeline Test

전체 4-Phase 파이프라인 통합 테스트:
  Phase 1: Avatar (YOLO, SAM3, 3D Body, FASHN)
  Phase 2: Wardrobe (SAM3 garment segmentation)
  Phase 3: Fitting (CatVTON-FLUX try-on)
  Phase 4: 3D Visualization (Hunyuan3D GLB generation)

비용 추적: H200 GPU $5.40/hr
"""

import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project root and data directories
PROJECT_ROOT = ROOT.parent
IMG_DATA = PROJECT_ROOT / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
WEAR_DIR = IMG_DATA / "wear"
OUTPUT_DIR = ROOT / "output" / "e2e_test"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Modal imports
try:
    import modal
    from worker.modal_app import (
        app as modal_app,
        run_light_models,
        run_catvton_batch,
        run_hunyuan3d,
    )
    MODAL_AVAILABLE = True
except ImportError as e:
    print(f"Modal not available: {e}")
    MODAL_AVAILABLE = False


# ── Utilities ────────────────────────────────────────────────────────

def image_to_b64(img_path: Path) -> str:
    """Read image file and encode as base64."""
    with open(img_path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("ascii")


def b64_to_image(b64: str) -> np.ndarray:
    """Decode base64 to BGR numpy array."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def b64_to_parsemap(b64: str) -> np.ndarray:
    """Decode base64 to grayscale parse map."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    pm = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return pm


def b64_to_ndarray(data: dict) -> np.ndarray:
    """Deserialize numpy array from base64 with metadata."""
    raw = base64.b64decode(data["data"])
    arr = np.frombuffer(raw, dtype=data["dtype"])
    return arr.reshape(data["shape"])


def save_b64_image(b64: str, path: Path):
    """Save base64 image to file."""
    img = b64_to_image(b64)
    cv2.imwrite(str(path), img)


def save_b64_parsemap(b64: str, path: Path):
    """Save base64 parse map to file."""
    pm = b64_to_parsemap(b64)
    cv2.imwrite(str(path), pm)


def save_glb(b64: str, path: Path):
    """Save base64 GLB to file."""
    raw = base64.b64decode(b64)
    path.write_bytes(raw)


def create_agnostic_mask(parsemap: np.ndarray) -> np.ndarray:
    """
    Create agnostic mask from FASHN parse map.

    FASHN labels (18 classes):
      0=background, 4=upper_clothes, 14=left_arm, 15=right_arm

    Mask convention for FluxFillPipeline:
      white(255) = inpaint region (try-on area)
      black(0) = preserve region (keep original)

    Args:
        parsemap: uint8 parse map (H, W)

    Returns:
        uint8 mask (H, W) where clothing area = white
    """
    # Upper body mask: upper_clothes + left_arm + right_arm
    upper_mask = np.zeros_like(parsemap, dtype=np.uint8)
    upper_mask[(parsemap == 4) | (parsemap == 14) | (parsemap == 15)] = 255

    # Morphological dilation (15px x 2)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(upper_mask, kernel, iterations=2)

    # Gaussian blur + threshold for smooth edges
    blurred = cv2.GaussianBlur(dilated, (21, 21), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return final_mask


def calculate_cost(elapsed_sec: float) -> float:
    """Calculate GPU cost in USD. H200 rate: $5.40/hr."""
    cost_per_hour = 5.40
    return (elapsed_sec / 3600.0) * cost_per_hour


# ── Phase 1: Avatar ──────────────────────────────────────────────────

def phase1_avatar(user_img_b64: str) -> dict:
    """
    Phase 1: Avatar creation from user image.

    Steps:
      1. YOLO person detection
      2. SAM3 segmentation
      3. SAM 3D Body reconstruction
      4. FASHN parsing

    Returns:
        {
            "detection": {...},
            "segmented_b64": str,
            "vertices": ndarray,
            "faces": ndarray,
            "joints": ndarray,
            "betas": ndarray,
            "parsemap_b64": str,
            "total_time": float,
            "total_cost": float,
        }
    """
    print("\n" + "="*60)
    print("Phase 1: Avatar")
    print("="*60)

    results = {}
    total_time = 0.0

    # Step 1: YOLO detection
    print("\n[1/4] Running YOLO person detection...")
    t0 = time.time()
    with modal_app.run():
        yolo_result = run_light_models.remote(
            task="detect_yolo",
            image_b64=user_img_b64
        )
    t_yolo = time.time() - t0
    total_time += t_yolo

    if "error" in yolo_result:
        raise RuntimeError(f"YOLO failed: {yolo_result['error']}")

    print(f"  Detected {yolo_result['num_persons']} person(s)")
    print(f"  Time: {t_yolo:.2f}s | Cost: ${calculate_cost(t_yolo):.4f}")
    results["detection"] = yolo_result

    # Step 2: SAM3 segmentation
    print("\n[2/4] Running SAM3 segmentation...")
    t0 = time.time()
    with modal_app.run():
        sam_result = run_light_models.remote(
            task="segment_sam3",
            image_b64=user_img_b64
        )
    t_sam = time.time() - t0
    total_time += t_sam

    if "error" in sam_result:
        raise RuntimeError(f"SAM3 failed: {sam_result['error']}")

    print(f"  Segmentation complete")
    print(f"  Time: {t_sam:.2f}s | Cost: ${calculate_cost(t_sam):.4f}")
    results["segmented_b64"] = sam_result["segmented_b64"]

    # Step 3: 3D Body reconstruction
    print("\n[3/4] Running SAM 3D Body reconstruction...")
    t0 = time.time()
    with modal_app.run():
        body3d_result = run_light_models.remote(
            task="reconstruct_3d",
            image_b64=user_img_b64
        )
    t_body3d = time.time() - t0
    total_time += t_body3d

    if "error" in body3d_result:
        raise RuntimeError(f"3D Body failed: {body3d_result['error']}")

    vertices = b64_to_ndarray(body3d_result["vertices"])
    faces = b64_to_ndarray(body3d_result["faces"])
    joints = b64_to_ndarray(body3d_result["joints"])
    betas = b64_to_ndarray(body3d_result["betas"])

    print(f"  Vertices: {vertices.shape}, Faces: {faces.shape}")
    print(f"  Joints: {joints.shape}, Betas: {betas.shape}")
    print(f"  Time: {t_body3d:.2f}s | Cost: ${calculate_cost(t_body3d):.4f}")

    results["vertices"] = vertices
    results["faces"] = faces
    results["joints"] = joints
    results["betas"] = betas

    # Step 4: FASHN parsing
    print("\n[4/4] Running FASHN parsing...")
    t0 = time.time()
    with modal_app.run():
        fashn_result = run_light_models.remote(
            task="parse_fashn",
            image_b64=user_img_b64
        )
    t_fashn = time.time() - t0
    total_time += t_fashn

    if "error" in fashn_result:
        raise RuntimeError(f"FASHN failed: {fashn_result['error']}")

    print(f"  Parse map complete")
    print(f"  Time: {t_fashn:.2f}s | Cost: ${calculate_cost(t_fashn):.4f}")
    results["parsemap_b64"] = fashn_result["parsemap_b64"]

    # Summary
    total_cost = calculate_cost(total_time)
    print(f"\n{'─'*60}")
    print(f"Phase 1 Total: {total_time:.2f}s | ${total_cost:.4f}")
    print(f"{'─'*60}")

    results["total_time"] = total_time
    results["total_cost"] = total_cost

    return results


# ── Phase 2: Wardrobe ────────────────────────────────────────────────

def phase2_wardrobe(garment_img_b64: str) -> dict:
    """
    Phase 2: Garment processing.

    Steps:
      1. SAM3 garment segmentation

    Returns:
        {
            "segmented_b64": str,
            "total_time": float,
            "total_cost": float,
        }
    """
    print("\n" + "="*60)
    print("Phase 2: Wardrobe")
    print("="*60)

    print("\n[1/1] Running SAM3 garment segmentation...")
    t0 = time.time()
    with modal_app.run():
        sam_result = run_light_models.remote(
            task="segment_sam3",
            image_b64=garment_img_b64
        )
    total_time = time.time() - t0

    if "error" in sam_result:
        raise RuntimeError(f"SAM3 garment failed: {sam_result['error']}")

    total_cost = calculate_cost(total_time)
    print(f"  Segmentation complete")
    print(f"  Time: {total_time:.2f}s | Cost: ${total_cost:.4f}")

    print(f"\n{'─'*60}")
    print(f"Phase 2 Total: {total_time:.2f}s | ${total_cost:.4f}")
    print(f"{'─'*60}")

    return {
        "segmented_b64": sam_result["segmented_b64"],
        "total_time": total_time,
        "total_cost": total_cost,
    }


# ── Phase 3: Fitting ─────────────────────────────────────────────────

def phase3_fitting(
    user_img_b64: str,
    garment_img_b64: str,
    parsemap_b64: str,
) -> dict:
    """
    Phase 3: Virtual try-on with CatVTON-FLUX.

    Steps:
      1. Create agnostic mask from parse map
      2. Run CatVTON batch (1 angle for cost savings)

    Returns:
        {
            "tryon_b64": str,
            "total_time": float,
            "total_cost": float,
        }
    """
    print("\n" + "="*60)
    print("Phase 3: Fitting (CatVTON-FLUX)")
    print("="*60)

    # Step 1: Create agnostic mask
    print("\n[1/2] Creating agnostic mask from parse map...")
    parsemap = b64_to_parsemap(parsemap_b64)
    agnostic_mask = create_agnostic_mask(parsemap)

    # Encode mask as PNG
    ok, buf = cv2.imencode(".png", agnostic_mask)
    if not ok:
        raise RuntimeError("Failed to encode agnostic mask")
    mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    print(f"  Mask shape: {agnostic_mask.shape}")
    print(f"  Inpaint pixels: {(agnostic_mask == 255).sum()}")

    # Step 2: Run CatVTON (single angle for cost savings)
    print("\n[2/2] Running CatVTON-FLUX try-on (1 angle)...")
    t0 = time.time()
    with modal_app.run():
        catvton_result = run_catvton_batch.remote(
            persons_b64=[user_img_b64],
            clothing_b64=garment_img_b64,
            masks_b64=[mask_b64],
            num_steps=30,
            guidance=30.0,
        )
    total_time = time.time() - t0

    if "error" in catvton_result:
        raise RuntimeError(f"CatVTON failed: {catvton_result['error']}")

    total_cost = calculate_cost(total_time)
    tryon_b64 = catvton_result["results_b64"][0]

    print(f"  Try-on complete")
    print(f"  Time: {total_time:.2f}s | Cost: ${total_cost:.4f}")

    print(f"\n{'─'*60}")
    print(f"Phase 3 Total: {total_time:.2f}s | ${total_cost:.4f}")
    print(f"{'─'*60}")

    return {
        "tryon_b64": tryon_b64,
        "total_time": total_time,
        "total_cost": total_cost,
    }


# ── Phase 4: 3D Visualization ────────────────────────────────────────

def phase4_visualization(tryon_b64: str) -> dict:
    """
    Phase 4: 3D model generation with Hunyuan3D.

    Steps:
      1. Generate GLB from try-on result

    Returns:
        {
            "glb_b64": str,
            "textured": bool,
            "total_time": float,
            "total_cost": float,
        }
    """
    print("\n" + "="*60)
    print("Phase 4: 3D Visualization (Hunyuan3D)")
    print("="*60)

    print("\n[1/1] Generating 3D GLB from try-on result...")
    t0 = time.time()
    with modal_app.run():
        hy3d_result = run_hunyuan3d.remote(
            front_image_b64=tryon_b64,
            reference_images_b64=None,
            shape_steps=30,  # Reduced from 50 for cost savings
            paint_steps=10,  # Reduced from 20 for cost savings
            texture_res=4096,
        )
    total_time = time.time() - t0

    if "error" in hy3d_result:
        raise RuntimeError(f"Hunyuan3D failed: {hy3d_result['error']}")

    total_cost = calculate_cost(total_time)
    glb_b64 = hy3d_result["glb_bytes_b64"]
    textured = hy3d_result.get("textured", False)

    # Verify GLB magic bytes
    glb_bytes = base64.b64decode(glb_b64)
    magic = glb_bytes[:4]
    is_valid_glb = magic == b"glTF"

    print(f"  GLB size: {len(glb_bytes) / 1024:.1f} KB")
    print(f"  Textured: {textured}")
    print(f"  Valid GLB: {is_valid_glb}")
    print(f"  Time: {total_time:.2f}s | Cost: ${total_cost:.4f}")

    print(f"\n{'─'*60}")
    print(f"Phase 4 Total: {total_time:.2f}s | ${total_cost:.4f}")
    print(f"{'─'*60}")

    return {
        "glb_b64": glb_b64,
        "textured": textured,
        "valid_glb": is_valid_glb,
        "total_time": total_time,
        "total_cost": total_cost,
    }


# ── Main E2E Test ────────────────────────────────────────────────────

def main():
    """Run full E2E pipeline test."""
    if not MODAL_AVAILABLE:
        print("ERROR: Modal not available. Cannot run E2E test.")
        return

    print("\n" + "="*60)
    print("StyleLens V6 E2E Pipeline Test")
    print("="*60)
    print(f"User images: {USER_IMG_DIR}")
    print(f"Garment images: {WEAR_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load test images
    user_imgs = sorted(USER_IMG_DIR.glob("*.jpg"))
    garment_imgs = sorted(WEAR_DIR.glob("*.png"))

    if not user_imgs:
        print("ERROR: No user images found")
        return
    if not garment_imgs:
        print("ERROR: No garment images found")
        return

    # Select first user image and 4th garment image
    user_img_path = user_imgs[0]
    garment_img_path = garment_imgs[3] if len(garment_imgs) > 3 else garment_imgs[0]

    print(f"\nUser image: {user_img_path.name}")
    print(f"Garment image: {garment_img_path.name}")

    # Encode images
    user_img_b64 = image_to_b64(user_img_path)
    garment_img_b64 = image_to_b64(garment_img_path)

    # Run all phases
    phase1_results = phase1_avatar(user_img_b64)
    phase2_results = phase2_wardrobe(garment_img_b64)
    phase3_results = phase3_fitting(
        user_img_b64,
        garment_img_b64,
        phase1_results["parsemap_b64"]
    )
    phase4_results = phase4_visualization(phase3_results["tryon_b64"])

    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    # Phase 1
    detection_path = OUTPUT_DIR / "phase1_detection.json"
    detection_path.write_text(json.dumps(phase1_results["detection"], indent=2))
    print(f"  {detection_path}")

    parsemap_path = OUTPUT_DIR / "phase1_parsemap.png"
    save_b64_parsemap(phase1_results["parsemap_b64"], parsemap_path)
    print(f"  {parsemap_path}")

    # Phase 2
    garment_seg_path = OUTPUT_DIR / "phase2_garment_seg.png"
    save_b64_image(phase2_results["segmented_b64"], garment_seg_path)
    print(f"  {garment_seg_path}")

    # Phase 3
    tryon_path = OUTPUT_DIR / "phase3_tryon_result.png"
    save_b64_image(phase3_results["tryon_b64"], tryon_path)
    print(f"  {tryon_path}")

    # Phase 4
    glb_path = OUTPUT_DIR / "phase4_model.glb"
    save_glb(phase4_results["glb_b64"], glb_path)
    print(f"  {glb_path}")

    # Cost report
    total_time = (
        phase1_results["total_time"] +
        phase2_results["total_time"] +
        phase3_results["total_time"] +
        phase4_results["total_time"]
    )
    total_cost = (
        phase1_results["total_cost"] +
        phase2_results["total_cost"] +
        phase3_results["total_cost"] +
        phase4_results["total_cost"]
    )

    report_lines = [
        "StyleLens V6 E2E Pipeline — Cost Report",
        "=" * 60,
        "",
        f"User Image: {user_img_path.name}",
        f"Garment Image: {garment_img_path.name}",
        "",
        "Phase 1: Avatar",
        f"  Time: {phase1_results['total_time']:.2f}s",
        f"  Cost: ${phase1_results['total_cost']:.4f}",
        "",
        "Phase 2: Wardrobe",
        f"  Time: {phase2_results['total_time']:.2f}s",
        f"  Cost: ${phase2_results['total_cost']:.4f}",
        "",
        "Phase 3: Fitting (CatVTON-FLUX)",
        f"  Time: {phase3_results['total_time']:.2f}s",
        f"  Cost: ${phase3_results['total_cost']:.4f}",
        "",
        "Phase 4: 3D Visualization (Hunyuan3D)",
        f"  Time: {phase4_results['total_time']:.2f}s",
        f"  Cost: ${phase4_results['total_cost']:.4f}",
        f"  Textured: {phase4_results['textured']}",
        f"  Valid GLB: {phase4_results['valid_glb']}",
        "",
        "=" * 60,
        f"TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} min)",
        f"TOTAL COST: ${total_cost:.4f}",
        "=" * 60,
        "",
        "GPU: NVIDIA H200 (141GB VRAM, 4.8TB/s)",
        "Rate: $5.40/hr",
        "",
        f"Output directory: {OUTPUT_DIR}",
    ]

    report_path = OUTPUT_DIR / "e2e_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("E2E Pipeline Complete")
    print("="*60)
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
