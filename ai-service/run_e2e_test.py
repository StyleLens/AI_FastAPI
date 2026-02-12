#!/usr/bin/env python3
"""
StyleLens V6 — End-to-End Pipeline Test
Runs all 4 phases with real sample data from IMG_Data/.
Saves per-phase results to output/e2e_results/ for index.html inspection.
"""

import asyncio
import base64
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e_test")

# ── Data Paths ────────────────────────────────────────────────
IMG_DATA = BASE_DIR.parent / "IMG_Data"
USER_IMG_DIR = IMG_DATA / "User_IMG"
USER_VOD_DIR = IMG_DATA / "User_VOD"
WEAR_DIR = IMG_DATA / "wear"
WEAR_SIZE_DIR = IMG_DATA / "wearSize"

# ── Output ────────────────────────────────────────────────────
RESULTS_DIR = BASE_DIR / "output" / "e2e_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_images_from_dir(directory: Path, max_count: int = 20) -> list[np.ndarray]:
    """Load all images from a directory (sorted by name)."""
    images = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(sorted(directory.glob(ext)))
    for p in paths[:max_count]:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Convert RGBA to BGR if needed
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            images.append(img)
            logger.info(f"  Loaded: {p.name} ({img.shape})")
    return images


def save_image(img: np.ndarray, name: str, subdir: str = ""):
    """Save an image to the results directory."""
    out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    cv2.imwrite(str(path), img)
    return str(path)


def image_to_base64(img: np.ndarray) -> str:
    """Convert BGR image to base64 data URI for HTML embedding."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


async def run_phase1(user_images: list[np.ndarray], video_path: str | None) -> dict:
    """Phase 1: Avatar — YOLO26 person detection + SAM 3D Body reconstruction."""
    from core.pipeline import generate_avatar, Metadata
    from core.gemini_client import GeminiClient
    from core.gemini_feedback import GeminiFeedbackInspector

    logger.info("=" * 60)
    logger.info("PHASE 1: Avatar Pipeline")
    logger.info("=" * 60)

    gemini = GeminiClient()
    inspector = GeminiFeedbackInspector()

    # Use photos as input (more reliable than video for still poses)
    metadata = Metadata(
        gender="female",
        height_cm=168.0,  # 카리나 approx height
        weight_kg=47.0,
        age=24,
        body_type="slim",
    )

    t0 = time.time()
    body_data = await generate_avatar(
        images=user_images[:3],  # Use first 3 photos
        metadata=metadata,
        inspector=inspector,
    )
    elapsed = time.time() - t0

    # Save results
    phase1_result = {
        "elapsed_sec": elapsed,
        "has_vertices": body_data.vertices is not None,
        "vertex_count": len(body_data.vertices) if body_data.vertices is not None else 0,
        "face_count": len(body_data.faces) if body_data.faces is not None else 0,
        "has_glb": len(body_data.glb_bytes) > 0,
        "glb_size_kb": len(body_data.glb_bytes) / 1024,
        "bbox": body_data.person_bbox,
        "gender": body_data.gender,
        "quality_gates": [],
        "renders": {},
    }

    for gate in body_data.quality_gates:
        phase1_result["quality_gates"].append({
            "stage": gate.stage,
            "score": gate.quality_score,
            "pass": gate.pass_check,
            "feedback": gate.feedback,
        })

    # Save mesh renders
    for angle, render in body_data.mesh_renders.items():
        fname = f"phase1_mesh_{angle}deg.jpg"
        save_image(render, fname, "phase1")
        phase1_result["renders"][angle] = f"phase1/{fname}"

    # Save person image
    if body_data.person_image is not None:
        save_image(body_data.person_image, "phase1_person_input.jpg", "phase1")

    # Save GLB
    if body_data.glb_bytes:
        glb_path = RESULTS_DIR / "phase1" / "avatar.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(body_data.glb_bytes)
        phase1_result["glb_file"] = "phase1/avatar.glb"

    logger.info(f"Phase 1 done: {phase1_result['vertex_count']} vertices, "
                f"{elapsed:.1f}s, GLB={phase1_result['glb_size_kb']:.0f}KB")

    return {"body_data": body_data, "result": phase1_result}


async def run_phase2(wear_images: list[np.ndarray],
                      size_images: list[np.ndarray]) -> dict:
    """Phase 2: Wardrobe — SAM3 segmentation + FASHN parsing + Gemini analysis."""
    from core.wardrobe import analyze_clothing
    from core.gemini_client import GeminiClient
    from core.gemini_feedback import GeminiFeedbackInspector

    logger.info("=" * 60)
    logger.info("PHASE 2: Wardrobe Pipeline")
    logger.info("=" * 60)

    gemini = GeminiClient()
    inspector = GeminiFeedbackInspector()

    t0 = time.time()
    clothing_item = await analyze_clothing(
        images=wear_images,
        gemini=gemini,
        size_chart_image=size_images[0] if size_images else None,
        product_info_images=size_images[1:] if len(size_images) > 1 else None,
        fitting_model_image=size_images[-1] if size_images else None,
        inspector=inspector,
    )
    elapsed = time.time() - t0

    phase2_result = {
        "elapsed_sec": elapsed,
        "analysis": {
            "name": clothing_item.analysis.name,
            "category": clothing_item.analysis.category,
            "subcategory": clothing_item.analysis.subcategory,
            "color": clothing_item.analysis.color,
            "color_hex": clothing_item.analysis.color_hex,
            "fabric": clothing_item.analysis.fabric,
            "fabric_composition": clothing_item.analysis.fabric_composition,
            "surface_texture": getattr(clothing_item.analysis, "surface_texture", ""),
            "fit_type": clothing_item.analysis.fit_type,
            "neck_style": clothing_item.analysis.neck_style,
            "sleeve_type": clothing_item.analysis.sleeve_type,
            "drape_style": clothing_item.analysis.drape_style,
            "pattern_type": clothing_item.analysis.pattern_type,
            "closure_type": clothing_item.analysis.closure_type,
            "button_count": clothing_item.analysis.button_count,
            "hem_style": clothing_item.analysis.hem_style,
            "thickness": clothing_item.analysis.thickness,
            "elasticity": clothing_item.analysis.elasticity,
            "transparency": clothing_item.analysis.transparency,
        },
        "has_segmented": clothing_item.segmented_image is not None,
        "has_garment_mask": clothing_item.garment_mask is not None,
        "has_parse_map": clothing_item.parse_map is not None,
        "size_chart": clothing_item.size_chart,
        "product_info": clothing_item.product_info,
        "fitting_model_info": clothing_item.fitting_model_info,
        "quality_gates": [],
    }

    for gate in clothing_item.quality_gates:
        phase2_result["quality_gates"].append({
            "stage": gate.stage,
            "score": gate.quality_score,
            "pass": gate.pass_check,
            "feedback": gate.feedback,
        })

    # Save segmented image
    if clothing_item.segmented_image is not None:
        save_image(clothing_item.segmented_image, "phase2_segmented.jpg", "phase2")

    # Save garment mask
    if clothing_item.garment_mask is not None:
        save_image(clothing_item.garment_mask, "phase2_garment_mask.jpg", "phase2")

    # Save parse map as colorized
    if clothing_item.parse_map is not None:
        cmap = cv2.applyColorMap(
            (clothing_item.parse_map * 14).astype(np.uint8), cv2.COLORMAP_JET
        )
        save_image(cmap, "phase2_parse_map.jpg", "phase2")

    # Save original clothing images
    for i, img in enumerate(wear_images[:5]):
        save_image(img, f"phase2_clothing_input_{i}.jpg", "phase2")

    logger.info(f"Phase 2 done: {clothing_item.analysis.name} "
                f"({clothing_item.analysis.category}), {elapsed:.1f}s")

    return {"clothing_item": clothing_item, "result": phase2_result}


async def run_phase3(body_data, clothing_item,
                      face_photos: list[np.ndarray]) -> dict:
    """Phase 3: Fitting — Gemini 8-angle try-on with face identity."""
    from core.fitting import generate_fitting
    from core.gemini_client import GeminiClient
    from core.gemini_feedback import GeminiFeedbackInspector

    logger.info("=" * 60)
    logger.info("PHASE 3: Fitting Pipeline (8 angles)")
    logger.info("=" * 60)

    gemini = GeminiClient()
    inspector = GeminiFeedbackInspector()

    # Use first face photo as primary
    face_photo = face_photos[0] if face_photos else None

    # Build Face Bank from user photos for identity preservation
    face_bank = None
    if face_photos and len(face_photos) >= 2:
        try:
            from core.face_bank import FaceBankBuilder
            from core.loader import registry as _reg
            face_app = _reg.load_insightface()
            builder = FaceBankBuilder(face_app)
            for i, photo in enumerate(face_photos[:8]):  # Up to 8 reference photos
                builder.add_reference(photo, label=f"user_photo_{i}")
            face_bank = builder.build()
            logger.info(f"Face Bank built: {len(face_bank.references)} references, "
                        f"coverage={face_bank.angle_coverage()}")
            _reg.unload_except()
        except Exception as e:
            logger.warning(f"Face Bank construction failed: {e}")
            face_bank = None

    t0 = time.time()
    fitting_result = await generate_fitting(
        body_data=body_data,
        clothing_item=clothing_item,
        gemini=gemini,
        face_photo=face_photo,
        inspector=inspector,
        face_bank=face_bank,
    )
    elapsed = time.time() - t0

    phase3_result = {
        "elapsed_sec": elapsed,
        "angles_generated": list(fitting_result.tryon_images.keys()),
        "method_used": fitting_result.method_used,
        "p2p_result": None,
        "quality_gates": [],
        "tryon_images": {},
    }

    if fitting_result.p2p_result:
        phase3_result["p2p_result"] = {
            "overall_tightness": fitting_result.p2p_result.overall_tightness.value,
            "mask_expansion_factor": fitting_result.p2p_result.mask_expansion_factor,
            "physics_prompt": fitting_result.p2p_result.physics_prompt[:200] if fitting_result.p2p_result.physics_prompt else "",
        }

    for gate in fitting_result.quality_gates:
        phase3_result["quality_gates"].append({
            "stage": gate.stage,
            "score": gate.quality_score,
            "pass": gate.pass_check,
            "feedback": gate.feedback,
        })

    # Save try-on images
    for angle, img in fitting_result.tryon_images.items():
        fname = f"phase3_tryon_{angle}deg.jpg"
        save_image(img, fname, "phase3")
        phase3_result["tryon_images"][angle] = f"phase3/{fname}"

    logger.info(f"Phase 3 done: {len(fitting_result.tryon_images)} angles, "
                f"{elapsed:.1f}s")

    return {"fitting_result": fitting_result, "result": phase3_result}


async def run_phase4(fitting_result) -> dict:
    """Phase 4: 3D Viewer — Hunyuan3D shape-only GLB."""
    from core.viewer3d import generate_3d_model
    from core.gemini_feedback import GeminiFeedbackInspector

    logger.info("=" * 60)
    logger.info("PHASE 4: 3D Viewer (Hunyuan3D Shape-Only)")
    logger.info("=" * 60)

    inspector = GeminiFeedbackInspector()

    t0 = time.time()
    viewer_result = await generate_3d_model(
        tryon_images=fitting_result.tryon_images,
        inspector=inspector,
    )
    elapsed = time.time() - t0

    phase4_result = {
        "elapsed_sec": elapsed,
        "has_glb": len(viewer_result.glb_bytes) > 0,
        "glb_size_kb": len(viewer_result.glb_bytes) / 1024,
        "glb_id": viewer_result.glb_id,
        "quality_gates": [],
        "preview_renders": {},
    }

    for gate in viewer_result.quality_gates:
        phase4_result["quality_gates"].append({
            "stage": gate.stage,
            "score": gate.quality_score,
            "pass": gate.pass_check,
            "feedback": gate.feedback,
        })

    # Save preview renders
    for angle, render in viewer_result.preview_renders.items():
        fname = f"phase4_preview_{angle}deg.jpg"
        save_image(render, fname, "phase4")
        phase4_result["preview_renders"][angle] = fname

    # Save GLB
    if viewer_result.glb_bytes:
        glb_path = RESULTS_DIR / "phase4" / f"model_{viewer_result.glb_id}.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(viewer_result.glb_bytes)
        phase4_result["glb_file"] = f"phase4/model_{viewer_result.glb_id}.glb"

    logger.info(f"Phase 4 done: GLB={phase4_result['glb_size_kb']:.0f}KB, {elapsed:.1f}s")

    return {"viewer_result": viewer_result, "result": phase4_result}


async def main():
    """Run full E2E test pipeline."""
    logger.info("=" * 60)
    logger.info("StyleLens V6 — E2E Pipeline Test")
    logger.info("=" * 60)

    # Verify data paths
    assert USER_IMG_DIR.exists(), f"Missing: {USER_IMG_DIR}"
    assert WEAR_DIR.exists(), f"Missing: {WEAR_DIR}"

    # Print model status
    from core.config import get_model_status, DEVICE, HAS_CUDA, GEMINI_ENABLED
    status = get_model_status()
    logger.info(f"Device: {DEVICE}, CUDA: {HAS_CUDA}, Gemini: {GEMINI_ENABLED}")
    for name, enabled in status.items():
        symbol = "✅" if enabled else "❌"
        logger.info(f"  {symbol} {name}")

    # Load sample data
    logger.info("\nLoading sample data...")
    user_images = load_images_from_dir(USER_IMG_DIR)
    wear_images = load_images_from_dir(WEAR_DIR)
    size_images = load_images_from_dir(WEAR_SIZE_DIR)
    logger.info(f"Loaded: {len(user_images)} user photos, "
                f"{len(wear_images)} clothing, {len(size_images)} size info")

    # Video path
    vod_files = list(USER_VOD_DIR.glob("*.mp4"))
    video_path = str(vod_files[0]) if vod_files else None
    logger.info(f"Video: {video_path}")

    all_results = {"phases": {}, "total_elapsed": 0.0}
    total_t0 = time.time()

    # ── Phase 1: Avatar ──────────────────────────────────────
    try:
        p1 = await run_phase1(user_images, video_path)
        all_results["phases"]["phase1"] = p1["result"]
        body_data = p1["body_data"]
    except Exception as e:
        logger.error(f"Phase 1 FAILED: {e}", exc_info=True)
        all_results["phases"]["phase1"] = {"error": str(e)}
        body_data = None

    # ── Phase 2: Wardrobe ────────────────────────────────────
    try:
        p2 = await run_phase2(wear_images, size_images)
        all_results["phases"]["phase2"] = p2["result"]
        clothing_item = p2["clothing_item"]
    except Exception as e:
        logger.error(f"Phase 2 FAILED: {e}", exc_info=True)
        all_results["phases"]["phase2"] = {"error": str(e)}
        clothing_item = None

    # ── Phase 3: Fitting ─────────────────────────────────────
    if body_data and clothing_item:
        try:
            p3 = await run_phase3(body_data, clothing_item, user_images)
            all_results["phases"]["phase3"] = p3["result"]
            fitting_result = p3["fitting_result"]
        except Exception as e:
            logger.error(f"Phase 3 FAILED: {e}", exc_info=True)
            all_results["phases"]["phase3"] = {"error": str(e)}
            fitting_result = None
    else:
        logger.warning("Skipping Phase 3: Phase 1 or 2 failed")
        all_results["phases"]["phase3"] = {"error": "Skipped — dependency failed"}
        fitting_result = None

    # ── Phase 4: 3D Viewer ───────────────────────────────────
    if fitting_result and fitting_result.tryon_images:
        try:
            p4 = await run_phase4(fitting_result)
            all_results["phases"]["phase4"] = p4["result"]
        except Exception as e:
            logger.error(f"Phase 4 FAILED: {e}", exc_info=True)
            all_results["phases"]["phase4"] = {"error": str(e)}
    else:
        logger.warning("Skipping Phase 4: Phase 3 failed or no images")
        all_results["phases"]["phase4"] = {"error": "Skipped — no try-on images"}

    all_results["total_elapsed"] = time.time() - total_t0

    # ── Save results JSON ────────────────────────────────────
    results_json_path = RESULTS_DIR / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_json_path}")

    # ── Save input references ────────────────────────────────
    ref_dir = RESULTS_DIR / "inputs"
    ref_dir.mkdir(exist_ok=True)
    for i, img in enumerate(user_images[:5]):
        save_image(img, f"user_photo_{i}.jpg", "inputs")
    for i, img in enumerate(wear_images[:5]):
        save_image(img, f"clothing_{i}.jpg", "inputs")
    for i, img in enumerate(size_images[:3]):
        save_image(img, f"size_info_{i}.jpg", "inputs")

    # ── Summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("E2E TEST SUMMARY")
    logger.info("=" * 60)
    for phase_name, phase_data in all_results["phases"].items():
        if "error" in phase_data:
            logger.info(f"  ❌ {phase_name}: {phase_data['error'][:80]}")
        else:
            elapsed = phase_data.get("elapsed_sec", 0)
            gates = phase_data.get("quality_gates", [])
            gate_str = ", ".join(
                f"{g['stage']}={'✅' if g['pass'] else '❌'}{g['score']:.2f}"
                for g in gates
            )
            logger.info(f"  ✅ {phase_name}: {elapsed:.1f}s | {gate_str}")

    logger.info(f"\nTotal: {all_results['total_elapsed']:.1f}s")
    logger.info(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
