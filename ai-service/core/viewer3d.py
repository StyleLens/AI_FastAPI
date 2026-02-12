"""
StyleLens V6 — Phase 4: 3D Viewer
Hunyuan3D 2.0 — 8 try-on images → textured 3D GLB.
"""

import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from core.config import (
    DEVICE, DTYPE, HAS_CUDA,
    HUNYUAN3D_ENABLED,
    HUNYUAN3D_PAINT_ENABLED,
    HUNYUAN3D_SHAPE_ONLY,
    HUNYUAN3D_SHAPE_STEPS,
    HUNYUAN3D_PAINT_STEPS,
    HUNYUAN3D_TEXTURE_RES,
    OUTPUT_DIR,
)
from core.loader import registry
from core.gemini_feedback import GeminiFeedbackInspector, InspectionResult
from core.sw_renderer import render_mesh_from_glb

logger = logging.getLogger("stylelens.viewer3d")


@dataclass
class Viewer3DResult:
    """Output of Phase 4 3D viewer pipeline."""
    glb_bytes: bytes = b""
    glb_id: str = ""
    glb_path: str = ""
    preview_renders: dict[int, np.ndarray] = field(default_factory=dict)
    quality_gates: list[InspectionResult] = field(default_factory=list)
    elapsed_sec: float = 0.0


def _remove_background(image_bgr: np.ndarray) -> np.ndarray:
    """Remove background from try-on image using rembg (U2Net model).

    Uses the rembg library for precise person segmentation, which is far more
    reliable than flood-fill. The flood-fill approach (Run 4) was too aggressive
    and destroyed the person, causing Hunyuan3D to produce empty geometry.

    Falls back to returning the original image if rembg fails.
    """
    try:
        from rembg import remove as rembg_remove
        from PIL import Image

        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_input = Image.fromarray(rgb)

        # rembg removes background → returns RGBA with transparent background
        pil_output = rembg_remove(pil_input)

        # Convert RGBA to RGB on white background
        if pil_output.mode == "RGBA":
            bg = Image.new("RGB", pil_output.size, (255, 255, 255))
            bg.paste(pil_output, mask=pil_output.split()[3])
            result_rgb = np.array(bg)
        else:
            result_rgb = np.array(pil_output.convert("RGB"))

        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        logger.info("Background removed via rembg (U2Net)")
        return result_bgr

    except Exception as e:
        logger.warning(f"rembg background removal failed ({e}), using original image")
        return image_bgr


def _select_best_front(tryon_images: dict[int, np.ndarray]) -> np.ndarray:
    """Select the best front-facing image as reference."""
    # Priority: 0° > 315° > 45°
    for angle in [0, 315, 45]:
        if angle in tryon_images:
            return tryon_images[angle]
    # Fallback: first available
    if tryon_images:
        return next(iter(tryon_images.values()))
    return np.zeros((512, 512, 3), dtype=np.uint8)


def _images_to_pil_list(tryon_images: dict[int, np.ndarray]) -> list[Image.Image]:
    """Convert BGR numpy images to PIL RGB list, sorted by angle."""
    result = []
    for angle in sorted(tryon_images.keys()):
        img = tryon_images[angle]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        result.append(pil)
    return result


async def generate_3d_model(
    tryon_images: dict[int, np.ndarray],
    inspector: GeminiFeedbackInspector | None = None,
) -> Viewer3DResult:
    """
    Phase 4: Hunyuan3D 2.0 — 8 try-on images → textured GLB.

    Steps:
        1. Select best front image as reference
        2. Hunyuan3D Shape pipeline → base mesh (50 steps)
        3. Hunyuan3D Paint pipeline → textured mesh (20 steps)
        3.5. Gemini Gate 6 — 3d_visualization
        4. Export as GLB with 4K textures
    """
    t0 = time.time()
    result = Viewer3DResult()
    result.glb_id = str(uuid.uuid4())[:8]

    if not tryon_images:
        logger.error("No try-on images provided for 3D generation")
        return result

    # ── Step 1: Select Reference Image ─────────────────────────
    logger.info("Phase 4 Step 1: Selecting reference image")
    front_image = _select_best_front(tryon_images)

    # Remove background to prevent Hunyuan3D from reconstructing it as geometry
    front_clean = _remove_background(front_image)
    front_pil = Image.fromarray(cv2.cvtColor(front_clean, cv2.COLOR_BGR2RGB))

    if not HUNYUAN3D_ENABLED:
        logger.warning("Hunyuan3D not available — generating placeholder GLB")
        result.glb_bytes = _generate_placeholder_glb()
        result.glb_path = _save_glb(result.glb_bytes, result.glb_id)
        result.elapsed_sec = time.time() - t0
        return result

    # ── Step 2: Hunyuan3D Shape Pipeline ───────────────────────
    logger.info("Phase 4 Step 2: Hunyuan3D shape generation")
    shape_pipe = registry.load_hunyuan3d_shape()

    shape_output = shape_pipe(
        image=front_pil,
        num_inference_steps=HUNYUAN3D_SHAPE_STEPS,
    )

    # Extract mesh: Hunyuan3D returns List[List[trimesh.Trimesh]]
    logger.info(f"Shape output type: {type(shape_output)}, "
                f"len: {len(shape_output) if hasattr(shape_output, '__len__') else 'N/A'}")
    mesh = None
    if isinstance(shape_output, list) and len(shape_output) > 0:
        batch_meshes = shape_output[0]
        if isinstance(batch_meshes, list) and len(batch_meshes) > 0:
            mesh = batch_meshes[0]
        elif hasattr(batch_meshes, 'vertices'):
            mesh = batch_meshes  # Direct trimesh object
    elif hasattr(shape_output, "meshes") and shape_output.meshes:
        mesh = shape_output.meshes[0]

    if mesh is None:
        logger.warning("Shape pipeline returned no mesh, using placeholder")
        registry.unload("hunyuan3d_shape")
        result.glb_bytes = _generate_placeholder_glb()
        result.glb_path = _save_glb(result.glb_bytes, result.glb_id)
        result.elapsed_sec = time.time() - t0
        return result

    registry.unload("hunyuan3d_shape")

    # ── Step 3: Hunyuan3D Paint Pipeline (or shape-only) ──────
    if HUNYUAN3D_PAINT_ENABLED:
        logger.info("Phase 4 Step 3: Hunyuan3D texture painting (CUDA)")
        paint_pipe = registry.load_hunyuan3d_paint()

        reference_images = _images_to_pil_list(tryon_images)

        paint_output = paint_pipe(
            mesh=mesh,
            image=front_pil,
            reference_images=reference_images,
            num_inference_steps=HUNYUAN3D_PAINT_STEPS,
            texture_resolution=HUNYUAN3D_TEXTURE_RES,
        )

        if hasattr(paint_output, "meshes") and paint_output.meshes:
            textured_mesh = paint_output.meshes[0]
        else:
            textured_mesh = mesh  # Use untextured shape

        registry.unload("hunyuan3d_paint")
    else:
        # Shape-only mode: export untextured mesh (MPS/CPU fallback)
        logger.info("Phase 4 Step 3: Shape-only mode (no CUDA) — skipping texture painting")
        textured_mesh = mesh
        # Apply a basic solid color so the GLB isn't invisible in viewers
        try:
            import trimesh
            if hasattr(textured_mesh, "visual"):
                textured_mesh.visual.face_colors = [180, 180, 190, 255]
        except Exception:
            pass

    # ── Export GLB ─────────────────────────────────────────────
    logger.info("Phase 4 Step 4: Exporting GLB")

    buf = io.BytesIO()
    if hasattr(textured_mesh, "export"):
        textured_mesh.export(buf, file_type="glb")
    else:
        # Try trimesh export
        import trimesh
        if isinstance(textured_mesh, trimesh.Trimesh):
            textured_mesh.export(buf, file_type="glb")
        else:
            buf.write(_generate_placeholder_glb())

    result.glb_bytes = buf.getvalue()
    result.glb_path = _save_glb(result.glb_bytes, result.glb_id)

    # Generate preview renders at higher resolution for quality inspection
    for angle in [0, 90, 180, 270]:
        try:
            render = render_mesh_from_glb(result.glb_bytes, angle, 768)
            result.preview_renders[angle] = render
        except Exception as e:
            logger.warning(f"Preview render failed for {angle}°: {e}")

    # ── Step 3.5: Gemini Gate 6 — 3D Visualization ────────────
    if inspector and result.preview_renders:
        tryon_list = [tryon_images.get(0, front_image), tryon_images.get(180, front_image)]
        glb_list = [result.preview_renders.get(0, front_image),
                    result.preview_renders.get(180, front_image)]
        gate6 = inspector.inspect_3d_visualization(tryon_list, glb_list)
        result.quality_gates.append(gate6)
        if not gate6.pass_check:
            logger.warning(f"Gate 6 failed: {gate6.feedback}")

    result.elapsed_sec = time.time() - t0
    logger.info(f"Phase 4 complete in {result.elapsed_sec:.1f}s")
    return result


def _save_glb(glb_bytes: bytes, glb_id: str) -> str:
    """Save GLB to output directory."""
    output_path = OUTPUT_DIR / f"model_{glb_id}.glb"
    output_path.write_bytes(glb_bytes)
    return str(output_path)


def _generate_placeholder_glb() -> bytes:
    """Generate a simple placeholder GLB when Hunyuan3D is unavailable."""
    import trimesh
    # Simple capsule as placeholder
    mesh = trimesh.creation.capsule(height=1.7, radius=0.15, count=[16, 8])
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


def get_glb_path(glb_id: str) -> str | None:
    """Get path to a generated GLB file by ID."""
    path = OUTPUT_DIR / f"model_{glb_id}.glb"
    if path.exists():
        return str(path)
    return None
