"""
StyleLens V6 — Modal GPU Worker (Tier 4)
Serverless GPU on NVIDIA H100 (80GB VRAM).
modal run 방식 — 배포 없이 GPU만 빌려 사용.

사용법:
  로컬 오케스트레이터에서 `modal.Function.lookup()` → `.remote()` 호출
  또는 `modal run worker/modal_app.py` 로 테스트

Three GPU functions:
  1. run_light_models — SAM3, SAM3D, FASHN (~4GB)
  2. run_catvton_batch — CatVTON-FLUX 8-angle try-on (~27GB)
  3. run_hunyuan3d — Hunyuan3D shape + paint (~20GB)
"""

import io
import logging
import time

logger = logging.getLogger("stylelens.worker")

# ── Attempt Modal import ──
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    logger.info("Modal not installed — worker runs in local-only mode")

from worker.serialization import (
    ndarray_to_b64, b64_to_ndarray,
    image_to_b64, b64_to_image,
    parsemap_to_b64, b64_to_parsemap,
    bytes_to_b64, b64_to_bytes,
)

from core.config import IS_LOCAL_DEV, MODEL_DIR

_MODEL_ROOT = str(MODEL_DIR)


# ── Modal App Definition ─────────────────────────────────────

if MODAL_AVAILABLE:
    app = modal.App("stylelens-v6-worker")

    # Persistent volume for model weights
    model_volume = modal.Volume.from_name(
        "stylelens-models", create_if_missing=True,
    )

    # Base image with all ML dependencies
    worker_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.6", "torchvision>=0.21",
            "transformers>=4.48", "diffusers>=0.32",
            "safetensors>=0.5", "accelerate>=1.3",
            "opencv-python-headless>=4.10", "Pillow>=11.0",
            "numpy>=2.2", "trimesh>=4.6",
            "ultralytics>=8.4",
            "peft>=0.15",
            "huggingface_hub>=0.28",
        )
    )
else:
    app = None
    model_volume = None
    worker_image = None


# ── Helper Functions ─────────────────────────────────────────

def _decode_input_image(image_b64: str):
    """Decode base64 to BGR numpy array."""
    return b64_to_image(image_b64)


def _encode_output_image(img_bgr, quality: int = 90) -> str:
    """Encode BGR numpy array to JPEG base64."""
    return image_to_b64(img_bgr, quality=quality)


# ── GPU Functions (modal run 방식) ───────────────────────────

if MODAL_AVAILABLE:

    @app.function(
        image=worker_image,
        gpu="H100",
        volumes={"/models": model_volume},
        timeout=300,
        memory=16384,
    )
    def run_light_models(task: str, image_b64: str) -> dict:
        """Light GPU tasks — SAM3, SAM3D, FASHN (~4GB VRAM).

        Args:
            task: "reconstruct_3d" | "segment_sam3" | "parse_fashn"
            image_b64: Base64-encoded input image

        Returns:
            Task-specific result dict
        """
        import torch
        import numpy as np
        import cv2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        img = _decode_input_image(image_b64)
        t0 = time.time()

        if task == "reconstruct_3d":
            from transformers import AutoModel, AutoProcessor
            model_path = f"{_MODEL_ROOT}/sam3d_body"
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(device)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            vertices = outputs.vertices[0].cpu().numpy().astype(np.float32)
            faces = outputs.faces[0].cpu().numpy().astype(np.int32) if hasattr(outputs, 'faces') else np.zeros((0, 3), dtype=np.int32)
            joints = outputs.joints[0].cpu().numpy().astype(np.float32) if hasattr(outputs, 'joints') else np.zeros((24, 3), dtype=np.float32)
            betas = outputs.betas[0].cpu().numpy().astype(np.float32) if hasattr(outputs, 'betas') else np.zeros(10, dtype=np.float32)

            del model, processor
            torch.cuda.empty_cache()

            return {
                "vertices": ndarray_to_b64(vertices),
                "faces": ndarray_to_b64(faces),
                "joints": ndarray_to_b64(joints),
                "betas": ndarray_to_b64(betas),
                "elapsed_sec": time.time() - t0,
            }

        elif task == "segment_sam3":
            from transformers import AutoModel, AutoProcessor
            model_path = f"{_MODEL_ROOT}/sam3"
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(device)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            masks = outputs.pred_masks[0].cpu().numpy()
            scores = outputs.iou_predictions[0].cpu().numpy()
            best_idx = scores.argmax()
            mask = (masks[best_idx] > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            segmented = img.copy()
            segmented[mask == 0] = [220, 220, 220]

            del model, processor
            torch.cuda.empty_cache()

            return {
                "segmented_b64": _encode_output_image(segmented),
                "mask_b64": parsemap_to_b64(mask),
                "elapsed_sec": time.time() - t0,
            }

        elif task == "parse_fashn":
            from transformers import AutoModelForSemanticSegmentation, AutoProcessor
            model_path = f"{_MODEL_ROOT}/fashn_parser"
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_path).to(device)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=img.shape[:2], mode="bilinear", align_corners=False,
            )
            parse_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            del model, processor
            torch.cuda.empty_cache()

            return {
                "parsemap_b64": parsemap_to_b64(parse_map),
                "elapsed_sec": time.time() - t0,
            }

        else:
            raise ValueError(f"Unknown task: {task}")

    @app.function(
        image=worker_image,
        gpu="H100",
        volumes={"/models": model_volume},
        timeout=600,
        memory=32768,
    )
    def run_catvton_batch(
        persons_b64: list[str],
        clothing_b64: str,
        masks_b64: list[str],
        num_steps: int = 30,
        guidance: float = 3.5,
        strength: float = 0.85,
    ) -> dict:
        """CatVTON-FLUX batch try-on — all 8 angles (~27GB VRAM).

        Returns:
            {"results_b64": [str, ...], "elapsed_sec": float}
        """
        import torch
        import numpy as np
        import cv2

        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CatVTON-FLUX pipeline
        logger.info("Loading CatVTON-FLUX pipeline...")
        from core.catvton_pipeline import CatVTONFluxPipeline
        from core.config import FLUX_GGUF_PATH, CATVTON_LORA_DIR, CATVTON_ATTN_DIR

        pipeline = CatVTONFluxPipeline(
            flux_gguf_path=str(FLUX_GGUF_PATH),
            lora_dir=str(CATVTON_LORA_DIR),
            attn_dir=str(CATVTON_ATTN_DIR),
            device=device,
        )

        clothing_img = _decode_input_image(clothing_b64)
        results = []

        for i, (person_b64, mask_b64) in enumerate(zip(persons_b64, masks_b64)):
            person_img = _decode_input_image(person_b64)
            mask_img = b64_to_parsemap(mask_b64)

            try:
                pil_result = pipeline.try_on(
                    person_img, clothing_img, mask_img,
                    num_steps=num_steps,
                    guidance_scale=guidance,
                    strength=strength,
                )
                result_bgr = cv2.cvtColor(np.array(pil_result), cv2.COLOR_RGB2BGR)
                results.append(_encode_output_image(result_bgr))
            except Exception as e:
                logger.error(f"CatVTON failed for angle {i}: {e}")
                results.append("")

        del pipeline
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        logger.info(f"CatVTON batch: {len(results)} angles in {elapsed:.1f}s")

        return {
            "results_b64": results,
            "elapsed_sec": elapsed,
        }

    @app.function(
        image=worker_image,
        gpu="H100",
        volumes={"/models": model_volume},
        timeout=600,
        memory=32768,
    )
    def run_hunyuan3d(
        front_image_b64: str,
        reference_images_b64: list[str] | None = None,
        shape_steps: int = 50,
        paint_steps: int = 20,
        texture_res: int = 4096,
    ) -> dict:
        """Hunyuan3D 2.0 shape + paint → GLB (~20GB VRAM).

        Returns:
            {"glb_bytes_b64": str, "elapsed_sec": float}
        """
        import torch
        import trimesh

        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        front_img = _decode_input_image(front_image_b64)
        reference_imgs = []
        if reference_images_b64:
            for ref_b64 in reference_images_b64:
                if ref_b64:
                    reference_imgs.append(_decode_input_image(ref_b64))

        # Step 1: Shape generation
        logger.info("Hunyuan3D: Shape pipeline...")
        # TODO: Load actual Hunyuan3D shape model from /models/hunyuan3d
        mesh = trimesh.primitives.Capsule(height=1.7, radius=0.2)

        if device == "cuda":
            torch.cuda.empty_cache()

        # Step 2: Paint (texture)
        logger.info("Hunyuan3D: Paint pipeline...")
        # TODO: Load actual Hunyuan3D paint model
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")
        glb_bytes = buf.getvalue()

        if device == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        logger.info(f"Hunyuan3D full pipeline: {elapsed:.1f}s")

        return {
            "glb_bytes_b64": bytes_to_b64(glb_bytes),
            "elapsed_sec": elapsed,
        }

    # ── Local entrypoint for testing ──────────────────────────

    @app.local_entrypoint()
    def main():
        """Test: modal run worker/modal_app.py"""
        import json

        print("=== StyleLens V6 GPU Worker Test ===")
        print(f"App: {app.name}")

        # Quick GPU health check
        @app.function(image=worker_image, gpu="H100", timeout=30)
        def gpu_health():
            import torch
            return {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
            }

        result = gpu_health.remote()
        print(f"GPU: {json.dumps(result, indent=2)}")
        print("✅ Worker ready!")
