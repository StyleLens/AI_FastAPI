"""
StyleLens V6 SOTA Pipeline — Model Loader
Singleton ModelRegistry with lazy load/unload + MPS memory management.
"""

import gc
import logging
import time
from typing import Any

import torch

from core.config import (
    BASE_DIR, DEVICE, DTYPE, HAS_CUDA,
    YOLO26_MODEL_PATH, YOLO26_ENABLED,
    SAM3_MODEL_DIR, SAM3_ENABLED,
    SAM3D_BODY_MODEL_DIR, SAM3D_BODY_ENABLED,
    SAM3D_BODY_CKPT_PATH, SAM3D_BODY_MHR_PATH,
    FASHN_PARSER_DIR, FASHN_PARSER_ENABLED,
    CATVTON_FLUX_DIR, CATVTON_FLUX_ENABLED,
    CATVTON_LORA_DIR, CATVTON_ATTN_DIR,
    FLUX_GGUF_PATH, FLUX_GGUF_ENABLED,
    HUNYUAN3D_DIR, HUNYUAN3D_ENABLED,
    HUNYUAN3D_PAINT_ENABLED,
    INSIGHTFACE_MODEL_DIR, INSIGHTFACE_ENABLED,
    MPS_FALLBACK_DEVICE, get_device_for_model,
)

logger = logging.getLogger("stylelens.loader")


class ModelRegistry:
    """Singleton registry for lazy model loading with sequential unload."""

    _instance = None
    _models: dict[str, Any] = {}
    _load_times: dict[str, float] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._load_times = {}
        return cls._instance

    # ── Memory Management ──────────────────────────────────────

    def _clear_cache(self):
        """Free MPS/CUDA memory."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self, name: str):
        """Unload a model and free memory."""
        if name in self._models:
            del self._models[name]
            self._clear_cache()
            logger.info(f"Unloaded model: {name}")

    def unload_all(self):
        """Unload all models."""
        names = list(self._models.keys())
        self._models.clear()
        self._clear_cache()
        if names:
            logger.info(f"Unloaded all models: {names}")

    def unload_except(self, *keep: str):
        """Unload all models except the specified ones."""
        to_remove = [n for n in self._models if n not in keep]
        for name in to_remove:
            del self._models[name]
        if to_remove:
            self._clear_cache()
            logger.info(f"Unloaded models: {to_remove}, kept: {list(keep)}")

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    # ── Generic Loader Wrapper ─────────────────────────────────

    def _load(self, name: str, loader_fn) -> Any:
        """Load model if not already cached."""
        if name in self._models:
            return self._models[name]
        logger.info(f"Loading model: {name}...")
        t0 = time.time()
        model = loader_fn()
        elapsed = time.time() - t0
        self._models[name] = model
        self._load_times[name] = elapsed
        logger.info(f"Loaded {name} in {elapsed:.1f}s")
        return model

    # ── 1. YOLO26-L ────────────────────────────────────────────

    def load_yolo26(self):
        """YOLO26-L for NMS-free person detection."""
        def _load():
            from ultralytics import YOLO
            model = YOLO(str(YOLO26_MODEL_PATH))
            return model
        return self._load("yolo26", _load)

    # ── 2. SAM 3 ──────────────────────────────────────────────

    def load_sam3(self):
        """SAM 3 for concept-aware segmentation."""
        def _load():
            from segment_anything_3 import sam_model_registry, SamPredictor
            checkpoint = next(SAM3_MODEL_DIR.glob("*.safetensors"))
            sam = sam_model_registry["default"](checkpoint=str(checkpoint))
            sam.to(DEVICE)
            predictor = SamPredictor(sam)
            return predictor
        return self._load("sam3", _load)

    # ── 3. SAM 3D Body DINOv3 ─────────────────────────────────

    def load_sam3d_body(self):
        """SAM 3D Body DINOv3 for single-image 3D body recovery.

        Uses native sam_3d_body package (GitHub: facebookresearch/sam-3d-body).
        Model: DINOv3-H+ backbone + SAM decoder + MHR head (1285M params).
        Returns SAM3DBodyEstimator instance with .process_one_image() API.
        CPU for MPS environments (no CUDA extensions needed for inference).
        """
        def _load():
            import sys
            # Add sam-3d-body repo to path
            sam3d_repo = str(BASE_DIR / "sam-3d-body")
            if sam3d_repo not in sys.path:
                sys.path.insert(0, sam3d_repo)

            from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
            device = get_device_for_model("sam3d_body")
            model, model_cfg = load_sam_3d_body(
                checkpoint_path=str(SAM3D_BODY_CKPT_PATH),
                device=device,
                mhr_path=str(SAM3D_BODY_MHR_PATH),
            )
            estimator = SAM3DBodyEstimator(
                sam_3d_body_model=model,
                model_cfg=model_cfg,
            )
            return {
                "estimator": estimator,
                "model": model,
                "config": model_cfg,
                "device": device,
            }
        return self._load("sam3d_body", _load)

    # ── 4. FASHN Parser ───────────────────────────────────────

    def load_fashn_parser(self):
        """FASHN SegFormer-B4 for 18-class fashion body parsing."""
        def _load():
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            processor = SegformerImageProcessor.from_pretrained(
                str(FASHN_PARSER_DIR), local_files_only=True
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                str(FASHN_PARSER_DIR), local_files_only=True,
                torch_dtype=DTYPE,
            ).to(DEVICE)
            model.eval()
            return {"model": model, "processor": processor}
        return self._load("fashn_parser", _load)

    # ── 5. CatVTON-FLUX (LoRA + attention weights) ────────────

    def load_catvton_flux(self):
        """CatVTON-FLUX LoRA + mix attention weights for virtual try-on."""
        def _load():
            from core.catvton_pipeline import CatVTONFluxPipeline
            pipeline = CatVTONFluxPipeline.from_pretrained(
                flux_gguf_path=str(FLUX_GGUF_PATH),
                catvton_lora_dir=str(CATVTON_LORA_DIR),
                catvton_attn_dir=str(CATVTON_ATTN_DIR),
                device=DEVICE,
                dtype=DTYPE,
            )
            return pipeline
        return self._load("catvton_flux", _load)

    # ── 6. FLUX.1-dev GGUF Q8 (loaded as part of CatVTON) ────
    # Note: FLUX base model is loaded internally by CatVTON-FLUX pipeline.
    # This loader is for standalone usage if needed.

    def load_flux_gguf(self):
        """FLUX.1-dev Q8 quantized base diffusion model."""
        def _load():
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_single_file(
                str(FLUX_GGUF_PATH),
                torch_dtype=DTYPE,
            ).to(DEVICE)
            return pipe
        return self._load("flux_gguf", _load)

    # ── 7. Hunyuan3D 2.0 Shape ────────────────────────────────

    def load_hunyuan3d_shape(self):
        """Hunyuan3D 2.0 shape generation pipeline (via hy3dgen)."""
        def _load():
            import os
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            # hy3dgen uses HY3DGEN_MODELS env to resolve local paths
            os.environ.setdefault("HY3DGEN_MODELS", str(HUNYUAN3D_DIR.parent.parent))
            device = get_device_for_model("hunyuan3d_shape")
            dtype = torch.float32 if device == "cpu" else torch.float16
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                str(HUNYUAN3D_DIR),
                subfolder="hunyuan3d-dit-v2-0-turbo",
                use_safetensors=True,
                variant="fp16",
                device=device,
                dtype=dtype,
            )
            return pipe
        return self._load("hunyuan3d_shape", _load)

    # ── 8. Hunyuan3D 2.0 Paint ────────────────────────────────

    def load_hunyuan3d_paint(self):
        """Hunyuan3D 2.0 texture painting pipeline (via hy3dgen).

        NOTE: Requires CUDA for custom rasterizer. On MPS/CPU, raises RuntimeError.
        Use HUNYUAN3D_PAINT_ENABLED to check before calling.
        """
        if not HAS_CUDA:
            raise RuntimeError(
                "Hunyuan3D Paint requires CUDA (custom rasterizer). "
                "On MPS/CPU, use shape-only mode (HUNYUAN3D_SHAPE_ONLY=True)."
            )
        def _load():
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            pipe = Hunyuan3DPaintPipeline.from_pretrained(
                str(HUNYUAN3D_DIR),
                subfolder="hunyuan3d-paint-v2-0",
            )
            pipe = pipe.to("cuda")
            return pipe
        return self._load("hunyuan3d_paint", _load)

    # ── 9. InsightFace (optional) ──────────────────────────────

    def load_insightface(self):
        """InsightFace buffalo_l for face detection/embedding."""
        def _load():
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(INSIGHTFACE_MODEL_DIR),
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        return self._load("insightface", _load)

    # ── Status Report ──────────────────────────────────────────

    def status_report(self) -> dict:
        """Model availability and load status."""
        available = {
            "yolo26": YOLO26_ENABLED,
            "sam3": SAM3_ENABLED,
            "sam3d_body": SAM3D_BODY_ENABLED,
            "fashn_parser": FASHN_PARSER_ENABLED,
            "catvton_flux": CATVTON_FLUX_ENABLED,
            "flux_gguf": FLUX_GGUF_ENABLED,
            "hunyuan3d": HUNYUAN3D_ENABLED,
            "hunyuan3d_paint": HUNYUAN3D_PAINT_ENABLED,
            "insightface": INSIGHTFACE_ENABLED,
        }
        loaded = {k: True for k in self._models}
        return {
            "available": available,
            "loaded": loaded,
            "load_times": dict(self._load_times),
            "device": DEVICE,
            "has_cuda": HAS_CUDA,
            "mps_fallback": MPS_FALLBACK_DEVICE,
        }


# Module-level singleton
registry = ModelRegistry()
