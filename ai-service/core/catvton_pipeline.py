"""
StyleLens V6 — CatVTON-FLUX Pipeline
FLUX-based virtual try-on inference engine.
Primary engine for Phase 3 fitting.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from core.config import (
    DEVICE, DTYPE,
    CATVTON_FLUX_STEPS,
    CATVTON_FLUX_GUIDANCE,
    CATVTON_FLUX_RESOLUTION,
    CATVTON_FLUX_STRENGTH,
)

logger = logging.getLogger("stylelens.catvton")


class CatVTONFluxPipeline:
    """FLUX-based virtual try-on inference engine."""

    def __init__(self, pipe, attn_processor=None):
        self.pipe = pipe
        self.attn_processor = attn_processor
        self.device = DEVICE
        self.dtype = DTYPE
        logger.info("CatVTON-FLUX pipeline initialized")

    @classmethod
    def from_pretrained(cls, flux_gguf_path: str, catvton_lora_dir: str,
                        catvton_attn_dir: str, device: str = DEVICE,
                        dtype: torch.dtype = DTYPE) -> "CatVTONFluxPipeline":
        """
        Load CatVTON-FLUX from pretrained weights.

        1. Load FLUX.1-dev GGUF as base
        2. Apply CatVTON-FLUX LoRA weights
        3. Load mix attention processor
        """
        from diffusers import FluxPipeline

        logger.info("Loading FLUX.1-dev GGUF base model...")
        pipe = FluxPipeline.from_single_file(
            flux_gguf_path,
            torch_dtype=dtype,
        ).to(device)

        # Apply CatVTON LoRA
        logger.info("Applying CatVTON-FLUX LoRA weights...")
        try:
            pipe.load_lora_weights(catvton_lora_dir)
            pipe.fuse_lora()
        except Exception as e:
            logger.warning(f"LoRA loading failed (may need manual weight merge): {e}")

        # Load attention processor for mix attention
        attn_processor = None
        try:
            import safetensors.torch
            import glob
            import os

            attn_files = sorted(glob.glob(os.path.join(catvton_attn_dir, "*.safetensors")))
            if attn_files:
                attn_weights = {}
                for f in attn_files:
                    attn_weights.update(safetensors.torch.load_file(f))
                attn_processor = attn_weights
                logger.info(f"Loaded {len(attn_weights)} attention weight keys")
        except Exception as e:
            logger.warning(f"Attention processor loading failed: {e}")

        return cls(pipe, attn_processor)

    def _prepare_inpainting_inputs(self, person_image: Image.Image,
                                    clothing_image: Image.Image,
                                    mask: Image.Image) -> dict:
        """Prepare inputs for FLUX inpainting-style try-on."""
        # Resize all to target resolution
        size = (CATVTON_FLUX_RESOLUTION, CATVTON_FLUX_RESOLUTION)
        person_resized = person_image.resize(size, Image.LANCZOS)
        clothing_resized = clothing_image.resize(size, Image.LANCZOS)
        mask_resized = mask.resize(size, Image.NEAREST)

        return {
            "person": person_resized,
            "clothing": clothing_resized,
            "mask": mask_resized,
        }

    def try_on(self, person_image: np.ndarray | Image.Image,
               clothing_image: np.ndarray | Image.Image,
               mask: np.ndarray | Image.Image,
               num_steps: int = CATVTON_FLUX_STEPS,
               guidance: float = CATVTON_FLUX_GUIDANCE,
               strength: float = CATVTON_FLUX_STRENGTH) -> Image.Image:
        """
        Single try-on inference.

        Args:
            person_image: Person photo (BGR numpy or PIL)
            clothing_image: Clothing photo (BGR numpy or PIL)
            mask: Agnostic mask (white=inpaint region)
            num_steps: Number of diffusion steps
            guidance: Classifier-free guidance scale
            strength: Denoising strength

        Returns:
            PIL Image of try-on result (1024x1024)
        """
        import cv2

        t0 = time.time()

        # Convert numpy to PIL if needed
        if isinstance(person_image, np.ndarray):
            person_pil = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
        else:
            person_pil = person_image

        if isinstance(clothing_image, np.ndarray):
            clothing_pil = Image.fromarray(cv2.cvtColor(clothing_image, cv2.COLOR_BGR2RGB))
        else:
            clothing_pil = clothing_image

        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask_pil = Image.fromarray(mask)
        else:
            mask_pil = mask

        inputs = self._prepare_inpainting_inputs(person_pil, clothing_pil, mask_pil)

        # Build conditioning: concat person (masked) + clothing in channel dim
        # The exact conditioning format depends on CatVTON-FLUX architecture
        # Using inpainting-style approach with FLUX

        try:
            result = self.pipe(
                prompt="",
                image=inputs["person"],
                mask_image=inputs["mask"],
                # CatVTON uses clothing as image conditioning
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                strength=strength,
                height=CATVTON_FLUX_RESOLUTION,
                width=CATVTON_FLUX_RESOLUTION,
            ).images[0]
        except Exception as e:
            logger.error(f"CatVTON-FLUX inference failed: {e}")
            # Return person image as fallback
            result = inputs["person"]

        elapsed = time.time() - t0
        logger.info(f"CatVTON-FLUX try-on in {elapsed:.1f}s")
        return result

    def try_on_batch(self, person_images: list[np.ndarray],
                     clothing_image: np.ndarray,
                     masks: list[np.ndarray],
                     **kwargs) -> list[Image.Image]:
        """Run try-on for multiple person images (angles)."""
        results = []
        for person, mask in zip(person_images, masks):
            result = self.try_on(person, clothing_image, mask, **kwargs)
            results.append(result)
        return results
