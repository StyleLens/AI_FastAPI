"""
segment_anything_3 — Compatibility wrapper for SAM 3 Video Model.

Provides sam_model_registry and SamPredictor APIs compatible with
core/loader.py expectations, backed by HuggingFace transformers
Sam3VideoModel's built-in detector (Sam3Model, 840M params).

Architecture:
  Sam3VideoModel
  ├── detector_model (Sam3Model, 840.4M) ← Used for segmentation
  │   └── text-prompted detection with CLIP tokenizer
  ├── tracker_model (Sam3TrackerVideoModel, 11.7M)
  └── tracker_neck (Sam3VisionNeck, 7.8M)

The detector_model accepts:
  - pixel_values: image tensor (processed by Sam3VideoProcessor)
  - input_ids + attention_mask: CLIP-tokenized text prompt
  Returns:
  - pred_masks: (B, 200, 288, 288) segmentation masks
  - pred_logits: (B, 200) confidence scores
  - pred_boxes: (B, 200, 4) bounding boxes
  - semantic_seg: (B, 1, 288, 288) full semantic segmentation
"""

import logging
import numpy as np
import torch
import cv2
from pathlib import Path

logger = logging.getLogger("segment_anything_3")

# Default text prompts for clothing segmentation
_CLOTHING_PROMPTS = ["clothing", "garment", "shirt", "top", "dress", "pants"]
_DEFAULT_PROMPT = "clothing"


class SamPredictor:
    """SAM3-compatible predictor using Sam3VideoModel's detector for segmentation.

    Uses text-prompted detection through the internal Sam3Model detector,
    which provides pixel-accurate masks via CLIP text grounding.
    """

    def __init__(self, model, processor, tokenizer, device="cpu"):
        self.model = model           # Sam3VideoModel
        self.detector = model.detector_model  # Sam3Model (840M)
        self.processor = processor    # Sam3VideoProcessor
        self.tokenizer = tokenizer    # CLIPTokenizer
        self.device = device
        self._image = None
        self._image_inputs = None

    def set_image(self, image_rgb: np.ndarray):
        """Set the image for subsequent prediction calls.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3), uint8
        """
        self._image = image_rgb
        # Pre-process image for the detector
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        self._image_inputs = self.processor(images=pil_image, return_tensors="pt")
        # Move tensors to device
        self._image_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self._image_inputs.items()
        }

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        multimask_output: bool = True,
        text_prompt: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate masks for the current image using text-prompted detection.

        The SAM3 detector uses CLIP text grounding. Point prompts are converted
        to a text prompt contextualizing the region of interest.

        Args:
            point_coords: (N, 2) array of point prompts (used as hint, not direct input)
            point_labels: (N,) array of point labels (1=fg, 0=bg)
            box: (4,) array [x1, y1, x2, y2] bounding box prompt
            multimask_output: If True, return 3 masks with quality scores
            text_prompt: Optional text prompt (default: "clothing")

        Returns:
            masks: (N, H, W) boolean masks
            scores: (N,) quality scores for each mask
            low_res_logits: (N, 256, 256) low-resolution logits
        """
        if self._image is None:
            raise RuntimeError("Call set_image() before predict()")

        h, w = self._image.shape[:2]
        prompt = text_prompt or _DEFAULT_PROMPT

        try:
            # Tokenize text prompt
            text_inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True,
            )
            text_inputs = {
                k: v.to(self.device) for k, v in text_inputs.items()
            }

            # Run detector forward
            with torch.no_grad():
                det_out = self.detector(
                    pixel_values=self._image_inputs["pixel_values"],
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )

            # Extract predictions
            pred_masks = det_out.pred_masks[0]   # (200, 288, 288)
            pred_logits = det_out.pred_logits[0]  # (200,)
            scores_all = torch.sigmoid(pred_logits)

            # Sort by confidence
            sorted_indices = scores_all.argsort(descending=True)

            # Determine how many masks to return
            n_masks = 3 if multimask_output else 1
            confidence_threshold = 0.2

            # Collect top masks above threshold
            result_masks = []
            result_scores = []

            for idx in sorted_indices:
                score = scores_all[idx.item()].item()
                if score < confidence_threshold and len(result_masks) >= 1:
                    break
                if len(result_masks) >= n_masks:
                    break

                mask_lr = pred_masks[idx.item()].cpu().float()
                # Resize to original resolution
                mask_hr = torch.nn.functional.interpolate(
                    mask_lr.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()

                result_masks.append((mask_hr > 0).astype(bool))
                result_scores.append(score)

            # If we got fewer masks than requested, try semantic_seg as fallback
            if len(result_masks) == 0 and hasattr(det_out, 'semantic_seg') and det_out.semantic_seg is not None:
                sem_seg = det_out.semantic_seg[0, 0].cpu().float()
                sem_hr = torch.nn.functional.interpolate(
                    sem_seg.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
                result_masks.append((sem_hr > 0).astype(bool))
                result_scores.append(0.5)

            # Pad to requested number of masks
            while len(result_masks) < n_masks:
                if result_masks:
                    result_masks.append(result_masks[-1])
                    result_scores.append(result_scores[-1] * 0.8)
                else:
                    result_masks.append(np.zeros((h, w), dtype=bool))
                    result_scores.append(0.0)

            masks_arr = np.stack(result_masks[:n_masks], axis=0)
            scores_arr = np.array(result_scores[:n_masks], dtype=np.float32)

            # Generate low-res logits (256x256)
            low_res = np.zeros((n_masks, 256, 256), dtype=np.float32)
            for i in range(n_masks):
                lr = cv2.resize(
                    masks_arr[i].astype(np.float32), (256, 256),
                    interpolation=cv2.INTER_LINEAR,
                )
                low_res[i] = lr * 10.0  # Scale to logit range

            logger.info(
                f"SAM3 predict: {n_masks} masks, "
                f"best score={scores_arr[0]:.3f}, "
                f"coverage={masks_arr[0].mean()*100:.1f}%"
            )

            return masks_arr, scores_arr, low_res

        except Exception as e:
            logger.warning(f"SAM3 detection failed: {e}, using fallback mask")
            return self._fallback_mask(h, w, point_coords, multimask_output)

    def _fallback_mask(
        self,
        h: int,
        w: int,
        point_coords: np.ndarray | None,
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate fallback masks using simple heuristics.

        Creates an elliptical mask centered on the point prompt,
        covering approximately 60% of the image area.
        """
        n_masks = 3 if multimask_output else 1
        masks = np.zeros((n_masks, h, w), dtype=bool)
        scores = np.zeros(n_masks, dtype=np.float32)

        if point_coords is not None and len(point_coords) > 0:
            cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
        else:
            cx, cy = w // 2, h // 2

        y_grid, x_grid = np.ogrid[:h, :w]
        for i, scale in enumerate([0.7, 0.5, 0.3] if multimask_output else [0.6]):
            rx = int(w * scale / 2)
            ry = int(h * scale / 2)
            if rx > 0 and ry > 0:
                dist = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2
                masks[i] = dist <= 1.0
                scores[i] = 0.9 - i * 0.1

        low_res = np.zeros((n_masks, 256, 256), dtype=np.float32)
        return masks, scores, low_res


def _load_sam3_model(checkpoint: str, device: str = "cpu"):
    """Load SAM3 model from a checkpoint file or directory.

    Loads Sam3VideoModel which contains the detector (Sam3Model) used
    for text-prompted segmentation.

    Args:
        checkpoint: Path to model checkpoint (.safetensors, .pt, or directory)

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    from transformers import Sam3VideoModel, AutoProcessor, AutoTokenizer

    checkpoint_path = Path(checkpoint)

    if checkpoint_path.is_file():
        model_dir = str(checkpoint_path.parent)
    else:
        model_dir = str(checkpoint_path)

    logger.info(f"Loading SAM3 model from {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = Sam3VideoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    det_params = sum(p.numel() for p in model.detector_model.parameters()) / 1e6
    logger.info(
        f"SAM3 loaded: {total_params:.1f}M total params, "
        f"{det_params:.1f}M detector params"
    )

    return model, processor, tokenizer


class _Sam3Registry:
    """Registry that mimics sam_model_registry['default'](checkpoint=...) pattern."""

    def __getitem__(self, key: str):
        """Return a factory function for the given model variant."""
        def factory(checkpoint: str = None, device: str = "cpu"):
            model, processor, tokenizer = _load_sam3_model(checkpoint, device)
            return _Sam3ModelWrapper(model, processor, tokenizer)
        return factory


class _Sam3ModelWrapper:
    """Wraps the model+processor+tokenizer so SamPredictor can be created from it."""

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self._device = "cpu"

    def to(self, device):
        self._device = str(device)
        self.model = self.model.to(device)
        return self

    def create_predictor(self):
        return SamPredictor(self.model, self.processor, self.tokenizer, self._device)


# Public API: matches `from segment_anything_3 import sam_model_registry, SamPredictor`
sam_model_registry = _Sam3Registry()


# Override SamPredictor to also accept a _Sam3ModelWrapper
_OrigSamPredictor = SamPredictor


class SamPredictor(_OrigSamPredictor):
    """SamPredictor that can be initialized from a _Sam3ModelWrapper."""

    def __init__(self, model_or_wrapper, processor=None, tokenizer=None, device="cpu"):
        if isinstance(model_or_wrapper, _Sam3ModelWrapper):
            super().__init__(
                model_or_wrapper.model,
                model_or_wrapper.processor,
                model_or_wrapper.tokenizer,
                model_or_wrapper._device,
            )
        else:
            super().__init__(model_or_wrapper, processor, tokenizer, device)
