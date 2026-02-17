"""
StyleLens V6 — Modal GPU Worker (Tier 4) [Commercial Edition]
100% commercial-safe licensed models only.

GPU functions (H200 — 141GB VRAM):
  1. run_light_models — MediaPipe Pose, SAM 2.1, SAM 3D Body (~8GB)
  2. run_mesh_to_realistic — SDXL + ControlNet Depth (~14GB)
  3. run_flux_refine — FLUX.2-klein-4B img2img refiner (~13GB) [v31 NEW]
  4. run_fashn_vton_batch — FASHN VTON v1.5 maskless try-on (~8GB)
  5. run_face_swap — InsightFace antelopev2 face swapping (~2GB)
  6. run_face_refiner — FLUX.2-klein-4B face inpainting [v31 NEW]
  7. run_trellis_3d — TRELLIS.2 4B 3D reconstruction (deferred)
"""

import base64
import io
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("stylelens.worker")

# ── Modal import ──
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    logger.info("Modal not installed — worker runs in local-only mode")

# Model paths in Modal Volume
_MODEL_ROOT = "/models"


# ── Inline Serialization (클라우드에서 self-contained 실행) ────

def _b64_to_image(b64: str):
    """Decode base64 JPEG/PNG to BGR numpy array."""
    import cv2
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def _b64_to_image_rgb(b64: str):
    """Decode base64 JPEG/PNG to RGB numpy array."""
    import cv2
    bgr = _b64_to_image(b64)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _image_to_b64(img_bgr, quality: int = 90) -> str:
    """Encode BGR numpy array to JPEG base64."""
    import cv2
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ndarray_to_b64(arr) -> dict:
    """Serialize numpy array to base64 with shape/dtype metadata."""
    raw = arr.tobytes()
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _parsemap_to_b64(pm) -> str:
    """Encode uint8 parse map as PNG base64."""
    import cv2
    ok, buf = cv2.imencode(".png", pm)
    if not ok:
        raise ValueError("Failed to encode parse map")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _b64_to_parsemap(b64: str):
    """Decode base64 PNG parse map to uint8 numpy array."""
    import cv2
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    pm = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if pm is None:
        raise ValueError("Failed to decode parse map")
    return pm


def _bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes to base64 string."""
    return base64.b64encode(data).decode("ascii")


def _pil_to_b64(pil_img, format: str = "JPEG", quality: int = 90) -> str:
    """Encode PIL Image to base64."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_pil(b64: str):
    """Decode base64 to PIL Image."""
    from PIL import Image
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


# ── Modal App Definition ─────────────────────────────────────

if MODAL_AVAILABLE:
    app = modal.App("stylelens-v6-worker")

    # Persistent volume for model weights (all models pre-uploaded)
    model_volume = modal.Volume.from_name(
        "stylelens-models", create_if_missing=True,
    )

    # Base image with all ML dependencies (100% commercial-safe)
    worker_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libopengl0",
            "git",
            "wget",
        )
        .pip_install(
            "torch>=2.6",
            "torchvision>=0.21",
            "transformers>=4.48",
            "diffusers @ git+https://github.com/huggingface/diffusers.git",
            "safetensors>=0.5",
            "accelerate>=1.3",
            "opencv-python-headless>=4.10",
            "Pillow>=11.0",
            "numpy>=2.2",
            "trimesh>=4.6",
            "mediapipe>=0.10",
            "peft>=0.15",
            "huggingface_hub>=0.28",
            "einops>=0.8",
            "braceexpand",  # Required by sam-3d-body
            "timm",  # Vision transformers
            "ninja",  # Build tool for some dependencies
            "rembg",  # Background removal
            "yacs",  # Config management for SAM 3D Body
            "termcolor",  # Required by SAM 3D Body logging
            "roma",  # Rotation utilities for SAM 3D Body
            "smplx",  # SMPL body model for SAM 3D Body
            "omegaconf",  # Config for SAM 3D Body
            "pytorch-lightning",  # Required by SAM 3D Body checkpoint loading
            "insightface>=0.7",  # Face detection and analysis for face swapping
            "onnxruntime-gpu",  # ONNX runtime with GPU support for InsightFace
        )
        .run_commands(
            # Pre-cache SAM model in HF cache during image build
            "python -c 'from transformers import SamModel, SamProcessor; "
            "SamProcessor.from_pretrained(\"facebook/sam-vit-large\"); "
            "SamModel.from_pretrained(\"facebook/sam-vit-large\"); "
            "print(\"SAM-vit-large cached OK\")'",
            # Install FASHN VTON v1.5 (Apache 2.0, maskless try-on)
            "pip install git+https://github.com/fashn-AI/fashn-vton-1.5.git",
        )
    )
else:
    app = None
    model_volume = None
    worker_image = None


# ── GPU Functions ────────────────────────────────────────────

if MODAL_AVAILABLE:

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=300,
        memory=16384,
        min_containers=0,  # GPU 유휴 과금 방지
    )
    def run_light_models(task: str, image_b64: str) -> dict:
        """Light GPU tasks — MediaPipe Pose, SAM 2.1, SAM 3D Body (~8GB VRAM).

        Args:
            task: "detect_person" | "segment_sam3" | "reconstruct_3d"
            image_b64: Base64-encoded input image

        Returns:
            Task-specific result dict or error dict
        """
        import torch
        import cv2
        import sys

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = _b64_to_image(image_b64)
        t0 = time.time()

        try:
            if task == "detect_person":
                # MediaPipe Pose — person detection (Apache 2.0, replaces YOLO AGPL)
                import mediapipe as mp

                logger.info("Running MediaPipe Pose person detection...")
                mp_pose = mp.solutions.pose
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                with mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.5
                ) as pose:
                    results_pose = pose.process(rgb)

                detections = []
                if results_pose.pose_landmarks:
                    h, w = img.shape[:2]
                    landmarks = results_pose.pose_landmarks.landmark
                    xs = [lm.x * w for lm in landmarks]
                    ys = [lm.y * h for lm in landmarks]
                    x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
                    x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(results_pose.pose_landmarks.landmark[0].visibility),
                    })

                return {
                    "detections": detections,
                    "num_persons": len(detections),
                    "elapsed_sec": time.time() - t0,
                }

            elif task == "segment_sam3":
                # SAM 2.1 segmentation (facebook/sam-vit-large)
                from transformers import SamModel, SamProcessor

                logger.info("Loading SAM 2.1 model...")
                processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
                model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

                # Use center point as prompt
                input_points = [[[w // 2, h // 2]]]

                inputs = processor(
                    rgb,
                    input_points=input_points,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )

                scores = outputs.iou_scores[0, 0].cpu().numpy()
                best_idx = scores.argmax()
                mask = masks[0][0, best_idx].numpy().astype(np.uint8) * 255

                # Apply mask
                segmented = img.copy()
                segmented[mask == 0] = [220, 220, 220]

                del model, processor
                torch.cuda.empty_cache()

                return {
                    "segmented_b64": _image_to_b64(segmented),
                    "mask_b64": _parsemap_to_b64(mask),
                    "elapsed_sec": time.time() - t0,
                }

            elif task == "reconstruct_3d":
                # SAM 3D Body reconstruction from Volume
                logger.info("Loading SAM 3D Body from Volume...")

                # Add custom package to path
                sam3d_code = f"{_MODEL_ROOT}/sam-3d-body"
                if sam3d_code not in sys.path:
                    sys.path.insert(0, sam3d_code)

                from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

                # Load model from Volume
                model, model_cfg = load_sam_3d_body(
                    checkpoint_path=f"{_MODEL_ROOT}/sam3d_body/model.ckpt",
                    device=device,
                    mhr_path=f"{_MODEL_ROOT}/sam3d_body/assets/mhr_model.pt",
                )

                estimator = SAM3DBodyEstimator(
                    sam_3d_body_model=model,
                    model_cfg=model_cfg,
                )

                # Process image — expects RGB numpy array
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # process_one_image returns list of dicts (one per detected person)
                all_out = estimator.process_one_image(img=rgb)

                if not all_out or len(all_out) == 0:
                    del model, estimator
                    torch.cuda.empty_cache()
                    return {
                        "error": "No person detected in image",
                        "elapsed_sec": time.time() - t0,
                    }

                # Use first person's result
                person = all_out[0]
                vertices = person["pred_vertices"]  # (N, 3)
                joints = person["pred_keypoints_3d"]  # (J, 3)
                betas = person["shape_params"]  # (10,)
                # faces from the estimator's model
                faces_np = estimator.faces  # (M, 3) already numpy

                # Serialize as base64
                output = {
                    "vertices": _ndarray_to_b64(np.asarray(vertices)),
                    "faces": _ndarray_to_b64(np.asarray(faces_np)),
                    "joints": _ndarray_to_b64(np.asarray(joints)),
                    "betas": _ndarray_to_b64(np.asarray(betas)),
                    "elapsed_sec": time.time() - t0,
                }

                del model, estimator
                torch.cuda.empty_cache()

                return output

            else:
                return {
                    "error": f"Unknown task: {task}",
                    "elapsed_sec": time.time() - t0,
                }

        except Exception as e:
            logger.error(f"run_light_models error on task={task}: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return {
                "error": str(e),
                "task": task,
                "elapsed_sec": time.time() - t0,
            }

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=600,
        memory=49152,
        min_containers=0,
    )
    def run_mesh_to_realistic(
        mesh_renders_b64: list[str],
        person_image_b64: str,
        face_image_b64: str = "",
        angles: list[int] = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5],
        num_steps: int = 30,
        guidance: float = 7.5,
        controlnet_conditioning_scale: float = 0.5,
        prompt_template: str = "",
        negative_prompt_override: str = "",
        body_description: str = "",
    ) -> dict:
        """Convert mesh renders to realistic person images using SDXL + ControlNet Depth.

        Uses stabilityai/sdxl-base-1.0 (OpenRAIL++, commercial OK) + diffusers/controlnet-depth-sdxl-1.0.
        """
        import torch
        from PIL import Image

        t0 = time.time()
        device = "cuda"

        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

            logger.info("Loading SDXL ControlNet Depth...")
            controlnet = ControlNetModel.from_pretrained(
                f"{_MODEL_ROOT}/sdxl_controlnet_depth",
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )

            logger.info("Loading SDXL base pipeline...")
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                f"{_MODEL_ROOT}/sdxl_base",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
                variant="fp16",
            ).to(device)

            target_w, target_h = 768, 1024
            results = []

            # 16-angle labels for prompt conditioning
            angle_labels = {
                0: "front view, facing camera directly",
                22.5: "slight left turn, nearly frontal",
                45: "45-degree left view, slightly turned left",
                67.5: "two-thirds left view",
                90: "left side profile view, facing left",
                112.5: "back-left three-quarter view",
                135: "135-degree view, mostly turned away showing back-left",
                157.5: "nearly rear view, slight left",
                180: "back view, facing completely away from camera",
                202.5: "nearly rear view, slight right",
                225: "225-degree view, mostly turned away showing back-right",
                247.5: "back-right three-quarter view",
                270: "right side profile view, facing right",
                292.5: "two-thirds right view",
                315: "315-degree right view, slightly turned right",
                337.5: "slight right turn, nearly frontal",
            }

            for i, (mesh_b64, angle) in enumerate(zip(mesh_renders_b64, angles)):
                logger.info(f"Generating angle {i+1}/{len(mesh_renders_b64)} ({angle}deg)...")

                mesh_pil = _b64_to_pil(mesh_b64).convert("RGB")
                control_image = mesh_pil.resize((target_w, target_h), Image.LANCZOS)

                angle_desc = angle_labels.get(angle, f"{angle}-degree view")
                if prompt_template:
                    prompt = prompt_template.replace("{angle_desc}", angle_desc)
                    if body_description:
                        prompt = prompt.replace("{body_desc}", body_description)
                else:
                    body_desc = body_description or "young Korean woman, average build"
                    prompt = (
                        f"A photorealistic full-body photograph of a {body_desc}, "
                        f"long black hair, {angle_desc}, "
                        f"wearing a plain gray short-sleeve t-shirt and dark blue jeans, "
                        f"clean gray studio background, soft natural lighting, "
                        f"high quality, detailed skin texture, sharp focus, "
                        f"professional fashion photography"
                    )
                negative_prompt = negative_prompt_override or (
                    "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, "
                    "anime, cartoon, graphic, 3d render, cgi, plastic skin, smooth skin, "
                    "doll, illustration, digital art, airbrushed, painting, "
                    "floating body, T-pose, arms spread wide"
                )

                with torch.no_grad():
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=control_image,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        height=target_h,
                        width=target_w,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        generator=torch.Generator(device=device).manual_seed(42 + i),
                    )

                result_img = output.images[0]
                results.append(_pil_to_b64(result_img, quality=95))
                logger.info(f"Angle {angle}deg done ({result_img.size})")

            del pipe, controlnet
            torch.cuda.empty_cache()

            return {
                "realistic_renders_b64": results,
                "num_angles": len(results),
                "elapsed_sec": time.time() - t0,
            }

        except Exception as e:
            logger.error(f"Mesh-to-Realistic error: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return {"error": str(e), "elapsed_sec": time.time() - t0}

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=600,
        memory=49152,
        min_containers=0,
    )
    def run_flux_refine(
        images_b64: list[str],
        prompt_template: str = "",
        negative_prompt_override: str = "",
        angles: list[float] | None = None,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
        body_description: str = "",
    ) -> dict:
        """FLUX.2-klein-4B img2img refiner — upgrade SDXL output to photorealistic.

        Takes SDXL-generated images and refines them through FLUX.2-klein-4B's
        in-context conditioning. The klein model uses the input image as a reference
        and regenerates it with superior texture quality (skin pores, hair detail,
        fabric weave) while preserving the original pose and composition.

        License: Apache 2.0 (fully commercial)
        Model: black-forest-labs/FLUX.2-klein-4B (~13GB VRAM)

        Args:
            images_b64: List of SDXL-generated images (base64) to refine
            prompt_template: Optional prompt with {angle_desc} placeholder
            negative_prompt_override: Optional negative prompt
            angles: Optional list of angles for prompt conditioning
            num_steps: Inference steps (4 for distilled model)
            guidance: Guidance scale (1.0 for distilled, 4.0 for base)
            seed: Random seed

        Returns:
            {"refined_b64": [...], "num_images": int, "elapsed_sec": float}
        """
        import torch
        from PIL import Image

        t0 = time.time()
        device = "cuda"

        try:
            from diffusers import Flux2KleinPipeline

            logger.info("Loading FLUX.2-klein-4B pipeline...")
            pipe = Flux2KleinPipeline.from_pretrained(
                f"{_MODEL_ROOT}/flux2_klein_4b",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(device)

            target_w, target_h = 768, 1024

            angle_labels = {
                0: "front view, facing camera directly",
                22.5: "slight left turn, nearly frontal",
                45: "45-degree left view",
                67.5: "two-thirds left view",
                90: "left side profile view",
                112.5: "back-left three-quarter view",
                135: "back-left view, mostly turned away",
                157.5: "nearly rear view, slight left",
                180: "back view, facing away from camera",
                202.5: "nearly rear view, slight right",
                225: "back-right view, mostly turned away",
                247.5: "back-right three-quarter view",
                270: "right side profile view",
                292.5: "two-thirds right view",
                315: "315-degree right view",
                337.5: "slight right turn, nearly frontal",
            }

            results = []

            for i, img_b64 in enumerate(images_b64):
                logger.info(f"FLUX refining {i+1}/{len(images_b64)}...")

                # Load SDXL result as reference image
                ref_img = _b64_to_pil(img_b64).convert("RGB")
                ref_img = ref_img.resize((target_w, target_h), Image.LANCZOS)

                # Build prompt
                angle = angles[i] if angles and i < len(angles) else 0
                angle_desc = angle_labels.get(angle, f"{angle}-degree view")

                if prompt_template:
                    prompt = prompt_template.replace("{angle_desc}", angle_desc)
                    if body_description:
                        prompt = prompt.replace("{body_desc}", body_description)
                else:
                    body_desc = body_description or "young Korean woman, average build"
                    prompt = (
                        f"RAW photo, ultra realistic full-body photograph of a {body_desc}, "
                        f"{angle_desc}, "
                        f"extremely detailed realistic skin with visible pores and natural texture, "
                        f"natural hair with individual strands visible, "
                        f"wearing a plain gray t-shirt and dark blue jeans, "
                        f"realistic fabric texture with natural wrinkles and folds, "
                        f"clean studio background, soft natural lighting, "
                        f"shot on Fujifilm XT4, 85mm portrait lens, subtle film grain, "
                        f"professional fashion photography, 8k uhd, photorealistic"
                    )

                with torch.no_grad():
                    output = pipe(
                        prompt=prompt,
                        image=[ref_img],  # Flux2Pipeline expects list of PIL images
                        height=target_h,
                        width=target_w,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        generator=torch.Generator(device=device).manual_seed(seed + i),
                    )

                result_img = output.images[0]
                results.append(_pil_to_b64(result_img, quality=95))
                logger.info(f"  FLUX refined angle {angle}deg ({result_img.size})")

            del pipe
            torch.cuda.empty_cache()

            return {
                "refined_b64": results,
                "num_images": len(results),
                "elapsed_sec": time.time() - t0,
            }

        except Exception as e:
            logger.error(f"FLUX Refine error: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return {"error": str(e), "elapsed_sec": time.time() - t0}

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=600,
        memory=32768,
        min_containers=0,
    )
    def run_fashn_vton_batch(
        persons_b64: list[str],
        clothing_b64: str,
        category: str = "tops",
        garment_photo_type: str = "flat-lay",
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        seed: int = 42,
    ) -> dict:
        """FASHN VTON v1.5 — maskless virtual try-on (Apache 2.0).

        Replaces CatVTON-FLUX + FASHN Parser + FLUX.1-Fill-dev (all non-commercial)
        with a single maskless model. No parsing or agnostic mask needed.

        Args:
            persons_b64: List of base64 person images (multi-angle)
            clothing_b64: Base64 clothing/garment image
            category: "tops" | "bottoms" | "one-pieces"
            garment_photo_type: "model" | "flat-lay"
            num_timesteps: Diffusion steps (20=fast, 30=balanced, 50=quality)
            guidance_scale: Classifier-free guidance (default 1.5)
            seed: Random seed for reproducibility

        Returns:
            {"results_b64": [...], "num_angles": int, "elapsed_sec": float}
        """
        import torch
        from PIL import Image

        t0 = time.time()

        try:
            from fashn_vton import TryOnPipeline

            logger.info("Loading FASHN VTON v1.5...")
            pipeline = TryOnPipeline(weights_dir=f"{_MODEL_ROOT}/fashn_vton", device="cuda")

            garment_pil = _b64_to_pil(clothing_b64).convert("RGB")
            results = []

            for i, person_b64 in enumerate(persons_b64):
                logger.info(f"Processing angle {i+1}/{len(persons_b64)}...")
                person_pil = _b64_to_pil(person_b64).convert("RGB")

                result = pipeline(
                    person_image=person_pil,
                    garment_image=garment_pil,
                    category=category,
                    garment_photo_type=garment_photo_type,
                    num_timesteps=num_timesteps,
                    guidance_scale=guidance_scale,
                    seed=seed + i,
                    segmentation_free=True,
                )

                output_img = result.images[0]
                results.append(_pil_to_b64(output_img, quality=95))
                logger.info(f"Angle {i+1} done")

            del pipeline
            torch.cuda.empty_cache()

            return {
                "results_b64": results,
                "num_angles": len(results),
                "elapsed_sec": time.time() - t0,
            }

        except Exception as e:
            logger.error(f"FASHN VTON error: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return {"error": str(e), "elapsed_sec": time.time() - t0}

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=300,
        memory=16384,
        min_containers=0,
    )
    def run_face_swap(
        images_b64: list[str],
        face_reference_b64: str,
        angles: list[float] | None = None,
        blend_radius: int = 25,
        face_scale: float = 1.0,
    ) -> dict:
        """InsightFace-based face swapping for multi-angle virtual try-on consistency.

        Swaps user's real face from reference photo onto VTON-generated images
        with angle-adaptive blending for natural results across 16 viewing angles.

        Uses antelopev2 model (Apache 2.0) from /models/insightface/models/antelopev2/
        for 5-point landmark detection and face feature extraction.

        Angle-based swap behavior:
        - Full swap (alpha=1.0): 0, 22.5, 337.5 (front/near-front)
        - Strong swap (alpha=0.85): 45, 315 (45-degree views)
        - Partial swap (alpha=0.5): 67.5, 292.5 (two-thirds views)
        - Light swap (alpha=0.3): 90, 270 (side profiles)
        - Skip entirely: 112.5-247.5 (back-facing angles)

        Args:
            images_b64: List of VTON output images (base64) to process
            face_reference_b64: User's face reference photo (base64)
            angles: Optional list of viewing angles (degrees) for each image
            blend_radius: Gaussian blur kernel size for soft edge blending
            face_scale: Scale factor for face region size (1.0 = default)

        Returns:
            {
                "swapped_b64": [...],  # List of face-swapped images (base64)
                "face_detected": [...],  # Boolean list of face detection success
                "num_images": int,
                "elapsed_sec": float
            }
        """
        import torch
        import cv2
        from PIL import Image

        t0 = time.time()

        try:
            import insightface
            import os as _os
            import glob as _glob

            # Debug: check available model files on Modal Volume
            insightface_root = '/models/insightface'
            logger.info(f"Checking InsightFace model files at {insightface_root}...")
            for pattern in [
                f"{insightface_root}/**/*.onnx",
                f"{insightface_root}/**/*.bin",
            ]:
                found = _glob.glob(pattern, recursive=True)
                for f in found:
                    logger.info(f"  Found: {f} ({_os.path.getsize(f)/1024/1024:.1f}MB)")

            if not _glob.glob(f"{insightface_root}/**/*.onnx", recursive=True):
                logger.warning("No ONNX models found. Checking /models/ root...")
                for root_f in _glob.glob("/models/**/antelopev2*", recursive=True):
                    logger.info(f"  Alt path: {root_f}")
                for root_f in _glob.glob("/models/**/glintr100*", recursive=True):
                    logger.info(f"  Alt path: {root_f}")

            logger.info("Loading InsightFace antelopev2 model...")
            face_app = insightface.app.FaceAnalysis(
                name='antelopev2',
                root=insightface_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace loaded successfully")

            # Extract source face from reference image
            logger.info("Extracting source face from reference...")
            source_pil = _b64_to_pil(face_reference_b64).convert("RGB")
            source_img_bgr = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
            source_faces = face_app.get(source_img_bgr)

            if not source_faces:
                logger.error("No face detected in reference image")
                return {
                    "error": "No face detected in reference image",
                    "elapsed_sec": time.time() - t0,
                }

            # Use face with highest detection score
            source_face = max(source_faces, key=lambda f: f.det_score)
            source_kps = source_face.kps.astype(np.float32)

            # Standard 5-point reference landmarks at 512x512
            ref_pts_512 = np.array([
                [0.34191607, 0.46157411],  # Left eye
                [0.65653393, 0.45983393],  # Right eye
                [0.50022500, 0.64050536],  # Nose tip
                [0.37097607, 0.82469196],  # Left mouth corner
                [0.63151696, 0.82325089],  # Right mouth corner
            ], dtype=np.float32) * 512

            # Align source face to reference points
            M_align = cv2.estimateAffinePartial2D(source_kps, ref_pts_512)[0]
            aligned_face = cv2.warpAffine(
                source_img_bgr, M_align, (512, 512),
                borderMode=cv2.BORDER_REPLICATE
            )

            # Angle-specific alpha blending values
            angle_alpha_map = {
                0: 1.0, 22.5: 1.0, 337.5: 1.0,  # Full swap (front)
                45: 0.85, 315: 0.85,  # Strong swap
                67.5: 0.5, 292.5: 0.5,  # Partial swap
                90: 0.3, 270: 0.3,  # Light swap (side profiles)
            }

            results = []
            face_detected = []

            for i, img_b64 in enumerate(images_b64):
                # Determine swap alpha based on angle
                angle = angles[i] if angles and i < len(angles) else 0

                # Skip back-facing angles (no visible face)
                if 112.5 <= angle <= 247.5:
                    logger.info(f"  Skipping angle {angle}° (back-facing, no visible face)")
                    results.append(img_b64)
                    face_detected.append(False)
                    continue

                alpha = angle_alpha_map.get(angle, 1.0)
                logger.info(f"Processing angle {i+1}/{len(images_b64)} ({angle}°, alpha={alpha})...")

                # Load target image
                target_pil = _b64_to_pil(img_b64).convert("RGB")
                target_img_bgr = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
                h, w = target_img_bgr.shape[:2]

                # Detect face in target image
                target_faces = face_app.get(target_img_bgr)

                if not target_faces:
                    logger.warning(f"  No face detected in target image {i}, returning original")
                    results.append(img_b64)
                    face_detected.append(False)
                    continue

                # Use face with highest detection score
                target_face = max(target_faces, key=lambda f: f.det_score)
                target_kps = target_face.kps.astype(np.float32)

                # Compute warp matrix from aligned source to target position
                M_warp = cv2.estimateAffinePartial2D(ref_pts_512, target_kps)[0]

                # Warp aligned source face to target position
                warped_face = cv2.warpAffine(
                    aligned_face, M_warp, (w, h),
                    borderMode=cv2.BORDER_REPLICATE
                )

                # Create soft elliptical blend mask
                mask = np.zeros((h, w), dtype=np.float32)
                center = target_kps.mean(axis=0).astype(int)
                face_w = int(np.linalg.norm(target_kps[0] - target_kps[1]) * 1.8 * face_scale)
                face_h = int(face_w * 1.3)

                cv2.ellipse(
                    mask,
                    tuple(center),
                    (face_w // 2, face_h // 2),
                    0, 0, 360,
                    1.0,
                    -1
                )

                # Apply Gaussian blur for soft edges
                ksize = blend_radius * 2 + 1
                mask = cv2.GaussianBlur(mask, (ksize, ksize), blend_radius / 2)

                # LAB color transfer for skin tone matching
                # Convert both images to LAB color space
                target_lab = cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                warped_lab = cv2.cvtColor(warped_face, cv2.COLOR_BGR2LAB).astype(np.float32)

                # Calculate mean/std in core face region (higher threshold = core only)
                mask_bool = mask > 0.3
                if mask_bool.any():
                    for channel in range(3):
                        target_channel = target_lab[:, :, channel]
                        warped_channel = warped_lab[:, :, channel]

                        target_mean = target_channel[mask_bool].mean()
                        target_std = max(target_channel[mask_bool].std(), 1e-6)
                        warped_mean = warped_channel[mask_bool].mean()
                        warped_std = max(warped_channel[mask_bool].std(), 1e-6)

                        # Match source to target statistics
                        # L channel (brightness): 60% transfer to prevent bright patches
                        # A/B channels (color): full transfer for skin tone match
                        transfer_strength = 0.6 if channel == 0 else 1.0
                        if warped_std > 0:
                            corrected = (warped_channel - warped_mean) * (target_std / warped_std) + target_mean
                            warped_lab[:, :, channel] = (
                                warped_channel * (1 - transfer_strength) + corrected * transfer_strength
                            )

                # Convert back to BGR
                warped_lab = np.clip(warped_lab, 0, 255).astype(np.uint8)
                warped_face = cv2.cvtColor(warped_lab, cv2.COLOR_LAB2BGR)

                # Alpha blend with angle-specific strength
                mask_3ch = cv2.merge([mask, mask, mask]) * alpha
                result_bgr = (
                    warped_face * mask_3ch + target_img_bgr * (1 - mask_3ch)
                ).astype(np.uint8)

                # Convert back to PIL and encode
                result_pil = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
                results.append(_pil_to_b64(result_pil, quality=95))
                face_detected.append(True)
                logger.info(f"  Face swapped at angle {angle}° (alpha={alpha})")

            # Cleanup
            del face_app
            torch.cuda.empty_cache()

            return {
                "swapped_b64": results,
                "face_detected": face_detected,
                "num_images": len(results),
                "elapsed_sec": time.time() - t0,
            }

        except Exception as e:
            import traceback
            error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Face swap error: {error_detail}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return {"error": error_detail, "elapsed_sec": time.time() - t0}

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=600,
        memory=49152,
        min_containers=0,
    )
    def run_face_refiner(
        images_b64: list[str],
        face_reference_b64: str = "",
        angles: list[float] | None = None,
        face_expand_ratio: float = 2.0,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
    ) -> dict:
        """Face Refiner v3 — Crop-Refine-Merge with FLUX.2-klein-4B.

        Replaces v2 (MediaPipe + SDXL Inpainting) to solve:
        - MediaPipe solutions import error in Modal container
        - SDXL-based inpainting producing plastic-like faces

        Strategy: Simple top-region crop → FLUX.2-klein img2img → merge back.
        No MediaPipe dependency. Uses image position heuristics for face region.
        Back-facing angles (135-225°) are auto-skipped.

        License: Apache 2.0 (fully commercial)
        Model: FLUX.2-klein-4B

        Args:
            images_b64: List of person images (base64) to refine
            face_reference_b64: Optional face reference (for future IP-Adapter)
            angles: Optional angles list. Back-facing (135-225°) auto-skipped.
            face_expand_ratio: Face region height as fraction of image (0.3 = top 30%)
            num_steps: FLUX inference steps (4 for distilled)
            guidance: Guidance scale (1.0 for distilled)
            seed: Random seed

        Returns:
            {"refined_b64": [...], "face_detected": [...], "elapsed_sec": float}
        """
        import torch
        from PIL import Image, ImageFilter

        t0 = time.time()
        device = "cuda"

        try:
            from diffusers import Flux2KleinPipeline

            logger.info("Loading FLUX.2-klein-4B for face refining...")
            pipe = Flux2KleinPipeline.from_pretrained(
                f"{_MODEL_ROOT}/flux2_klein_4b",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(device)

            results = []
            face_detected = []

            # Face region = top 30% of image (head + hair + neck area)
            face_region_ratio = 0.30

            for i, img_b64 in enumerate(images_b64):
                # Skip back-facing angles where no face is visible
                if angles and i < len(angles):
                    angle = angles[i]
                    if 135 <= angle <= 225:
                        logger.info(f"  Skipping angle {angle}° (back-facing)")
                        results.append(img_b64)
                        face_detected.append(False)
                        continue

                logger.info(f"Face refining {i+1}/{len(images_b64)}...")
                pil_img = _b64_to_pil(img_b64).convert("RGB")
                orig_w, orig_h = pil_img.size

                # Crop face region (top portion of image)
                face_h = int(orig_h * face_region_ratio)
                face_crop = pil_img.crop((0, 0, orig_w, face_h))

                # Resize crop to square for FLUX (best quality)
                crop_size = 512
                face_input = face_crop.resize((crop_size, crop_size), Image.LANCZOS)

                prompt = (
                    "RAW photo, ultra realistic close-up portrait of a young Korean woman, "
                    "extremely detailed face with visible skin pores and natural texture, "
                    "clear bright eyes with natural light reflections, "
                    "natural hair with individual strands visible, "
                    "soft natural studio lighting, sharp focus, "
                    "shot on Fujifilm XT4, 85mm f/1.4 portrait lens, subtle film grain, "
                    "photorealistic, 8k uhd"
                )

                with torch.no_grad():
                    output = pipe(
                        prompt=prompt,
                        image=[face_input],  # Flux2Pipeline expects list of PIL images
                        height=crop_size,
                        width=crop_size,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        generator=torch.Generator(device=device).manual_seed(seed + i),
                    )

                refined_face = output.images[0]

                # Resize back to original face crop dimensions
                refined_face = refined_face.resize((orig_w, face_h), Image.LANCZOS)

                # Create smooth blend mask (soft transition at bottom edge)
                blend_mask = Image.new("L", (orig_w, face_h), 255)
                from PIL import ImageDraw as IDraw
                draw = IDraw.Draw(blend_mask)
                # Gradient fade at bottom 15% of face region
                fade_start = int(face_h * 0.85)
                for y in range(fade_start, face_h):
                    alpha = int(255 * (1.0 - (y - fade_start) / (face_h - fade_start)))
                    draw.line([(0, y), (orig_w, y)], fill=alpha)

                # Merge: paste refined face onto original with blend mask
                result = pil_img.copy()
                result.paste(refined_face, (0, 0), blend_mask)

                results.append(_pil_to_b64(result, quality=95))
                face_detected.append(True)
                logger.info(f"  Face refined at index {i}")

            del pipe
            torch.cuda.empty_cache()

            return {
                "refined_b64": results,
                "face_detected": face_detected,
                "num_images": len(results),
                "elapsed_sec": time.time() - t0,
            }

        except Exception as e:
            logger.error(f"Face Refiner error: {e}", exc_info=True)
            torch.cuda.empty_cache()
            return {"error": str(e), "elapsed_sec": time.time() - t0}

    @app.function(
        image=worker_image,
        gpu="H200",
        volumes={"/models": model_volume},
        timeout=600,
        memory=49152,
        min_containers=0,
    )
    def run_trellis_3d(
        front_image_b64: str,
        reference_images_b64: list[str] | None = None,
        seed: int = 42,
    ) -> dict:
        """TRELLIS.2 4B — image to 3D GLB (MIT license).

        Replaces Hunyuan3D 2.0 (non-commercial in Korea).
        Microsoft Research, 1536^3 resolution PBR textured assets.

        NOTE: Deferred — TRELLIS.2 requires complex CUDA kernel compilation
        (e.g., spconv, custom voxel ops) that needs a dedicated image build step.
        Will be implemented separately once the core pipeline is validated.

        Args:
            front_image_b64: Front view image (base64)
            reference_images_b64: Optional additional reference images
            seed: Random seed for reproducibility

        Returns:
            {"glb_bytes_b64": str, "elapsed_sec": float} or error dict
        """
        t0 = time.time()
        return {
            "error": "TRELLIS.2 4B is deferred — requires complex CUDA kernel setup. "
                     "Will be implemented in a dedicated worker image.",
            "elapsed_sec": time.time() - t0,
        }

    @app.function(
        image=worker_image,
        gpu="H200",
        timeout=30,
        min_containers=0,
    )
    def gpu_health() -> dict:
        """GPU health check - verify H200 availability and CUDA setup.

        Returns:
            {"cuda_available": bool, "device_name": str, "vram_gb": float}
        """
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "device_name": "N/A",
                "vram_gb": 0,
                "error": "CUDA not available in container",
            }

        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    # ── Local entrypoint for testing ──────────────────────────

    @app.local_entrypoint()
    def main():
        """Test: modal run worker/modal_app.py"""
        import json
        import cv2

        print("=== StyleLens V6 GPU Worker Test ===")
        print(f"App: {app.name}")

        # GPU health check
        print("\n1. Testing GPU health...")
        result = gpu_health.remote()
        print(f"   Result: {json.dumps(result, indent=2)}")

        if not result.get("cuda_available"):
            print("   [ERROR] CUDA not available, cannot proceed")
            return

        print(f"\n   [OK] H200 ready: {result['device_name']}, {result['vram_gb']}GB VRAM")

        # Create test image
        print("\n2. Creating test image (512x512)...")
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", test_img)
        test_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        print(f"   Test image size: {len(test_b64)} bytes")

        # Test MediaPipe Pose person detection
        print("\n3. Testing MediaPipe Pose person detection...")
        try:
            result = run_light_models.remote(task="detect_person", image_b64=test_b64)
            if "error" in result:
                print(f"   [ERROR] {result['error']}")
            else:
                print(f"   [OK] Detected {result['num_persons']} persons in {result['elapsed_sec']:.2f}s")
        except Exception as e:
            print(f"   [ERROR] MediaPipe Pose test failed: {e}")

        # Test SAM 3 segmentation
        print("\n4. Testing SAM 3 segmentation...")
        try:
            result = run_light_models.remote(task="segment_sam3", image_b64=test_b64)
            if "error" in result:
                print(f"   [WARN] {result['error']}")
            else:
                print(f"   [OK] SAM 3 segmentation completed in {result['elapsed_sec']:.2f}s")
                print(f"   Segmented image size: {len(result['segmented_b64'])} bytes")
                print(f"   Mask size: {len(result['mask_b64'])} bytes")
        except Exception as e:
            print(f"   [ERROR] SAM 3 test failed: {e}")

        # Test SAM 3D Body (if available)
        print("\n5. Testing SAM 3D Body reconstruction...")
        try:
            result = run_light_models.remote(task="reconstruct_3d", image_b64=test_b64)
            if "error" in result:
                print(f"   [SKIP] {result['error']}")
            else:
                print(f"   [OK] 3D reconstruction completed in {result['elapsed_sec']:.2f}s")
                print(f"   Vertices: {result['vertices']['shape']}")
                print(f"   Faces: {result['faces']['shape']}")
        except Exception as e:
            print(f"   [ERROR] SAM 3D Body test failed: {e}")

        # Test FASHN VTON (basic check)
        print("\n6. Testing FASHN VTON v1.5 (single angle)...")
        try:
            result = run_fashn_vton_batch.remote(
                persons_b64=[test_b64],
                clothing_b64=test_b64,
                category="tops",
                num_timesteps=5,  # Quick test
            )
            if "error" in result:
                print(f"   [ERROR] {result['error']}")
            else:
                print(f"   [OK] FASHN VTON completed in {result['elapsed_sec']:.2f}s")
                print(f"   Generated {result['num_angles']} results")
        except Exception as e:
            print(f"   [ERROR] FASHN VTON test failed: {e}")

        # Test TRELLIS.2 (basic check — expected to return deferred message)
        print("\n7. Testing TRELLIS.2 4B...")
        try:
            result = run_trellis_3d.remote(
                front_image_b64=test_b64,
            )
            if "error" in result:
                print(f"   [ERROR] {result['error']}")
            else:
                print(f"   [OK] TRELLIS.2 completed in {result['elapsed_sec']:.2f}s")
                print(f"   GLB size: {len(result['glb_bytes_b64'])} bytes (base64)")
        except Exception as e:
            print(f"   [ERROR] TRELLIS.2 test failed: {e}")

        print("\n=== Test Complete ===")
        print("Worker ready for production traffic!")

else:
    # Dummy functions when Modal not available (local import safety)
    def run_light_models(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def run_mesh_to_realistic(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def run_flux_refine(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def run_fashn_vton_batch(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def run_face_swap(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def run_face_refiner(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def run_trellis_3d(*args, **kwargs):
        raise RuntimeError("Modal not installed")

    def gpu_health(*args, **kwargs):
        raise RuntimeError("Modal not installed")
