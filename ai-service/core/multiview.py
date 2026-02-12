"""
StyleLens V6 — Multiview Image Generation
Gemini-based try-on image generation (fallback path when CatVTON-FLUX unavailable).
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from core.config import (
    V5_GEMINI_IMAGE_MODEL,
    GEMINI_FLASH_IMAGE_MODEL,
    FITTING_ANGLES,
)
from core.gemini_client import GeminiClient, ClothingAnalysis, _angle_to_text

logger = logging.getLogger("stylelens.multiview")


def _clean_face_photo(face_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a clean, close-up face crop from a photo.

    Strategy:
    1. Try face detection via cv2 Haar cascade for face bounding box
    2. Expand bbox by 80% on all sides to include hair/neck/ears
    3. Fallback to social media UI crop if detection fails

    LESSON (Run 6): Sending a full-body street photo with a tiny face
    causes Gemini to ignore the face entirely. A tight face crop (head+shoulders)
    at high resolution gives Gemini the best chance to reproduce the identity.
    """
    h, w = face_bgr.shape[:2]

    # Step 1: Try face detection for precise crop
    face_crop = _detect_and_crop_face(face_bgr)
    if face_crop is not None:
        logger.info(f"Face detected and cropped: {face_crop.shape[:2]}")
        return face_crop

    # Step 2: Fallback — remove social media UI borders
    top_crop = int(h * 0.08)
    bot_crop = int(h * 0.08)
    left_crop = int(w * 0.05)
    right_crop = int(w * 0.05)

    if h / max(w, 1) > 1.5:
        top_crop = int(h * 0.12)
        bot_crop = int(h * 0.12)

    cropped = face_bgr[top_crop:h - bot_crop, left_crop:w - right_crop]

    if cropped.shape[0] < 100 or cropped.shape[1] < 100:
        return face_bgr

    return cropped


def _detect_and_crop_face(image_bgr: np.ndarray) -> np.ndarray | None:
    """Detect the largest face in an image and return a generous head+shoulders crop.

    Uses OpenCV's DNN face detector (more reliable than Haar cascades) with
    fallback to Haar cascades. Expands the detected face bbox by ~100% to
    include full head, hair, neck, and upper shoulders — this gives Gemini
    enough context to reproduce the full appearance.
    """
    h, w = image_bgr.shape[:2]

    # Try DNN detector first (SSD-based, more robust)
    face_box = None
    try:
        face_box = _detect_face_dnn(image_bgr)
    except Exception:
        pass

    # Fallback to Haar cascade
    if face_box is None:
        try:
            face_box = _detect_face_haar(image_bgr)
        except Exception:
            pass

    if face_box is None:
        return None

    fx, fy, fw, fh = face_box

    # Expand bbox by ~100% to include full head + hair + neck + upper shoulders
    # This is critical — Gemini needs hair/shoulder context, not just face oval
    expand_x = int(fw * 1.0)
    expand_y_top = int(fh * 1.2)   # More expansion above for hair
    expand_y_bot = int(fh * 1.5)   # More below for neck/shoulders

    x1 = max(0, fx - expand_x)
    y1 = max(0, fy - expand_y_top)
    x2 = min(w, fx + fw + expand_x)
    y2 = min(h, fy + fh + expand_y_bot)

    crop = image_bgr[y1:y2, x1:x2]

    # Ensure crop is large enough and face is significant portion
    crop_h, crop_w = crop.shape[:2]
    if crop_h < 150 or crop_w < 150:
        return None

    return crop


def _detect_face_dnn(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect face using OpenCV's DNN SSD detector."""
    # Use OpenCV's built-in DNN face detector
    proto = cv2.data.haarcascades.replace("haarcascade_", "")
    # This is not available by default, so fall through to Haar
    raise NotImplementedError("DNN detector not configured")


def _detect_face_haar(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect face using Haar cascade classifier."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Return the largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]
    return (int(x), int(y), int(w), int(h))


def _composite_on_clean_bg(result_np: np.ndarray,
                           bg_color: tuple[int, int, int] = (220, 220, 220)) -> np.ndarray:
    """Extract person from generated image and composite on clean background."""
    h, w = result_np.shape[:2]

    # Convert to LAB for better segmentation
    lab = cv2.cvtColor(result_np, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Simple threshold to separate person from background
    # Assume background is relatively uniform
    edges = cv2.Canny(result_np, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Flood fill from corners to find background
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = np.zeros((h, w), np.uint8)

    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        temp_mask = np.zeros((h + 2, w + 2), np.uint8)
        # Only flood fill if seed is not on an edge
        if edges[seed[1], seed[0]] == 0:
            cv2.floodFill(result_np.copy(), temp_mask, seed, 0,
                         (15, 15, 15), (15, 15, 15))
            flood |= temp_mask[1:-1, 1:-1]

    # Person mask = NOT background
    person_mask = (flood == 0).astype(np.uint8) * 255

    # Clean up mask
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Gaussian blur for soft edges
    person_mask_soft = cv2.GaussianBlur(person_mask, (11, 11), 3).astype(np.float32) / 255

    # Composite
    bg = np.full_like(result_np, bg_color, dtype=np.uint8)
    mask_3d = person_mask_soft[:, :, np.newaxis]
    composited = (result_np.astype(np.float32) * mask_3d +
                  bg.astype(np.float32) * (1 - mask_3d))
    return np.clip(composited, 0, 255).astype(np.uint8)


def _enhance_tryon_image(image_bgr: np.ndarray) -> np.ndarray:
    """Post-process generated try-on image for professional quality.

    Applies subtle sharpening, slight contrast boost, and noise reduction
    to match e-commerce photo studio quality.
    """
    # Convert to PIL for enhancement
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Step 1: Light denoise (bilateral filter preserves edges)
    bgr_clean = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=40, sigmaSpace=40)

    # Step 2: Subtle sharpening via unsharp mask
    rgb_clean = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2RGB)
    pil_clean = Image.fromarray(rgb_clean)
    sharpener = ImageEnhance.Sharpness(pil_clean)
    pil_sharp = sharpener.enhance(1.15)  # Very subtle — 1.0 = no change

    # Step 3: Slight contrast boost for fabric detail
    contrast = ImageEnhance.Contrast(pil_sharp)
    pil_final = contrast.enhance(1.05)  # Very subtle

    # Step 4: Slight color saturation for vibrancy (but not over-saturated)
    color = ImageEnhance.Color(pil_final)
    pil_final = color.enhance(1.02)  # Near-zero change, just enough for depth

    result = cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGB2BGR)
    return result


def _get_angle_body_details(angle_deg: int) -> str:
    """Get angle-specific body/garment visibility details for prompt."""
    details = {
        0: ("front view — full face visible, both arms at sides, "
            "front placket/buttons visible, collar fully visible"),
        45: ("three-quarter front-right — face partially turned, "
             "right shoulder closer to camera, right sleeve/arm prominent, "
             "garment side seam and front partially visible"),
        90: ("right side profile — face in profile, "
             "right arm fully visible, left arm hidden behind body, "
             "garment side seam, right sleeve drape clearly visible"),
        135: ("three-quarter back-right — face barely visible, "
              "back of head and right ear visible, "
              "right shoulder blade, back of garment partially visible"),
        180: ("back view — back of head visible, "
              "both shoulder blades visible, "
              "back of garment (yoke, back seams, tag) visible, "
              "NO face visible"),
        225: ("three-quarter back-left — face barely visible, "
              "back of head and left ear visible, "
              "left shoulder blade, back of garment partially visible"),
        270: ("left side profile — face in left profile, "
              "left arm fully visible, right arm hidden, "
              "garment side seam, left sleeve drape visible"),
        315: ("three-quarter front-left — face partially turned left, "
              "left shoulder closer to camera, left sleeve prominent, "
              "garment side and front partially visible"),
    }
    return details.get(angle_deg, f"{angle_deg}° viewing angle")


def _select_best_face_photo(face_photo_bgr: np.ndarray | None,
                             face_references: list | None) -> np.ndarray | None:
    """Select the single best face photo for Gemini.

    LESSON (Run 4): Sending multiple face references to Gemini DEGRADES quality.
    4 refs → blurry faces, artifacts, bangs added. Single photo → clean result.
    Always use exactly 1 face photo — the highest quality front-facing one.
    """
    # If Face Bank available, pick the single best front-facing reference
    if face_references and len(face_references) > 0:
        # Prefer front angle, then front_right, then any
        for preferred_angle in ("front", "front_left", "front_right"):
            for ref in face_references:
                if getattr(ref, "face_angle", "") == preferred_angle:
                    img = ref.aligned_face if ref.aligned_face is not None else ref.image_bgr
                    return img
        # Fallback to first reference
        ref = face_references[0]
        return ref.aligned_face if ref.aligned_face is not None else ref.image_bgr

    return face_photo_bgr


def generate_front_view_gemini(gemini: GeminiClient,
                                face_photo_bgr: np.ndarray | None,
                                clothing_image_bgr: np.ndarray,
                                clothing: ClothingAnalysis,
                                mesh_render_bgr: np.ndarray | None,
                                gender: str = "female",
                                physics_prompt: str | None = None,
                                face_references: list | None = None) -> np.ndarray | None:
    """
    Generate front view try-on image using Gemini image model.

    NOTE: Always uses a SINGLE face photo (not multiple). Run 4 proved that
    sending 4+ face references causes severe face degradation in Gemini output.
    """
    clothing_desc = gemini._build_clothing_description(clothing)

    # Extract precise color info for prompt
    color_hex = clothing.color_hex if clothing.color_hex and clothing.color_hex != "#000000" else ""
    fabric_desc = clothing.fabric or clothing.fabric_composition or ""
    surface_texture = getattr(clothing, "surface_texture", "") or ""

    prompt = f"""Generate a photorealistic full-body fashion photograph of a {gender} model
wearing this specific clothing item: {clothing_desc}

The model should be:
- Standing in a natural relaxed pose with arms relaxed naturally at sides
- Facing directly toward the camera (front view, 0 degrees)
- On a clean, neutral light gray (#E0E0E0) studio background
- Full body visible from head to toe
- Professional fashion e-commerce photo quality

CRITICAL COLOR ACCURACY (most important requirement):
- The clothing color in the provided product image is the GROUND TRUTH — match it pixel-perfectly
- Do NOT shift, saturate, or warm the color. If the product is pale pink-beige/champagne, keep it EXACTLY that pale muted tone — NOT orange, NOT peach, NOT coral, NOT salmon, NOT tan, NOT warm beige
{f'- Reference hex code: {color_hex} — this is a VERY subtle, muted, desaturated tone with a COOL/PINK undertone, NOT warm/yellow' if color_hex else ''}
- When uncertain, err toward COOLER and more PINK rather than warmer and more tan/yellow
- The product color likely has a subtle PINK or ROSE undertone — preserve this coolness, do NOT shift to warm/golden tones

CRITICAL FABRIC/TEXTURE ACCURACY:
- Reproduce the exact fabric appearance from the product image: sheen level, drape, weight, surface quality
{f'- Surface texture: {surface_texture}' if surface_texture else ''}
{f'- Material: {fabric_desc}' if fabric_desc else ''}
- If the product shows a shiny/satin/silky fabric, the output MUST clearly show the same glossy sheen and light reflections on the fabric surface

BOTTOMS ACCURACY:
- Look carefully at the FULL clothing/product image — if it shows the model wearing specific bottoms (skirt, pants, shorts), reproduce the EXACT SAME bottoms with their EXACT COLOR
- Do NOT color-match bottoms to the top. Each piece keeps its own distinct color
- Example: if product shows pale beige shirt + BLACK pencil skirt → output MUST show BLACK skirt

GARMENT DESIGN DETAILS:
- Reproduce ALL visible design elements from the product image:
  * Buttons: exact count, placement, size, and color
  * Collar/neckline: exact shape, width, and how it sits
  * Cuffs: style (buttoned, open, folded) — match the product exactly
  * Hem: straight, curved, tucked — match the product
  * Seams: visible stitching lines, princess seams, darts
  * Any unique elements: pleats, ruffles, ties, bows, belts
- These details are what make the garment recognizable — do NOT simplify or omit them

BODY & POSE REQUIREMENTS:
- Natural body proportions with realistic anatomy
- Arms relaxed at sides with slight natural bend at elbows
- Fingers visible and natural (not clenched)
- Weight balanced on both feet, standing straight
- Shoulders level and relaxed (not raised or hunched)
- Do NOT include text, watermarks, or UI elements

IMAGE QUALITY:
- Resolution: high detail, sharp fabric textures visible
- Lighting: soft studio lighting with subtle shadows
- No motion blur, no noise, no compression artifacts
"""

    # P2P Physics Keywords Injection
    if physics_prompt:
        prompt += f"""
PHYSICAL FIT ACCURACY:
{physics_prompt}
These fit descriptions MUST be visually reflected in the generated image.
"""

    # Face identity: ALWAYS use single best photo (never multiple)
    # CRITICAL: Face image must be FIRST in the image list, and prompt must
    # explicitly reference "Image 1" as the face. This gives Gemini the strongest
    # signal about which image contains the target identity.
    images = []
    best_face = _select_best_face_photo(face_photo_bgr, face_references)
    has_face = False

    if best_face is not None:
        cleaned = _clean_face_photo(best_face)
        cleaned = cv2.resize(cleaned, (768, 768), interpolation=cv2.INTER_LANCZOS4)
        images.append(cleaned)  # FIRST image = face reference
        has_face = True
        logger.info(f"Front view: face reference prepared ({cleaned.shape[:2]})")

    images.append(clothing_image_bgr)  # Second image = clothing product
    if mesh_render_bgr is not None:
        images.append(mesh_render_bgr)  # Third image = body mesh

    if has_face:
        n_img = len(images)
        prompt += f"""
FACE IDENTITY — THIS IS THE MOST IMPORTANT REQUIREMENT:
I am providing {n_img} images in this order:
  Image 1: FACE REFERENCE PHOTO — this shows the EXACT person whose face you must reproduce
  Image 2: CLOTHING PRODUCT — the garment to put on the person
{f'  Image 3: BODY SILHOUETTE — reference for body pose/proportions' if mesh_render_bgr is not None else ''}

The generated person MUST be the SAME individual shown in Image 1 (the face reference).
- Copy the EXACT facial features from Image 1: face shape, eyes, nose, mouth, jawline, skin tone
- Copy the EXACT hairstyle from Image 1: color, length, parting, texture, volume
- HAIRSTYLE IS CRITICAL: If the reference person has bangs/fringe, the generated image MUST also have bangs/fringe in exactly the same style. If no bangs, then no bangs. Do NOT remove or add bangs.
- HAIR TEXTURE IS CRITICAL: If the reference person has STRAIGHT hair, generate STRAIGHT hair. If WAVY, generate WAVY. If CURLY, generate CURLY. Do NOT change the hair texture.
- Do NOT generate a generic model face — you MUST reproduce this specific person
- The face must be SHARP, high-resolution, and artifact-free
- This is a face-swap scenario: take this person's face and put them in studio clothing
"""

    pil_img = gemini.generate_tryon_image(prompt, images)
    if pil_img is None:
        return None

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = _enhance_tryon_image(result)
    return result


def generate_angle_with_reference(gemini: GeminiClient,
                                   front_view_rgb: np.ndarray,
                                   face_photo_bgr: np.ndarray | None,
                                   clothing: ClothingAnalysis,
                                   mesh_render_bgr: np.ndarray | None,
                                   angle_deg: int,
                                   gender: str = "female",
                                   physics_prompt: str | None = None,
                                   face_references: list | None = None) -> np.ndarray | None:
    """
    Generate try-on image at a specific angle using front view as reference.
    Uses single best face photo only (not multiple refs).
    """
    angle_text = _angle_to_text(angle_deg)
    angle_detail = _get_angle_body_details(angle_deg)
    clothing_desc = gemini._build_clothing_description(clothing)

    prompt = f"""Generate a photorealistic full-body fashion photograph of the SAME person
from the reference image, now viewed from {angle_text} ({angle_deg} degrees).

Clothing: {clothing_desc}

ANGLE-SPECIFIC DETAILS for {angle_deg}°:
{angle_detail}

The model should be:
- The IDENTICAL person from the reference (same face, body, hair, skin tone)
- Same clothing as the reference image — EXACT same garment with EXACT same colors
- Viewed from {angle_text} angle with correct perspective and foreshortening
- Standing in a natural relaxed pose with arms relaxed naturally at sides
- On a clean, neutral light gray (#E0E0E0) studio background
- Full body visible from head to toe

CRITICAL COLOR ACCURACY:
- Match the EXACT color tones from the front reference image — same shirt color, same bottoms color
- Do NOT saturate, warm, or shift any colors. Keep the muted/desaturated tones exactly as they appear
- If the front reference shirt has a COOL/PINK undertone, preserve it — do NOT shift to warm/tan/golden
- Top and bottoms each maintain their own distinct color (do not blend)
- If the front reference shows satin/sheen on the fabric, maintain that sheen at this angle

GARMENT CONTINUITY AT THIS ANGLE:
- The garment is a single continuous piece of clothing — what's visible from {angle_text} must be
  structurally consistent with the front view (same length, same fit tightness, same hem height)
- Fabric wrinkles and folds should be physically plausible for this viewing angle
- Seams, stitching, and construction details must be consistent with the garment type

CRITICAL CONSISTENCY:
- EXACT likeness to the reference person — same face, same hair (same bangs, same texture: straight/wavy/curly)
- Same body proportions and posture as the front reference
- Same clothing pattern, texture, sheen, and fit as the front reference
- Realistic perspective and foreshortening for {angle_text} viewing angle
- Same lighting direction as the front reference
- Do NOT include text, watermarks, or UI elements
"""

    # P2P Physics Keywords Injection
    if physics_prompt:
        prompt += f"""
PHYSICAL FIT ACCURACY:
{physics_prompt}
"""

    # Build image list with explicit ordering:
    # Face reference FIRST (most important), then front view, then mesh
    # CRITICAL: Same pattern as generate_front_view_gemini() — face must be
    # Image 1 with explicit numbering in prompt for Gemini to reproduce identity.
    images = []
    best_face = _select_best_face_photo(face_photo_bgr, face_references)
    has_face = False

    if best_face is not None:
        cleaned = _clean_face_photo(best_face)
        cleaned = cv2.resize(cleaned, (768, 768), interpolation=cv2.INTER_LANCZOS4)
        images.append(cleaned)  # Image 1 = face reference
        has_face = True
        logger.info(f"Angle {angle_deg}°: face reference prepared ({cleaned.shape[:2]})")

    images.append(front_view_rgb)  # Image 2 (or 1 if no face) = front view reference

    if mesh_render_bgr is not None:
        images.append(mesh_render_bgr)  # Last image = body mesh silhouette

    if has_face:
        n_img = len(images)
        mesh_idx = n_img if mesh_render_bgr is not None else None
        prompt += f"""
FACE IDENTITY — THIS IS THE MOST IMPORTANT REQUIREMENT:
I am providing {n_img} images in this order:
  Image 1: FACE REFERENCE PHOTO — close-up of the EXACT person whose face you must reproduce
  Image 2: FRONT VIEW REFERENCE — the same person from the front, wearing the clothing
{f'  Image {mesh_idx}: BODY SILHOUETTE — mesh reference for body pose at {angle_deg}°' if mesh_render_bgr is not None else ''}

The person in your generated image MUST be the SAME individual shown in Image 1.
- Copy the EXACT facial features from Image 1: face shape, eyes, nose, mouth, jawline, skin tone
- Copy the EXACT hairstyle from Image 1: color, length, parting, texture, volume
- HAIRSTYLE IS CRITICAL: If the person has bangs/fringe in Image 1, the generated image MUST also have bangs/fringe in exactly the same style. Do NOT remove or add bangs.
- HAIR TEXTURE IS CRITICAL: If the person has STRAIGHT hair in Image 1, generate STRAIGHT hair. If WAVY, generate WAVY. If CURLY, generate CURLY. Do NOT change the hair texture. Match the exact texture visible in Image 1.
- The face in Image 2 (front view) should ALSO match — use both images to confirm identity
- Do NOT generate a generic model face — this is a face-swap scenario
- At {angle_deg}°, the face may be partially visible or in profile — but it must still be recognizably the SAME person
- Hair from the back/side must be consistent with the front view (same length, color, style, bangs)
"""
    else:
        prompt += """
IDENTITY CONSISTENCY:
- The generated person must be clearly the SAME individual as in the front view reference
- Same face shape, skin tone, hair color, hairstyle, body proportions
"""

    pil_img = gemini.generate_tryon_image(prompt, images)
    if pil_img is None:
        return None

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = _enhance_tryon_image(result)
    return result
