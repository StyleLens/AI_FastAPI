"""
StyleLens V6 SOTA Pipeline — Gemini Client
Full Gemini API client for clothing analysis, video/photo analysis,
size chart extraction, and image generation fallback.
Uses google-genai SDK (new pattern).
"""

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from core.config import (
    GEMINI_API_KEY,
    GEMINI_ENABLED,
    GEMINI_MODEL_NAME,
    GEMINI_PRO_MODEL_NAME,
    V5_GEMINI_IMAGE_MODEL,
    GEMINI_FLASH_IMAGE_MODEL,
    GEMINI_FLASH_TEXT_MODEL,
)

logger = logging.getLogger("stylelens.gemini")


# ── Dataclasses ────────────────────────────────────────────────

@dataclass
class GeminiBodyAnalysis:
    height_cm: float = 170.0
    weight_kg: float = 65.0
    bmi: float = 22.5
    body_fat_pct: float = 20.0
    shoulder_width_cm: float = 42.0
    chest_cm: float = 90.0
    waist_cm: float = 75.0
    hip_cm: float = 95.0
    bust_cup: str = "B"
    gender: str = "female"
    body_type: str = "standard"
    confidence: float = 0.5


@dataclass
class GeminiPhotoAnalysis:
    gender: str = "female"
    age_range: str = "25-30"
    skin_tone: str = "medium"
    hair_color: str = "dark brown"
    hair_length: str = "medium"
    hair_style: str = "straight"
    face_shape: str = "oval"
    body_type: str = "standard"
    height_estimate_cm: float = 165.0
    confidence: float = 0.5


@dataclass
class SupervisedResult:
    score: float = 0.0
    passed: bool = False
    feedback: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ClothingAnalysis:
    name: str = ""
    category: str = ""
    color: str = ""
    color_hex: str = "#000000"
    fabric: str = ""
    fit_type: str = ""
    subcategory: str = ""
    neck_style: str = ""
    sleeve_type: str = ""
    thickness: str = ""
    elasticity: str = ""
    fabric_composition: str = ""
    fabric_weight_gsm: int = 0
    surface_texture: str = ""
    secondary_colors: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    body_interaction: str = ""
    unique_elements: list[str] = field(default_factory=list)
    transparency: str = "opaque"
    has_lining: bool = False
    care_instructions: str = ""
    size_info: dict = field(default_factory=dict)
    button_count: int = 0
    button_positions: str = ""
    button_type: str = ""
    wrinkle_locations: str = ""
    wrinkle_intensity: str = ""
    drape_style: str = ""
    pocket_count: int = 0
    pocket_positions: str = ""
    pocket_type: str = ""
    zipper_type: str = ""
    zipper_position: str = ""
    seam_details: str = ""
    pattern_type: str = ""
    pattern_description: str = ""
    logo_text: str = ""
    logo_position: str = ""
    hem_style: str = ""
    closure_type: str = ""
    view_angles_analyzed: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    camera_angle: str = ""
    confidence: float = 0.0


# ── Utility Functions ──────────────────────────────────────────

def _validate_hex(value: str) -> str:
    """Validate hex color code, return #000000 if invalid."""
    if not value or not isinstance(value, str):
        return "#000000"
    value = value.strip()
    if re.match(r"^#[0-9a-fA-F]{6}$", value):
        return value
    if re.match(r"^[0-9a-fA-F]{6}$", value):
        return f"#{value}"
    return "#000000"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = _validate_hex(hex_color).lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _angle_to_text(angle_deg: int) -> str:
    """Convert angle to human-readable text."""
    mapping = {
        0: "front", 45: "front-right", 90: "right side",
        135: "back-right", 180: "back", 225: "back-left",
        270: "left side", 315: "front-left",
    }
    return mapping.get(angle_deg, f"{angle_deg} degrees")


# Field name normalization for fuzzy matching
_FIELD_ALIASES = {
    "name": ["name", "garment_name", "item_name", "product_name"],
    "category": ["category", "garment_type", "type", "clothing_type"],
    "color": ["color", "main_color", "primary_color", "colour"],
    "color_hex": ["color_hex", "hex_color", "hex", "color_code"],
    "fabric": ["fabric", "material", "fabric_type", "main_fabric"],
    "fit_type": ["fit_type", "fit", "silhouette", "fit_style"],
    "subcategory": ["subcategory", "sub_category", "sub_type"],
    "neck_style": ["neck_style", "neckline", "neck_type", "collar"],
    "sleeve_type": ["sleeve_type", "sleeve", "sleeve_length", "sleeves"],
    "thickness": ["thickness", "weight", "fabric_thickness"],
    "elasticity": ["elasticity", "stretch", "stretchiness"],
    "fabric_composition": ["fabric_composition", "composition", "material_composition"],
    "fabric_weight_gsm": ["fabric_weight_gsm", "gsm", "weight_gsm", "fabric_weight"],
    "surface_texture": ["surface_texture", "texture", "surface_finish", "finish"],
    "secondary_colors": ["secondary_colors", "secondary_color"],
    "accent_colors": ["accent_colors", "accent_color"],
    "body_interaction": ["body_interaction", "body_fit", "how_it_fits"],
    "unique_elements": ["unique_elements", "unique_features", "special_features"],
    "transparency": ["transparency", "opacity", "see_through"],
    "has_lining": ["has_lining", "lining", "lined"],
    "care_instructions": ["care_instructions", "care", "washing"],
    "button_count": ["button_count", "buttons", "number_of_buttons"],
    "button_positions": ["button_positions", "button_placement"],
    "button_type": ["button_type", "button_style"],
    "wrinkle_locations": ["wrinkle_locations", "wrinkles", "wrinkle_areas"],
    "wrinkle_intensity": ["wrinkle_intensity", "wrinkle_level"],
    "drape_style": ["drape_style", "drape", "draping"],
    "pocket_count": ["pocket_count", "pockets", "number_of_pockets"],
    "pocket_positions": ["pocket_positions", "pocket_placement"],
    "pocket_type": ["pocket_type", "pocket_style"],
    "zipper_type": ["zipper_type", "zipper", "zip_type"],
    "zipper_position": ["zipper_position", "zip_position"],
    "seam_details": ["seam_details", "seams", "stitching"],
    "pattern_type": ["pattern_type", "pattern", "print_type"],
    "pattern_description": ["pattern_description", "print_description"],
    "logo_text": ["logo_text", "logo", "brand_text", "branding"],
    "logo_position": ["logo_position", "logo_placement"],
    "hem_style": ["hem_style", "hem", "hemline"],
    "closure_type": ["closure_type", "closure", "fastening"],
    "style_tags": ["style_tags", "tags", "styles", "keywords"],
    "camera_angle": ["camera_angle", "angle", "view_angle", "shooting_angle"],
    "confidence": ["confidence", "confidence_score", "conf"],
}


def _fuzzy_match_field(raw_key: str) -> str:
    """Match a raw key from Gemini response to a canonical field name."""
    normalized = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    for canonical, aliases in _FIELD_ALIASES.items():
        if normalized in aliases:
            return canonical
    return "unknown"


def _image_to_base64(image_bgr: np.ndarray, fmt: str = "JPEG") -> str:
    """Convert BGR numpy array to base64 string."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _pil_to_bytes(pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, quality=92)
    return buf.getvalue()


# ── Prompts ────────────────────────────────────────────────────

_VIDEO_ANALYSIS_PROMPT = """Analyze this video of a person for body measurement estimation.

Return a JSON object with these fields:
{
  "height_cm": <estimated height in cm>,
  "weight_kg": <estimated weight in kg>,
  "bmi": <estimated BMI>,
  "body_fat_pct": <estimated body fat percentage>,
  "shoulder_width_cm": <shoulder width cm>,
  "chest_cm": <chest circumference cm>,
  "waist_cm": <waist circumference cm>,
  "hip_cm": <hip circumference cm>,
  "bust_cup": "<A/B/C/D or N/A for male>",
  "gender": "<male/female>",
  "body_type": "<slim/standard/average/athletic/bulky>",
  "confidence": <0.0-1.0>
}

Be precise with measurements. Return ONLY the JSON object, no other text."""

_PHOTO_ANALYSIS_PROMPT = """Analyze this photo of a person for physical appearance.

Return a JSON object:
{
  "gender": "<male/female>",
  "age_range": "<e.g. 25-30>",
  "skin_tone": "<fair/light/medium/olive/tan/dark/deep>",
  "hair_color": "<description>",
  "hair_length": "<short/medium/long>",
  "hair_style": "<straight/wavy/curly/coily>",
  "face_shape": "<oval/round/square/heart/oblong>",
  "body_type": "<slim/standard/average/athletic/bulky>",
  "height_estimate_cm": <number>,
  "confidence": <0.0-1.0>
}

Return ONLY the JSON object."""

_IMAGE_FASHION_PROMPT = """Analyze this clothing image in extreme detail.

Return a JSON object with ALL of these fields:
{
  "name": "<descriptive garment name>",
  "category": "<top/bottom/dress/outerwear/activewear/swimwear/underwear/accessories>",
  "color": "<primary color name>",
  "color_hex": "<#RRGGBB hex code of main color>",
  "fabric": "<primary fabric type>",
  "fit_type": "<slim/regular/relaxed/oversized>",
  "subcategory": "<specific type, e.g. t-shirt, blazer, jeans>",
  "neck_style": "<crew/v-neck/turtleneck/scoop/boat/collar/hoodie/off-shoulder/N/A>",
  "sleeve_type": "<sleeveless/short/3-4/long/N/A>",
  "thickness": "<thin/medium/thick/heavy>",
  "elasticity": "<none/slight/moderate/high>",
  "fabric_composition": "<e.g. 95% cotton 5% elastane>",
  "fabric_weight_gsm": <estimated GSM number>,
  "surface_texture": "<smooth/satin/matte/glossy/textured/ribbed/quilted/rough/suede/velvet>",
  "transparency": "<opaque/semi-transparent/transparent>",
  "has_lining": <true/false>,
  "care_instructions": "<general care advice>",
  "button_count": <number or 0>,
  "button_positions": "<front/cuff/collar/N/A>",
  "button_type": "<flat/shank/snap/toggle/N/A>",
  "wrinkle_locations": "<description or none>",
  "wrinkle_intensity": "<none/light/moderate/heavy>",
  "drape_style": "<structured/semi-structured/flowing/stiff>",
  "pocket_count": <number or 0>,
  "pocket_positions": "<chest/side/back/cargo/N/A>",
  "pocket_type": "<patch/welt/slash/flap/N/A>",
  "zipper_type": "<metal/plastic/invisible/N/A>",
  "zipper_position": "<front/side/back/N/A>",
  "seam_details": "<description>",
  "pattern_type": "<solid/striped/plaid/floral/graphic/abstract/animal/geometric/N/A>",
  "pattern_description": "<detailed pattern description>",
  "logo_text": "<visible text or N/A>",
  "logo_position": "<chest/back/sleeve/hem/N/A>",
  "hem_style": "<straight/curved/raw/ribbed/banded/asymmetric>",
  "closure_type": "<buttons/zipper/pullover/wrap/hook/snap/N/A>",
  "style_tags": ["<tag1>", "<tag2>", ...],
  "camera_angle": "<front/back/side/flat-lay/detail>",
  "confidence": <0.0-1.0>
}

Be extremely precise. Return ONLY the JSON object."""

_DETAILED_FASHION_PROMPT = """Analyze this clothing item with maximum detail for virtual try-on.

IMPORTANT: This analysis is for the PRIMARY garment (shirt/blouse/top/jacket/dress) only.
If the image shows a model wearing multiple pieces (e.g., a shirt + skirt), analyze the
TOP/PRIMARY garment. Do NOT use the bottoms' color as the primary color_hex.

CRITICAL for color_hex field:
- "color_hex" MUST be the hex color of the PRIMARY garment (the top/shirt/blouse)
- Do NOT use black (#000000) unless the primary garment itself is actually black
- If the model wears a beige shirt + black skirt, color_hex should be the BEIGE hex, NOT black
- Sample the color from the largest area of the PRIMARY garment fabric

Return a JSON object with ALL standard analysis fields PLUS:
{
  "color": "<primary garment color name>",
  "color_hex": "<#hex of the PRIMARY garment's dominant color — NOT the bottoms>",
  "secondary_colors": ["#hex of any secondary colors on the PRIMARY garment"],
  "accent_colors": ["#hex of accents"],
  "surface_texture": "<smooth/satin/matte/glossy/textured/ribbed/quilted/etc>",
  "fabric": "<fabric type like satin, cotton, polyester, silk, chiffon, etc>",
  "body_interaction": "<how the garment drapes/sits on body>",
  "drape_style": "<how fabric falls — flowing, structured, clingy, etc>",
  "unique_elements": ["<element1>", "<element2>"],
  "neck_style": "<collar type>",
  "sleeve_type": "<sleeve style>",
  ... (all other standard fields)
}

Return ONLY the JSON object."""

_HTML_FASHION_PROMPT = """Extract clothing product information from this webpage HTML.

Return a JSON object:
{
  "name": "<product name>",
  "category": "<category>",
  "color": "<color>",
  "fabric_composition": "<materials>",
  "price": "<price if visible>",
  "size_options": ["<S>", "<M>", "<L>", ...],
  "fit_type": "<fit description>",
  "care_instructions": "<care info>"
}

Return ONLY the JSON object."""

_VIEW_CLASSIFICATION_PROMPT = """Classify the viewing angle of this clothing photograph.

Return EXACTLY one of these values:
- "front" (full front view)
- "back" (full back view)
- "left-side" (left profile)
- "right-side" (right profile)
- "45-front-left"
- "45-front-right"
- "45-back-left"
- "45-back-right"
- "flat-lay" (laid flat on surface)
- "detail-closeup" (zoomed in on detail)

Return ONLY the classification string, nothing else."""

_SIZE_CHART_PROMPT = """Extract size chart information from this image.

IMPORTANT: Determine if measurements are "flat-lay" (단면, half-width laid flat)
or "full circumference" (둘레, full wrap-around). Korean clothing sites typically
use flat-lay (단면) measurements where chest/waist/hip values are HALF the
circumference. Look for keywords: 단면, flat, 가슴단면, 허리단면.
If unclear, assume flat-lay for Asian size charts with small values (e.g. chest < 70cm).

Return a JSON object:
{
  "brand": "<brand name if visible>",
  "size_system": "<US/EU/UK/Asian>",
  "measurement_type": "<flat-lay or full-circumference>",
  "sizes": {
    "<size label>": {
      "chest_cm": <number or null>,
      "waist_cm": <number or null>,
      "hip_cm": <number or null>,
      "length_cm": <number or null>,
      "shoulder_cm": <number or null>,
      "sleeve_cm": <number or null>
    }
  },
  "notes": "<any additional sizing notes>"
}

Return ONLY the JSON object."""

_PRODUCT_INFO_PROMPT = """Extract product information from this image.
This may be a product page screenshot, label, tag, or packaging.

Return a JSON object:
{
  "brand": "<brand name>",
  "product_name": "<full product name>",
  "material": "<material/fabric composition>",
  "size": "<size if visible>",
  "price": "<price if visible>",
  "color_name": "<official color name>",
  "sku": "<SKU/product code if visible>",
  "care": "<care instructions if visible>",
  "country_of_origin": "<if visible>",
  "additional_info": "<any other relevant info>"
}

Return ONLY the JSON object."""

_FITTING_MODEL_INFO_PROMPT = """Analyze this photo of a fitting model / mannequin display.

Return a JSON object:
{
  "model_height_cm": <estimated height>,
  "model_size_wearing": "<size being worn, e.g. M, L>",
  "model_body_type": "<slim/standard/average/athletic>",
  "model_chest_cm": <estimated chest>,
  "model_waist_cm": <estimated waist>,
  "model_hip_cm": <estimated hip>,
  "garment_fit_on_model": "<tight/fitted/regular/loose/oversized>",
  "confidence": <0.0-1.0>
}

Return ONLY the JSON object."""

_HAIR_ANALYSIS_PROMPT = """Analyze this person's hair for 3D avatar generation.

Return a JSON object:
{
  "hair_length": "<short/medium/long/very-long>",
  "hair_texture": "<straight/wavy/curly/coily>",
  "hair_color": "<natural color description>",
  "hair_style": "<specific style name>",
  "has_bangs": <true/false>,
  "parting": "<center/left/right/none>",
  "volume": "<flat/normal/voluminous>",
  "confidence": <0.0-1.0>
}

Return ONLY the JSON object."""


def _build_spatial_estimate_prompt(measurements: dict) -> str:
    """Build a prompt for garment size estimation from body measurements."""
    return f"""Given these body measurements:
- Height: {measurements.get('height_cm', 170)} cm
- Chest: {measurements.get('chest_cm', 90)} cm
- Waist: {measurements.get('waist_cm', 75)} cm
- Hip: {measurements.get('hip_cm', 95)} cm

Estimate the best clothing size for this person.

Return a JSON object:
{{
  "recommended_size": "<XS/S/M/L/XL/XXL>",
  "size_confidence": <0.0-1.0>,
  "notes": "<any fitting notes>"
}}

Return ONLY the JSON object."""


# ── GeminiClient ───────────────────────────────────────────────

class GeminiClient:
    """Gemini API client for all analysis and image generation tasks."""

    def __init__(self):
        if not GEMINI_ENABLED:
            logger.warning("Gemini API key not set — client will not function")
            self._client = None
            return

        from google import genai
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")

    def _call_text(self, prompt: str, images: list[np.ndarray] | None = None,
                   model: str | None = None) -> str:
        """Call Gemini text model with optional images.

        Falls back to Flash model if Pro model quota is exhausted (429 error).
        """
        if not self._client:
            raise RuntimeError("Gemini client not initialized (missing API key)")

        from google.genai import types
        model = model or GEMINI_MODEL_NAME

        contents = []
        if images:
            for img in images:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=90)
                contents.append(types.Part.from_bytes(
                    data=buf.getvalue(), mime_type="image/jpeg"
                ))
        contents.append(prompt)

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            text = response.text
            if text is None:
                logger.warning(f"Gemini returned None text for model={model}")
            return text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Fall back to Flash model
                fallback = GEMINI_FLASH_TEXT_MODEL
                if model == fallback:
                    raise  # Both exhausted, give up
                logger.warning(f"Pro model quota exhausted, falling back to {fallback}")
                response = self._client.models.generate_content(
                    model=fallback,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=4096,
                    ),
                )
                text = response.text
                if text is None:
                    logger.warning(f"Gemini returned None text for fallback model={fallback}")
                return text
            raise

    def _call_image(self, prompt: str,
                    images: list[np.ndarray] | None = None,
                    model: str | None = None) -> tuple[Image.Image | None, str]:
        """Call Gemini image generation model. Returns (PIL Image or None, text)."""
        if not self._client:
            raise RuntimeError("Gemini client not initialized")

        from google.genai import types
        model = model or V5_GEMINI_IMAGE_MODEL

        contents = []
        if images:
            for img in images:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                buf = io.BytesIO()
                # High quality JPEG for maximum detail in reference images
                pil.save(buf, format="JPEG", quality=95)
                contents.append(types.Part.from_bytes(
                    data=buf.getvalue(), mime_type="image/jpeg"
                ))
        contents.append(prompt)

        # Try primary model, then fallbacks
        models_to_try = [model, GEMINI_FLASH_IMAGE_MODEL]
        for m in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=m,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
                # Check for image in response
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            img_bytes = part.inline_data.data
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            text = ""
                            for p in response.candidates[0].content.parts:
                                if hasattr(p, "text") and p.text:
                                    text += p.text
                            return pil_img, text
                logger.warning(f"Model {m} returned no image, trying next")
            except Exception as e:
                logger.warning(f"Model {m} failed: {e}")
                continue

        return None, "All image models failed"

    def _parse_json(self, text: str | None) -> dict:
        """Extract JSON from Gemini response text."""
        if text is None:
            logger.warning("_parse_json received None text")
            return {}
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Try finding first { ... }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from response: {text[:200]}")
        return {}

    # ── Analysis Methods ───────────────────────────────────────

    def analyze_video(self, frames: list[np.ndarray]) -> GeminiBodyAnalysis:
        """Analyze video frames for body measurements."""
        # Send 3-5 evenly spaced frames
        step = max(1, len(frames) // 5)
        selected = frames[::step][:5]
        text = self._call_text(_VIDEO_ANALYSIS_PROMPT, selected)
        data = self._parse_json(text)
        return self._parse_video_response(data)

    def analyze_photo(self, image_bgr: np.ndarray) -> GeminiPhotoAnalysis:
        """Analyze a single photo for appearance."""
        text = self._call_text(_PHOTO_ANALYSIS_PROMPT, [image_bgr])
        data = self._parse_json(text)
        return self._parse_photo_response(data)

    def analyze_image_for_fashion(self, image_bgr: np.ndarray) -> ClothingAnalysis:
        """Standard clothing analysis from single image."""
        text = self._call_text(_IMAGE_FASHION_PROMPT, [image_bgr])
        data = self._parse_json(text)
        return self._parse_fashion_response(data)

    def analyze_detailed_fashion(self, image_bgr: np.ndarray) -> ClothingAnalysis:
        """Detailed clothing analysis for virtual try-on."""
        text = self._call_text(_DETAILED_FASHION_PROMPT, [image_bgr])
        data = self._parse_json(text)
        return self._parse_detailed_fashion_response(data)

    def analyze_html_for_fashion(self, html: str) -> dict:
        """Extract fashion info from HTML."""
        text = self._call_text(f"{_HTML_FASHION_PROMPT}\n\n{html[:8000]}")
        return self._parse_json(text)

    def analyze_size_chart_image(self, image_bgr: np.ndarray) -> dict:
        """Extract size chart from image."""
        text = self._call_text(_SIZE_CHART_PROMPT, [image_bgr])
        return self._parse_json(text)

    def analyze_product_info_image(self, image_bgr: np.ndarray) -> dict:
        """Extract product info from single image."""
        text = self._call_text(_PRODUCT_INFO_PROMPT, [image_bgr])
        return self._parse_json(text)

    def analyze_product_info_images(self, images_bgr: list[np.ndarray]) -> dict:
        """Extract product info from multiple images."""
        combined_prompt = (
            f"{_PRODUCT_INFO_PROMPT}\n\n"
            f"I'm providing {len(images_bgr)} images of the same product. "
            "Combine information from all images into a single JSON response."
        )
        text = self._call_text(combined_prompt, images_bgr)
        return self._parse_json(text)

    def extract_fitting_model_info(self, image_bgr: np.ndarray) -> dict:
        """Extract fitting model body info from photo."""
        text = self._call_text(_FITTING_MODEL_INFO_PROMPT, [image_bgr])
        return self._parse_json(text)

    def estimate_garment_size_from_image(self, image_bgr: np.ndarray,
                                          measurements: dict) -> dict:
        """Estimate garment size based on body measurements."""
        prompt = _build_spatial_estimate_prompt(measurements)
        text = self._call_text(prompt, [image_bgr])
        return self._parse_json(text)

    def analyze_hair(self, image_bgr: np.ndarray) -> dict:
        """Analyze hair for avatar generation."""
        text = self._call_text(_HAIR_ANALYSIS_PROMPT, [image_bgr])
        return self._parse_json(text)

    def classify_view(self, image_bgr: np.ndarray) -> str:
        """Classify clothing image viewing angle."""
        text = self._call_text(_VIEW_CLASSIFICATION_PROMPT, [image_bgr])
        if text is None:
            logger.warning("classify_view: Gemini returned None, defaulting to 'front'")
            return "front"
        result = text.strip().strip('"').strip("'").lower()
        valid = [
            "front", "back", "left-side", "right-side",
            "45-front-left", "45-front-right", "45-back-left", "45-back-right",
            "flat-lay", "detail-closeup",
        ]
        return result if result in valid else "front"

    def generate_tryon_image(self, prompt: str,
                              images: list[np.ndarray] | None = None,
                              model: str | None = None) -> Image.Image | None:
        """Generate a try-on image using Gemini image model (fallback path)."""
        pil_img, text = self._call_image(prompt, images, model)
        return pil_img

    def _build_clothing_description(self, analysis: ClothingAnalysis) -> str:
        """Build a rich text description of clothing for prompts.

        Includes hex color codes, surface texture, fabric details, and design
        elements so Gemini can reproduce the garment accurately.
        """
        parts = []

        # Precise color with hex code
        if analysis.color:
            if analysis.color_hex and analysis.color_hex != "#000000":
                parts.append(f"{analysis.color} (exact color: {analysis.color_hex})")
            else:
                parts.append(analysis.color)

        # Surface texture / fabric description
        surface = getattr(analysis, "surface_texture", "")
        if surface:
            parts.append(f"{surface}-finish")
        if analysis.fabric:
            parts.append(analysis.fabric)
        elif analysis.fabric_composition:
            parts.append(analysis.fabric_composition)

        # Garment type
        if analysis.name:
            parts.append(analysis.name)
        elif analysis.subcategory:
            parts.append(analysis.subcategory)
        elif analysis.category:
            parts.append(analysis.category)

        # Fit & style details
        if analysis.fit_type and analysis.fit_type != "regular":
            parts.append(f"({analysis.fit_type} fit)")
        if analysis.pattern_type and analysis.pattern_type not in ("solid", "N/A", ""):
            parts.append(f"with {analysis.pattern_type} pattern")

        # Collar / neckline
        if analysis.neck_style:
            parts.append(f"with {analysis.neck_style} collar/neckline")

        # Sleeve
        if analysis.sleeve_type:
            parts.append(f"and {analysis.sleeve_type} sleeves")

        # Closure type (buttons, zipper, pullover, etc.)
        if analysis.closure_type and analysis.closure_type not in ("N/A", ""):
            if analysis.button_count and analysis.button_count > 0:
                parts.append(f"with {analysis.button_count} {analysis.closure_type}")
            else:
                parts.append(f"({analysis.closure_type} closure)")

        # Hem style
        if analysis.hem_style and analysis.hem_style not in ("N/A", ""):
            parts.append(f"with {analysis.hem_style} hem")

        # Drape / unique elements
        if analysis.drape_style:
            parts.append(f"— {analysis.drape_style} drape")

        # Unique design elements
        if analysis.unique_elements:
            elements_str = ", ".join(analysis.unique_elements[:3])
            parts.append(f"[details: {elements_str}]")

        return " ".join(parts) if parts else "clothing item"

    # ── Response Parsers ───────────────────────────────────────

    def _parse_video_response(self, data: dict) -> GeminiBodyAnalysis:
        result = GeminiBodyAnalysis()
        if not data:
            return result
        result.height_cm = float(data.get("height_cm", 170))
        result.weight_kg = float(data.get("weight_kg", 65))
        result.bmi = float(data.get("bmi", 22.5))
        result.body_fat_pct = float(data.get("body_fat_pct", 20))
        result.shoulder_width_cm = float(data.get("shoulder_width_cm", 42))
        result.chest_cm = float(data.get("chest_cm", 90))
        result.waist_cm = float(data.get("waist_cm", 75))
        result.hip_cm = float(data.get("hip_cm", 95))
        result.bust_cup = str(data.get("bust_cup", "B"))
        result.gender = str(data.get("gender", "female")).lower()
        result.body_type = str(data.get("body_type", "standard"))
        result.confidence = _clamp(float(data.get("confidence", 0.5)), 0, 1)
        return result

    def _parse_photo_response(self, data: dict) -> GeminiPhotoAnalysis:
        result = GeminiPhotoAnalysis()
        if not data:
            return result
        result.gender = str(data.get("gender", "female")).lower()
        result.age_range = str(data.get("age_range", "25-30"))
        result.skin_tone = str(data.get("skin_tone", "medium"))
        result.hair_color = str(data.get("hair_color", "dark brown"))
        result.hair_length = str(data.get("hair_length", "medium"))
        result.hair_style = str(data.get("hair_style", "straight"))
        result.face_shape = str(data.get("face_shape", "oval"))
        result.body_type = str(data.get("body_type", "standard"))
        result.height_estimate_cm = float(data.get("height_estimate_cm", 165))
        result.confidence = _clamp(float(data.get("confidence", 0.5)), 0, 1)
        return result

    def _parse_fashion_response(self, data: dict) -> ClothingAnalysis:
        """Parse standard fashion analysis response."""
        result = ClothingAnalysis()
        if not data:
            return result
        for raw_key, value in data.items():
            canonical = _fuzzy_match_field(raw_key)
            if canonical == "unknown":
                continue
            try:
                if canonical == "color_hex":
                    setattr(result, canonical, _validate_hex(str(value)))
                elif canonical in ("fabric_weight_gsm", "button_count", "pocket_count"):
                    setattr(result, canonical, int(value) if value else 0)
                elif canonical == "has_lining":
                    setattr(result, canonical, bool(value))
                elif canonical == "confidence":
                    setattr(result, canonical, _clamp(float(value), 0, 1))
                elif canonical in ("style_tags", "view_angles_analyzed",
                                   "secondary_colors", "accent_colors", "unique_elements"):
                    if isinstance(value, list):
                        setattr(result, canonical, value)
                    elif isinstance(value, str):
                        setattr(result, canonical, [v.strip() for v in value.split(",")])
                elif canonical == "size_info":
                    setattr(result, canonical, value if isinstance(value, dict) else {})
                else:
                    setattr(result, canonical, str(value) if value else "")
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse field {canonical}={value}: {e}")
        return result

    def _parse_detailed_fashion_response(self, data: dict) -> ClothingAnalysis:
        """Parse detailed fashion analysis (superset of standard)."""
        result = self._parse_fashion_response(data)
        # Additional detail fields handled by standard parser via fuzzy matching
        return result
