"""
StyleLens V6 SOTA Pipeline — Gemini Feedback Inspector
6 quality gates using Gemini 3 Pro Preview for inter-stage quality inspection.
Uses google-genai SDK (new pattern).
"""

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
    GEMINI_FLASH_TEXT_MODEL,
    STAGE_THRESHOLDS,
)

logger = logging.getLogger("stylelens.feedback")


@dataclass
class InspectionResult:
    stage: str = ""
    quality_score: float = 0.0
    pass_check: bool = False
    feedback: str = ""
    retry_suggested: bool = False
    issues: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    raw_response: str = ""


class GeminiFeedbackInspector:
    """Quality gate inspector using Gemini 3 Pro Preview."""

    def __init__(self):
        self._client = None
        self._inspection_log: list[InspectionResult] = []

        if not GEMINI_ENABLED:
            logger.warning("Gemini not available — feedback inspector disabled")
            return

        from google import genai
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini Feedback Inspector initialized")

    def _call_gemini(self, prompt: str,
                     images: list[np.ndarray] | None = None) -> str:
        """Call Gemini text model with optional images.

        Uses Flash model for quality gate evaluations to conserve Pro quota.
        Pro model daily limits are tight; Flash has much higher quotas.
        """
        if not self._client:
            return '{"quality_score": 0.5, "pass": true, "feedback": "Inspector disabled"}'

        from google.genai import types

        contents = []
        if images:
            for img in images:
                if img is None:
                    continue
                if len(img.shape) == 3 and img.shape[2] == 3:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    rgb = img
                pil = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=85)
                contents.append(types.Part.from_bytes(
                    data=buf.getvalue(), mime_type="image/jpeg"
                ))
        contents.append(prompt)

        # Use Flash for evaluations (high quota), fall back to Pro if needed
        model = GEMINI_FLASH_TEXT_MODEL
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(f"Flash model quota exhausted, trying Pro model...")
                model = GEMINI_MODEL_NAME
            else:
                raise

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        return response.text

    def _parse_response(self, text: str, stage: str) -> InspectionResult:
        """Parse Gemini inspection response.

        Handles various response formats:
        - Pure JSON
        - JSON wrapped in ```json ... ``` markdown code fences
        - JSON embedded in surrounding text
        - Nested JSON objects (greedy vs non-greedy matching)
        """
        result = InspectionResult(stage=stage, raw_response=text)

        # Extract JSON — try multiple strategies
        data = None

        # Strategy 1: Direct parse (clean JSON)
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code fence (```json ... ```)
        if data is None:
            fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if fence_match:
                try:
                    data = json.loads(fence_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Find JSON object with balanced braces
        if data is None:
            # Find the outermost { ... } with simple brace counting
            start = text.find('{')
            if start >= 0:
                depth = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        # Strategy 4: Fallback — greedy regex
        if data is None:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if data is None:
            data = {}
            logger.warning(f"Gate [{stage}]: Failed to parse JSON from response: {text[:200]}")

        threshold = STAGE_THRESHOLDS.get(stage, 0.75)
        result.quality_score = min(1.0, max(0.0, float(data.get("quality_score", 0.5))))
        result.pass_check = result.quality_score >= threshold
        result.feedback = str(data.get("feedback", ""))
        result.issues = data.get("issues", [])
        if isinstance(result.issues, str):
            result.issues = [result.issues]
        result.retry_suggested = data.get("retry_suggested", not result.pass_check)
        return result

    # ── Gate 1: Person Detection ───────────────────────────────

    def inspect_person_detection(self, image: np.ndarray,
                                  boxes: list[dict]) -> InspectionResult:
        """Gate 1: Validate person detection results."""
        t0 = time.time()
        box_desc = json.dumps(boxes[:5]) if boxes else "[]"
        prompt = f"""You are a quality inspector for a virtual try-on pipeline.
Evaluate the person detection results on this image.

Detection boxes (up to 5): {box_desc}
Total detections: {len(boxes)}

Score the detection quality (0.0-1.0) considering:
- Is the primary person fully detected?
- Is the bounding box tight and accurate?
- Are there false positives?

Return JSON:
{{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}}"""

        text = self._call_gemini(prompt, [image])
        result = self._parse_response(text, "person_detection")

        # Retry once if empty response
        if not result.feedback and result.quality_score == 0.5:
            logger.warning("Gate 1 (person_detection): empty response, retrying...")
            text2 = self._call_gemini(prompt, [image])
            result2 = self._parse_response(text2, "person_detection")
            if result2.feedback:
                result = result2

        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Gate 2: Body Segmentation ──────────────────────────────

    def inspect_body_segmentation(self, image: np.ndarray,
                                   mask: np.ndarray) -> InspectionResult:
        """Gate 2: Validate body segmentation quality."""
        t0 = time.time()

        # Create visualization: overlay mask on image
        vis = image.copy()
        if mask is not None and mask.shape[:2] == image.shape[:2]:
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, color_mask, 0.3, 0)

        prompt = """You are a quality inspector for a virtual try-on pipeline.
Evaluate this body segmentation result. The green overlay shows the segmented region.

Score the segmentation quality (0.0-1.0) considering:
- Is the person's body completely segmented?
- Are edges clean and accurate?
- Is the background properly excluded?
- Are limbs and extremities included?

Return JSON:
{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}"""

        text = self._call_gemini(prompt, [vis])
        result = self._parse_response(text, "body_segmentation")

        # Retry once if empty response
        if not result.feedback and result.quality_score == 0.5:
            logger.warning("Gate 2 (body_segmentation): empty response, retrying...")
            text2 = self._call_gemini(prompt, [vis])
            result2 = self._parse_response(text2, "body_segmentation")
            if result2.feedback:
                result = result2

        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Gate 3: 3D Body Reconstruction ─────────────────────────

    def inspect_body_3d_reconstruction(self, image: np.ndarray,
                                        mesh_render: np.ndarray) -> InspectionResult:
        """Gate 3: Validate 3D body reconstruction against input."""
        t0 = time.time()

        prompt = """You are a quality inspector for a virtual try-on pipeline.
Compare the original person image (first) with the 3D body reconstruction render (second).

Score the reconstruction quality (0.0-1.0) considering:
- Does the body shape match the person?
- Are proportions (height, width, limb length) correct?
- Is the pose natural and appropriate?
- Does the mesh appear well-formed (no artifacts)?

Return JSON:
{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}"""

        text = self._call_gemini(prompt, [image, mesh_render])
        result = self._parse_response(text, "body_3d_reconstruction")
        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Gate 4: Clothing Analysis ──────────────────────────────

    def inspect_clothing_analysis(self, image: np.ndarray,
                                   analysis_dict: dict) -> InspectionResult:
        """Gate 4: Validate clothing analysis accuracy."""
        t0 = time.time()

        # Truncate analysis for prompt
        analysis_str = json.dumps(analysis_dict, indent=2)[:3000]
        prompt = f"""You are a quality inspector for a virtual try-on pipeline.
Compare this clothing image with the analysis results.

Analysis results:
{analysis_str}

Score the analysis quality (0.0-1.0) on these specific criteria:

COLOR ACCURACY (30% of score):
- Does the color name match what you see? (e.g., "pale pink beige" not "orange")
- Is the hex code (#RRGGBB) accurate for the PRIMARY garment? (NOT bottoms)
- If hex is #000000 but the garment is not black, deduct heavily

FABRIC/MATERIAL (20% of score):
- Is the fabric type correct? (satin vs cotton vs polyester etc)
- Would you describe the surface texture the same way?

GARMENT IDENTIFICATION (20% of score):
- Is the category correct? (top/bottom/dress/outerwear)
- Is the subcategory accurate? (blouse vs shirt vs t-shirt)

DESIGN DETAILS (30% of score):
- Are buttons correctly counted and positioned?
- Is the collar/neckline type accurate?
- Is the sleeve type correct?
- Are pockets, zippers, patterns correctly identified?
- Is the fit type reasonable? (slim/regular/oversized)

Return JSON:
{{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}}"""

        text = self._call_gemini(prompt, [image])
        result = self._parse_response(text, "clothing_analysis")
        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Gate 5: Virtual Try-On ─────────────────────────────────

    def inspect_virtual_tryon(self, original: np.ndarray,
                               tryon_result: np.ndarray,
                               clothing: np.ndarray | None = None,
                               physics_prompt: str | None = None) -> InspectionResult:
        """Gate 5: Validate virtual try-on result with optional P2P physics check."""
        t0 = time.time()

        images = [original, tryon_result]
        if clothing is not None:
            images.append(clothing)

        prompt = """You are a quality inspector for a virtual try-on pipeline.
The FIRST image is a reference photo of the person (for identity reference ONLY —
the pose, background, and clothing in this image are IRRELEVANT).
The SECOND image is the generated virtual try-on result.
"""
        if clothing is not None:
            prompt += "The THIRD image is the original clothing product image from the shopping mall.\n"

        prompt += """
IMPORTANT: This pipeline generates STUDIO-STYLE try-on images. Do NOT penalize for:
- Different pose from the reference photo (the try-on uses a standard pose)
- Different background (studio backgrounds are expected)
- Different lighting (studio lighting is expected)
- The person wearing different clothes than in the reference

Score the try-on quality (0.0-1.0) using this detailed rubric:

FACE IDENTITY (25% of score):
- Does the face match the reference person? (facial structure, eye shape, jawline)
- Is the skin tone consistent?
- Is the face sharp and clear? (no blur, no smudges, no gray artifacts on cheeks/jaw)
- Is the hairstyle preserved? (same length, color, parting, bangs presence/absence)

CLOTHING ACCURACY (30% of score):
- Does the clothing COLOR match the product image? (exact shade, not shifted warmer/cooler)
- Does the FABRIC TEXTURE match? (satin sheen, matte cotton, etc)
- Are DESIGN DETAILS preserved? (buttons, collar shape, cuffs, hem, seams)
- If bottoms are visible, do they match the product image? (correct color, correct style)

BODY & POSE (15% of score):
- Are body proportions natural and anatomically correct?
- Is the pose relaxed and natural? (not stiff, not T-pose, not A-pose)
- Are hands/fingers natural looking?

IMAGE QUALITY (15% of score):
- Is the resolution sufficient? (no pixelation)
- Are there any artifacts? (noise, banding, weird textures, uncanny valley)
- Is the lighting professional studio quality?

OVERALL REALISM (15% of score):
- Would a viewer believe this is a real studio photograph?
- Does the image look commercially viable for an e-commerce site?
- Is the garment-body interaction realistic? (fabric drape, wrinkles, fit)"""

        if physics_prompt:
            prompt += f"""

OPTIONAL FIT REFERENCE (P2P Engine) — for context only, do NOT heavily penalize:
The P2P engine predicted the following physical fit based on size chart math:
{physics_prompt}

NOTE: This is a mathematical prediction that may not match the actual product styling.
The PRODUCT IMAGE (Image 3 if provided) is the GROUND TRUTH for how the garment should look.
If the try-on matches the product image styling (e.g., tucked in, fitted, loose), that is CORRECT
even if the P2P prediction says otherwise. Only lightly note fit discrepancies — do NOT use them
to heavily reduce the score. The product image appearance always takes priority."""

        prompt += """

Return JSON:
{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}"""

        text = self._call_gemini(prompt, images)
        result = self._parse_response(text, "virtual_tryon")

        # Retry once if response was empty/unparseable (stochasticity fix)
        if not result.feedback and result.quality_score == 0.5:
            logger.warning("Gate 5 (virtual_tryon): empty response, retrying...")
            text2 = self._call_gemini(prompt, images)
            result2 = self._parse_response(text2, "virtual_tryon")
            if result2.feedback:  # Use retry result if it has actual content
                result = result2

        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Gate 5.5: Face Consistency ──────────────────────────────

    def inspect_face_consistency(self, face_references: list[np.ndarray],
                                  generated_angles: list[np.ndarray]) -> InspectionResult:
        """Gate 5.5: Validate face identity preservation across generated angles.

        Sends face reference images alongside generated angle images to Gemini
        and asks it to score identity preservation.

        Args:
            face_references: 1-4 face reference images (BGR)
            generated_angles: 1-4 generated try-on images at various angles (BGR)
        """
        t0 = time.time()

        images = []
        # Include up to 2 face references
        for ref in face_references[:2]:
            images.append(ref)
        # Include up to 2 generated angles
        for gen in generated_angles[:2]:
            images.append(gen)

        n_refs = min(2, len(face_references))
        n_gens = min(2, len(generated_angles))

        prompt = f"""You are a face identity expert for a virtual try-on pipeline.

The first {n_refs} image(s) are REFERENCE face photos of a specific person.
The next {n_gens} image(s) are GENERATED try-on results that should depict the SAME person.

Evaluate whether the generated images preserve the person's facial identity.

Score the face consistency (0.0-1.0) using this detailed rubric:

FACIAL STRUCTURE (35% of score):
- Eye shape, size, and spacing: do they match?
- Nose shape, bridge width, tip: are they consistent?
- Jawline and chin: same shape?
- Lip shape and fullness: preserved?
- Overall face proportions: forehead height, cheekbone width

SKIN & COMPLEXION (15% of score):
- Skin tone: same shade and undertone?
- No gray patches, smudges, or discoloration artifacts?
- Skin texture: natural and smooth?

HAIR (20% of score):
- Same hair color?
- Same length and volume?
- Same style (parting, bangs presence/absence, texture)?
- No unnatural hair additions or changes?

IMAGE QUALITY (15% of score):
- Is the face sharp and high-resolution? (no blur)
- Are there any artifacts on the face? (smudges, gray patches, ghosting)
- Are eyes clear and well-defined?

IDENTITY RECOGNITION (15% of score):
- Would a stranger confidently identify both images as the same person?
- Could this pass as the same person on an ID comparison?

Be STRICT — even small deviations in facial structure should lower the score.
A score of 0.8+ means the person is clearly recognizable as the same individual.

Return JSON:
{{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment of identity preservation>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}}"""

        text = self._call_gemini(prompt, images)
        result = self._parse_response(text, "face_consistency")
        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Gate 6: 3D Visualization ───────────────────────────────

    def inspect_3d_visualization(self, tryon_images: list[np.ndarray],
                                  glb_renders: list[np.ndarray]) -> InspectionResult:
        """Gate 6: Validate 3D model against try-on images."""
        t0 = time.time()

        # Send 2 try-on + 2 GLB renders for comparison
        images = []
        for img in tryon_images[:2]:
            images.append(img)
        for img in glb_renders[:2]:
            images.append(img)

        prompt = f"""You are a quality inspector for a virtual try-on pipeline.
Compare the 2D try-on images (first {min(2, len(tryon_images))}) with the 3D model renders (last {min(2, len(glb_renders))}).

Score the 3D visualization quality (0.0-1.0) considering:
- Does the 3D model match the clothing appearance from try-on images?
- Is the texture quality acceptable (no blurring, stretching)?
- Is the mesh geometry reasonable (no holes, artifacts)?
- Does the 3D model look like a real garment from multiple angles?

Return JSON:
{{
  "quality_score": <0.0-1.0>,
  "feedback": "<brief assessment>",
  "issues": ["<issue1>", ...],
  "retry_suggested": <true/false>
}}"""

        text = self._call_gemini(prompt, images)
        result = self._parse_response(text, "3d_visualization")
        result.elapsed_sec = time.time() - t0
        self._inspection_log.append(result)
        return result

    # ── Summary ────────────────────────────────────────────────

    def get_inspection_log(self) -> list[InspectionResult]:
        return list(self._inspection_log)

    def get_summary(self) -> dict:
        """Aggregated inspection report."""
        if not self._inspection_log:
            return {
                "total_inspections": 0,
                "overall_score": 0.0,
                "pass_rate": 0.0,
                "failed_stages": [],
            }

        total = len(self._inspection_log)
        scores = [r.quality_score for r in self._inspection_log]
        passed = sum(1 for r in self._inspection_log if r.pass_check)
        failed = [r.stage for r in self._inspection_log if not r.pass_check]

        return {
            "total_inspections": total,
            "overall_score": sum(scores) / total if total else 0.0,
            "pass_rate": passed / total if total else 0.0,
            "failed_stages": failed,
            "total_time_sec": sum(r.elapsed_sec for r in self._inspection_log),
        }

    def clear_log(self):
        self._inspection_log.clear()
