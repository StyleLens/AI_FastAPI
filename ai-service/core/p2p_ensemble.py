"""
StyleLens V6 — P2P Multi-Agent Gemini Ensemble
3-agent workflow: Analyst (Flash) → Critic (Flash) → Final Corrector (Pro)

Sequential A→B→C flow with timeout fallback to deterministic P2P engine.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from core.config import (
    GEMINI_FLASH_TEXT_MODEL,
    GEMINI_MODEL_NAME,
    P2P_ENSEMBLE_TIMEOUT_SEC,
)
from core.gemini_client import GeminiClient
from core.p2p_engine import (
    BodyMeasurements,
    GarmentMeasurements,
    P2PResult,
    TightnessLevel,
    calculate_deltas,
    calculate_mask_expansion,
    generate_physics_prompt,
    _determine_overall_tightness,
    _classify_tightness,
    _get_visual_keywords,
    BodyPartDelta,
)

logger = logging.getLogger("stylelens.p2p_ensemble")


# ── Dataclasses ───────────────────────────────────────────────

@dataclass
class P2PEnsembleResult:
    """Result of multi-agent P2P ensemble."""
    p2p_result: P2PResult = field(default_factory=P2PResult)
    agent_a_output: dict = field(default_factory=dict)
    agent_b_output: dict = field(default_factory=dict)
    agent_c_output: dict = field(default_factory=dict)
    ensemble_confidence: float = 0.0
    elapsed_sec: float = 0.0
    method: str = "ensemble"  # "ensemble" or "fallback"


# ── Agent Prompts ─────────────────────────────────────────────

_AGENT_A_PROMPT = """You are a garment fit analyst (Agent A).
Given body measurements and garment measurements, analyze how the garment will fit.

Body measurements (cm):
{body_json}

Garment measurements (cm):
{garment_json}

Clothing description: {clothing_desc}

For each body part where both measurements are available, calculate:
  delta = garment_measurement - body_measurement
  (positive delta = garment is larger = loose, negative = garment is smaller = tight)

Classify each delta:
  Δ < -5cm: critical_tight (buttons strained, tension rays, fabric pulling)
  -5 ≤ Δ < -2cm: tight (slightly tight, minor pulling)
  -2 ≤ Δ < +5cm: optimal (proper fit, natural drape)
  +5 ≤ Δ < +10cm: loose (relaxed fit, slight bunching)
  Δ ≥ +10cm: very_loose (excessive fabric, hanging)

Return JSON:
{{
  "deltas": [
    {{"body_part": "chest", "body_cm": 90, "garment_cm": 88, "delta_cm": -2.0, "tightness": "optimal", "visual_keywords": ["proper chest fit"]}},
    ...
  ],
  "overall_assessment": "<brief overall fit description>",
  "confidence": <0.0-1.0>
}}"""

_AGENT_B_PROMPT = """You are a physics accuracy critic (Agent B).
Review the fit analysis from Agent A and verify its accuracy.

Agent A's analysis:
{agent_a_json}

Original body measurements (cm):
{body_json}

Original garment measurements (cm):
{garment_json}

Verify:
1. Are the delta calculations correct (garment - body)?
2. Are the tightness classifications correct for each delta?
3. Do the visual keywords match the tightness level?
4. Are there any physically impossible descriptions?
5. Are there contradictions (e.g., "loose" with negative delta)?

Return JSON:
{{
  "corrections": [
    {{"body_part": "chest", "issue": "<what's wrong>", "corrected_tightness": "tight", "corrected_keywords": ["slight tension across chest"]}},
    ...
  ],
  "validation_passed": <true/false>,
  "overall_feedback": "<brief critique>",
  "confidence": <0.0-1.0>
}}"""

_AGENT_C_PROMPT = """You are the final fit corrector (Agent C).
Synthesize the analyst and critic outputs into a final physics-accurate description.

Agent A (Analyst) output:
{agent_a_json}

Agent B (Critic) output:
{agent_b_json}

Clothing description: {clothing_desc}

Your task:
1. Apply any corrections from Agent B to Agent A's analysis
2. Generate a final, coherent physics description suitable for image generation prompts
3. The description should be specific, visual, and physically accurate
4. Do NOT beautify or idealize the fit — describe it exactly as physics dictates

Return JSON:
{{
  "final_physics_prompt": "<a multi-line description of how this garment physically fits, suitable for injection into an AI image generation prompt>",
  "validated_deltas": [
    {{"body_part": "chest", "delta_cm": -2.0, "tightness": "optimal", "description": "proper chest fit with natural fabric drape"}},
    ...
  ],
  "overall_tightness": "<critical_tight|tight|optimal|loose|very_loose>",
  "confidence": <0.0-1.0>
}}"""


# ── Agent Functions ───────────────────────────────────────────

def _measurements_to_dict(meas: BodyMeasurements | GarmentMeasurements) -> dict:
    """Convert measurements dataclass to dict, filtering zero values."""
    from dataclasses import asdict
    return {k: v for k, v in asdict(meas).items() if v > 0}


async def _run_agent_a(
    gemini: GeminiClient,
    body_meas: BodyMeasurements,
    garment_meas: GarmentMeasurements,
    clothing_desc: str,
) -> dict:
    """Agent A (Flash - Analyst): Calculate deltas and generate physics keywords."""
    body_json = json.dumps(_measurements_to_dict(body_meas), indent=2)
    garment_json = json.dumps(_measurements_to_dict(garment_meas), indent=2)

    prompt = _AGENT_A_PROMPT.format(
        body_json=body_json,
        garment_json=garment_json,
        clothing_desc=clothing_desc,
    )

    # Run synchronous Gemini call in thread pool
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        lambda: gemini._call_text(prompt, model=GEMINI_FLASH_TEXT_MODEL),
    )

    try:
        return gemini._parse_json(text)
    except Exception:
        logger.warning("Agent A: Failed to parse response as JSON")
        return {"raw_text": text, "error": "parse_failed"}


async def _run_agent_b(
    gemini: GeminiClient,
    agent_a_output: dict,
    body_meas: BodyMeasurements,
    garment_meas: GarmentMeasurements,
) -> dict:
    """Agent B (Flash - Critic): Verify physics accuracy of Agent A's output."""
    body_json = json.dumps(_measurements_to_dict(body_meas), indent=2)
    garment_json = json.dumps(_measurements_to_dict(garment_meas), indent=2)
    agent_a_json = json.dumps(agent_a_output, indent=2, default=str)[:3000]

    prompt = _AGENT_B_PROMPT.format(
        agent_a_json=agent_a_json,
        body_json=body_json,
        garment_json=garment_json,
    )

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        lambda: gemini._call_text(prompt, model=GEMINI_FLASH_TEXT_MODEL),
    )

    try:
        return gemini._parse_json(text)
    except Exception:
        logger.warning("Agent B: Failed to parse response as JSON")
        return {"raw_text": text, "error": "parse_failed"}


async def _run_agent_c(
    gemini: GeminiClient,
    agent_a_output: dict,
    agent_b_output: dict,
    clothing_desc: str,
) -> dict:
    """Agent C (Pro - Final Corrector): Synthesize A+B into final prompt."""
    agent_a_json = json.dumps(agent_a_output, indent=2, default=str)[:3000]
    agent_b_json = json.dumps(agent_b_output, indent=2, default=str)[:3000]

    prompt = _AGENT_C_PROMPT.format(
        agent_a_json=agent_a_json,
        agent_b_json=agent_b_json,
        clothing_desc=clothing_desc,
    )

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        lambda: gemini._call_text(prompt, model=GEMINI_MODEL_NAME),
    )

    try:
        return gemini._parse_json(text)
    except Exception:
        logger.warning("Agent C: Failed to parse response as JSON")
        return {"raw_text": text, "error": "parse_failed"}


def _build_fallback_result(
    body_meas: BodyMeasurements,
    garment_meas: GarmentMeasurements,
    clothing_desc: str = "",
) -> P2PEnsembleResult:
    """Build result from deterministic P2P engine (fallback)."""
    deltas = calculate_deltas(body_meas, garment_meas)
    physics_prompt = generate_physics_prompt(deltas, clothing_desc)
    overall = _determine_overall_tightness(deltas)
    mask_factor = calculate_mask_expansion(deltas, "none")

    p2p = P2PResult(
        deltas=deltas,
        overall_tightness=overall,
        physics_prompt=physics_prompt,
        mask_expansion_factor=mask_factor,
        confidence=0.4,
        method="fallback",
    )
    return P2PEnsembleResult(
        p2p_result=p2p,
        method="fallback",
    )


def _build_ensemble_result(
    agent_a: dict,
    agent_b: dict,
    agent_c: dict,
    body_meas: BodyMeasurements,
    garment_meas: GarmentMeasurements,
) -> P2PEnsembleResult:
    """Build P2PResult from Agent C's synthesized output."""
    # Extract final physics prompt from Agent C
    physics_prompt = agent_c.get("final_physics_prompt", "")
    validated_deltas = agent_c.get("validated_deltas", [])
    overall_str = agent_c.get("overall_tightness", "optimal")
    confidence = float(agent_c.get("confidence", 0.7))

    # Convert validated deltas to BodyPartDelta objects
    deltas = []
    for vd in validated_deltas:
        part = vd.get("body_part", "")
        delta_cm = float(vd.get("delta_cm", 0))
        tightness_str = vd.get("tightness", "optimal")
        desc = vd.get("description", "")

        try:
            tightness = TightnessLevel(tightness_str)
        except ValueError:
            tightness = _classify_tightness(delta_cm)

        keywords = _get_visual_keywords(part, tightness)

        deltas.append(BodyPartDelta(
            body_part=part,
            body_cm=0,  # Not tracked in ensemble output
            garment_cm=0,
            delta_cm=delta_cm,
            tightness=tightness,
            visual_keywords=keywords,
            prompt_fragment=desc or f"{part}: {', '.join(keywords)}",
        ))

    # Determine overall tightness
    try:
        overall = TightnessLevel(overall_str)
    except ValueError:
        overall = _determine_overall_tightness(deltas) if deltas else TightnessLevel.OPTIMAL

    # Use deterministic deltas for mask expansion (more reliable)
    det_deltas = calculate_deltas(body_meas, garment_meas)
    mask_factor = calculate_mask_expansion(det_deltas, "none")

    # If Agent C didn't produce a good physics prompt, fall back to deterministic
    if not physics_prompt or len(physics_prompt) < 10:
        physics_prompt = generate_physics_prompt(det_deltas or deltas)

    p2p = P2PResult(
        deltas=deltas or det_deltas,
        overall_tightness=overall,
        physics_prompt=physics_prompt,
        mask_expansion_factor=mask_factor,
        confidence=confidence,
        method="ensemble",
    )

    return P2PEnsembleResult(
        p2p_result=p2p,
        agent_a_output=agent_a,
        agent_b_output=agent_b,
        agent_c_output=agent_c,
        ensemble_confidence=confidence,
        method="ensemble",
    )


# ── Main Orchestrator ─────────────────────────────────────────

async def run_p2p_ensemble(
    gemini: GeminiClient,
    body_meas: BodyMeasurements,
    garment_meas: GarmentMeasurements,
    clothing_desc: str,
    timeout_sec: float = P2P_ENSEMBLE_TIMEOUT_SEC,
) -> P2PEnsembleResult:
    """
    Orchestrate 3-agent P2P ensemble.

    Flow:
        1. Agent A (Flash) — analyst: calculates deltas + generates keywords
        2. Agent B (Flash) — critic: verifies A's physics accuracy
        3. Agent C (Pro)   — corrector: synthesizes A+B into final prompt

    On timeout or error, falls back to deterministic P2P engine.
    """
    t0 = time.time()
    per_agent_timeout = timeout_sec / 3

    try:
        # Step 1: Agent A (Analyst)
        logger.info("P2P Ensemble: Running Agent A (Analyst - Flash)")
        agent_a = await asyncio.wait_for(
            _run_agent_a(gemini, body_meas, garment_meas, clothing_desc),
            timeout=per_agent_timeout,
        )

        if agent_a.get("error"):
            logger.warning(f"Agent A failed: {agent_a.get('error')}")
            result = _build_fallback_result(body_meas, garment_meas, clothing_desc)
            result.elapsed_sec = time.time() - t0
            return result

        # Step 2: Agent B (Critic)
        logger.info("P2P Ensemble: Running Agent B (Critic - Flash)")
        agent_b = await asyncio.wait_for(
            _run_agent_b(gemini, agent_a, body_meas, garment_meas),
            timeout=per_agent_timeout,
        )

        if agent_b.get("error"):
            logger.warning(f"Agent B failed: {agent_b.get('error')}, continuing with A only")
            agent_b = {"validation_passed": True, "corrections": []}

        # Step 3: Agent C (Final Corrector)
        logger.info("P2P Ensemble: Running Agent C (Corrector - Pro)")
        agent_c = await asyncio.wait_for(
            _run_agent_c(gemini, agent_a, agent_b, clothing_desc),
            timeout=per_agent_timeout,
        )

        if agent_c.get("error"):
            logger.warning(f"Agent C failed: {agent_c.get('error')}, using fallback")
            result = _build_fallback_result(body_meas, garment_meas, clothing_desc)
            result.agent_a_output = agent_a
            result.agent_b_output = agent_b
            result.elapsed_sec = time.time() - t0
            return result

        # Build ensemble result
        result = _build_ensemble_result(agent_a, agent_b, agent_c, body_meas, garment_meas)
        result.elapsed_sec = time.time() - t0

        logger.info(f"P2P Ensemble: Complete in {result.elapsed_sec:.1f}s, "
                    f"confidence={result.ensemble_confidence:.2f}")
        return result

    except asyncio.TimeoutError:
        logger.warning(f"P2P Ensemble: Timed out after {timeout_sec}s, using fallback")
        result = _build_fallback_result(body_meas, garment_meas, clothing_desc)
        result.elapsed_sec = time.time() - t0
        return result

    except Exception as e:
        logger.error(f"P2P Ensemble error: {e}")
        result = _build_fallback_result(body_meas, garment_meas, clothing_desc)
        result.elapsed_sec = time.time() - t0
        return result
