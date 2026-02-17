"""
StyleLens V6 SOTA Pipeline — Configuration
All model paths, device config, pipeline constants.
"""

import importlib
import os
import torch
from pathlib import Path
from dotenv import load_dotenv


def _has_package(name: str) -> bool:
    """Check if a Python package is importable (without actually importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False

# ── MODE SWITCH ────────────────────────────────────────────────
# 로컬 테스트 시 True, Modal 배포 준비 시 False로 주석 전환
IS_LOCAL_DEV = True
# IS_LOCAL_DEV = False

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

if IS_LOCAL_DEV:
    # 로컬 경로 (M4 Mac에 다운로드된 모델)
    MODEL_DIR = BASE_DIR / "model"
else:
    # Modal 클라우드 경로 (Modal Volume 마운트 위치)
    MODEL_DIR = Path("/models")

# ── Device ─────────────────────────────────────────────────────
if IS_LOCAL_DEV:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # fp16 unstable on MPS
HAS_CUDA = torch.cuda.is_available()

# ── MPS Fallback Strategy ─────────────────────────────────
# CUDA가 없는 환경(M4 Mac 등)에서도 모든 파이프라인을 테스트 가능하도록
# CPU fallback을 사용. 시간은 오래 걸리지만 동작은 보장됨.
# - SAM 3D Body: CPU로 실행 (MPS 미지원, detectron2 불필요)
# - Hunyuan3D Shape: CPU로 실행 (MPS fp16 불안정)
# - Hunyuan3D Paint: CUDA 전용 (로컬에서는 shape-only 모드)
# - CatVTON-FLUX: MPS에서 실행 가능 (메모리 ~32GB 필요)
MPS_FALLBACK_DEVICE = "cpu"  # CUDA 전용 모델의 MPS 대체 디바이스

def get_device_for_model(model_name: str) -> str:
    """Get the appropriate device for a model, with MPS→CPU fallback."""
    # Models that need CPU fallback on MPS
    _CPU_FALLBACK_MODELS = {"sam3d_body", "hunyuan3d_shape", "hunyuan3d_paint"}
    if model_name in _CPU_FALLBACK_MODELS and DEVICE == "mps":
        return MPS_FALLBACK_DEVICE
    return DEVICE

# ── Gemini ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-3-pro-preview"           # text / analysis
GEMINI_PRO_MODEL_NAME = "gemini-3-pro-preview"       # alias
V5_GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"  # image generation
GEMINI_FLASH_IMAGE_MODEL = "gemini-2.5-flash-image"   # image fallback
GEMINI_FLASH_TEXT_MODEL = "gemini-3-flash-preview"     # text-only (NO images)

# ── P2P (Physics-to-Prompt) Engine ───────────────────────────
P2P_ENABLED = True
P2P_ENSEMBLE_ENABLED = True
P2P_ENSEMBLE_TIMEOUT_SEC = 30.0
P2P_TIGHTNESS_THRESHOLDS = {
    "critical_tight": (-999.0, -5.0),
    "tight":          (-5.0,  -2.0),
    "optimal":        (-2.0,   5.0),
    "loose":          ( 5.0,  10.0),
    "very_loose":     (10.0, 999.0),
}
P2P_BODY_PARTS = ["shoulder", "chest", "waist", "hip", "sleeve"]

# ── Model 1: YOLO26-L (NMS-free person detection) ─────────────
YOLO26_MODEL_DIR = MODEL_DIR / "yolo26"
YOLO26_MODEL_PATH = YOLO26_MODEL_DIR / "yolo26l.pt"
YOLO26_ENABLED = YOLO26_MODEL_PATH.exists()
YOLO26_CONF_THRESHOLD = 0.5

# ── Model 2: SAM 3 (concept-aware segmentation) ───────────────
SAM3_MODEL_DIR = MODEL_DIR / "sam3"
SAM3_ENABLED = (
    SAM3_MODEL_DIR.exists()
    and any(SAM3_MODEL_DIR.glob("*.safetensors"))
    and _has_package("segment_anything_3")
)
SAM3_POINTS_PER_SIDE = 32
SAM3_PRED_IOU_THRESH = 0.88

# ── Model 3: SAM 3D Body DINOv3 (single-image → 3D mesh) ─────
SAM3D_BODY_MODEL_DIR = MODEL_DIR / "sam3d_body"
SAM3D_BODY_CKPT_PATH = SAM3D_BODY_MODEL_DIR / "model.ckpt"
SAM3D_BODY_MHR_PATH = SAM3D_BODY_MODEL_DIR / "assets" / "mhr_model.pt"
SAM3D_BODY_ENABLED = (
    SAM3D_BODY_MODEL_DIR.exists()
    and (
        any(SAM3D_BODY_MODEL_DIR.glob("*.safetensors"))
        or SAM3D_BODY_CKPT_PATH.exists()
    )
)

# ── Model 4: FASHN Parser (18-class fashion body parsing) ─────
FASHN_PARSER_DIR = MODEL_DIR / "fashn_parser"
FASHN_PARSER_ENABLED = FASHN_PARSER_DIR.exists() and any(
    FASHN_PARSER_DIR.glob("*.safetensors")
)
FASHN_CLASSES = [
    "background", "hat", "hair", "sunglasses", "upper_clothes",
    "skirt", "pants", "dress", "belt", "left_shoe",
    "right_shoe", "head", "left_leg", "right_leg", "left_arm",
    "right_arm", "bag", "scarf",
]

# ── Model 5: CatVTON-FLUX (FLUX-based virtual try-on) ─────────
CATVTON_FLUX_DIR = MODEL_DIR / "catvton_flux"
CATVTON_FLUX_ENABLED = CATVTON_FLUX_DIR.exists()
CATVTON_FLUX_STEPS = 30
CATVTON_FLUX_GUIDANCE = 3.5
CATVTON_FLUX_RESOLUTION = 1024
CATVTON_FLUX_STRENGTH = 0.85

# CatVTON sub-paths (DensePose, SCHP, LoRA, attention)
CATVTON_DIR = MODEL_DIR / "catvton"
CATVTON_LORA_DIR = CATVTON_DIR / "flux-lora"
CATVTON_ATTN_DIR = CATVTON_DIR / "mix-48k-1024" / "attention"
CATVTON_DENSEPOSE_DIR = CATVTON_DIR / "DensePose"
CATVTON_SCHP_DIR = CATVTON_DIR / "SCHP"

# ── Model 6: FLUX.1-dev GGUF Q8 (base diffusion model) ────────
FLUX_GGUF_DIR = MODEL_DIR / "flux_gguf"
FLUX_GGUF_PATH = FLUX_GGUF_DIR / "flux1-dev-Q8_0.gguf"
FLUX_GGUF_ENABLED = FLUX_GGUF_PATH.exists()

# ── Model 7: Hunyuan3D 2.0 (multi-view → textured 3D GLB) ────
# Uses hy3dgen package (GitHub: Tencent-Hunyuan/Hunyuan3D-2)
# Shape: MPS + CUDA. Paint: CUDA only (custom rasterizer).
HUNYUAN3D_DIR = MODEL_DIR / "hunyuan3d"
HUNYUAN3D_SHAPE_DIR = HUNYUAN3D_DIR / "hunyuan3d-dit-v2-0-turbo"
HUNYUAN3D_VAE_DIR = HUNYUAN3D_DIR / "hunyuan3d-vae-v2-0"
HUNYUAN3D_ENABLED = (
    HUNYUAN3D_DIR.exists()
    and HUNYUAN3D_SHAPE_DIR.exists()
    and _has_package("hy3dgen")
)
HUNYUAN3D_PAINT_ENABLED = HUNYUAN3D_ENABLED and HAS_CUDA
HUNYUAN3D_SHAPE_ONLY = HUNYUAN3D_ENABLED and not HAS_CUDA  # MPS/CPU: shape만 가능
HUNYUAN3D_SHAPE_STEPS = 5   # turbo variant: 4-8 steps
HUNYUAN3D_PAINT_STEPS = 20
HUNYUAN3D_TEXTURE_RES = 4096  # 4K texture maps

# ── Optional: InsightFace (face identity preservation) ─────────
INSIGHTFACE_MODEL_DIR = MODEL_DIR / "insightface"
INSIGHTFACE_ENABLED = INSIGHTFACE_MODEL_DIR.exists()

# ── Pipeline Constants ─────────────────────────────────────────
FITTING_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Phase 1.5: Realistic Virtual Model ─────────────────────────
REALISTIC_MODEL_ENABLED = True   # Convert mesh renders to realistic person images
REALISTIC_MODEL_STEPS = 28       # FLUX inference steps for mesh-to-realistic
REALISTIC_MODEL_GUIDANCE = 30.0  # FLUX guidance scale

# ── V35+ Pipeline Flags ────────────────────────────────────────
# SDXL + ControlNet Depth (Phase 1.5A)
SDXL_NUM_STEPS = 30
SDXL_GUIDANCE = 6.5
SDXL_DEFAULT_CN_SCALE = 0.60     # Fallback if no bust-specific cn_scale

# FLUX.2-klein img2img refiner (Phase 1.5B) — OPTIONAL
FLUX_REFINE_ENABLED = False      # Skip FLUX to save ~40s GPU / ~140원 per test
FLUX_REFINE_STEPS = 4
FLUX_REFINE_GUIDANCE = 1.0

# FASHN VTON v1.5 (Phase 3)
FASHN_VTON_TIMESTEPS = 30
FASHN_VTON_GUIDANCE = 1.5
FASHN_VTON_CATEGORY = "tops"
FASHN_VTON_GARMENT_TYPE = "model"

# InsightFace Face Swap (Phase 4) — post-processing on VTON results
FACE_SWAP_ENABLED = True
FACE_SWAP_BLEND_RADIUS = 25
FACE_SWAP_SCALE = 1.0

# ── Bust Cup → Dynamic ControlNet Scale Mapping ───────────────
# Larger bust → stronger ControlNet adherence to preserve bust silhouette
# through SDXL→VTON pipeline. KEY FINDING from v35 testing.
BUST_CUP_CN_SCALE = {
    "AA": 0.50,
    "A":  0.55,
    "B":  0.60,   # baseline
    "C":  0.65,
    "D":  0.70,
    "DD": 0.72,
    "E":  0.75,
    "F":  0.78,
    "G":  0.80,
    "H":  0.82,
}

# Bust cup → mesh cup_scale multiplier (B-cup = 1.0 baseline)
BUST_CUP_SCALE = {
    "AA": 0.3,
    "A":  0.6,
    "B":  1.0,
    "C":  1.4,
    "D":  1.8,
    "DD": 2.2,
    "E":  2.2,
    "F":  3.0,
    "G":  3.4,
    "H":  3.8,
}


def get_bust_cn_scale(cup: str) -> float:
    """Get dynamic ControlNet conditioning scale for a bust cup size."""
    return BUST_CUP_CN_SCALE.get(cup.upper().strip(), SDXL_DEFAULT_CN_SCALE)


def get_bust_cup_scale(cup: str) -> float:
    """Get mesh vertex bust scaling multiplier for a cup size."""
    return BUST_CUP_SCALE.get(cup.upper().strip(), 1.0)

# ── Gemini Feedback Quality Gates ──────────────────────────────
STAGE_THRESHOLDS = {
    "person_detection": 0.70,
    "body_segmentation": 0.75,
    "body_3d_reconstruction": 0.70,
    "clothing_analysis": 0.75,
    "virtual_tryon": 0.80,
    "face_consistency": 0.75,
    "3d_visualization": 0.80,
}

# ── Face Identity ──────────────────────────────────────────────
V5_FACE_SWAP_BLEND_RADIUS = 15
V5_FACE_SWAP_SCALE = 1.0
V5_FACE_CROP_SIZE = 512

# ── Face Bank (Multi-Reference Identity) ─────────────────────
FACE_BANK_MAX_REFERENCES = 11        # 10 past + 1 current
FACE_BANK_MAX_REFS_PER_ANGLE = 4     # max face refs sent to Gemini per angle
FACE_BANK_MIN_DET_SCORE = 0.5        # minimum InsightFace detection confidence
FACE_BANK_SIMILARITY_THRESHOLD = 0.3 # cosine similarity for same-person check
FACE_BANK_ENABLED = INSIGHTFACE_ENABLED

# ── Reference Body Models ──────────────────────────────────────
REFERENCE_MODELS = {
    "female": {
        "slim":     {"height_cm": 170, "weight_kg": 52, "bmi": 18.0, "bust_cup": "A"},
        "standard": {"height_cm": 165, "weight_kg": 55, "bmi": 20.2, "bust_cup": "B"},
        "average":  {"height_cm": 163, "weight_kg": 62, "bmi": 23.3, "bust_cup": "C"},
    },
    "male": {
        "slim":     {"height_cm": 178, "weight_kg": 65, "bmi": 20.5},
        "standard": {"height_cm": 175, "weight_kg": 72, "bmi": 23.5},
        "bulky":    {"height_cm": 180, "weight_kg": 90, "bmi": 27.8},
    },
}

# ── Shoulder / Pose ────────────────────────────────────────────
A_POSE_SHOULDER_ANGLE = 65  # degrees, relaxed natural A-pose


def get_model_status() -> dict:
    """Return availability status for all models."""
    return {
        "yolo26": YOLO26_ENABLED,
        "sam3": SAM3_ENABLED,
        "sam3d_body": SAM3D_BODY_ENABLED,
        "fashn_parser": FASHN_PARSER_ENABLED,
        "catvton_flux": CATVTON_FLUX_ENABLED,
        "flux_gguf": FLUX_GGUF_ENABLED,
        "hunyuan3d": HUNYUAN3D_ENABLED,
        "hunyuan3d_paint": HUNYUAN3D_PAINT_ENABLED,
        "hunyuan3d_shape_only": HUNYUAN3D_SHAPE_ONLY,
        "insightface": INSIGHTFACE_ENABLED,
        "face_bank": FACE_BANK_ENABLED,
        "gemini": GEMINI_ENABLED,
    }
