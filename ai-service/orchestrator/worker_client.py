"""
StyleLens V6 — Worker Client (Tier 3 → Tier 4 Modal remote call)
modal.Function.lookup() → .remote() 방식으로 GPU 함수 직접 호출.
배포 없이 modal run 방식으로 GPU만 빌려 사용.
"""

import logging
from typing import Any

logger = logging.getLogger("stylelens.worker_client")

# ── Modal import ──
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


class WorkerUnavailableError(Exception):
    """Raised when the GPU worker is not reachable."""
    pass


class WorkerClient:
    """Modal remote call client for GPU tasks.

    HTTP 호출 대신 modal.Function.lookup().remote() 로 직접 호출.
    GPU 할당 → 함수 실행 → 결과 반환 → GPU 즉시 해제.
    """

    def __init__(self):
        self._app_name = "stylelens-v6-worker"

    def is_configured(self) -> bool:
        """Check if Modal is available."""
        return MODAL_AVAILABLE

    async def is_available(self) -> bool:
        """Check if Modal is available and authenticated."""
        if not MODAL_AVAILABLE:
            return False
        try:
            # Simple check — Modal SDK installed and token valid
            modal.config._profile()
            return True
        except Exception:
            return False

    async def close(self):
        """No-op (no persistent session to close)."""
        pass

    def _lookup(self, func_name: str):
        """Lookup a remote Modal function."""
        if not MODAL_AVAILABLE:
            raise WorkerUnavailableError("Modal not installed")
        try:
            return modal.Function.from_name(self._app_name, func_name)
        except Exception as e:
            raise WorkerUnavailableError(f"Cannot lookup {func_name}: {e}")

    # ── Phase 1: Body Reconstruction ────────────────────────

    async def reconstruct_3d_body(self, image_b64: str) -> dict:
        """SAM 3D Body — single image → 3D mesh."""
        try:
            fn = self._lookup("run_light_models")
            return fn.remote("reconstruct_3d", image_b64)
        except Exception as e:
            raise WorkerUnavailableError(f"reconstruct_3d failed: {e}")

    # ── Phase 2: Segmentation & Parsing ─────────────────────

    async def segment_sam3(self, image_b64: str) -> dict:
        """SAM 3 — concept-aware segmentation."""
        try:
            fn = self._lookup("run_light_models")
            return fn.remote("segment_sam3", image_b64)
        except Exception as e:
            raise WorkerUnavailableError(f"segment_sam3 failed: {e}")

    async def parse_fashn(self, image_b64: str) -> dict:
        """FASHN Parser — 18-class body parsing."""
        try:
            fn = self._lookup("run_light_models")
            return fn.remote("parse_fashn", image_b64)
        except Exception as e:
            raise WorkerUnavailableError(f"parse_fashn failed: {e}")

    # ── Phase 3: Virtual Try-On ────────────────────────────

    async def tryon_catvton_batch(
        self,
        persons_b64: list[str],
        clothing_b64: str,
        masks_b64: list[str],
        num_steps: int = 30,
        guidance: float = 3.5,
        strength: float = 0.85,
    ) -> dict:
        """CatVTON-FLUX — batch virtual try-on for all angles."""
        try:
            fn = self._lookup("run_catvton_batch")
            return fn.remote(
                persons_b64, clothing_b64, masks_b64,
                num_steps, guidance, strength,
            )
        except Exception as e:
            raise WorkerUnavailableError(f"tryon_catvton_batch failed: {e}")

    # ── Phase 4: 3D Generation ─────────────────────────────

    async def generate_3d_full(
        self,
        front_image_b64: str,
        reference_images_b64: list[str] | None = None,
        shape_steps: int = 50,
        paint_steps: int = 20,
        texture_res: int = 4096,
    ) -> dict:
        """Hunyuan3D — shape + texture in one call."""
        try:
            fn = self._lookup("run_hunyuan3d")
            return fn.remote(
                front_image_b64, reference_images_b64,
                shape_steps, paint_steps, texture_res,
            )
        except Exception as e:
            raise WorkerUnavailableError(f"generate_3d_full failed: {e}")

    # ── Health ─────────────────────────────────────────────

    async def health(self) -> dict:
        """Get worker health status."""
        if not self.is_configured():
            return {"status": "not_configured"}
        try:
            available = await self.is_available()
            return {
                "status": "ready" if available else "unavailable",
                "mode": "modal_remote",
                "gpu": "H100",
            }
        except Exception:
            return {"status": "unavailable"}
