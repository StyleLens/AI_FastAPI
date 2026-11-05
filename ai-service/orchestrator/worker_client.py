"""
StyleLens V6 — Worker Client (Tier 3 → Tier 4 Modal GPU)

app.run() 방식: 배포 없이 GPU만 빌려 사용.
keep_warm=0 -> 유휴 GPU 과금 방지.
Worker 미설치/비활성화 시 안전한 Local fallback 및 WorkerUnavailableError.
"""

import asyncio
import logging

logger = logging.getLogger("stylelens.worker_client")

try:
    import modal  # noqa: F401
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


class WorkerUnavailableError(Exception):
    """Raised when the GPU worker is not reachable."""
    pass


class WorkerClient:
    """Modal ephemeral GPU client with graceful fallback."""

    _app_name = "stylelens-v6-worker"

    def __init__(self):
        self._modal_app = None
        self._fn_light = None
        self._fn_catvton = None
        self._fn_hunyuan = None
        self._fn_fashn_vton = None
        self._fn_mesh_realistic = None
        self._fn_flux_refine = None
        self._fn_face_swap = None
        self._fn_trellis = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load Modal app and function references from worker module."""
        if self._loaded:
            return
        import orchestrator.worker_client as _mod
        if not getattr(_mod, "MODAL_AVAILABLE", False):
            raise WorkerUnavailableError("Modal not installed")
        try:
            import sys
            modal_mod = sys.modules.get("worker.modal_app")
            if modal_mod is None:
                import worker.modal_app as modal_mod
            self._modal_app = getattr(modal_mod, "app", None)
            self._fn_light = getattr(modal_mod, "run_light_models", None)
            self._fn_catvton = getattr(modal_mod, "run_catvton_batch", None)
            self._fn_hunyuan = getattr(modal_mod, "run_hunyuan3d", None)
            self._fn_fashn_vton = getattr(modal_mod, "run_fashn_vton_batch", None)
            self._fn_mesh_realistic = getattr(modal_mod, "run_mesh_to_realistic", None)
            self._fn_flux_refine = getattr(modal_mod, "run_flux_refine", None)
            self._fn_face_swap = getattr(modal_mod, "run_face_swap", None)
            self._fn_trellis = getattr(modal_mod, "run_trellis_3d", None)
            self._loaded = True
            logger.info("Modal app and functions loaded")
        except Exception as e:
            raise WorkerUnavailableError(f"Failed to load worker module: {e}")

    def _call_gpu(self, fn, *args, **kwargs):
        """Execute a Modal function inside ephemeral app.run() context."""
        self._ensure_loaded()
        try:
            with self._modal_app.run():
                result = fn.remote(*args, **kwargs)
            if isinstance(result, dict) and "error" in result:
                raise WorkerUnavailableError(
                    f"Worker returned error: {result['error']}"
                )
            return result
        except WorkerUnavailableError:
            raise
        except Exception as e:
            raise WorkerUnavailableError(f"GPU call failed: {e}")

    def is_configured(self) -> bool:
        """Check if Modal GPU worker is enabled in orchestrator config."""
        import orchestrator.worker_client as _mod
        return getattr(_mod, "MODAL_AVAILABLE", False)

    async def is_available(self) -> bool:
        """Check if Modal SDK is installed and authenticated."""
        import orchestrator.worker_client as _mod
        if not getattr(_mod, "MODAL_AVAILABLE", False):
            return False
        try:
            import modal
            _ = modal.config._profile
            return True
        except Exception:
            return False

    async def close(self):
        """Reset loaded state."""
        self._loaded = False
        self._modal_app = None
        self._fn_light = None
        self._fn_catvton = None
        self._fn_hunyuan = None
        self._fn_fashn_vton = None
        self._fn_mesh_realistic = None
        self._fn_flux_refine = None
        self._fn_face_swap = None
        self._fn_trellis = None

    async def detect_yolo(self, image_b64: str) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_light, "detect_yolo", image_b64)

    async def reconstruct_3d_body(self, image_b64: str) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_light, "reconstruct_3d", image_b64)

    async def segment_sam3(self, image_b64: str) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_light, "segment_sam3", image_b64)

    async def parse_fashn(self, image_b64: str) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_light, "parse_fashn", image_b64)

    async def mesh_to_realistic(self, *args, **kwargs) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_mesh_realistic, *args, **kwargs)

    async def flux_refine(self, *args, **kwargs) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_flux_refine, *args, **kwargs)

    async def tryon_fashn_batch(self, *args, **kwargs) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_fashn_vton, *args, **kwargs)

    async def face_swap(self, *args, **kwargs) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_face_swap, *args, **kwargs)

    async def generate_3d_full(self, *args, **kwargs) -> dict:
        return await asyncio.to_thread(self._call_gpu, self._fn_trellis, *args, **kwargs)

    async def health(self) -> dict:
        if not self.is_configured():
            return {"status": "not_configured"}
        try:
            available = await self.is_available()
            return {
                "status": "ready" if available else "unavailable",
                "mode": "modal_ephemeral",
                "gpu": "H200",
            }
        except Exception:
            return {"status": "unavailable"}
