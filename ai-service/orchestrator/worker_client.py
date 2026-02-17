"""
StyleLens V6 — Worker Client (Tier 3 → Tier 4 Modal GPU)

app.run() 방식: 배포 없이 GPU만 빌려 사용.
  1. app.run() 컨텍스트 진입 → Modal에 함수 등록
  2. fn.remote() 호출 → H200 컨테이너 할당 → 함수 실행
  3. 함수 반환 → 컨테이너 즉시 해제 (GPU 과금 중지)
  4. app.run() 컨텍스트 종료

keep_warm=0 → 유휴 GPU 과금 방지.
"""

import asyncio
import logging

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
    """Modal ephemeral GPU client.

    app.run() 컨텍스트 안에서 fn.remote() 호출.
    GPU 할당 → 함수 실행 → 결과 반환 → GPU 즉시 해제.
    """

    _app_name = "stylelens-v6-worker"

    def __init__(self):
        self._modal_app = None
        self._fn_light = None
        self._fn_fashn_vton = None
        self._fn_mesh_realistic = None
        self._fn_flux_refine = None
        self._fn_face_swap = None
        self._fn_trellis = None
        self._loaded = False

    # ── Setup ─────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load Modal app and function references from worker module."""
        if self._loaded:
            return
        if not MODAL_AVAILABLE:
            raise WorkerUnavailableError("Modal not installed")
        try:
            from worker.modal_app import (
                app as modal_app,
                run_light_models,
                run_fashn_vton_batch,
                run_mesh_to_realistic,
                run_flux_refine,
                run_face_swap,
                run_trellis_3d,
            )
            self._modal_app = modal_app
            self._fn_light = run_light_models
            self._fn_fashn_vton = run_fashn_vton_batch
            self._fn_mesh_realistic = run_mesh_to_realistic
            self._fn_flux_refine = run_flux_refine
            self._fn_face_swap = run_face_swap
            self._fn_trellis = run_trellis_3d
            self._loaded = True
            logger.info("Modal app and functions loaded (including FLUX refine + face swap)")
        except Exception as e:
            raise WorkerUnavailableError(f"Failed to load worker module: {e}")

    def _call_gpu(self, fn, *args, **kwargs):
        """Execute a Modal function inside ephemeral app.run() context.

        GPU allocated → function runs → result returned → GPU freed.
        """
        self._ensure_loaded()
        try:
            with self._modal_app.run():
                result = fn.remote(*args, **kwargs)
            # Check for error response from worker
            if isinstance(result, dict) and "error" in result:
                raise WorkerUnavailableError(
                    f"Worker returned error: {result['error']}"
                )
            return result
        except WorkerUnavailableError:
            raise  # Re-raise our own errors
        except Exception as e:
            raise WorkerUnavailableError(f"GPU call failed: {e}")

    def is_configured(self) -> bool:
        """Check if Modal GPU worker is enabled in orchestrator config."""
        from orchestrator.config import MODAL_AVAILABLE as CONFIG_MODAL
        return CONFIG_MODAL and MODAL_AVAILABLE

    async def is_available(self) -> bool:
        """Check if Modal SDK is installed and authenticated."""
        if not MODAL_AVAILABLE:
            return False
        try:
            _ = modal.config._profile
            return True
        except Exception:
            return False

    async def close(self):
        """Reset loaded state."""
        self._loaded = False
        self._modal_app = None
        self._fn_light = None
        self._fn_fashn_vton = None
        self._fn_mesh_realistic = None
        self._fn_flux_refine = None
        self._fn_face_swap = None
        self._fn_trellis = None

    # ── Phase 1: Person Detection & Body Reconstruction ─────

    async def detect_yolo(self, image_b64: str) -> dict:
        """YOLO26-L — NMS-free person detection."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_light, "detect_yolo", image_b64,
        )

    async def reconstruct_3d_body(self, image_b64: str) -> dict:
        """SAM 3D Body — single image → 3D mesh."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_light, "reconstruct_3d", image_b64,
        )

    # ── Phase 2: Segmentation & Parsing ─────────────────────

    async def segment_sam3(self, image_b64: str) -> dict:
        """SAM 3 — concept-aware segmentation."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_light, "segment_sam3", image_b64,
        )

    async def parse_fashn(self, image_b64: str) -> dict:
        """FASHN Parser — 18-class body parsing."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_light, "parse_fashn", image_b64,
        )

    # ── Phase 1.5: Mesh to Realistic Virtual Model ──────────

    async def mesh_to_realistic(
        self,
        mesh_renders_b64: list[str],
        person_image_b64: str,
        angles: list[float] | None = None,
        num_steps: int = 30,
        guidance: float = 6.5,
        controlnet_conditioning_scale: float = 0.6,
        prompt_template: str = "",
        negative_prompt_override: str = "",
        body_description: str = "",
    ) -> dict:
        """Convert gray mesh renders to realistic person images via SDXL + ControlNet Depth."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_mesh_realistic,
            mesh_renders_b64,
            person_image_b64,
            "",                # face_image_b64 (unused)
            angles or [0, 45, 90, 135, 180, 225, 270, 315],
            num_steps,
            guidance,
            controlnet_conditioning_scale,
            prompt_template,
            negative_prompt_override,
            body_description,
        )

    # ── Phase 1.5B: FLUX Texture Refine (Optional) ─────────

    async def flux_refine(
        self,
        images_b64: list[str],
        prompt_template: str = "",
        angles: list[float] | None = None,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
        body_description: str = "",
    ) -> dict:
        """FLUX.2-klein-4B img2img refiner — upgrade SDXL textures to photorealistic."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_flux_refine,
            images_b64,
            prompt_template,
            "",                # negative_prompt_override
            angles,
            num_steps,
            guidance,
            seed,
            body_description,
        )

    # ── Phase 3: Virtual Try-On ────────────────────────────

    async def tryon_fashn_batch(
        self,
        persons_b64: list[str],
        clothing_b64: str,
        category: str = "tops",
        garment_photo_type: str = "flat-lay",
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        seed: int = 42,
    ) -> dict:
        """FASHN VTON v1.5 — maskless virtual try-on (Apache 2.0)."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_fashn_vton,
            persons_b64, clothing_b64,
            category, garment_photo_type,
            num_timesteps, guidance_scale, seed,
        )

    # ── Phase 4: Face Swap ───────────────────────────────────

    async def face_swap(
        self,
        images_b64: list[str],
        face_reference_b64: str,
        angles: list[float] | None = None,
        blend_radius: int = 25,
        face_scale: float = 1.0,
    ) -> dict:
        """InsightFace antelopev2 face swap — apply user face to all angle images."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_face_swap,
            images_b64,
            face_reference_b64,
            angles,
            blend_radius,
            face_scale,
        )

    # ── Phase 5: 3D Generation ─────────────────────────────

    async def generate_3d_full(
        self,
        front_image_b64: str,
        reference_images_b64: list[str] | None = None,
        seed: int = 42,
    ) -> dict:
        """TRELLIS.2 4B — image to 3D GLB (MIT license)."""
        return await asyncio.to_thread(
            self._call_gpu, self._fn_trellis,
            front_image_b64, reference_images_b64,
            seed,
        )

    # ── Health ─────────────────────────────────────────────

    async def health(self) -> dict:
        """Get worker health status."""
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
