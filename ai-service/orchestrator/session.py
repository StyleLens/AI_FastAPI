"""
StyleLens V6 — Session State Manager
Replaces the module-level globals from main.py with proper session tracking.
Supports multiple concurrent pipeline sessions with TTL-based cleanup.
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.pipeline import BodyData
from core.wardrobe import ClothingItem
from core.fitting import FittingResult
from core.viewer3d import Viewer3DResult
from core.face_bank import FaceBank

logger = logging.getLogger("stylelens.session")


@dataclass
class PipelineSession:
    """State of a single pipeline session."""
    session_id: str
    created_at: float
    body_data: Optional[BodyData] = None
    clothing_item: Optional[ClothingItem] = None
    fitting_result: Optional[FittingResult] = None
    viewer3d_result: Optional[Viewer3DResult] = None
    face_bank: Optional[FaceBank] = None
    status: dict = field(default_factory=lambda: {
        "phase1": "pending",
        "phase2": "pending",
        "phase3": "pending",
        "phase4": "pending",
    })
    progress: dict = field(default_factory=lambda: {
        "phase1": 0.0,
        "phase2": 0.0,
        "phase3": 0.0,
        "phase4": 0.0,
    })
    last_accessed: float = 0.0

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at


class SessionManager:
    """Manages pipeline sessions with TTL and max capacity."""

    def __init__(self, max_sessions: int = 10, ttl_sec: int = 3600):
        self._sessions: dict[str, PipelineSession] = {}
        self._max_sessions = max_sessions
        self._ttl_sec = ttl_sec

    def create(self) -> str:
        """Create a new session, evicting oldest if at capacity.

        Returns:
            session_id (UUID string)
        """
        self.cleanup_expired()

        # Evict oldest if at capacity
        if len(self._sessions) >= self._max_sessions:
            oldest_id = min(
                self._sessions,
                key=lambda sid: self._sessions[sid].last_accessed,
            )
            logger.info(f"Session limit reached, evicting {oldest_id}")
            del self._sessions[oldest_id]

        session_id = str(uuid.uuid4())
        now = time.time()
        self._sessions[session_id] = PipelineSession(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
        )
        logger.info(f"Created session {session_id}")
        return session_id

    def get(self, session_id: str) -> PipelineSession:
        """Retrieve a session by ID.

        Raises:
            KeyError: If session not found or expired
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        session = self._sessions[session_id]

        # Check TTL
        if time.time() - session.last_accessed > self._ttl_sec:
            del self._sessions[session_id]
            raise KeyError(f"Session {session_id} expired")

        session.last_accessed = time.time()
        return session

    def exists(self, session_id: str) -> bool:
        """Check if session exists and is not expired."""
        try:
            self.get(session_id)
            return True
        except KeyError:
            return False

    def update_status(self, session_id: str, phase: str, status: str,
                      progress: float = 0.0):
        """Update phase status and progress."""
        session = self.get(session_id)
        session.status[phase] = status
        session.progress[phase] = progress

    def update_body_data(self, session_id: str, body_data: BodyData):
        """Store Phase 1 result."""
        session = self.get(session_id)
        session.body_data = body_data
        session.status["phase1"] = "done"
        session.progress["phase1"] = 1.0

    def update_clothing(self, session_id: str, item: ClothingItem):
        """Store Phase 2 result."""
        session = self.get(session_id)
        session.clothing_item = item
        session.status["phase2"] = "done"
        session.progress["phase2"] = 1.0

    def update_fitting(self, session_id: str, result: FittingResult):
        """Store Phase 3 result."""
        session = self.get(session_id)
        session.fitting_result = result
        session.status["phase3"] = "done"
        session.progress["phase3"] = 1.0

    def update_viewer3d(self, session_id: str, result: Viewer3DResult):
        """Store Phase 4 result."""
        session = self.get(session_id)
        session.viewer3d_result = result
        session.status["phase4"] = "done"
        session.progress["phase4"] = 1.0

    def update_face_bank(self, session_id: str, face_bank: FaceBank):
        """Store Face Bank for a session."""
        session = self.get(session_id)
        session.face_bank = face_bank
        logger.info(f"Session {session_id}: Face Bank updated "
                    f"({len(face_bank.references)} refs)")

    def cleanup_expired(self):
        """Remove all expired sessions."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_accessed > self._ttl_sec
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug(f"Cleaned up expired session {sid}")

    @property
    def active_count(self) -> int:
        """Number of active (non-expired) sessions."""
        self.cleanup_expired()
        return len(self._sessions)

    def list_sessions(self) -> list[dict]:
        """List all active sessions with status summary."""
        self.cleanup_expired()
        result = []
        for s in self._sessions.values():
            result.append({
                "session_id": s.session_id,
                "created_at": s.created_at,
                "status": s.status.copy(),
                "progress": s.progress.copy(),
                "has_body_data": s.body_data is not None,
                "has_clothing": s.clothing_item is not None,
                "has_fitting": s.fitting_result is not None,
                "has_3d": s.viewer3d_result is not None,
                "has_face_bank": s.face_bank is not None,
            })
        return result
