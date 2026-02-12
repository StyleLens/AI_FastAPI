"""
Tests for orchestrator/session.py — session state management.
"""

import time
import pytest
from unittest.mock import MagicMock

from orchestrator.session import PipelineSession, SessionManager


class TestPipelineSession:
    """PipelineSession dataclass tests."""

    def test_defaults(self):
        """Session starts with all phases pending."""
        s = PipelineSession(session_id="test-123", created_at=time.time())
        assert s.body_data is None
        assert s.clothing_item is None
        assert s.fitting_result is None
        assert s.viewer3d_result is None
        assert s.status["phase1"] == "pending"
        assert s.status["phase2"] == "pending"
        assert s.progress["phase1"] == 0.0

    def test_last_accessed_auto(self):
        """last_accessed auto-set from created_at."""
        now = time.time()
        s = PipelineSession(session_id="test", created_at=now)
        assert s.last_accessed == now


class TestSessionManager:
    """SessionManager lifecycle tests."""

    def test_create_session(self):
        """Create session returns valid UUID."""
        mgr = SessionManager(max_sessions=5)
        sid = mgr.create()
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID format
        assert mgr.active_count == 1

    def test_get_session(self):
        """Retrieve created session."""
        mgr = SessionManager()
        sid = mgr.create()
        session = mgr.get(sid)
        assert session.session_id == sid
        assert session.status["phase1"] == "pending"

    def test_get_nonexistent_raises(self):
        """Getting unknown session raises KeyError."""
        mgr = SessionManager()
        with pytest.raises(KeyError):
            mgr.get("nonexistent-id")

    def test_exists(self):
        """Check session existence."""
        mgr = SessionManager()
        sid = mgr.create()
        assert mgr.exists(sid)
        assert not mgr.exists("fake-id")

    def test_update_body_data(self):
        """Store BodyData marks phase1 as done."""
        mgr = SessionManager()
        sid = mgr.create()

        mock_body = MagicMock()
        mgr.update_body_data(sid, mock_body)

        session = mgr.get(sid)
        assert session.body_data is mock_body
        assert session.status["phase1"] == "done"
        assert session.progress["phase1"] == 1.0

    def test_update_clothing(self):
        """Store ClothingItem marks phase2 as done."""
        mgr = SessionManager()
        sid = mgr.create()

        mock_item = MagicMock()
        mgr.update_clothing(sid, mock_item)

        session = mgr.get(sid)
        assert session.clothing_item is mock_item
        assert session.status["phase2"] == "done"

    def test_update_fitting(self):
        """Store FittingResult marks phase3 as done."""
        mgr = SessionManager()
        sid = mgr.create()

        mock_fitting = MagicMock()
        mgr.update_fitting(sid, mock_fitting)

        session = mgr.get(sid)
        assert session.fitting_result is mock_fitting
        assert session.status["phase3"] == "done"

    def test_update_viewer3d(self):
        """Store Viewer3DResult marks phase4 as done."""
        mgr = SessionManager()
        sid = mgr.create()

        mock_3d = MagicMock()
        mgr.update_viewer3d(sid, mock_3d)

        session = mgr.get(sid)
        assert session.viewer3d_result is mock_3d
        assert session.status["phase4"] == "done"

    def test_update_status(self):
        """Update phase status and progress."""
        mgr = SessionManager()
        sid = mgr.create()

        mgr.update_status(sid, "phase1", "running", 0.5)
        session = mgr.get(sid)
        assert session.status["phase1"] == "running"
        assert session.progress["phase1"] == 0.5

    def test_session_ttl_cleanup(self):
        """Expired sessions are cleaned up."""
        mgr = SessionManager(ttl_sec=1)
        sid = mgr.create()
        assert mgr.exists(sid)

        # Manually set last_accessed to past
        mgr._sessions[sid].last_accessed = time.time() - 2

        # Should be cleaned up
        assert not mgr.exists(sid)
        assert mgr.active_count == 0

    def test_max_sessions_eviction(self):
        """Oldest session evicted when limit reached."""
        mgr = SessionManager(max_sessions=3)
        s1 = mgr.create()
        s2 = mgr.create()
        s3 = mgr.create()
        assert mgr.active_count == 3

        # Creating 4th should evict s1 (oldest)
        s4 = mgr.create()
        assert mgr.active_count == 3
        assert not mgr.exists(s1)
        assert mgr.exists(s2)
        assert mgr.exists(s4)

    def test_session_state_progression(self):
        """Full Phase 1→2→3→4 state progression."""
        mgr = SessionManager()
        sid = mgr.create()

        # Phase 1
        mgr.update_status(sid, "phase1", "running", 0.5)
        mgr.update_body_data(sid, MagicMock())

        # Phase 2
        mgr.update_status(sid, "phase2", "running", 0.3)
        mgr.update_clothing(sid, MagicMock())

        # Phase 3
        mgr.update_status(sid, "phase3", "running", 0.8)
        mgr.update_fitting(sid, MagicMock())

        # Phase 4
        mgr.update_viewer3d(sid, MagicMock())

        session = mgr.get(sid)
        assert all(s == "done" for s in session.status.values())
        assert all(p == 1.0 for p in session.progress.values())

    def test_list_sessions(self):
        """List sessions returns correct summary."""
        mgr = SessionManager()
        s1 = mgr.create()
        s2 = mgr.create()
        mgr.update_body_data(s1, MagicMock())

        listings = mgr.list_sessions()
        assert len(listings) == 2

        s1_info = next(l for l in listings if l["session_id"] == s1)
        assert s1_info["has_body_data"] is True
        assert s1_info["has_clothing"] is False
