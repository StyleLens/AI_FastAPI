"""
StyleLens V6 — Orchestrator Route: Progress (WebSocket & SSE)
WS  /ws/progress/{session_id}   — Real-time progress updates via WebSocket
GET /progress/stream/{session_id} — Server-Sent Events (SSE) progress stream
"""

import asyncio
import json
import logging
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

logger = logging.getLogger("stylelens.routes.progress")

router = APIRouter(tags=["Real-time Progress"])


@router.websocket("/ws/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time pipeline progress updates."""
    await websocket.accept()
    sm = websocket.app.state.session_manager
    queue = sm.subscribe(session_id)
    logger.info(f"WebSocket client connected for session {session_id}")

    try:
        # Send initial status
        if sm.exists(session_id):
            session = sm.get(session_id)
            await websocket.send_json({
                "type": "init",
                "sessionId": session_id,
                "status": session.status,
                "progress": session.progress,
            })

        while True:
            # Wait for event from queue
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for session {session_id}: {e}")
    finally:
        sm.unsubscribe(session_id, queue)


@router.get("/progress/stream/{session_id}")
async def sse_progress(request: Request, session_id: str):
    """Server-Sent Events (SSE) endpoint for progress streaming."""
    sm = request.app.state.session_manager
    queue = sm.subscribe(session_id)

    async def event_generator():
        try:
            # Send initial state
            if sm.exists(session_id):
                session = sm.get(session_id)
                init_data = json.dumps({
                    "type": "init",
                    "sessionId": session_id,
                    "status": session.status,
                    "progress": session.progress,
                })
                yield f"data: {init_data}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive ping
                    yield ": ping\n\n"
        finally:
            sm.unsubscribe(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
