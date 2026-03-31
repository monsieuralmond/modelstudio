from __future__ import annotations

from fastapi import WebSocket

from .models import PipelineStatus


class StatusBroadcaster:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast_status(self, status: PipelineStatus) -> None:
        stale: list[WebSocket] = []
        payload = {"type": "status", "status": status.model_dump(mode="json")}
        for websocket in self._connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)
