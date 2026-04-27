import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["bbox"])


class BboxManager:
    def __init__(self):
        # camera_id → 연결된 프론트 WebSocket 목록
        self._clients: dict[str, list[WebSocket]] = {}

    async def connect(self, camera_id: str, ws: WebSocket):
        await ws.accept()
        self._clients.setdefault(camera_id, []).append(ws)

    def disconnect(self, camera_id: str, ws: WebSocket):
        clients = self._clients.get(camera_id, [])
        if ws in clients:
            clients.remove(ws)

    async def broadcast(self, camera_id: str, data: Any):
        dead = []
        for ws in self._clients.get(camera_id, []):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(camera_id, ws)


bbox_manager = BboxManager()


@router.websocket("/ws/bbox/{camera_id}")
async def bbox_ws(camera_id: str, websocket: WebSocket):
    """프론트엔드 연결 — bbox 데이터를 수신해서 화면에 그림"""
    await bbox_manager.connect(camera_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # 연결 유지용
    except WebSocketDisconnect:
        bbox_manager.disconnect(camera_id, websocket)


@router.websocket("/ws/bbox/{camera_id}/push")
async def bbox_push_ws(camera_id: str, websocket: WebSocket):
    """AI 서버 연결 — bbox 데이터를 push하면 프론트로 브로드캐스트"""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            await bbox_manager.broadcast(camera_id, data)
    except WebSocketDisconnect:
        pass
