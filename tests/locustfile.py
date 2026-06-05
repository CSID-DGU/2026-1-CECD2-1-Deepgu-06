"""
BE API 부하 테스트
사용법:
    pip install locust
    locust -f locustfile.py --users 30 --spawn-rate 3 --host http://43.201.17.169:8000

환경변수:
    CALLBACK_SECRET  - X-Callback-Secret 값
"""
import os
import random
from datetime import datetime

from locust import HttpUser, between, task

CALLBACK_SECRET = os.getenv("CALLBACK_SECRET", "")
CAMERA_IDS = [str(i) for i in range(1, 6)]  # camera1~5


class FEPollingUser(HttpUser):
    """FE가 스트림 상태를 5초마다 폴링하는 시뮬"""
    wait_time = between(4, 6)

    @task(3)
    def poll_stream_status(self):
        camera_id = random.choice(CAMERA_IDS)
        self.client.get(
            f"/api/cameras/{camera_id}/stream",
            name="/api/cameras/[id]/stream",
        )

    @task(1)
    def get_cameras(self):
        self.client.get("/api/cameras")

    @task(1)
    def get_events(self):
        self.client.get("/api/events?page=1&size=20")


class AIWorkerUser(HttpUser):
    """AI Worker가 이벤트를 전송하는 시뮬"""
    wait_time = between(20, 40)  # POST_COOLDOWN_SEC 기준

    @task
    def post_event(self):
        camera_id = random.choice(CAMERA_IDS)
        self.client.post(
            "/internal/events",
            headers={"X-Callback-Secret": CALLBACK_SECRET},
            data={
                "camera_id": camera_id,
                "detected_at": datetime.utcnow().isoformat(),
                "anomaly_type": "fight",
                "confidence": round(random.uniform(0.5, 0.95), 2),
            },
            name="/internal/events",
        )


class MediaCallbackUser(HttpUser):
    """MEDIA 서버가 콜백을 보내는 시뮬"""
    wait_time = between(2, 4)

    @task
    def stream_callback(self):
        camera_id = random.choice(CAMERA_IDS)
        status = random.choice(["RUNNING", "STARTING"])
        self.client.post(
            "/internal/streams/callback",
            headers={"X-Callback-Secret": CALLBACK_SECRET},
            json={
                "camera_id": camera_id,
                "status": status,
                "hls_url": f"/live/camera{camera_id}/index.m3u8",
            },
            name="/internal/streams/callback",
        )
