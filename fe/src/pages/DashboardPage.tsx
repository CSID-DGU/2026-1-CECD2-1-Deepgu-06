import { useEffect, useState, useCallback } from "react";
import { useAuth } from "../context/AuthContext";
import { getCameras, startStream, stopStream, getStreamStatus } from "../api/camera";
import type { Camera, StreamStatus } from "../api/camera";
import VideoPlayer from "../components/VideoPlayer";
import styles from "./DashboardPage.module.css";

const MEDIA_SERVER_URL = import.meta.env.VITE_MEDIA_SERVER_URL || "http://localhost:9000";

const statusLabel: Record<string, { label: string; color: string }> = {
  RUNNING:  { label: "스트리밍 중", color: "#10b981" },
  STARTING: { label: "시작 중",    color: "#f59e0b" },
  STOPPED:  { label: "정지",       color: "#6b7280" },
  FAILED:   { label: "오류",       color: "#ef4444" },
};

interface AnomalyLog {
  id: number;
  time: string;
  message: string;
}

const DashboardPage = () => {
  const { user, logout } = useAuth();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selected, setSelected] = useState<Camera | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamStatus | null>(null);
  const [anomalyLogs] = useState<AnomalyLog[]>([]);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    getCameras().then(setCameras).catch(console.error);
  }, []);

  const loadStatus = useCallback(async (camera: Camera) => {
    try {
      const s = await getStreamStatus(camera.cameraId);
      setStreamStatus(s);
    } catch {
      setStreamStatus(null);
    }
  }, []);

  useEffect(() => {
    if (!selected) return;
    loadStatus(selected);
    const id = setInterval(() => loadStatus(selected), 5000);
    return () => clearInterval(id);
  }, [selected, loadStatus]);

  const handleStart = async () => {
    if (!selected) return;
    setActionLoading(true);
    try {
      await startStream(selected.cameraId);
      await loadStatus(selected);
    } catch (err: any) {
      alert(err.response?.data?.detail || "스트림 시작 실패");
    } finally {
      setActionLoading(false);
    }
  };

  const handleStop = async () => {
    if (!selected) return;
    setActionLoading(true);
    try {
      await stopStream(selected.cameraId);
      await loadStatus(selected);
    } catch (err: any) {
      alert(err.response?.data?.detail || "스트림 종료 실패");
    } finally {
      setActionLoading(false);
    }
  };

  const camStatus = streamStatus?.cameraStatus || selected?.status || "STOPPED";
  const hlsUrl = streamStatus?.currentSession?.hlsUrl || null;
  const isRunning = camStatus === "RUNNING" || camStatus === "STARTING";
  const st = statusLabel[camStatus] || statusLabel["STOPPED"];

  return (
    <div className={styles.layout}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <span>🎥</span>
          <span className={styles.logoText}>DEEPGU</span>
        </div>

        <div className={styles.sectionTitle}>카메라 목록</div>
        <ul className={styles.cameraList}>
          {cameras.map((cam) => {
            const s = statusLabel[cam.status] || statusLabel["STOPPED"];
            return (
              <li key={cam.cameraId}
                className={`${styles.cameraItem} ${selected?.cameraId === cam.cameraId ? styles.selected : ""}`}
                onClick={() => { setSelected(cam); setStreamStatus(null); }}>
                <span className={styles.dot} style={{ background: s.color }} />
                <div>
                  <div className={styles.camName}>{cam.name}</div>
                  <div className={styles.camLoc}>{cam.location}</div>
                </div>
              </li>
            );
          })}
          {cameras.length === 0 && <li className={styles.empty}>할당된 카메라가 없습니다</li>}
        </ul>

        <div className={styles.sidebarBottom}>
          {user?.role === "ADMIN" && (
            <a href="/admin" className={styles.adminLink}>⚙️ 관리자 페이지</a>
          )}
          <button className={styles.logoutBtn} onClick={logout}>로그아웃</button>
        </div>
      </aside>

      <main className={styles.main}>
        {selected ? (
          <>
            <div className={styles.mainHeader}>
              <div>
                <h2 className={styles.camTitle}>{selected.name}</h2>
                <span className={styles.camLocTag}>{selected.location}</span>
              </div>
              <span className={styles.statusBadge} style={{ color: st.color }}>● {st.label}</span>
            </div>

            <div className={styles.videoWrapper}>
              {hlsUrl
                ? <VideoPlayer hlsUrl={hlsUrl} mediaServerUrl={MEDIA_SERVER_URL} />
                : <div className={styles.noStream}>{camStatus === "STARTING" ? "스트림 시작 중..." : "스트림이 없습니다"}</div>
              }
            </div>

            <div className={styles.controls}>
              <button className={styles.startBtn} onClick={handleStart} disabled={actionLoading || isRunning}>▶ 스트림 시작</button>
              <button className={styles.stopBtn} onClick={handleStop} disabled={actionLoading || !isRunning}>⏹ 스트림 종료</button>
            </div>

            <div className={styles.logSection}>
              <h3 className={styles.logTitle}>⚠️ 이상행동 감지 로그</h3>
              <div className={styles.logList}>
                {anomalyLogs.length === 0
                  ? <div className={styles.logEmpty}>감지된 이상행동이 없습니다</div>
                  : anomalyLogs.map((log) => (
                    <div key={log.id} className={styles.logItem}>
                      <span className={styles.logTime}>{log.time}</span>
                      <span>{log.message}</span>
                    </div>
                  ))
                }
              </div>
            </div>
          </>
        ) : (
          <div className={styles.emptyMain}>
            <span>👈</span>
            <p>왼쪽에서 카메라를 선택하세요</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default DashboardPage;
