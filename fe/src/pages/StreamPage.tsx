import { useEffect, useState, useCallback } from "react";
import { getCameras, startStream, stopStream, getStreamStatus } from "../api/camera";
import type { Camera, StreamStatus } from "../api/camera";
import VideoPlayer from "../components/VideoPlayer";
import Layout from "../components/Layout";

const MEDIA_SERVER_URL = import.meta.env.VITE_MEDIA_SERVER_URL || "";

const StreamPage = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selected, setSelected] = useState<Camera | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamStatus | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [search, setSearch] = useState("");

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

  const camStatus = streamStatus?.camera_status || selected?.status || "STOPPED";
  const hlsUrl = camStatus === "RUNNING" ? (streamStatus?.current_session?.hls_url || null) : null;
  const isRunning = camStatus === "RUNNING" || camStatus === "STARTING";

  const filteredCameras = cameras.filter(c =>
    c.name.toLowerCase().includes(search.toLowerCase()) ||
    c.location.toLowerCase().includes(search.toLowerCase())
  );

  const statusColors: Record<string, string> = {
    RUNNING: "var(--ok)", STARTING: "#f59e0b", STOPPED: "var(--ink-4)", FAILED: "var(--danger-ink)",
  };
  const statusLabels: Record<string, string> = {
    RUNNING: "스트리밍 중", STARTING: "시작 중", STOPPED: "정지", FAILED: "오류",
  };

  return (
    <Layout
      title={selected ? selected.name : "실시간 스트리밍"}
      subtitle={selected ? selected.location : undefined}
      topbarRight={
        selected && isRunning ? (
          <span className="chip ok dot">라이브</span>
        ) : undefined
      }
    >
      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 20, height: "calc(100vh - 56px - 48px)" }}>
        {/* Camera list */}
        <div className="card" style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ padding: "14px 14px 10px" }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-2)", marginBottom: 8 }}>카메라 목록</div>
            <div className="search">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <input placeholder="검색..." value={search} onChange={e => setSearch(e.target.value)} />
            </div>
          </div>
          <div style={{ flex: 1, overflowY: "auto" }}>
            {filteredCameras.map(cam => (
              <div
                key={cam.cameraId}
                onClick={() => { setSelected(cam); setStreamStatus(null); }}
                style={{
                  padding: "10px 14px", cursor: "pointer", borderBottom: "1px solid var(--line-soft)",
                  background: selected?.cameraId === cam.cameraId ? "var(--primary-soft)" : undefined,
                  transition: "background .1s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="dot" style={{ background: statusColors[cam.status] || "var(--ink-4)" }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: selected?.cameraId === cam.cameraId ? "var(--primary-ink)" : "var(--ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {cam.name}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {cam.location}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {filteredCameras.length === 0 && (
              <div style={{ padding: "24px 14px", textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>
                카메라가 없습니다
              </div>
            )}
          </div>
        </div>

        {/* Main panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
          {selected ? (
            <>
              {/* Video */}
              <div className="media" style={{ flex: "none" }}>
                {isRunning && (
                  <div className="meta">
                    <span className="rec">● REC</span>
                    <span style={{ color: "#fff", fontSize: 11, background: "rgba(0,0,0,.4)", padding: "2px 6px", borderRadius: 3 }}>{selected.name}</span>
                  </div>
                )}
                {hlsUrl
                  ? <VideoPlayer hlsUrl={hlsUrl} mediaServerUrl={MEDIA_SERVER_URL} />
                  : (
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 8, color: "rgba(255,255,255,.5)" }}>
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                      </svg>
                      <span style={{ fontSize: 13 }}>{camStatus === "STARTING" ? "스트림 시작 중..." : "스트림이 없습니다"}</span>
                    </div>
                  )
                }
              </div>

              {/* Controls */}
              <div className="card" style={{ padding: "14px 16px" }}>
                <div className="row between">
                  <div className="row" style={{ gap: 10 }}>
                    <span className="chip dot" style={{ color: statusColors[camStatus] }}>
                      {statusLabels[camStatus] || camStatus}
                    </span>
                    {streamStatus?.current_session && (
                      <span className="chip mono" style={{ fontSize: 11 }}>
                        세션 #{streamStatus.current_session.session_id}
                      </span>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button className="btn primary sm" onClick={handleStart} disabled={actionLoading || isRunning}>
                      ▶ 시작
                    </button>
                    <button className="btn sm" onClick={handleStop} disabled={actionLoading || !isRunning}>
                      ⏹ 종료
                    </button>
                  </div>
                </div>
              </div>

              {/* Info */}
              <div className="card" style={{ padding: "14px 16px" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-3)", marginBottom: 10 }}>카메라 정보</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 16px" }}>
                  {[
                    ["카메라 ID", selected.cameraId],
                    ["위치", selected.location],
                    ["상태", statusLabels[selected.status] || selected.status],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <div style={{ fontSize: 11, color: "var(--ink-4)", marginBottom: 2 }}>{k}</div>
                      <div style={{ fontSize: 13, color: "var(--ink)", fontWeight: 500 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="card" style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12 }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/>
              </svg>
              <div style={{ fontSize: 14, color: "var(--ink-3)" }}>왼쪽에서 카메라를 선택하세요</div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default StreamPage;
