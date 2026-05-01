import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getEvent, updateEventStatus, type EventDetail } from "../api/event";
import Layout from "../components/Layout";

const statusLabel: Record<string, string> = {
  UNREVIEWED: "미검토", REVIEWED: "검토 완료", FALSE_POSITIVE: "오탐",
};
const statusClass: Record<string, string> = {
  UNREVIEWED: "danger", REVIEWED: "ok", FALSE_POSITIVE: "",
};

const EventDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [ev, setEv] = useState<EventDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    getEvent(Number(id))
      .then(setEv)
      .catch(() => setEv(null))
      .finally(() => setLoading(false));
  }, [id]);

  const handleStatusUpdate = async (status: "REVIEWED" | "FALSE_POSITIVE") => {
    if (!ev) return;
    setUpdating(true);
    try {
      const updated = await updateEventStatus(ev.id, status);
      setEv((prev) => prev ? { ...prev, status: updated.status } : prev);
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <Layout title="이벤트 상세">
        <div className="card" style={{ padding: 40, textAlign: "center", color: "var(--ink-4)", fontSize: 14 }}>불러오는 중...</div>
      </Layout>
    );
  }

  if (!ev) {
    return (
      <Layout title="이벤트 상세">
        <div className="card" style={{ padding: 40, textAlign: "center" }}>
          <div style={{ color: "var(--ink-3)", fontSize: 14 }}>이벤트를 찾을 수 없습니다.</div>
          <button className="btn primary" style={{ marginTop: 16 }} onClick={() => navigate("/events")}>목록으로</button>
        </div>
      </Layout>
    );
  }

  return (
    <Layout
      title="이벤트 상세"
      subtitle={`이벤트 #${ev.id}`}
      topbarRight={<button className="btn sm" onClick={() => navigate("/events")}>← 목록</button>}
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 20 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card" style={{ overflow: "hidden" }}>
            {ev.video_url ? (
              <video
                src={ev.video_url}
                controls
                autoPlay
                style={{ width: "100%", display: "block", background: "#000" }}
              />
            ) : (
              <div className="media">
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 8, color: "rgba(255,255,255,.4)" }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="23 7 16 12 23 17 23 7" /><rect x="1" y="5" width="15" height="14" rx="2" />
                  </svg>
                  <span style={{ fontSize: 13 }}>클립 없음</span>
                </div>
                <div className="meta"><span className="rec">이상행동 클립</span></div>
              </div>
            )}
          </div>

          {ev.description && (
            <div className="card" style={{ padding: "18px 20px" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 10 }}>AI 분석 결과</div>
              <p style={{ fontSize: 13, color: "var(--ink)", lineHeight: 1.6, margin: 0 }}>{ev.description}</p>
            </div>
          )}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card" style={{ padding: "18px 20px" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 14 }}>이벤트 정보</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                ["이벤트 ID", `#${ev.id}`],
                ["카메라 ID", ev.camera_id],
                ["감지 시각", new Date(ev.detected_at).toLocaleString("ko-KR")],
                ["이상행동", ev.anomaly_type],
              ].map(([k, v]) => (
                <div key={k}>
                  <div style={{ fontSize: 11, color: "var(--ink-4)", marginBottom: 2 }}>{k}</div>
                  <div style={{ fontSize: 13, color: "var(--ink)", fontWeight: 500 }}>{v}</div>
                </div>
              ))}
              <div>
                <div style={{ fontSize: 11, color: "var(--ink-4)", marginBottom: 4 }}>신뢰도</div>
                <div style={{ height: 6, background: "var(--bg-soft)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ width: `${ev.confidence * 100}%`, height: "100%", background: ev.confidence > 0.8 ? "var(--danger-ink)" : "var(--primary)", borderRadius: 3 }} />
                </div>
                <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 4 }}>{(ev.confidence * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "var(--ink-4)", marginBottom: 4 }}>상태</div>
                <span className={`chip dot ${statusClass[ev.status]}`}>{statusLabel[ev.status]}</span>
              </div>
            </div>
          </div>

          {ev.status === "UNREVIEWED" && (
            <div className="card" style={{ padding: "18px 20px" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 12 }}>검토</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <button
                  className="btn block"
                  style={{ background: "var(--ok-soft)", color: "#276749", borderColor: "transparent" }}
                  disabled={updating}
                  onClick={() => handleStatusUpdate("REVIEWED")}
                >
                  검토 완료 처리
                </button>
                <button
                  className="btn block"
                  style={{ color: "var(--ink-3)" }}
                  disabled={updating}
                  onClick={() => handleStatusUpdate("FALSE_POSITIVE")}
                >
                  오탐으로 처리
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default EventDetailPage;
