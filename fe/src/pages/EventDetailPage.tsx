import { useParams, useNavigate } from "react-router-dom";
import Layout from "../components/Layout";
import type { EventItem } from "./EventLogsPage";

const MOCK_EVENTS: Record<number, EventItem & { description?: string }> = {
  1: { id: 1, camera_id: "CAM-001", camera_name: "정문 카메라", detected_at: "2026-04-20T10:23:45", anomaly_type: "폭행", confidence: 0.91, status: "UNREVIEWED", description: "두 명의 인물이 격렬하게 충돌하는 행동이 감지되었습니다." },
  2: { id: 2, camera_id: "CAM-002", camera_name: "후문 카메라", detected_at: "2026-04-20T09:15:12", anomaly_type: "배회", confidence: 0.74, status: "REVIEWED", description: "한 인물이 지정된 구역 주변을 반복적으로 배회하는 행동이 감지되었습니다." },
  3: { id: 3, camera_id: "CAM-001", camera_name: "정문 카메라", detected_at: "2026-04-19T22:05:33", anomaly_type: "쓰러짐", confidence: 0.88, status: "UNREVIEWED", description: "인물이 갑자기 쓰러지는 행동이 감지되었습니다." },
  4: { id: 4, camera_id: "CAM-003", camera_name: "주차장 카메라", detected_at: "2026-04-19T18:44:01", anomaly_type: "침입", confidence: 0.62, status: "FALSE_POSITIVE", description: "허가되지 않은 구역 접근이 감지되었습니다." },
};

const statusLabel: Record<string, string> = {
  UNREVIEWED: "미검토", REVIEWED: "검토 완료", FALSE_POSITIVE: "오탐",
};
const statusClass: Record<string, string> = {
  UNREVIEWED: "danger", REVIEWED: "ok", FALSE_POSITIVE: "",
};

const EventDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const ev = MOCK_EVENTS[Number(id)];

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
            <div className="media">
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 8, color: "rgba(255,255,255,.4)" }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                </svg>
                <span style={{ fontSize: 13 }}>클립 없음 (AI 연동 예정)</span>
              </div>
              <div className="meta"><span className="rec">이상행동 클립</span></div>
            </div>
          </div>

          {ev.description && (
            <div className="card" style={{ padding: "18px 20px" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 10 }}>AI 분석 결과</div>
              <p style={{ fontSize: 13, color: "var(--ink)", lineHeight: 1.6, margin: 0 }}>{ev.description}</p>
            </div>
          )}

          <div className="card" style={{ padding: "18px 20px" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 12 }}>주요 프레임</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
              {[1, 2, 3, 4].map(i => (
                <div key={i} style={{ aspectRatio: "16/9", background: "var(--bg-soft)", borderRadius: 6, border: "1px solid var(--line)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ fontSize: 10, color: "var(--ink-4)" }}>프레임 {i}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card" style={{ padding: "18px 20px" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 14 }}>이벤트 정보</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                ["이벤트 ID", `#${ev.id}`],
                ["카메라", ev.camera_name],
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

          <div className="card" style={{ padding: "18px 20px" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink-2)", marginBottom: 12 }}>검토</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button className="btn block" style={{ background: "var(--ok-soft)", color: "#276749", borderColor: "transparent" }}>
                검토 완료 처리
              </button>
              <button className="btn block" style={{ color: "var(--ink-3)" }}>
                오탐으로 처리
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default EventDetailPage;
