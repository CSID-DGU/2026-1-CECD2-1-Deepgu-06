import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "../components/Layout";

export interface EventItem {
  id: number;
  camera_id: string;
  camera_name: string;
  detected_at: string;
  anomaly_type: string;
  confidence: number;
  status: "UNREVIEWED" | "REVIEWED" | "FALSE_POSITIVE";
}

const MOCK_EVENTS: EventItem[] = [
  { id: 1, camera_id: "CAM-001", camera_name: "정문 카메라", detected_at: "2026-04-20T10:23:45", anomaly_type: "폭행", confidence: 0.91, status: "UNREVIEWED" },
  { id: 2, camera_id: "CAM-002", camera_name: "후문 카메라", detected_at: "2026-04-20T09:15:12", anomaly_type: "배회", confidence: 0.74, status: "REVIEWED" },
  { id: 3, camera_id: "CAM-001", camera_name: "정문 카메라", detected_at: "2026-04-19T22:05:33", anomaly_type: "쓰러짐", confidence: 0.88, status: "UNREVIEWED" },
  { id: 4, camera_id: "CAM-003", camera_name: "주차장 카메라", detected_at: "2026-04-19T18:44:01", anomaly_type: "침입", confidence: 0.62, status: "FALSE_POSITIVE" },
];

const statusLabel: Record<string, string> = {
  UNREVIEWED: "미검토", REVIEWED: "검토 완료", FALSE_POSITIVE: "오탐",
};
const statusClass: Record<string, string> = {
  UNREVIEWED: "danger", REVIEWED: "ok", FALSE_POSITIVE: "",
};

const EventLogsPage = () => {
  const navigate = useNavigate();
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [search, setSearch] = useState("");

  const filtered = MOCK_EVENTS.filter(e => {
    if (statusFilter !== "ALL" && e.status !== statusFilter) return false;
    if (search && !e.camera_name.toLowerCase().includes(search.toLowerCase()) &&
        !e.anomaly_type.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const unreviewed = MOCK_EVENTS.filter(e => e.status === "UNREVIEWED").length;

  return (
    <Layout
      title="이벤트 로그"
      subtitle="이상행동 감지 기록"
      topbarRight={unreviewed > 0 ? <span className="chip danger dot">{unreviewed}개 미검토</span> : undefined}
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginBottom: 20 }}>
        {[
          { label: "전체 이벤트", value: MOCK_EVENTS.length, color: "var(--primary)" },
          { label: "미검토", value: MOCK_EVENTS.filter(e => e.status === "UNREVIEWED").length, color: "var(--danger-ink)" },
          { label: "검토 완료", value: MOCK_EVENTS.filter(e => e.status === "REVIEWED").length, color: "var(--ok)" },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: "16px 18px" }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 2 }}>{s.label}</div>
          </div>
        ))}
      </div>

      <div className="card">
        <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div className="search" style={{ width: 240 }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input placeholder="카메라, 이상행동 검색..." value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <select className="input" style={{ width: "auto", height: 34, fontSize: 12 }} value={statusFilter}
            onChange={e => setStatusFilter(e.target.value)}>
            <option value="ALL">전체 상태</option>
            <option value="UNREVIEWED">미검토</option>
            <option value="REVIEWED">검토 완료</option>
            <option value="FALSE_POSITIVE">오탐</option>
          </select>
          <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--ink-3)" }}>{filtered.length}건</span>
        </div>

        <table className="table">
          <thead>
            <tr>
              <th>감지 시각</th>
              <th>카메라</th>
              <th>이상행동 유형</th>
              <th>신뢰도</th>
              <th>상태</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(ev => (
              <tr key={ev.id} onClick={() => navigate(`/events/${ev.id}`)}>
                <td>
                  <div className="mono" style={{ fontSize: 12 }}>
                    {new Date(ev.detected_at).toLocaleString("ko-KR")}
                  </div>
                </td>
                <td>
                  <div style={{ fontWeight: 600 }}>{ev.camera_name}</div>
                  <div style={{ fontSize: 11, color: "var(--ink-3)" }}>{ev.camera_id}</div>
                </td>
                <td>
                  <span className={`chip ${ev.confidence > 0.85 ? "danger" : ev.confidence > 0.65 ? "primary" : ""}`}>
                    {ev.anomaly_type}
                  </span>
                </td>
                <td>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 60, height: 4, background: "var(--bg-soft)", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ width: `${ev.confidence * 100}%`, height: "100%", background: ev.confidence > 0.8 ? "var(--danger-ink)" : "var(--primary)", borderRadius: 2 }} />
                    </div>
                    <span style={{ fontSize: 12, color: "var(--ink-2)" }}>{(ev.confidence * 100).toFixed(0)}%</span>
                  </div>
                </td>
                <td>
                  <span className={`chip dot ${statusClass[ev.status]}`}>{statusLabel[ev.status]}</span>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--ink-4)", padding: "32px 0" }}>이벤트가 없습니다</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </Layout>
  );
};

export default EventLogsPage;
