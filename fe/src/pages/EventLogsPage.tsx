import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listEvents, type EventItem } from "../api/event";
import Layout from "../components/Layout";

const statusLabel: Record<string, string> = {
  UNREVIEWED: "미검토", REVIEWED: "검토 완료", FALSE_POSITIVE: "오탐",
};
const statusClass: Record<string, string> = {
  UNREVIEWED: "danger", REVIEWED: "ok", FALSE_POSITIVE: "",
};

const EventLogsPage = () => {
  const navigate = useNavigate();
  const [events, setEvents] = useState<EventItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  const SIZE = 20;

  useEffect(() => {
    setLoading(true);
    listEvents({
      page,
      size: SIZE,
      status: statusFilter !== "ALL" ? statusFilter : undefined,
    })
      .then((res) => {
        setEvents(res.items);
        setTotal(res.total);
      })
      .finally(() => setLoading(false));
  }, [page, statusFilter]);

  const filtered = search
    ? events.filter(
        (e) =>
          e.camera_id.toLowerCase().includes(search.toLowerCase()) ||
          e.anomaly_type.toLowerCase().includes(search.toLowerCase())
      )
    : events;

  const unreviewed = events.filter((e) => e.status === "UNREVIEWED").length;
  const totalPages = Math.ceil(total / SIZE);

  return (
    <Layout
      title="이벤트 로그"
      subtitle="이상행동 감지 기록"
      topbarRight={unreviewed > 0 ? <span className="chip danger dot">{unreviewed}개 미검토</span> : undefined}
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginBottom: 20 }}>
        {[
          { label: "전체 이벤트", value: total, color: "var(--primary)" },
          { label: "미검토 (현재 페이지)", value: events.filter((e) => e.status === "UNREVIEWED").length, color: "var(--danger-ink)" },
          { label: "검토 완료 (현재 페이지)", value: events.filter((e) => e.status === "REVIEWED").length, color: "var(--ok)" },
        ].map((s) => (
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
              <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input placeholder="카메라 ID, 이상행동 검색..." value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <select className="input" style={{ width: "auto", height: 34, fontSize: 12 }} value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}>
            <option value="ALL">전체 상태</option>
            <option value="UNREVIEWED">미검토</option>
            <option value="REVIEWED">검토 완료</option>
            <option value="FALSE_POSITIVE">오탐</option>
          </select>
          <span style={{ marginLeft: "auto", fontSize: 12, color: "var(--ink-3)" }}>{total}건</span>
        </div>

        <table className="table">
          <thead>
            <tr>
              <th>감지 시각</th>
              <th>카메라 ID</th>
              <th>이상행동 유형</th>
              <th>신뢰도</th>
              <th>상태</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--ink-4)", padding: "32px 0" }}>불러오는 중...</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--ink-4)", padding: "32px 0" }}>이벤트가 없습니다</td></tr>
            ) : filtered.map((ev) => (
              <tr key={ev.id} onClick={() => navigate(`/events/${ev.id}`)}>
                <td>
                  <div className="mono" style={{ fontSize: 12 }}>
                    {new Date(ev.detected_at).toLocaleString("ko-KR")}
                  </div>
                </td>
                <td>
                  <div style={{ fontSize: 13 }}>{ev.camera_id}</div>
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
          </tbody>
        </table>

        {totalPages > 1 && (
          <div style={{ display: "flex", justifyContent: "center", gap: 8, padding: "12px 0" }}>
            <button className="btn sm" disabled={page === 1} onClick={() => setPage((p) => p - 1)}>이전</button>
            <span style={{ fontSize: 12, color: "var(--ink-3)", lineHeight: "30px" }}>{page} / {totalPages}</span>
            <button className="btn sm" disabled={page === totalPages} onClick={() => setPage((p) => p + 1)}>다음</button>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default EventLogsPage;
