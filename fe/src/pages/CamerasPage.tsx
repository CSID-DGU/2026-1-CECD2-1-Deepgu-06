import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getCameras, deleteCamera } from "../api/camera";
import type { Camera } from "../api/camera";
import Layout from "../components/Layout";
import { useAuth } from "../context/AuthContext";

const statusColors: Record<string, string> = {
  RUNNING: "ok", STARTING: "primary", STOPPED: "", FAILED: "danger",
};
const statusLabels: Record<string, string> = {
  RUNNING: "스트리밍 중", STARTING: "시작 중", STOPPED: "정지", FAILED: "오류",
};

const CamerasPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    getCameras().then(setCameras).catch(console.error).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (cam: Camera) => {
    if (!confirm(`"${cam.name}" 카메라를 삭제하시겠습니까?`)) return;
    setDeleting(cam.cameraId);
    try {
      await deleteCamera(cam.cameraId);
      load();
    } catch (err: any) {
      alert(err.response?.data?.detail || "삭제 실패");
    } finally {
      setDeleting(null);
    }
  };

  const filtered = cameras.filter(c =>
    c.name.toLowerCase().includes(search.toLowerCase()) ||
    c.location.toLowerCase().includes(search.toLowerCase()) ||
    c.cameraId.toLowerCase().includes(search.toLowerCase())
  );

  const running = cameras.filter(c => c.status === "RUNNING").length;

  return (
    <Layout
      title="카메라 관리"
      subtitle="등록된 카메라 목록"
      topbarRight={
        user?.role === "ADMIN" ? (
          <button className="btn primary sm" onClick={() => navigate("/cameras/register")}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            카메라 등록
          </button>
        ) : undefined
      }
    >
      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginBottom: 20 }}>
        {[
          { label: "전체 카메라", value: cameras.length, color: "var(--primary)" },
          { label: "스트리밍 중", value: running, color: "var(--ok)" },
          { label: "정지", value: cameras.length - running, color: "var(--ink-3)" },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: "16px 18px" }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 2 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Table */}
      <div className="card">
        <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", gap: 12 }}>
          <div className="search" style={{ width: 260 }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <input placeholder="이름, 위치, ID 검색..." value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <span style={{ fontSize: 12, color: "var(--ink-3)" }}>{filtered.length}개</span>
        </div>

        {loading ? (
          <div style={{ padding: 40, textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>불러오는 중...</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>카메라명</th>
                <th>위치</th>
                <th>카메라 ID</th>
                <th>상태</th>
                {user?.role === "ADMIN" && <th style={{ width: 80 }}>관리</th>}
              </tr>
            </thead>
            <tbody>
              {filtered.map(cam => (
                <tr key={cam.cameraId} onClick={() => navigate(`/stream`)} style={{ cursor: "pointer" }}>
                  <td>
                    <div style={{ fontWeight: 600, color: "var(--ink)" }}>{cam.name}</div>
                  </td>
                  <td style={{ color: "var(--ink-2)" }}>{cam.location}</td>
                  <td><span className="chip mono" style={{ fontSize: 11 }}>{cam.cameraId}</span></td>
                  <td>
                    <span className={`chip dot ${statusColors[cam.status] || ""}`}>
                      {statusLabels[cam.status] || cam.status}
                    </span>
                  </td>
                  {user?.role === "ADMIN" && (
                    <td onClick={e => e.stopPropagation()}>
                      <button
                        className="btn sm"
                        style={{ color: "var(--danger-ink)", borderColor: "transparent" }}
                        onClick={() => handleDelete(cam)}
                        disabled={deleting === cam.cameraId}
                      >
                        삭제
                      </button>
                    </td>
                  )}
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--ink-4)", padding: "32px 0" }}>카메라가 없습니다</td></tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </Layout>
  );
};

export default CamerasPage;
