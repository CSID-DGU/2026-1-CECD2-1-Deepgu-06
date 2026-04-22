import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getCameras } from "../api/camera";
import { getUsers } from "../api/user";
import type { Camera } from "../api/camera";
import type { User } from "../api/user";
import Layout from "../components/Layout";

const AdminDashboardPage = () => {
  const navigate = useNavigate();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getCameras(), getUsers()])
      .then(([c, u]) => { setCameras(c); setUsers(u); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const runningCams = cameras.filter(c => c.status === "RUNNING").length;
  const pendingUsers = users.filter(u => u.status === "PENDING").length;
  const activeUsers = users.filter(u => u.status === "ACTIVE").length;

  const stats = [
    { label: "전체 카메라", value: cameras.length, sub: `${runningCams}개 스트리밍 중`, color: "var(--primary)", path: "/cameras" },
    { label: "전체 사용자", value: users.length, sub: `${activeUsers}명 활성`, color: "var(--ok)", path: "/users" },
    { label: "승인 대기", value: pendingUsers, sub: "신규 가입 요청", color: pendingUsers > 0 ? "var(--danger-ink)" : "var(--ink-3)", path: "/users" },
    { label: "이상행동 감지", value: 4, sub: "오늘 기준", color: "var(--danger-ink)", path: "/events" },
  ];

  return (
    <Layout title="관리자 대시보드" subtitle="시스템 현황 요약">
      {/* Stat cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 24 }}>
        {stats.map(s => (
          <div key={s.label} className="card" style={{ padding: "18px 20px", cursor: "pointer", transition: "box-shadow .15s" }}
            onClick={() => navigate(s.path)}
            onMouseEnter={e => (e.currentTarget.style.boxShadow = "var(--shadow-lg)")}
            onMouseLeave={e => (e.currentTarget.style.boxShadow = "")}>
            <div style={{ fontSize: 26, fontWeight: 700, color: s.color }}>{loading ? "—" : s.value}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)", marginTop: 2 }}>{s.label}</div>
            <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 2 }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Camera status */}
        <div className="card">
          <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>카메라 현황</span>
            <button className="btn sm ghost" onClick={() => navigate("/cameras")}>전체 보기</button>
          </div>
          {loading ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>불러오는 중...</div>
          ) : (
            <div>
              {cameras.slice(0, 5).map(cam => (
                <div key={cam.cameraId} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderBottom: "1px solid var(--line-soft)" }}>
                  <span className={`dot ${cam.status === "RUNNING" ? "green" : cam.status === "FAILED" ? "red" : ""}`} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{cam.name}</div>
                    <div style={{ fontSize: 11, color: "var(--ink-3)" }}>{cam.location}</div>
                  </div>
                  <span className={`chip ${cam.status === "RUNNING" ? "ok" : cam.status === "FAILED" ? "danger" : ""}`} style={{ fontSize: 11 }}>
                    {cam.status === "RUNNING" ? "스트리밍" : cam.status === "STARTING" ? "시작 중" : cam.status === "FAILED" ? "오류" : "정지"}
                  </span>
                </div>
              ))}
              {cameras.length === 0 && (
                <div style={{ padding: "24px 16px", textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>카메라가 없습니다</div>
              )}
            </div>
          )}
        </div>

        {/* Users */}
        <div className="card">
          <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>사용자 현황</span>
            <button className="btn sm ghost" onClick={() => navigate("/users")}>전체 보기</button>
          </div>
          {loading ? (
            <div style={{ padding: 32, textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>불러오는 중...</div>
          ) : (
            <div>
              {users.slice(0, 5).map(user => (
                <div key={user.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", borderBottom: "1px solid var(--line-soft)" }}>
                  <div className="avatar" style={{ width: 28, height: 28, fontSize: 10 }}>{user.name[0]}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{user.name}</div>
                    <div style={{ fontSize: 11, color: "var(--ink-3)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{user.email}</div>
                  </div>
                  <span className={`chip dot ${user.status === "ACTIVE" ? "ok" : user.status === "PENDING" ? "primary" : ""}`} style={{ fontSize: 11 }}>
                    {user.status === "ACTIVE" ? "활성" : user.status === "PENDING" ? "대기" : "비활성"}
                  </span>
                </div>
              ))}
              {users.length === 0 && (
                <div style={{ padding: "24px 16px", textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>사용자가 없습니다</div>
              )}
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default AdminDashboardPage;
