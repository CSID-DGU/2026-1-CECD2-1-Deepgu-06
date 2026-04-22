import { useEffect, useState } from "react";
import { getUsers, approveUser, deleteUser, assignCamera } from "../api/user";
import { getCameras } from "../api/camera";
import type { User } from "../api/user";
import type { Camera } from "../api/camera";
import Layout from "../components/Layout";

const roleLabel: Record<string, string> = { ADMIN: "관리자", USER: "사용자" };
const statusLabel: Record<string, string> = { ACTIVE: "활성", PENDING: "승인 대기", INACTIVE: "비활성" };
const statusClass: Record<string, string> = { ACTIVE: "ok", PENDING: "primary", INACTIVE: "" };

const UsersPage = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<User | null>(null);
  const [search, setSearch] = useState("");
  const [actionLoading, setActionLoading] = useState(false);
  const [assigningCam, setAssigningCam] = useState("");

  const load = async () => {
    setLoading(true);
    try {
      const [u, c] = await Promise.all([getUsers(), getCameras()]);
      setUsers(u); setCameras(c);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleApprove = async (user: User) => {
    setActionLoading(true);
    try { await approveUser(user.id); await load(); }
    catch (e: any) { alert(e.response?.data?.detail || "승인 실패"); }
    finally { setActionLoading(false); }
  };

  const handleDelete = async (user: User) => {
    if (!confirm(`"${user.name}" 사용자를 삭제하시겠습니까?`)) return;
    setActionLoading(true);
    try { await deleteUser(user.id); setSelected(null); await load(); }
    catch (e: any) { alert(e.response?.data?.detail || "삭제 실패"); }
    finally { setActionLoading(false); }
  };

  const handleAssign = async (camId: string) => {
    if (!selected || !camId) return;
    setAssigningCam(camId);
    try { await assignCamera(camId, selected.id); await load(); }
    catch (e: any) { alert(e.response?.data?.detail || "할당 실패"); }
    finally { setAssigningCam(""); }
  };

  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  );

  const pending = users.filter(u => u.status === "PENDING").length;

  return (
    <Layout
      title="사용자 관리"
      subtitle="사용자 계정 및 카메라 권한 관리"
      topbarRight={pending > 0 ? <span className="chip danger dot">{pending}명 승인 대기</span> : undefined}
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 20 }}>
        {/* User list */}
        <div className="card">
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--line)", display: "flex", gap: 10 }}>
            <div className="search" style={{ flex: 1 }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <input placeholder="이름, 이메일 검색..." value={search} onChange={e => setSearch(e.target.value)} />
            </div>
          </div>

          {loading ? (
            <div style={{ padding: 40, textAlign: "center", color: "var(--ink-4)", fontSize: 13 }}>불러오는 중...</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>이름</th>
                  <th>이메일</th>
                  <th>역할</th>
                  <th>상태</th>
                  <th style={{ width: 120 }}>액션</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(user => (
                  <tr key={user.id} onClick={() => setSelected(user)} style={{ background: selected?.id === user.id ? "var(--primary-soft)" : undefined }}>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div className="avatar" style={{ width: 28, height: 28, fontSize: 10 }}>{user.name[0]}</div>
                        <span style={{ fontWeight: 600 }}>{user.name}</span>
                      </div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--ink-2)" }}>{user.email}</td>
                    <td><span className="chip">{roleLabel[user.role] || user.role}</span></td>
                    <td><span className={`chip dot ${statusClass[user.status]}`}>{statusLabel[user.status] || user.status}</span></td>
                    <td onClick={e => e.stopPropagation()}>
                      <div style={{ display: "flex", gap: 6 }}>
                        {user.status === "PENDING" && (
                          <button className="btn sm primary" onClick={() => handleApprove(user)} disabled={actionLoading}>승인</button>
                        )}
                        <button className="btn sm" style={{ color: "var(--danger-ink)", borderColor: "transparent" }}
                          onClick={() => handleDelete(user)} disabled={actionLoading}>삭제</button>
                      </div>
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--ink-4)", padding: "32px 0" }}>사용자가 없습니다</td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Detail panel */}
        <div>
          {selected ? (
            <div className="card" style={{ padding: "18px 20px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                <div className="avatar">{selected.name[0]}</div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--ink)" }}>{selected.name}</div>
                  <div style={{ fontSize: 12, color: "var(--ink-3)" }}>{selected.email}</div>
                </div>
              </div>

              <div style={{ display: "flex", gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
                <span className="chip">{roleLabel[selected.role]}</span>
                <span className={`chip dot ${statusClass[selected.status]}`}>{statusLabel[selected.status]}</span>
              </div>

              <hr className="hr" style={{ margin: "14px 0" }} />

              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-2)", marginBottom: 10 }}>카메라 할당</div>

              <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                <select className="input" style={{ flex: 1, height: 32, fontSize: 12 }}
                  onChange={e => { if (e.target.value) handleAssign(e.target.value); e.target.value = ""; }}
                  disabled={!!assigningCam}>
                  <option value="">카메라 할당...</option>
                  {cameras.map(c => (
                    <option key={c.cameraId} value={c.cameraId}>{c.name}</option>
                  ))}
                </select>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ fontSize: 11, color: "var(--ink-4)" }}>할당된 카메라</div>
                <div style={{ fontSize: 13, color: "var(--ink-3)" }}>— 구현 예정</div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: 40, textAlign: "center" }}>
              <div style={{ color: "var(--ink-4)", fontSize: 13 }}>사용자를 선택하면<br />상세 정보가 표시됩니다</div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default UsersPage;
