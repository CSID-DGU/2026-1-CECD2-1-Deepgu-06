import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getUsers, approveUser, deleteUser, assignCamera } from "../api/user";
import type { User } from "../api/user";
import { getCameras, createCamera, deleteCamera } from "../api/camera";
import type { Camera } from "../api/camera";
import styles from "./AdminPage.module.css";

type Tab = "users" | "cameras";

const AdminPage = () => {
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>("users");
  const [users, setUsers] = useState<User[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [newCamera, setNewCamera] = useState({ cameraId: "", name: "", location: "", streamKey: "", description: "" });
  const [assignForm, setAssignForm] = useState({ cameraId: "", userId: "" });
  const [msg, setMsg] = useState("");

  useEffect(() => {
    getUsers().then(setUsers).catch(console.error);
    getCameras().then(setCameras).catch(console.error);
  }, []);

  const handleApprove = async (userId: number) => {
    try {
      await approveUser(userId);
      setUsers((prev) => prev.map((u) => u.id === userId ? { ...u, status: "ACTIVE" } : u));
      setMsg("승인 완료");
    } catch { setMsg("승인 실패"); }
  };

  const handleDeleteUser = async (userId: number) => {
    if (!window.confirm("정말 삭제하시겠습니까?")) return;
    try {
      await deleteUser(userId);
      setUsers((prev) => prev.filter((u) => u.id !== userId));
      setMsg("삭제 완료");
    } catch { setMsg("삭제 실패"); }
  };

  const handleCreateCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const created = await createCamera(newCamera);
      setCameras((prev) => [...prev, created]);
      setNewCamera({ cameraId: "", name: "", location: "", streamKey: "", description: "" });
      setMsg("카메라 등록 완료");
    } catch (err: any) {
      setMsg(err.response?.data?.detail || "카메라 등록 실패");
    }
  };

  const handleDeleteCamera = async (cameraId: string) => {
    if (!window.confirm("정말 삭제하시겠습니까?")) return;
    try {
      await deleteCamera(cameraId);
      setCameras((prev) => prev.filter((c) => c.cameraId !== cameraId));
      setMsg("카메라 삭제 완료");
    } catch { setMsg("카메라 삭제 실패"); }
  };

  const handleAssign = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await assignCamera(assignForm.cameraId, Number(assignForm.userId));
      setAssignForm({ cameraId: "", userId: "" });
      setMsg("카메라 할당 완료");
    } catch { setMsg("할당 실패"); }
  };

  const statusColor: Record<string, string> = {
    ACTIVE: "#10b981", PENDING: "#f59e0b", INACTIVE: "#6b7280",
  };
  const statusText: Record<string, string> = {
    ACTIVE: "활성", PENDING: "승인 대기", INACTIVE: "비활성",
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <button className={styles.backBtn} onClick={() => navigate("/")}>← 대시보드</button>
        <h1 className={styles.title}>관리자 페이지</h1>
      </div>

      {msg && <div className={styles.toast} onClick={() => setMsg("")}>{msg} ✕</div>}

      <div className={styles.tabs}>
        <button className={`${styles.tab} ${tab === "users" ? styles.activeTab : ""}`} onClick={() => setTab("users")}>👤 사용자 관리</button>
        <button className={`${styles.tab} ${tab === "cameras" ? styles.activeTab : ""}`} onClick={() => setTab("cameras")}>🎥 카메라 관리</button>
      </div>

      {tab === "users" && (
        <div className={styles.section}>
          <table className={styles.table}>
            <thead>
              <tr><th>이름</th><th>이메일</th><th>권한</th><th>상태</th><th>가입일</th><th>액션</th></tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.name}</td>
                  <td>{u.email}</td>
                  <td><span className={u.role === "ADMIN" ? styles.badgeAdmin : styles.badgeUser}>{u.role}</span></td>
                  <td><span style={{ color: statusColor[u.status] || "#6b7280" }}>{statusText[u.status] || u.status}</span></td>
                  <td>{new Date(u.createdAt).toLocaleDateString()}</td>
                  <td className={styles.actions}>
                    {u.status === "PENDING" && <button className={styles.approveBtn} onClick={() => handleApprove(u.id)}>승인</button>}
                    {u.role !== "ADMIN" && <button className={styles.deleteBtn} onClick={() => handleDeleteUser(u.id)}>삭제</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className={styles.card}>
            <h3 className={styles.cardTitle}>카메라 할당</h3>
            <form onSubmit={handleAssign} className={styles.inlineForm}>
              <select value={assignForm.cameraId} onChange={(e) => setAssignForm({ ...assignForm, cameraId: e.target.value })} required>
                <option value="">카메라 선택</option>
                {cameras.map((c) => <option key={c.cameraId} value={c.cameraId}>{c.name}</option>)}
              </select>
              <select value={assignForm.userId} onChange={(e) => setAssignForm({ ...assignForm, userId: e.target.value })} required>
                <option value="">사용자 선택</option>
                {users.filter((u) => u.role === "USER").map((u) => (
                  <option key={u.id} value={u.id}>{u.name} ({u.email})</option>
                ))}
              </select>
              <button type="submit" className={styles.submitBtn}>할당</button>
            </form>
          </div>
        </div>
      )}

      {tab === "cameras" && (
        <div className={styles.section}>
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>카메라 등록</h3>
            <form onSubmit={handleCreateCamera} className={styles.formGrid}>
              <div className={styles.field}>
                <label>카메라 ID</label>
                <input value={newCamera.cameraId} onChange={(e) => setNewCamera({ ...newCamera, cameraId: e.target.value })} placeholder="cam-01" required />
              </div>
              <div className={styles.field}>
                <label>이름</label>
                <input value={newCamera.name} onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })} placeholder="정문 카메라" required />
              </div>
              <div className={styles.field}>
                <label>위치</label>
                <input value={newCamera.location} onChange={(e) => setNewCamera({ ...newCamera, location: e.target.value })} placeholder="1층 정문" required />
              </div>
              <div className={styles.field}>
                <label>스트림 키</label>
                <input value={newCamera.streamKey} onChange={(e) => setNewCamera({ ...newCamera, streamKey: e.target.value })} placeholder="stream-key" required />
              </div>
              <div className={`${styles.field} ${styles.fullWidth}`}>
                <label>설명 (선택)</label>
                <input value={newCamera.description} onChange={(e) => setNewCamera({ ...newCamera, description: e.target.value })} placeholder="카메라 설명" />
              </div>
              <button type="submit" className={`${styles.submitBtn} ${styles.fullWidth}`}>등록</button>
            </form>
          </div>

          <table className={styles.table}>
            <thead>
              <tr><th>ID</th><th>이름</th><th>위치</th><th>상태</th><th>액션</th></tr>
            </thead>
            <tbody>
              {cameras.map((cam) => (
                <tr key={cam.cameraId}>
                  <td>{cam.cameraId}</td>
                  <td>{cam.name}</td>
                  <td>{cam.location}</td>
                  <td>{cam.status}</td>
                  <td><button className={styles.deleteBtn} onClick={() => handleDeleteCamera(cam.cameraId)}>삭제</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AdminPage;
