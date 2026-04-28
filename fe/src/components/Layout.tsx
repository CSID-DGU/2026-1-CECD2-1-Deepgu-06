import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

interface Props {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
  topbarRight?: React.ReactNode;
}

const VideoIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/>
  </svg>
);
const LogIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
  </svg>
);
const CameraIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>
  </svg>
);
const UsersIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
  </svg>
);
const DashIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
  </svg>
);
const LogoutIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
  </svg>
);

const Layout = ({ children, title, subtitle, topbarRight }: Props) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const initials = user?.id ? String(user.id).slice(0, 2) : "U";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">
            <div className="logo-mark" />
            <div>
              <div className="logo-wordmark">DeepGu</div>
              <div className="logo-sub">v1.0 · 이상행동 감지</div>
            </div>
          </div>
        </div>

        <nav className="nav-section">
          <div className="section-label">모니터링</div>
          <NavLink to="/stream" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <span className="ico"><VideoIcon /></span>실시간 스트리밍
          </NavLink>
          <NavLink to="/events" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <span className="ico"><LogIcon /></span>이벤트 로그
          </NavLink>

          <div className="section-label" style={{ marginTop: 8 }}>관리</div>
          <NavLink to="/cameras" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <span className="ico"><CameraIcon /></span>카메라 관리
          </NavLink>
          {user?.role === "ADMIN" && (
            <>
              <NavLink to="/users" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
                <span className="ico"><UsersIcon /></span>사용자 관리
              </NavLink>
              <NavLink to="/admin" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
                <span className="ico"><DashIcon /></span>관리자 대시보드
              </NavLink>
            </>
          )}
        </nav>

        <div className="sb-foot">
          <div className="avatar" style={{ fontSize: 10 }}>{initials}</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {user?.role === "ADMIN" ? "관리자" : "사용자"} #{user?.id}
            </div>
            <div style={{ fontSize: 11, color: "var(--ink-3)" }}>{user?.role}</div>
          </div>
          <button
            onClick={handleLogout}
            style={{ background: "none", border: "none", cursor: "pointer", color: "var(--ink-3)", padding: 4, display: "flex", alignItems: "center" }}
            title="로그아웃"
          >
            <LogoutIcon />
          </button>
        </div>
      </aside>

      <div className="content">
        <header className="topbar">
          <div>
            {subtitle && <div className="topbar-sub">{subtitle}</div>}
            <div className="topbar-title">{title}</div>
          </div>
          {topbarRight && <div style={{ display: "flex", alignItems: "center", gap: 8 }}>{topbarRight}</div>}
        </header>
        <main className="main">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
