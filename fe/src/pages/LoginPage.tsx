import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { login } from "../api/auth";

const LoginPage = () => {
  const { setToken } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const token = await login({ email, password });
      setToken(token);
      navigate("/stream");
    } catch (err: any) {
      const code = err.response?.data?.code;
      if (code === "PENDING_APPROVAL") setError("관리자 승인 대기 중입니다.");
      else if (code === "INVALID_CREDENTIALS") setError("이메일 또는 비밀번호가 올바르지 않습니다.");
      else setError(err.response?.data?.detail || "로그인에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="logo" style={{ marginBottom: 24 }}>
          <div className="logo-mark" />
          <div>
            <div className="logo-wordmark">DeepGu</div>
            <div className="logo-sub">이상행동 감지 시스템</div>
          </div>
        </div>

        <h1 className="auth-title">로그인</h1>
        <p className="auth-sub">계정 정보를 입력해 주세요</p>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <label className="label">이메일</label>
            <input className="input" type="email" placeholder="이메일 입력" value={email}
              onChange={e => setEmail(e.target.value)} required autoFocus />
          </div>
          <div>
            <label className="label">비밀번호</label>
            <input className="input" type="password" placeholder="••••••••" value={password}
              onChange={e => setPassword(e.target.value)} required />
          </div>

          {error && (
            <div style={{ padding: "10px 12px", background: "var(--danger-soft)", borderRadius: 8, color: "var(--danger-ink)", fontSize: 13 }}>
              {error}
            </div>
          )}

          <button className="btn primary block lg" type="submit" disabled={loading} style={{ marginTop: 4 }}>
            {loading ? "로그인 중..." : "로그인"}
          </button>
        </form>

        <div className="auth-foot">
          계정이 없으신가요? <Link to="/signup">회원가입</Link>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
