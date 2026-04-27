import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { register } from "../api/auth";

const SignupPage = () => {
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (form.password !== form.confirm) { setError("비밀번호가 일치하지 않습니다."); return; }
    setLoading(true);
    try {
      await register({ email: form.email, password: form.password, name: form.name });
      setDone(true);
    } catch (err: any) {
      const code = err.response?.data?.code;
      if (code === "EMAIL_ALREADY_EXISTS") setError("이미 사용 중인 이메일입니다.");
      else setError(err.response?.data?.detail || "오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  if (done) {
    return (
      <div className="auth-page">
        <div className="auth-card" style={{ textAlign: "center" }}>
          <div style={{ width: 48, height: 48, borderRadius: "50%", background: "var(--ok-soft)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--ok)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          </div>
          <h2 className="auth-title">가입 신청 완료</h2>
          <p className="auth-sub" style={{ marginBottom: 24 }}>관리자 승인 후 로그인하실 수 있습니다.</p>
          <button className="btn primary block" onClick={() => navigate("/login")}>로그인 페이지로</button>
        </div>
      </div>
    );
  }

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

        <h1 className="auth-title">회원가입</h1>
        <p className="auth-sub">계정 정보를 입력해 주세요</p>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <label className="label">이름</label>
            <input className="input" name="name" type="text" placeholder="이름 입력" value={form.name}
              onChange={handleChange} required autoFocus />
          </div>
          <div>
            <label className="label">이메일</label>
            <input className="input" name="email" type="email" placeholder="이메일 입력" value={form.email}
              onChange={handleChange} required />
          </div>
          <div>
            <label className="label">비밀번호</label>
            <input className="input" name="password" type="password" placeholder="••••••••" value={form.password}
              onChange={handleChange} required />
          </div>
          <div>
            <label className="label">비밀번호 확인</label>
            <input className="input" name="confirm" type="password" placeholder="••••••••" value={form.confirm}
              onChange={handleChange} required />
          </div>

          {error && (
            <div style={{ padding: "10px 12px", background: "var(--danger-soft)", borderRadius: 8, color: "var(--danger-ink)", fontSize: 13 }}>
              {error}
            </div>
          )}

          <button className="btn primary block lg" type="submit" disabled={loading} style={{ marginTop: 4 }}>
            {loading ? "처리 중..." : "가입 신청"}
          </button>
        </form>

        <div className="auth-foot">
          이미 계정이 있으신가요? <Link to="/login">로그인</Link>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
