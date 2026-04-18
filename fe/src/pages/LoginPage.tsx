import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, register } from "../api/auth";
import { useAuth } from "../context/AuthContext";
import styles from "./LoginPage.module.css";

const LoginPage = () => {
  const { setToken } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [form, setForm] = useState({ email: "", password: "", name: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [registered, setRegistered] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      if (mode === "login") {
        const token = await login({ email: form.email, password: form.password });
        setToken(token);
        navigate("/");
      } else {
        await register({ email: form.email, password: form.password, name: form.name });
        setRegistered(true);
      }
    } catch (err: any) {
      const code = err.response?.data?.code;
      if (code === "PENDING_APPROVAL") setError("관리자 승인 대기 중입니다.");
      else if (code === "INVALID_CREDENTIALS") setError("이메일 또는 비밀번호가 올바르지 않습니다.");
      else if (code === "EMAIL_ALREADY_EXISTS") setError("이미 사용 중인 이메일입니다.");
      else setError("오류가 발생했습니다. 다시 시도해주세요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>🎥</span>
          <h1 className={styles.logoText}>DEEPGU</h1>
          <p className={styles.logoSub}>이상행동 감지 시스템</p>
        </div>

        {registered ? (
          <div className={styles.successBox}>
            <p>회원가입이 완료되었습니다.</p>
            <p>관리자 승인 후 로그인할 수 있습니다.</p>
            <button className={styles.btn} onClick={() => { setRegistered(false); setMode("login"); }}>
              로그인으로 돌아가기
            </button>
          </div>
        ) : (
          <>
            <div className={styles.tabs}>
              <button className={`${styles.tab} ${mode === "login" ? styles.activeTab : ""}`}
                onClick={() => { setMode("login"); setError(""); }}>로그인</button>
              <button className={`${styles.tab} ${mode === "register" ? styles.activeTab : ""}`}
                onClick={() => { setMode("register"); setError(""); }}>회원가입</button>
            </div>
            <form onSubmit={handleSubmit} className={styles.form}>
              {mode === "register" && (
                <div className={styles.field}>
                  <label>이름</label>
                  <input name="name" type="text" placeholder="이름 입력" value={form.name} onChange={handleChange} required />
                </div>
              )}
              <div className={styles.field}>
                <label>이메일</label>
                <input name="email" type="email" placeholder="이메일 입력" value={form.email} onChange={handleChange} required />
              </div>
              <div className={styles.field}>
                <label>비밀번호</label>
                <input name="password" type="password" placeholder="비밀번호 입력" value={form.password} onChange={handleChange} required />
              </div>
              {error && <p className={styles.error}>{error}</p>}
              <button type="submit" className={styles.btn} disabled={loading}>
                {loading ? "처리 중..." : mode === "login" ? "로그인" : "회원가입"}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
};

export default LoginPage;
