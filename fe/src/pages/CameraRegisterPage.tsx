import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCamera } from "../api/camera";
import Layout from "../components/Layout";

const steps = ["기본 정보", "스트림 설정", "확인"];

const CameraRegisterPage = () => {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState({ cameraId: "", name: "", location: "", streamKey: "", description: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError("");
  };

  const handleNext = () => {
    if (step === 0 && (!form.name || !form.location || !form.cameraId)) {
      setError("모든 필드를 입력해 주세요.");
      return;
    }
    if (step === 1 && !form.streamKey) {
      setError("스트림 키를 입력해 주세요.");
      return;
    }
    setError("");
    setStep(s => s + 1);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      await createCamera({
        cameraId: form.cameraId,
        name: form.name,
        location: form.location,
        streamKey: form.streamKey,
        description: form.description,
      });
      navigate("/cameras");
    } catch (err: any) {
      setError(err.response?.data?.detail || "카메라 등록에 실패했습니다.");
      setStep(0);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout title="카메라 등록" subtitle="새 카메라를 시스템에 등록합니다">
      <div style={{ maxWidth: 560, margin: "0 auto" }}>
        {/* Step indicator */}
        <div style={{ display: "flex", alignItems: "center", marginBottom: 28 }}>
          {steps.map((s, i) => (
            <div key={s} style={{ display: "flex", alignItems: "center", flex: i < steps.length - 1 ? 1 : undefined }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: 700,
                  background: i < step ? "var(--ok)" : i === step ? "var(--primary)" : "var(--bg-soft)",
                  color: i <= step ? "#fff" : "var(--ink-3)",
                  border: i === step ? "2px solid var(--primary)" : "2px solid transparent",
                }}>
                  {i < step ? "✓" : i + 1}
                </div>
                <span style={{ fontSize: 13, fontWeight: i === step ? 600 : 400, color: i === step ? "var(--ink)" : "var(--ink-3)" }}>
                  {s}
                </span>
              </div>
              {i < steps.length - 1 && (
                <div style={{ flex: 1, height: 1, background: i < step ? "var(--ok)" : "var(--line)", margin: "0 12px" }} />
              )}
            </div>
          ))}
        </div>

        <div className="card" style={{ padding: 28 }}>
          {error && (
            <div style={{ padding: "10px 12px", background: "var(--danger-soft)", borderRadius: 8, color: "var(--danger-ink)", fontSize: 13, marginBottom: 16 }}>
              {error}
            </div>
          )}

          {step === 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>기본 정보</h3>
              <div>
                <label className="label">카메라 ID <span style={{ color: "var(--danger)" }}>*</span></label>
                <input className="input" name="cameraId" placeholder="예: CAM-001" value={form.cameraId} onChange={handleChange} />
                <div style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 4 }}>고유한 카메라 식별자 (영문, 숫자, 하이픈)</div>
              </div>
              <div>
                <label className="label">카메라명 <span style={{ color: "var(--danger)" }}>*</span></label>
                <input className="input" name="name" placeholder="예: 정문 카메라" value={form.name} onChange={handleChange} />
              </div>
              <div>
                <label className="label">설치 위치 <span style={{ color: "var(--danger)" }}>*</span></label>
                <input className="input" name="location" placeholder="예: 1층 출입구" value={form.location} onChange={handleChange} />
              </div>
              <div>
                <label className="label">설명</label>
                <input className="input" name="description" placeholder="카메라에 대한 설명 (선택)" value={form.description} onChange={handleChange} />
              </div>
            </div>
          )}

          {step === 1 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>스트림 설정</h3>
              <div>
                <label className="label">스트림 키 <span style={{ color: "var(--danger)" }}>*</span></label>
                <input className="input mono" name="streamKey" placeholder="스트림 키 입력" value={form.streamKey} onChange={handleChange} />
                <div style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 4 }}>RTMP 스트림 키 — OBS 또는 카메라 설정에서 확인하세요</div>
              </div>
              <div style={{ padding: 14, background: "var(--bg-soft)", borderRadius: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-2)", marginBottom: 6 }}>RTMP 서버 주소</div>
                <div className="mono" style={{ fontSize: 12, color: "var(--primary-ink)" }}>rtmp://your-server/live</div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>등록 확인</h3>
              <div style={{ background: "var(--bg-soft)", borderRadius: 8, overflow: "hidden" }}>
                {[
                  ["카메라 ID", form.cameraId],
                  ["카메라명", form.name],
                  ["설치 위치", form.location],
                  ["스트림 키", form.streamKey.replace(/./g, "•")],
                  ["설명", form.description || "—"],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", padding: "10px 14px", borderBottom: "1px solid var(--line-soft)" }}>
                    <div style={{ width: 120, fontSize: 12, color: "var(--ink-3)", flexShrink: 0 }}>{k}</div>
                    <div style={{ fontSize: 13, color: "var(--ink)", fontWeight: 500 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: 10, marginTop: 24, justifyContent: "flex-end" }}>
            {step > 0 && (
              <button className="btn" onClick={() => setStep(s => s - 1)} disabled={loading}>이전</button>
            )}
            <button className="btn ghost" onClick={() => navigate("/cameras")} disabled={loading}>취소</button>
            {step < steps.length - 1 ? (
              <button className="btn primary" onClick={handleNext}>다음</button>
            ) : (
              <button className="btn primary" onClick={handleSubmit} disabled={loading}>
                {loading ? "등록 중..." : "카메라 등록"}
              </button>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default CameraRegisterPage;
