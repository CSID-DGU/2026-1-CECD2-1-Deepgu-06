import { useEffect, useState } from "react";

interface Notification {
  id: number;
  camera_id: string;
  anomaly_type: string;
  confidence: number;
}

interface Props {
  cameraId: string | null;
}

let _id = 0;
const DURATION = 30000;

const EventNotification = ({ cameraId }: Props) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  useEffect(() => {
    if (!cameraId) return;

    const apiBase = import.meta.env.VITE_API_URL || "";
    const token = localStorage.getItem("token") || "";
    const es = new EventSource(`${apiBase}/api/events/stream?camera_id=${encodeURIComponent(cameraId)}&token=${encodeURIComponent(token)}`);

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const note: Notification = { id: ++_id, ...data };
      setNotifications((prev) => [...prev, note]);
      setTimeout(() => {
        setNotifications((prev) => prev.filter((n) => n.id !== note.id));
      }, DURATION);
    };

    return () => es.close();
  }, [cameraId]);

  const dismiss = (id: number) =>
    setNotifications((prev) => prev.filter((n) => n.id !== id));

  if (notifications.length === 0) return null;

  return (
    <>
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(120%); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
        @keyframes timerShrink {
          from { width: 100%; }
          to   { width: 0%; }
        }
      `}</style>

      <div style={{
        position: "fixed",
        right: 24,
        bottom: 32,
        zIndex: 9999,
        display: "flex",
        flexDirection: "column-reverse",
        gap: 10,
        pointerEvents: "none",
      }}>
        {notifications.map((n) => (
          <div
            key={n.id}
            style={{
              width: 300,
              background: "var(--surface, #fff)",
              border: "1px solid var(--line-soft, #e5e7eb)",
              borderLeft: "4px solid #dc2626",
              borderRadius: 8,
              boxShadow: "0 8px 24px rgba(0,0,0,.12)",
              overflow: "hidden",
              pointerEvents: "auto",
              animation: `slideInRight .25s ease`,
            }}
          >
            {/* 헤더 */}
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "10px 12px 6px",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 14 }}>⚠</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: "#dc2626", letterSpacing: "0.02em" }}>
                  이상행동 감지
                </span>
              </div>
              <button
                onClick={() => dismiss(n.id)}
                style={{
                  background: "none", border: "none", cursor: "pointer",
                  color: "var(--ink-4, #9ca3af)", fontSize: 13, padding: 0, lineHeight: 1,
                }}
              >✕</button>
            </div>

            {/* 본문 */}
            <div style={{ padding: "0 12px 12px" }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--ink, #111)", marginBottom: 4 }}>
                {n.anomaly_type}
              </div>
              <div style={{ fontSize: 12, color: "var(--ink-3, #6b7280)", display: "flex", gap: 6 }}>
                <span>신뢰도 {Math.round(n.confidence * 100)}%</span>
                <span>·</span>
                <span>카메라 {n.camera_id}</span>
              </div>
            </div>

            {/* 타이머 바 */}
            <div style={{ height: 3, background: "#fee2e2" }}>
              <div style={{
                height: "100%",
                background: "#dc2626",
                animation: `timerShrink ${DURATION}ms linear forwards`,
              }} />
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default EventNotification;
