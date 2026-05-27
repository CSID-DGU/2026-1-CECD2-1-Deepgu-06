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

const EventNotification = ({ cameraId }: Props) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  useEffect(() => {
    if (!cameraId) return;

    const apiBase = import.meta.env.VITE_API_URL || "";
    const es = new EventSource(`${apiBase}/api/events/stream?camera_id=${encodeURIComponent(cameraId)}`);

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const note: Notification = { id: ++_id, ...data };
      setNotifications((prev) => [...prev, note]);
      setTimeout(() => {
        setNotifications((prev) => prev.filter((n) => n.id !== note.id));
      }, 6000);
    };

    return () => es.close();
  }, [cameraId]);

  if (notifications.length === 0) return null;

  return (
    <div style={{ position: "fixed", top: 16, right: 16, zIndex: 9999, display: "flex", flexDirection: "column", gap: 8 }}>
      {notifications.map((n) => (
        <div
          key={n.id}
          style={{
            background: "var(--danger-ink, #dc2626)",
            color: "#fff",
            borderRadius: 8,
            padding: "12px 16px",
            minWidth: 260,
            boxShadow: "0 4px 16px rgba(0,0,0,.25)",
            display: "flex",
            flexDirection: "column",
            gap: 4,
            animation: "slideIn .2s ease",
          }}
        >
          <div style={{ fontWeight: 700, fontSize: 13 }}>⚠ 이상행동 감지</div>
          <div style={{ fontSize: 12, opacity: 0.9 }}>
            {n.anomaly_type} &nbsp;·&nbsp; 신뢰도 {Math.round(n.confidence * 100)}%
          </div>
          <div style={{ fontSize: 11, opacity: 0.7 }}>카메라 {n.camera_id}</div>
        </div>
      ))}
    </div>
  );
};

export default EventNotification;
