import { useEffect, useRef, useState } from "react";

interface Box {
  x: number; y: number; w: number; h: number;
  label: string; conf: number;
}

interface BboxFrame {
  wall_time: number;
  boxes: Box[];
}

interface Props {
  cameraId: string;
  getLatency: () => number;
}

const WS_BASE = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/^http/, "ws")
  : `ws://${window.location.host}`;

const BboxOverlay = ({ cameraId, getLatency }: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufferRef = useRef<BboxFrame[]>([]);
  const getLatencyRef = useRef(getLatency);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    getLatencyRef.current = getLatency;
  }, [getLatency]);

  // WebSocket: fill buffer
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/bbox/${cameraId}`);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const boxes: Box[] = data.boxes ?? [];

      if (data.wall_time !== undefined) {
        const frame: BboxFrame = { wall_time: data.wall_time, boxes };
        const buf = bufferRef.current;
        // Insert maintaining ascending wall_time order
        let i = buf.length;
        while (i > 0 && buf[i - 1].wall_time > frame.wall_time) i--;
        buf.splice(i, 0, frame);
        // Prune entries older than 15s
        const cutoff = Date.now() / 1000 - 15;
        while (buf.length > 0 && buf[0].wall_time < cutoff) buf.shift();
      } else {
        // No timestamp: treat as "now" for backward compat
        bufferRef.current = [{ wall_time: Date.now() / 1000, boxes }];
      }
    };

    return () => ws.close();
  }, [cameraId]);

  // Animation loop: find frame matching current displayed wall time
  useEffect(() => {
    let rafId: number;

    const drawBoxes = (boxes: Box[]) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      boxes.forEach((box) => {
        const x = box.x * canvas.width;
        const y = box.y * canvas.height;
        const w = box.w * canvas.width;
        const h = box.h * canvas.height;

        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        const label = `${box.label} ${(box.conf * 100).toFixed(0)}%`;
        ctx.font = "bold 12px monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(x, y - 18, tw + 8, 18);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 4, y - 4);
      });
    };

    const loop = () => {
      const buf = bufferRef.current;
      if (buf.length > 0) {
        // The frame currently shown on screen was captured at this wall time
        const displayedWallTime = Date.now() / 1000 - getLatencyRef.current();

        // Find the last buffered frame at or before the displayed wall time
        let target: BboxFrame | null = null;
        for (let i = buf.length - 1; i >= 0; i--) {
          if (buf[i].wall_time <= displayedWallTime) {
            target = buf[i];
            break;
          }
        }

        if (target) {
          drawBoxes(target.boxes);
        } else {
          // All frames are ahead of playback; clear canvas
          const canvas = canvasRef.current;
          const ctx = canvas?.getContext("2d");
          if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, []);

  return (
    <>
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        style={{
          position: "absolute", inset: 0,
          width: "100%", height: "100%",
          pointerEvents: "none", zIndex: 2,
        }}
      />
      {connected && (
        <div style={{
          position: "absolute", bottom: 8, right: 8,
          fontSize: 10, color: "#4ade80", background: "rgba(0,0,0,.5)",
          padding: "2px 6px", borderRadius: 3, zIndex: 3,
        }}>
          AI 연결됨
        </div>
      )}
    </>
  );
};

export default BboxOverlay;
