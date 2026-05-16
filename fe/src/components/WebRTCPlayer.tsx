import { useEffect, useRef } from "react";

interface Props {
  whepUrl: string;
}

const WebRTCPlayer = ({ whepUrl }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);

  useEffect(() => {
    if (!whepUrl || !videoRef.current) return;

    const pc = new RTCPeerConnection({ iceServers: [] });
    pcRef.current = pc;

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    pc.ontrack = (e) => {
      if (videoRef.current) videoRef.current.srcObject = e.streams[0];
    };

    (async () => {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === "complete") return resolve();
        const handler = () => {
          if (pc.iceGatheringState === "complete") {
            pc.removeEventListener("icegatheringstatechange", handler);
            resolve();
          }
        };
        pc.addEventListener("icegatheringstatechange", handler);
      });

      try {
        const res = await fetch(whepUrl, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: pc.localDescription?.sdp,
        });
        if (!res.ok) throw new Error(`WHEP ${res.status}`);
        const sdp = await res.text();
        await pc.setRemoteDescription({ type: "answer", sdp });
        videoRef.current?.play().catch(() => {});
      } catch (err) {
        console.error("[WebRTCPlayer] failed:", err);
      }
    })();

    return () => {
      pc.close();
      pcRef.current = null;
    };
  }, [whepUrl]);

  return (
    <video
      ref={videoRef}
      style={{ width: "100%", height: "100%", objectFit: "contain", background: "#000" }}
      autoPlay
      muted
      playsInline
    />
  );
};

export default WebRTCPlayer;
