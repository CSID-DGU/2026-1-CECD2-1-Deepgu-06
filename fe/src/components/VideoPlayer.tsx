import { useEffect, useRef } from "react";
import Hls from "hls.js";

interface Props {
  hlsUrl: string;
  mediaServerUrl: string;
}

const VideoPlayer = ({ hlsUrl, mediaServerUrl }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !hlsUrl) return;

    const fullUrl = `${mediaServerUrl}${hlsUrl}`;

    if (Hls.isSupported()) {
      const hls = new Hls({ lowLatencyMode: true });
      hlsRef.current = hls;
      hls.loadSource(fullUrl);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(() => {});
      });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = fullUrl;
      video.play().catch(() => {});
    }

    return () => {
      hlsRef.current?.destroy();
      hlsRef.current = null;
    };
  }, [hlsUrl, mediaServerUrl]);

  return (
    <video
      ref={videoRef}
      style={{ width: "100%", height: "100%", objectFit: "contain", background: "#000" }}
      muted
      playsInline
    />
  );
};

export default VideoPlayer;
