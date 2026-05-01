import { useEffect, useRef } from "react";
import Hls from "hls.js";

interface Props {
  hlsUrl: string;
  mediaServerUrl: string;
}

const VideoPlayer = ({ hlsUrl, mediaServerUrl }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !hlsUrl) return;

    const fullUrl = `${mediaServerUrl}${hlsUrl}`;

    if (Hls.isSupported()) {
      const hls = new Hls({
        lowLatencyMode: true,
        backBufferLength: 0,
        maxBufferLength: 4,
        maxMaxBufferLength: 8,
        liveSyncDurationCount: 2,
        liveMaxLatencyDurationCount: 4,
      });
      hls.loadSource(fullUrl);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (video.duration === Infinity || isNaN(video.duration)) {
          video.currentTime = 1e10;
        }
        video.play().catch(() => {});
      });
      hls.on(Hls.Events.LEVEL_LOADED, (_, data) => {
        if (data.details.live) {
          const liveEdge = data.details.totalduration - 2;
          if (video.currentTime < liveEdge) video.currentTime = liveEdge;
        }
      });
      return () => hls.destroy();
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = fullUrl;
      video.play().catch(() => {});
    }
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
