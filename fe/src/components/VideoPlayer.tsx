import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import Hls from "hls.js";

interface Props {
  hlsUrl: string;
  mediaServerUrl: string;
}

export interface VideoPlayerHandle {
  getLatency: () => number;
}

const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(({ hlsUrl, mediaServerUrl }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);

  useImperativeHandle(ref, () => ({
    getLatency: () => hlsRef.current?.latency ?? 5,
  }));

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
      hlsRef.current = hls;
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
          if (video.currentTime < liveEdge) {
            video.currentTime = liveEdge;
          }
        }
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
});

VideoPlayer.displayName = "VideoPlayer";
export default VideoPlayer;
