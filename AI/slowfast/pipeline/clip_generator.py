def build_sliding_clips(frames, fps, temporal_window_sec, stride_sec):
    clips = []
    window_frames = max(2, int(round(float(temporal_window_sec) * float(fps))))
    stride_frames = max(1, int(round(float(stride_sec) * float(fps))))
    if len(frames) < window_frames:
        return clips

    clip_id = 0
    for start_frame in range(0, len(frames) - window_frames + 1, stride_frames):
        end_frame = start_frame + window_frames - 1
        clip_frames = frames[start_frame : end_frame + 1]
        clips.append(
            {
                "clip_id": clip_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / float(fps),
                "end_time": (end_frame + 1) / float(fps),
                "frames": clip_frames,
            }
        )
        clip_id += 1
    return clips
