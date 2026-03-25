from config import CONTEXT_WINDOW

def generate_candidate_clips(frames, segments):

    clips = []
    total_frames = len(frames)

    for start,end in segments:

        clip_start = max(0,start-CONTEXT_WINDOW)
        clip_end = min(total_frames-1,end+CONTEXT_WINDOW)

        clip_frames = frames[clip_start:clip_end+1]

        clips.append({
            "start":clip_start,
            "end":clip_end,
            "frames":clip_frames
        })

    return clips