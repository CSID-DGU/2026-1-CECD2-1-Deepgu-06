import sys
import os

_AI_PIPELINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KEYFRAME = os.path.normpath(os.path.join(_AI_PIPELINE, "../keyframe"))
sys.path.insert(0, _AI_PIPELINE)
sys.path.insert(0, _KEYFRAME)

from pipeline.clip_generator import generate_clips

video_path = "/home/deepgu/test/data/raw_videos/Abuse007_x264.mp4"
output_dir = "/home/deepgu/test/data/clips"

clips = generate_clips(
    video_path=video_path,
    output_dir=output_dir,
    clip_len=16,
    stride=8,
    save=True
)

print(f"생성된 clip 개수: {len(clips)}")