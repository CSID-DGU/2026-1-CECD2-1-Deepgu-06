from pipeline.main_pipeline import run_pipeline

video_path = "/home/deepgu/test/data/raw_videos/Abuse007_x264.mp4"
clip_dir = "/home/deepgu/test/data/clips"

results = run_pipeline(video_path, clip_dir)

print(f"최종 candidate 수: {len(results)}")

for r in results[:5]:
    print("clip_id:", r["clip_id"])
    print("num_sampled_frames:", len(r["sampled_frames"]))
    print("frame shape:", r["sampled_frames"][0].shape)
    print("VLM output:", r["vlm_output"])
    print("-" * 50)

"""
for r in results[:5]:
    print("clip_id:", r["clip_id"])
    print("num_sampled_frames:", len(r["sampled_frames"]))

    # 추가 디버깅
    print("frame shape:", r["sampled_frames"][0].shape)
    print("-" * 30)
"""