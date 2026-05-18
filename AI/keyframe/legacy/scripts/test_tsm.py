from models.tsm.inference import TSMInference
from pipeline.clip_generator import generate_clips

VIDEO_PATH = "/home/deepgu/test/data/raw_videos/Abuse007_x264.mp4"
CLIP_DIR = "/home/deepgu/test/data/clips"

MODEL_PATH = "models/tsm/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth"


def main():
    # 1. clip 생성
    clips = generate_clips(VIDEO_PATH, CLIP_DIR, save=False)

    print(f"총 clip 개수: {len(clips)}")

    # 2. TSM 로드
    tsm = TSMInference(MODEL_PATH)

    # 3. 첫 번째 clip 테스트
    clip = clips[0]

    print(f"clip frame 개수: {len(clip)}")

    # 4. inference
    probs = tsm.predict(clip)

    print("probs shape:", len(probs))

    # 5. top-5 출력
    top5 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]

    print("\nTop-5 결과:")
    for idx, score in top5:
        print(f"class index: {idx}, score: {score:.4f}")


if __name__ == "__main__":
    main()
