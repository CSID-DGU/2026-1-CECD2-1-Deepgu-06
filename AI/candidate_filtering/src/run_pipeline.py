import cv2
import json
import os

from candidate_pipeline import candidate_pipeline

VIDEO_PATH = "/home/deepgu/CECD2/yolo_tracking/data/Abuse007_x264.mp4"
JSON_PATH = "/home/deepgu/CECD2/yolo_tracking/results/tracking_results_0324.json"
OUTPUT_CLIP_DIR = "/home/deepgu/2026-1-CECD2-1-Deepgu-06/AI/candidate_filtering/result/clips"


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def load_frame_objects(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["frames"]


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps == 0:
        print("[WARN] FPS=0 → fallback 30")
        return 30

    return int(fps)


def save_clips(frames, clips, output_dir, fps):

    os.makedirs(output_dir, exist_ok=True)

    if len(frames) == 0:
        print("프레임 없음")
        return

    height, width = frames[0].shape[:2]

    # 코덱 안정성 위해 AVI + XVID 사용
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    print(f"[save] frame size: {(width, height)}, fps: {fps}")
    print(f"[save] total clips: {len(clips)}")

    for idx, clip in enumerate(clips):

        output_path = os.path.join(output_dir, f"clip_{idx}.avi")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[save] writer opened: {out.isOpened()} -> {output_path}")

        written = 0

        if isinstance(clip, dict) and "frames" in clip:

            print(f"[save] clip[{idx}] length:", len(clip["frames"]))

            for f_i, frame in enumerate(clip["frames"]):

                if frame is None:
                    continue

                if not hasattr(frame, "shape"):
                    print(f"[save] invalid frame at {f_i}: {type(frame)}")
                    continue

                # size mismatch 방지
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))

                # dtype 보정
                if frame.dtype != 'uint8':
                    frame = frame.astype('uint8')

                out.write(frame)
                written += 1

        else:
            print(f"[save] Unknown clip format: {type(clip)}")

        out.release()

        print(f"[save] written frames: {written}")

        if written == 0:
            print(f"[WARNING] empty clip: {output_path}")

        else:
            print(f"저장 완료: {output_path}")


def main():
    print("=== START ===")

    print("프레임 로딩 중...")
    frames = load_frames(VIDEO_PATH)
    print("[main] frame count:", len(frames))

    if frames:
        print("[main] frame shape:", frames[0].shape, frames[0].dtype)

    print("JSON 로딩 중...")
    frame_objects = load_frame_objects(JSON_PATH)
    print("[main] frame_objects sample:", list(frame_objects.keys())[:5])

    print("candidate pipeline 실행...")
    clips = candidate_pipeline(frames, frame_objects)

    print("[main] clips:", len(clips))

    fps = get_fps(VIDEO_PATH)
    print("[main] fps:", fps)

    print("clip 저장 시작...")
    save_clips(frames, clips, OUTPUT_CLIP_DIR, fps)


if __name__ == "__main__":
    main()