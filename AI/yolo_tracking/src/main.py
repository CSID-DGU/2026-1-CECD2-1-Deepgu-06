import cv2
import json
import supervision as sv

from yolo_detection import YOLODetector
from tracking import Tracker

# 경로 설정
VIDEO_PATH = "/home/deepgu/CECD2/yolo_tracking/data/Abuse007_x264.mp4"
OUTPUT_VIDEO_PATH = "/home/deepgu/CECD2/yolo_tracking/results/output_tracking_0324.mp4"
OUTPUT_JSON_PATH = "/home/deepgu/CECD2/yolo_tracking/results/tracking_results_0324.json"


def main():
    detector = YOLODetector()
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(" 영상 열기 실패")
        return

    # 영상 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(" Tracking 시작")

    frame_dict = {}   # 변경 (list → dict)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # YOLO detection
        results = detector.detect(frame)

        # YOLO → supervision format
        detections = sv.Detections.from_ultralytics(results[0])

        # Tracking
        tracked = tracker.update(detections)

        objects = []

        for i in range(len(tracked.xyxy)):
            # 사람만 필터링
            if tracked.class_id[i] != 0:
                continue

            x1, y1, x2, y2 = map(int, tracked.xyxy[i])

            track_id = tracked.tracker_id[i]
            if track_id is None:
                continue

            track_id = int(track_id)
            conf = float(tracked.confidence[i])

            # 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # JSON 저장용
            objects.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": 0
            })

        # frame_id를 key로 저장
        frame_dict[str(frame_id)] = objects

        # 영상 저장
        out.write(frame)

    cap.release()
    out.release()

    # JSON 저장
    output_data = {
        "frames": frame_dict
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f" 영상 저장 완료: {OUTPUT_VIDEO_PATH}")
    print(f" JSON 저장 완료: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()