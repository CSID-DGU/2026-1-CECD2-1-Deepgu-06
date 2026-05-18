"""
X3D-S feature 추출 진행률 확인 스크립트.

사용법:
  python scripts/check_progress.py
  python scripts/check_progress.py --log outputs/logs/x3d_extract.log
"""

import re
import argparse
from pathlib import Path

DEFAULT_LOG = Path(__file__).parent.parent / "outputs/logs/x3d_extract.log"


def parse_log(log_path: Path):
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    # [v_idx/total] 패턴
    video_matches = re.findall(r"\[(\d+)/(\d+)\]", text)
    # 누적 clip 라인: clips=N  누적 clip=M (anomaly=A)  Xs/video  ETA H:MM:SS
    stat_matches  = re.findall(
        r"clips=(\d+)\s+누적 clip=(\d+) \(anomaly=(\d+)\)\s+([\d.]+)s/video\s+ETA ([\d:]+)",
        text
    )
    # 타임스탬프
    timestamps = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", text)

    if not video_matches or not stat_matches:
        print("아직 진행 데이터가 없습니다.")
        return

    cur_v, total_v = map(int, video_matches[-1])
    last_clips, total_clips, anom_clips, spv, eta = stat_matches[-1]
    total_clips = int(total_clips)
    anom_clips  = int(anom_clips)
    last_ts     = timestamps[-1] if timestamps else "-"

    pct = cur_v / int(total_v) * 100
    bar_len = 30
    filled  = int(bar_len * cur_v / int(total_v))
    bar     = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'='*50}")
    print(f"  X3D-S Feature 추출 진행률")
    print(f"{'='*50}")
    print(f"  [{bar}] {pct:.1f}%")
    print(f"  비디오  : {cur_v:,} / {total_v} 개")
    print(f"  클립    : {total_clips:,} 개")
    print(f"    anomaly : {anom_clips:,}  |  normal : {total_clips - anom_clips:,}")
    print(f"  속도    : {spv} s/video")
    print(f"  ETA     : {eta}")
    print(f"  마지막  : {last_ts}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"로그 파일 없음: {log_path}")
    else:
        parse_log(log_path)
