"""단일 영상 end-to-end 폭행 탐지 + 이벤트별 자연어 설명.

클립 추출 → X3D-S fast model → event 생성 → event-level VLM(Qwen3-VL, adaptive 샘플링)
까지 전체 파이프라인을 한 영상에 대해 실행하고, 폭행으로 판정된 구간(시간)과 각 이벤트의
VLM 판단 근거(reasoning)를 출력한다.

사용:
  python3 scripts/detect_video.py /path/to/video.mp4
  python3 scripts/detect_video.py /path/to/video.mp4 --show-rejected   # 기각 이벤트도 표시
  python3 scripts/detect_video.py /path/to/video.mp4 --config configs/other.yaml
  python3 scripts/detect_video.py /path/to/video.mp4 --json out.json    # 결과 JSON 저장
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.main_pipeline import run_single_video_pipeline
from utils.config import load_config

DEFAULT_CONFIG = "configs/eval_event_v3_start040_bedrock_qwen.yaml"  # 운영(adaptive20)


def mmss(sec):
    sec = float(sec)
    m = int(sec // 60)
    return f"{m:d}:{sec - 60 * m:05.2f}"


def parse_reasoning(raw):
    """vlm_raw(JSON 문자열)에서 reasoning/label/confidence 추출."""
    if not raw:
        return None, None, None
    try:
        j = json.loads(raw)
        return j.get("label"), j.get("confidence"), j.get("reasoning")
    except Exception:
        return None, None, str(raw)[:160]


def describe(event, fps, idx, kind):
    s, e = int(event["start_frame"]), int(event["end_frame"])
    st, en = s / fps, (e + 1) / fps
    dur = event.get("duration_sec", (e - s + 1) / fps)
    label, conf, reasoning = parse_reasoning(event.get("vlm_raw", ""))
    return {
        "idx": idx, "kind": kind,
        "start_sec": round(st, 2), "end_sec": round(en, 2),
        "start_mmss": mmss(st), "end_mmss": mmss(en),
        "duration_sec": round(float(dur), 1),
        "vlm_decision": event.get("vlm_decision"),
        "vlm_score": (round(float(event["vlm_score"]), 3)
                      if event.get("vlm_score") is not None else None),
        "fast_peak_score": round(float(event.get("fast_peak_score", 0.0)), 3),
        "vlm_label": label, "vlm_confidence": conf,
        "reasoning": reasoning,
        "sampling_used": event.get("_sampling_used"),
        "sampling_frames": event.get("_sampling_k"),
        "start_frame": s, "end_frame": e,
    }


def print_event(d):
    print(f"  [{d['idx']}] {d['start_mmss']} – {d['end_mmss']}  ({d['duration_sec']}s)")
    print(f"      decision={d['vlm_decision']}  vlm_score={d['vlm_score']}  "
          f"fast_peak={d['fast_peak_score']}  sampling={d['sampling_used']}/{d['sampling_frames']}f")
    print(f"      → {d['reasoning']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--show-rejected", action="store_true", help="기각된 이벤트도 출력")
    ap.add_argument("--show-clip-scores", action="store_true",
                    help="fast model 클립별 score 타임라인 출력")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    vpath = Path(args.video_path)
    if not vpath.exists():
        print(f"[error] 파일 없음: {vpath}")
        sys.exit(1)

    config = load_config(args.config)
    # 단일 테스트라 산출물 저장은 끄고 빠르게
    config.setdefault("outputs", {}).update(
        save_run_artifacts=False, save_event_media=False, save_clip_manifest=False)

    print(f"[run] {vpath.name}  (config={Path(args.config).name})")
    res = run_single_video_pipeline(str(vpath), config, run_name=vpath.stem, verbose=True)
    fps = float(res["manifest"]["fps"])
    kept = res["events"]
    rejected = res["rejected_events"]

    kept_d = [describe(ev, fps, i + 1, "fight") for i, ev in enumerate(
        sorted(kept, key=lambda x: x["start_frame"]))]
    rej_d = [describe(ev, fps, i + 1, "rejected") for i, ev in enumerate(
        sorted(rejected, key=lambda x: x["start_frame"]))]

    print("\n" + "=" * 64)
    print(f"  폭행 탐지 결과: {len(kept_d)}건  (기각 {len(rej_d)}건)")
    print(f"  영상 길이 {res['manifest']['num_frames'] / fps:.1f}s, fps={fps:.2f}")
    print("=" * 64)
    if kept_d:
        print("\n● 폭행으로 판정된 구간:")
        for d in kept_d:
            print_event(d)
    else:
        print("\n● 폭행으로 판정된 구간 없음.")

    if args.show_rejected and rej_d:
        print("\n○ 기각된 이벤트(후보였으나 VLM이 non_fight 판정):")
        for d in rej_d:
            print_event(d)

    clip_rows = []
    if args.show_clip_scores or args.json:
        th = config["thresholds"]
        start_s, end_s = float(th["start_score"]), float(th["end_score"])
        for c in sorted(res["clip_scores"], key=lambda x: x["start_frame"]):
            sc = float(c.get("final_score", 0.0))
            clip_rows.append({
                "clip_id": int(c["clip_id"]), "time_sec": round(c["start_frame"] / fps, 1),
                "time_mmss": mmss(c["start_frame"] / fps), "fast_score": round(sc, 3),
            })
    if args.show_clip_scores and clip_rows:
        print(f"\n■ Fast model 클립 score 타임라인 (start={config['thresholds']['start_score']}, "
              f"end={config['thresholds']['end_score']}, {len(clip_rows)}개 클립)")
        print("   (█ = >=start, ▒ = >=end, · = below)")
        for c in clip_rows:
            sc = c["fast_score"]
            mark = "█" if sc >= float(config["thresholds"]["start_score"]) else (
                "▒" if sc >= float(config["thresholds"]["end_score"]) else "·")
            bar = "▉" * int(round(sc * 30))
            print(f"   {c['time_mmss']:>7}  {sc:0.3f} {mark} {bar}")

    if args.json:
        out = {
            "video_path": str(vpath), "fps": fps,
            "num_frames": res["manifest"]["num_frames"],
            "fights": kept_d, "rejected": rej_d,
            "clip_scores": clip_rows,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
