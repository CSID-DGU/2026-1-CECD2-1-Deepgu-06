"""Fighting 탐지 결과 그래프.

fast model 점수(시계열) + event-level VLM 점수(이벤트 구간)를 한 그래프에 표시.
- 파란 실선: fast model fighting_prob (폭행 구간에서 상승)
- 초록 구간선: 각 이벤트의 VLM 판단 점수
- 회색 점선: event start threshold (0.40)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def load(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=(
        "outputs/eval/event_compare/overlap030_focal_bedrock_qwen_event_v3/verify_Fighting049"))
    ap.add_argument("--out", default="outputs/figures/fighting049_detection.png")
    ap.add_argument("--start-th", type=float, default=0.40)
    args = ap.parse_args()

    run = Path(args.run_dir)
    clips = load(run / "clip_scores.json")
    events = load(run / "events.json")
    payload = load(run / "event_payload.json")
    rej_path = run / "rejected_events.json"
    rejected = load(rej_path) if rej_path.exists() else []

    # fast model 시계열: clip 중심 시각 vs fighting_prob
    t = [(c["start_time"] + c["end_time"]) / 2.0 for c in clips]
    fast = [c["fighting_prob"] for c in clips]

    # clip_id -> (start_time, end_time) (기각 이벤트 구간 계산용)
    clip_time = {c["clip_id"]: (c["start_time"], c["end_time"]) for c in clips}

    # accept 이벤트 구간 + VLM 점수
    ev_spans = []
    for ev, pe in zip(events, payload["events"]):
        ev_spans.append((pe["start_time_sec"], pe["end_time_sec"],
                         float(ev.get("vlm_score") or 0.0)))

    # reject 이벤트 구간 + VLM 점수 (clip_ids 로 시각 환산)
    rej_spans = []
    for ev in rejected:
        ids = ev.get("clip_ids", [])
        times = [clip_time[i] for i in ids if i in clip_time]
        if not times:
            continue
        s = min(a for a, _ in times)
        e = max(b for _, b in times)
        rej_spans.append((s, e, float(ev.get("vlm_score") or 0.0)))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # 탐지(accept) 이벤트 구간 음영
    for s, e, _ in ev_spans:
        ax.axvspan(s, e, color="#ffd9b3", alpha=0.55, zorder=0)
    # 기각(reject) 이벤트 구간 음영 (회색)
    for s, e, _ in rej_spans:
        ax.axvspan(s, e, color="#bdbdbd", alpha=0.45, zorder=0)

    # fast model 점수 라인
    ax.plot(t, fast, color="#1f77b4", lw=2.2, label="Fast model score (X3D-S)",
            zorder=3)
    ax.fill_between(t, fast, color="#1f77b4", alpha=0.08, zorder=1)

    # VLM 판단 점수(accept): 이벤트 구간 위 수평선
    for s, e, v in ev_spans:
        ax.hlines(v, s, e, color="#2ca02c", lw=4, zorder=4)
        ax.plot([(s + e) / 2.0], [v], "o", color="#2ca02c", ms=7, zorder=5)
        ax.annotate(f"{v:.2f}", ((s + e) / 2.0, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#2ca02c",
                    fontweight="bold")

    # VLM 판단 점수(reject): 회색 수평선 + X 마커
    for s, e, v in rej_spans:
        ax.hlines(v, s, e, color="#555555", lw=4, zorder=4)
        ax.plot([(s + e) / 2.0], [v], "x", color="#555555", ms=9, mew=2.5, zorder=5)
        ax.annotate(f"{v:.2f}  (rejected)", ((s + e) / 2.0, v),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=9, color="#555555", fontweight="bold")

    # threshold
    ax.axhline(args.start_th, color="gray", ls="--", lw=1.3, zorder=2)
    ax.text(t[-1], args.start_th + 0.015, f"event threshold = {args.start_th:.2f}",
            ha="right", va="bottom", fontsize=9, color="gray")

    video_name = payload.get("video_id", run.name)
    ax.set_xlim(0, max(t) + 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title(f"Violence Detection Scores for {video_name}")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.2, label="Fast model score (X3D-S)"),
        Line2D([0], [0], color="#2ca02c", lw=4, marker="o",
               label="VLM confidence (accepted)"),
        Line2D([0], [0], color="#555555", lw=4, marker="x",
               label="VLM confidence (rejected)"),
        Patch(facecolor="#ffd9b3", alpha=0.55, label="Accepted event span"),
        Patch(facecolor="#bdbdbd", alpha=0.45, label="Rejected event span"),
        Line2D([0], [0], color="gray", ls="--", lw=1.3, label="Event threshold"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[done] saved -> {out}  ({len(clips)} clips, {len(ev_spans)} events)")


if __name__ == "__main__":
    main()
