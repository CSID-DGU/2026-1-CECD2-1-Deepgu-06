"""
VLM description 검사 스크립트

C1/C2/C3 실험 완료 후 실행. 흥미로운 케이스(FP 많은 영상, FN 영상, 실험 간 결과 불일치)를
선별해서 save_run_artifacts=True로 재실행하고, VLM이 실제로 뭐라고 했는지 확인용 MD 리포트 생성.

사용법:
    cd AI/slowfast
    python scripts/inspect_vlm_descriptions.py \
        --c1 outputs/eval/keyframe_vlm/internvl4b_vlmlabel_uniform/results.json \
        --c2 outputs/eval/keyframe_vlm/internvl4b_vlmlabel_4f/results.json \
        --c3 outputs/eval/keyframe_vlm/internvl4b_vlmlabel_6f/results.json \
        --config configs/exp_C1_internvl4b_uniform.yaml \
        --output-dir outputs/eval/keyframe_vlm/inspection \
        --top-n 15
"""

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# 흥미로운 영상 선별
# ──────────────────────────────────────────────

def select_interesting_videos(results_list, top_n=15):
    """
    C1/C2/C3 결과에서 흥미로운 video_id를 선별.
    - FP 많은 영상 (VLM이 없는 걸 있다고 함)
    - FN만 있는 영상 (VLM이 있는 걸 놓침)
    - 실험 간 결과 불일치 영상 (C1 vs C2/C3 판정이 다른 것)
    """
    # video_id → 실험별 metrics 매핑
    video_data = {}
    for exp_name, result in results_list:
        for v in result.get("per_video", []):
            vid = v["video_id"]
            if vid not in video_data:
                video_data[vid] = {}
            video_data[vid][exp_name] = {
                "fp": v["false_positive"],
                "fn": v["false_negative"],
                "tp": v["true_positive"],
                "f1": v["event_f1"],
                "pred": v["predicted_event_count"],
                "gt": v["gt_event_count"],
            }

    # 점수 계산: FP 많은 영상 + FN만 있는 영상 + 불일치 영상
    scored = []
    for vid, exps in video_data.items():
        # C1 기준 FP, FN
        c1 = exps.get(results_list[0][0], {})
        fp = c1.get("fp", 0)
        fn = c1.get("fn", 0)
        tp = c1.get("tp", 0)

        # 실험 간 F1 불일치: max-min 차이
        f1_vals = [e.get("f1", 0) for e in exps.values()]
        f1_spread = max(f1_vals) - min(f1_vals) if len(f1_vals) > 1 else 0

        # 점수: FP 많을수록, FN만 있을수록, 불일치 클수록 우선
        score = fp * 2 + (fn if tp == 0 else 0) + f1_spread * 3
        category = []
        if fp >= 2:
            category.append(f"FP={fp}")
        if fn > 0 and tp == 0:
            category.append(f"FN-only(fn={fn})")
        if f1_spread > 0.15:
            category.append(f"실험불일치(spread={f1_spread:.2f})")
        if not category:
            category.append(f"tp={tp},fp={fp},fn={fn}")

        scored.append({
            "video_id": vid,
            "score": score,
            "category": ", ".join(category),
            "fp": fp, "fn": fn, "tp": tp,
            "f1_spread": f1_spread,
            "exps": exps,
        })

    scored.sort(key=lambda x: -x["score"])
    selected = scored[:top_n]
    print(f"\n[선별] 총 {len(video_data)}개 영상 중 {len(selected)}개 선택")
    for s in selected:
        print(f"  {s['video_id']:20s} score={s['score']:.2f}  {s['category']}")
    return [s["video_id"] for s in selected], scored


# ──────────────────────────────────────────────
# 재실행 (save_run_artifacts=True)
# ──────────────────────────────────────────────

def run_with_artifacts(config_path, video_ids, output_dir, gt_json, dataset_root):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "results.json"
    cmd = [
        sys.executable, "scripts/evaluate_event_batch.py",
        "--config", config_path,
        "--ground-truth-json", gt_json,
        "--dataset-root", dataset_root,
        "--output-json", str(results_json),
        "--run-prefix", "inspect",
        "--video-ids", *video_ids,
        # --summary-only 없음 → save_run_artifacts 활성화
    ]
    print(f"\n[재실행] {len(video_ids)}개 영상, 결과 → {output_dir}")
    print(f"  cmd: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    return results_json


# ──────────────────────────────────────────────
# events.json 파싱 → descriptions 수집
# ──────────────────────────────────────────────

def collect_descriptions(output_dir, video_ids):
    """
    각 video의 output 디렉토리에서 events.json을 읽어 VLM 응답 수집.
    """
    output_dir = Path(output_dir)
    descriptions = {}

    for vid in video_ids:
        # evaluate_event_batch는 run_name = f"{run_prefix}_{video_id}" 로 output 저장
        # config의 output_root 하위에 run_name/ 디렉토리 생성
        candidate_dirs = list(output_dir.glob(f"*{vid}*")) + list(
            (output_dir.parent).glob(f"**/*{vid}/events.json")
        )

        # config의 output_root 아래서 직접 찾기
        events_paths = list(output_dir.rglob(f"*{vid}*/events.json"))
        if not events_paths:
            # config output_root가 다른 경로일 수 있으므로 상위까지 탐색
            events_paths = list(Path("outputs/eval/keyframe_vlm").rglob(f"inspect_{vid}/events.json"))

        if not events_paths:
            descriptions[vid] = [{"error": "events.json not found"}]
            continue

        events = load_json(events_paths[0])
        vid_descriptions = []
        for ev in events:
            vid_descriptions.append({
                "start_sec": ev.get("start_sec"),
                "end_sec": ev.get("end_sec"),
                "fast_peak_score": ev.get("fast_peak_score"),
                "vlm_score": ev.get("vlm_score"),
                "vlm_decision": ev.get("vlm_decision"),
                "vlm_raw": ev.get("vlm_raw", ""),
            })
        descriptions[vid] = vid_descriptions

    return descriptions


# ──────────────────────────────────────────────
# 마크다운 리포트 생성
# ──────────────────────────────────────────────

def build_report(descriptions, scored_meta, output_dir, results_list):
    lines = []
    lines.append("# VLM Description 검사 리포트\n")
    lines.append(f"재실행 config: C1 (InternVL2-4B, 균등 6장)  \n")
    lines.append(f"선별 영상 수: {len(descriptions)}개\n")
    lines.append("")

    # 실험 결과 요약 테이블
    lines.append("## 실험 요약 (선별 영상 기준)\n")
    lines.append("| video_id | 선별 이유 | C1 tp/fp/fn | C2 tp/fp/fn | C3 tp/fp/fn |")
    lines.append("|---|---|---|---|---|")
    meta_map = {s["video_id"]: s for s in scored_meta}
    for vid in descriptions:
        m = meta_map.get(vid, {})
        exps = m.get("exps", {})
        def fmt(name):
            e = exps.get(name, {})
            return f"{e.get('tp','-')}/{e.get('fp','-')}/{e.get('fn','-')}"
        names = [r[0] for r in results_list]
        cols = [fmt(n) for n in names]
        while len(cols) < 3:
            cols.append("N/A")
        lines.append(f"| {vid} | {m.get('category','')} | {cols[0]} | {cols[1]} | {cols[2]} |")
    lines.append("")

    # 영상별 VLM 응답 상세
    lines.append("## 영상별 VLM 응답 상세\n")
    for vid, events in descriptions.items():
        m = meta_map.get(vid, {})
        lines.append(f"### {vid}  _{m.get('category', '')}_\n")
        if not events:
            lines.append("_이벤트 없음_\n")
            continue
        for i, ev in enumerate(events, 1):
            if "error" in ev:
                lines.append(f"- **오류**: {ev['error']}\n")
                continue
            start = ev.get("start_sec", "?")
            end = ev.get("end_sec", "?")
            fast = ev.get("fast_peak_score")
            vlm_s = ev.get("vlm_score")
            decision = ev.get("vlm_decision", "?")
            raw = ev.get("vlm_raw", "").strip()

            lines.append(f"**Event {i}** | {start:.1f}s ~ {end:.1f}s | "
                         f"fast={fast:.3f} | vlm_score={vlm_s if vlm_s is not None else 'N/A'} | "
                         f"decision=**{decision}**\n")
            if raw:
                lines.append("```")
                lines.append(raw[:1000] + ("..." if len(raw) > 1000 else ""))
                lines.append("```")
            lines.append("")

    report_path = Path(output_dir) / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[리포트] {report_path}")
    return report_path


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM description 검사 리포트 생성")
    parser.add_argument("--c1", required=True, help="C1 results.json 경로")
    parser.add_argument("--c2", default=None, help="C2 results.json 경로 (선택)")
    parser.add_argument("--c3", default=None, help="C3 results.json 경로 (선택)")
    parser.add_argument("--config", required=True, help="재실행에 사용할 실험 config (save_run_artifacts 활성화)")
    parser.add_argument("--output-dir", default="outputs/eval/keyframe_vlm/inspection")
    parser.add_argument("--gt-json", default="/home/hyrn2/github/archive_extracted/dataset/ground-truth.json")
    parser.add_argument("--dataset-root", default="/home/hyrn2/github/archive_extracted/dataset")
    parser.add_argument("--top-n", type=int, default=15, help="선별할 영상 수")
    parser.add_argument("--skip-rerun", action="store_true", help="재실행 없이 기존 artifacts로 리포트만 생성")
    args = parser.parse_args()

    results_list = [("C1", load_json(args.c1))]
    if args.c2:
        results_list.append(("C2", load_json(args.c2)))
    if args.c3:
        results_list.append(("C3", load_json(args.c3)))

    # 흥미로운 영상 선별
    selected_ids, scored_meta = select_interesting_videos(results_list, top_n=args.top_n)

    if not args.skip_rerun:
        # save_run_artifacts=True로 재실행
        run_with_artifacts(
            config_path=args.config,
            video_ids=selected_ids,
            output_dir=args.output_dir,
            gt_json=args.gt_json,
            dataset_root=args.dataset_root,
        )

    # events.json에서 VLM 응답 수집
    descriptions = collect_descriptions(args.output_dir, selected_ids)

    # 마크다운 리포트 생성
    report_path = build_report(descriptions, scored_meta, args.output_dir, results_list)
    print(f"\n완료. 리포트: {report_path}")


if __name__ == "__main__":
    main()
