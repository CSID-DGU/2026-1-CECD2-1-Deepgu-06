"""v3 baseline(peak8) vs v3+uniform12 비교.

- 두 분석 JSON(per_event_all)을 (video, start_frame, end_frame) 근접 매칭으로 정렬.
- 성능표(TP/FP/FN, P/R/F1) 나란히.
- baseline 대비: 새로 회수된 TP(reject→keep), 새로 증가한 FP(reject→keep & not GT),
  새로 잃은 TP(keep→reject), 제거된 FP(keep→reject) 목록.
- 결정이 바뀐 모든 이벤트 목록을 CSV로 저장.

사용: python3 scripts/compare_v3_uniform12.py
"""
import csv
import json
from pathlib import Path

BASE = "outputs/eval/analyze_qwen_fp_removal_61.json"          # v3 peak8 baseline
NEW = "outputs/eval/analyze_qwen_fp_removal_61_v3_uniform12.json"  # v3 uniform12
OUT_CSV = "outputs/eval/compare_v3_uniform12_changes.csv"


def load_events(path):
    d = json.load(open(path))
    m = {}
    for r in d["per_event_all"]:
        m[(r["video_id"], r["start_frame"], r["end_frame"])] = r
    return d, m


def match(new_key, base_map):
    """근접 매칭(±30프레임): 샘플링이 달라도 이벤트 경계는 fast model 동일이라 보통 정확히 일치."""
    if new_key in base_map:
        return base_map[new_key]
    v, s, e = new_key
    for (bv, bs, be), r in base_map.items():
        if bv == v and abs(bs - s) <= 30 and abs(be - e) <= 30:
            return r
    return None


def kept(r):
    return r.get("kept") if r else None


def main():
    bd, bmap = load_events(BASE)
    nd, nmap = load_events(NEW)

    print("=" * 64)
    print("[성능] v3 baseline(peak8) vs v3+uniform12 (61, GT-centric, IoU>=0.10)")
    print("| Setting          | TP | FP | FN | Prec | Rec | F1 |")
    print("|------------------|----|----|----|------|-----|-----|")
    for name, d in [("v3 peak8 (base)", bd), ("v3 uniform12", nd)]:
        m = d["metrics"]["fast_qwen"]
        print(f"| {name:16} | {m['tp']} | {m['fp']} | {m['fn']} | "
              f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    print(f"  TP retention: base {bd['tp_retention_rate']:.4f} -> new {nd['tp_retention_rate']:.4f}")
    print(f"  FP removal  : base {bd['fp_removal_rate']:.4f} -> new {nd['fp_removal_rate']:.4f}")

    recovered_tp, new_fp, lost_tp, removed_fp, other = [], [], [], [], []
    for key, nr in nmap.items():
        br = match(key, bmap)
        if br is None:
            other.append((key, nr, "no-baseline-match"))
            continue
        bk, nk = kept(br), kept(nr)
        if bk == nk:
            continue
        row = {
            "video_id": key[0], "start_frame": key[1], "end_frame": key[2],
            "gt_match": nr["gt_match"],
            "base_decision": br.get("vlm_decision"), "base_score": br.get("vlm_score"),
            "new_decision": nr.get("vlm_decision"), "new_score": nr.get("vlm_score"),
            "duration_sec": nr.get("duration_sec"), "fast_peak_score": nr.get("fast_peak_score"),
            "new_reasoning": (nr.get("reasoning", "") or "")[:200],
        }
        if nr["gt_match"] and not bk and nk:
            row["change"] = "RECOVERED_TP"; recovered_tp.append(row)
        elif not nr["gt_match"] and not bk and nk:
            row["change"] = "NEW_FP"; new_fp.append(row)
        elif nr["gt_match"] and bk and not nk:
            row["change"] = "LOST_TP"; lost_tp.append(row)
        elif not nr["gt_match"] and bk and not nk:
            row["change"] = "REMOVED_FP"; removed_fp.append(row)

    def show(title, rows):
        print(f"\n[{title}] {len(rows)}건")
        for r in sorted(rows, key=lambda x: -(x["duration_sec"] or 0)):
            print(f"  {r['video_id']:12} [{r['start_frame']}-{r['end_frame']}] "
                  f"{r['duration_sec']}s peak{r['fast_peak_score']} "
                  f"{r['base_decision']}({r['base_score']})->{r['new_decision']}({r['new_score']}) "
                  f"| {r['new_reasoning'][:90]}")

    show("새로 회수된 TP (reject->keep)", recovered_tp)
    show("새로 증가한 FP (reject->keep)", new_fp)
    show("새로 잃은 TP (keep->reject)", lost_tp)
    show("추가 제거된 FP (keep->reject)", removed_fp)
    if other:
        print(f"\n[경고] baseline 매칭 실패 {len(other)}건 (경계 변동)")
        for key, nr, why in other[:20]:
            print(f"  {key}  gt={nr['gt_match']} kept={nr['kept']}")

    all_rows = recovered_tp + new_fp + lost_tp + removed_fp
    cols = ["change", "video_id", "start_frame", "end_frame", "gt_match", "duration_sec",
            "fast_peak_score", "base_decision", "base_score", "new_decision", "new_score",
            "new_reasoning"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["change"], x["video_id"])):
            w.writerow({c: r.get(c) for c in cols})
    print(f"\n[summary] recovered_tp={len(recovered_tp)} new_fp={len(new_fp)} "
          f"lost_tp={len(lost_tp)} removed_fp={len(removed_fp)} | changed total={len(all_rows)}")
    print(f"[output] {OUT_CSV}")


if __name__ == "__main__":
    main()
