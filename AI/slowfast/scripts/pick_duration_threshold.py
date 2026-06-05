"""duration-adaptive 샘플링 임계 선택 도구.

두 전체평가 결과(peak8 baseline, uniform12)의 per-event 결정을 합쳐, 가상의 하이브리드
정책 "duration >= T 면 uniform12 결정, 아니면 peak8 결정"을 임계 T별로 적용했을 때의
GT-centric TP/FP/FN/P/R/F1을 직접 계산한다(추가 VLM 호출 없음).

또한 후보 이벤트 길이 분포 + 변경(회수/손실/FP) 이벤트의 길이 위치를 출력.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_event_batch import build_gt_events, interval_iou

BASE = "outputs/eval/analyze_qwen_fp_removal_61.json"               # peak8
NEW = "outputs/eval/analyze_qwen_fp_removal_61_v3_uniform12.json"   # uniform12
GT_JSON = "/home/deepgu/test/cctv/dataset/ground-truth.json"
SCOPE = str(PROJECT_ROOT / "data/manifests/test_service_scope.json")
IOU = 0.10
THRESHOLDS = [0, 10, 12, 15, 18, 20, 22, 25, 30, 1e9]  # 0=항상uniform, inf=항상peak


def key(r):
    return (r["video_id"], r["start_frame"], r["end_frame"])


def load_map(path):
    return {key(r): r for r in json.load(open(path))["per_event_all"]}


def match(k, m):
    if k in m:
        return m[k]
    v, s, e = k
    for (bv, bs, be), r in m.items():
        if bv == v and abs(bs - s) <= 30 and abs(be - e) <= 30:
            return r
    return None


def metrics_for(decisions, gt_db, scope):
    """decisions: dict[(vid,s,e)] -> (kept(bool), gt_match(bool), s, e). GT-centric 집계."""
    by_vid = {}
    for (vid, s, e), (kp, gtm, ss, ee) in decisions.items():
        by_vid.setdefault(vid, []).append((kp, gtm, ss, ee))
    tp_gt = fp = fn = n_pred = 0
    for vid in scope:
        gts = build_gt_events(gt_db[vid]) if vid in gt_db else []
        kept = [(ss, ee) for (kp, gtm, ss, ee) in by_vid.get(vid, []) if kp]
        # GT-centric TP: 각 GT가 kept 이벤트와 IoU>=0.10 겹치면 1
        covered = 0
        for g in gts:
            if any(interval_iou(ss, ee, g["start_frame"], g["end_frame"]) >= IOU for ss, ee in kept):
                covered += 1
        tp_gt += covered
        fn += max(0, len(gts) - covered)
        n_pred += len(kept)
        fp += sum(1 for (kp, gtm, ss, ee) in by_vid.get(vid, []) if kp and not gtm)
    tp_pred = max(0, n_pred - fp)
    prec = tp_pred / max(1, n_pred)
    rec = tp_gt / max(1, tp_gt + fn)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return tp_gt, fp, fn, n_pred, prec, rec, f1


def main():
    bmap = load_map(BASE)
    nmap = load_map(NEW)
    gt_db = json.load(open(GT_JSON))["database"]
    scope = sorted(json.load(open(SCOPE))["video_ids"])

    # canonical 이벤트 = uniform12 set (fast model 동일이라 base와 일치)
    events = []
    for k, nr in nmap.items():
        br = match(k, bmap)
        dur = nr.get("duration_sec", 0.0)
        events.append({
            "key": k, "dur": dur, "gt": nr["gt_match"],
            "kp_uniform": nr["kept"],
            "kp_peak": (br["kept"] if br else nr["kept"]),
            "s": nr["start_frame"], "e": nr["end_frame"],
        })

    # ---- 길이 분포 ----
    durs = sorted(ev["dur"] for ev in events)
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 1e9)]
    print(f"[후보 이벤트 길이 분포] n={len(durs)}  중앙값={durs[len(durs)//2]:.1f}s  최대={durs[-1]:.1f}s")
    print("  bin(s)      all   GT(TP)  nonGT(FP)")
    for lo, hi in bins:
        sub = [ev for ev in events if lo <= ev["dur"] < hi]
        gtc = sum(ev["gt"] for ev in sub)
        print(f"  {lo:>3}-{(hi if hi<1e9 else '∞'):<4}  {len(sub):>4}   {gtc:>5}   {len(sub)-gtc:>6}")

    # ---- 임계별 하이브리드 시뮬레이션 ----
    print("\n[duration-adaptive 시뮬레이션]  dur>=T -> uniform12, else peak8")
    print("| T(s)  | TP | FP | FN | Prec | Rec | F1 | n_pred |")
    print("|-------|----|----|----|------|-----|----|--------|")
    rows = []
    for T in THRESHOLDS:
        dec = {}
        for ev in events:
            kp = ev["kp_uniform"] if ev["dur"] >= T else ev["kp_peak"]
            dec[ev["key"]] = (kp, ev["gt"], ev["s"], ev["e"])
        tp, fp, fn, npred, p, r, f1 = metrics_for(dec, gt_db, scope)
        tag = "∞(all peak8)" if T == 1e9 else ("0(all uniform)" if T == 0 else f"{T:g}")
        print(f"| {tag:>5} | {tp} | {fp} | {fn} | {p:.4f} | {r:.4f} | {f1:.4f} | {npred} |")
        rows.append((T, f1, p, r, tp, fp))

    best = max(rows, key=lambda x: x[1])
    bt = "∞" if best[0] == 1e9 else ("0" if best[0] == 0 else f"{best[0]:g}")
    print(f"\n  baseline peak8 F1=0.5650, uniform12 F1=0.5579")
    print(f"  >> 최적 임계 T={bt}s : F1={best[1]:.4f} (P={best[2]:.4f} R={best[3]:.4f} TP={best[4]} FP={best[5]})")

    # ---- 변경 이벤트의 길이 위치 ----
    print("\n[전환으로 달라지는 이벤트의 길이]")
    rec_tp = sorted(ev["dur"] for ev in events if ev["gt"] and not ev["kp_peak"] and ev["kp_uniform"])
    lost_tp = sorted(ev["dur"] for ev in events if ev["gt"] and ev["kp_peak"] and not ev["kp_uniform"])
    new_fp = sorted(ev["dur"] for ev in events if not ev["gt"] and not ev["kp_peak"] and ev["kp_uniform"])
    rem_fp = sorted(ev["dur"] for ev in events if not ev["gt"] and ev["kp_peak"] and not ev["kp_uniform"])
    print(f"  회수 TP 길이(s): {rec_tp}")
    print(f"  손실 TP 길이(s): {lost_tp}")
    print(f"  신규 FP 길이(s): {new_fp}")
    print(f"  제거 FP 길이(s): {rem_fp}")


if __name__ == "__main__":
    main()
