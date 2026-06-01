"""
Step 3/4: merge 해결 후보 3종 비교.
  A. split_threshold 상승 (gap=>=2 consecutive clips < T 를 잘라냄)  T=0.40/0.42/0.44
  B. max_event_duration (D 초과 시 최저점에서 재귀 강제 분할)        D=60/45/30
  C. score valley split (내부 local minimum < V 에서 분할)          V=0.55/0.50/0.45
각 방법: TP/tp_pred/FP/FN/P/R/F1 + 22개 merge GT 중 회수 수.
focal, start=0.48 end=0.36, GT-centric precision(B), IoU=0.10.
"""
import json, math, sys
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.event_builder import build_events
from copy import deepcopy

CSV = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips_service_scope.csv"
SERVICE = PROJECT_ROOT / "data/manifests/test_service_scope.json"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
SCORES = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"
START, END, IOU, MINDUR = 0.48, 0.36, 0.10, 0.5

def iou(a0,a1,b0,b1):
    l=max(a0,b0); r=min(a1,b1); it=max(0,r-l+1); uni=(a1-a0+1)+(b1-b0+1)-it
    return it/uni if uni>0 else 0.0
def isect(a0,a1,b0,b1): return max(0,min(a1,b1)-max(a0,b0)+1)
def s2f(s,e,fps,nbf):
    sf=max(0,int(math.floor(s*fps))); ef=min(int(nbf)-1,int(math.ceil(e*fps))-1); return sf,max(sf,ef)

sids=sorted(json.loads(SERVICE.read_text())["video_ids"])
smap={r["clip_id"]:r["fighting_prob"] for r in json.loads(SCORES.read_text())["rows"]}
df=pd.read_csv(CSV); df=df[df.video_id.isin(sids)].copy()
vm=df.groupby("video_id").first()[["fps","nb_frames"]].to_dict("index")
gtdb=json.loads(GT_JSON.read_text())["database"]
TH={"start_score":START,"end_score":END,"min_event_duration_sec":0.0,"mean_score_threshold":0.0,
    "split":{"enabled":False},"score_smoothing":{"enabled":True,"window_size":3,"method":"moving_average"}}

# 비디오별: base events(분할 전) clip_seq + gt
videos=[]
for vid in sids:
    fps=float(vm[vid]["fps"]); nbf=int(vm[vid]["nb_frames"])
    grp=df[df.video_id==vid].sort_values("start_frame")
    clips=[{"clip_id":i,"start_frame":int(r.start_frame),"end_frame":int(r.end_frame),"final_score":smap.get(r.clip_id,0.0)}
           for i,(_,r) in enumerate(grp.iterrows())]
    ev,sm=build_events(clips,TH,fps)
    sm_by={int(c["clip_id"]):float(c["final_score"]) for c in sm}
    cb={int(c["clip_id"]):(c["start_frame"],c["end_frame"]) for c in clips}
    base=[]
    for e in ev:
        seq=[{"cid":c,"score":sm_by[c],"sf":cb[c][0],"ef":cb[c][1]} for c in e["clip_ids"]]
        base.append(seq)
    gts=[s2f(*ann["segment"],fps,nbf) for ann in gtdb[vid].get("annotations",[])]
    videos.append({"vid":vid,"fps":fps,"base":base,"gts":gts})

# ── 분할 전략 ──
def seg_to_event(seq):
    return {"sf":seq[0]["sf"],"ef":seq[-1]["ef"],"clips":len(seq)}

def split_A(seq,T,minc=2):
    gap=[s["score"]<T for s in seq]
    # >=minc 연속 gap 표시
    isgap=[False]*len(seq); i=0
    while i<len(seq):
        if gap[i]:
            j=i
            while j<len(seq) and gap[j]: j+=1
            if j-i>=minc:
                for k in range(i,j): isgap[k]=True
            i=j
        else: i+=1
    segs=[]; cur=[]
    for s,gp in zip(seq,isgap):
        if gp:
            if cur: segs.append(cur); cur=[]
        else: cur.append(s)
    if cur: segs.append(cur)
    return segs if segs else [seq]

def split_B(seq,D,fps):
    def dur(sq): return (sq[-1]["ef"]-sq[0]["sf"]+1)/fps
    out=[]; stack=[seq]
    while stack:
        sq=stack.pop()
        if dur(sq)<=D or len(sq)<2: out.append(sq); continue
        # 내부 최저점(양끝 제외)에서 분할
        lo=range(1,len(sq)-1) if len(sq)>2 else range(0,len(sq))
        mi=min(lo,key=lambda k:sq[k]["score"])
        stack.append(sq[:mi+1]); stack.append(sq[mi+1:])
    return out

def split_C(seq,V):
    cuts=[]
    for i in range(1,len(seq)-1):
        if seq[i]["score"]<=seq[i-1]["score"] and seq[i]["score"]<=seq[i+1]["score"] and seq[i]["score"]<V:
            cuts.append(i)
    if not cuts: return [seq]
    segs=[]; prev=0
    for c in cuts:
        segs.append(seq[prev:c+1]); prev=c+1
    if prev<len(seq): segs.append(seq[prev:])
    return [s for s in segs if s]

# merge GT 집합 (baseline 기준) 식별
def evaluate(split_fn):
    tp=tp_pred=fn=npred=0
    tp_gtset=set()
    for V in videos:
        fps=V["fps"]; events=[]
        for seq in V["base"]:
            for seg in (split_fn(seq,fps) if split_fn else [seq]):
                if not seg: continue
                if (seg[-1]["ef"]-seg[0]["sf"]+1)/fps < MINDUR: continue
                events.append((seg[0]["sf"],seg[-1]["ef"]))
        npred+=len(events)
        for gi,g in enumerate(V["gts"]):
            if any(iou(a,b,g[0],g[1])>=IOU for a,b in events):
                tp+=1; tp_gtset.add((V["vid"],gi))
            else: fn+=1
        for a,b in events:
            if any(iou(a,b,g[0],g[1])>=IOU for g in V["gts"]): tp_pred+=1
    fp=npred-tp_pred; P=tp_pred/max(1,npred); R=tp/max(1,tp+fn); F1=2*P*R/max(P+R,1e-12)
    return dict(tp=tp,tp_pred=tp_pred,fp=fp,fn=fn,npred=npred,P=P,R=R,F1=F1,gtset=tp_gtset)

base=evaluate(None)
# merge GT 22개: baseline FN이면서 merge였던 것 (Step1 결과 재식별)
merge_gts=set()
for V in videos:
    fps=V["fps"]; events=[]
    for seq in V["base"]:
        if (seq[-1]["ef"]-seq[0]["sf"]+1)/fps<MINDUR: continue
        events.append((seq[0]["sf"],seq[-1]["ef"],seq))
    span=[sum(1 for g in V["gts"] if isect(e[0],e[1],g[0],g[1])>0) for e in events]
    for gi,g in enumerate(V["gts"]):
        best=0.0; bi=-1
        for ei,e in enumerate(events):
            iv=iou(e[0],e[1],g[0],g[1])
            if iv>best: best=iv; bi=ei
        if 0<best<IOU and bi>=0 and span[bi]>=2: merge_gts.add((V["vid"],gi))

def recovered(res): return len(merge_gts & res["gtset"])

print(f"merge GT 집합 크기: {len(merge_gts)}")
print(f"\n{'method':<22}{'TP':>4}{'tpP':>5}{'FP':>5}{'FN':>4}{'P':>7}{'R':>7}{'F1':>7}{'merge회수':>9}")
def row(name,res,base):
    dr=res['R']-base['R']; df1=res['F1']-base['F1']
    print(f"{name:<22}{res['tp']:>4}{res['tp_pred']:>5}{res['fp']:>5}{res['fn']:>4}"
          f"{res['P']:>7.3f}{res['R']:>7.3f}{res['F1']:>7.3f}{recovered(res):>6}/{len(merge_gts)}"
          f"   dR={dr:+.3f} dF1={df1:+.3f}")
row("baseline(0.48/0.36)",base,base)
print("-- A. split_threshold --")
for T in [0.40,0.42,0.44]:
    row(f"A split<{T}", evaluate(lambda s,fps,T=T: split_A(s,T)), base)
print("-- B. max_event_duration --")
for D in [60,45,30]:
    row(f"B maxdur={D}s", evaluate(lambda s,fps,D=D: split_B(s,D,fps)), base)
print("-- C. valley split --")
for Vv in [0.55,0.50,0.45]:
    row(f"C valley<{Vv}", evaluate(lambda s,fps,Vv=Vv: split_C(s,Vv)), base)
print("-- B+ 조합 (maxdur 30 + smoothing off 효과는 별도) --")
