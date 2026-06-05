#!/bin/bash
# GPU1: adaptive20 || bigru_12f 동시 실행
set -e
BASE="/home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/slowfast"
GT_JSON="/home/hyrn2/github/archive_extracted/dataset/ground-truth.json"
DATASET="/home/hyrn2/github/archive_extracted/dataset"
LOG="$BASE/outputs/eval/keyframe_vlm/batch_8b_gpu1.log"

cd "$BASE"

run_exp() {
    local name=$1 config=$2
    mkdir -p "outputs/eval/keyframe_vlm/$name"
    python scripts/evaluate_event_batch.py \
        --config "$config" \
        --ground-truth-json "$GT_JSON" \
        --dataset-root "$DATASET" \
        --output-json "outputs/eval/keyframe_vlm/$name/results.json" \
        --run-prefix "$name" \
        --summary-only \
        2>&1 | tee "outputs/eval/keyframe_vlm/$name/run.log"
    echo "[$(date +%H:%M:%S)] 완료: $name" | tee -a "$LOG"
}

echo "[$(date +%H:%M:%S)] GPU1 병렬 시작: adaptive20 || bigru_12f" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | tee -a "$LOG"

run_exp "internvl8b_adaptive20"  "configs/exp_8b_adaptive20.yaml"  &
PID_A=$!
run_exp "internvl8b_bigru_12f"   "configs/exp_8b_bigru_12f.yaml"   &
PID_B=$!

wait $PID_A $PID_B
echo "[$(date +%H:%M:%S)] GPU1 모두 완료" | tee -a "$LOG"
