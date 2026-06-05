#!/bin/bash
# GPU0: bigru_4f → bigru_16f
set -e
BASE="/home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/slowfast"
GT_JSON="/home/hyrn2/github/archive_extracted/dataset/ground-truth.json"
DATASET="/home/hyrn2/github/archive_extracted/dataset"
LOG="$BASE/outputs/eval/keyframe_vlm/batch_8b_gpu0.log"

cd "$BASE"

run_exp() {
    local name=$1 config=$2
    echo "[$(date +%H:%M:%S)] GPU0 시작: $name" | tee -a "$LOG"
    mkdir -p "outputs/eval/keyframe_vlm/$name"
    python scripts/evaluate_event_batch.py \
        --config "$config" \
        --ground-truth-json "$GT_JSON" \
        --dataset-root "$DATASET" \
        --output-json "outputs/eval/keyframe_vlm/$name/results.json" \
        --run-prefix "$name" \
        --summary-only \
        2>&1 | tee "outputs/eval/keyframe_vlm/$name/run.log"
    echo "[$(date +%H:%M:%S)] GPU0 완료: $name" | tee -a "$LOG"
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 10
}

echo "[$(date +%H:%M:%S)] GPU0 배치 시작" | tee -a "$LOG"
run_exp "internvl8b_bigru_4f"  "configs/exp_8b_bigru_4f.yaml"
run_exp "internvl8b_bigru_16f" "configs/exp_8b_bigru_16f.yaml"
echo "[$(date +%H:%M:%S)] GPU0 모두 완료" | tee -a "$LOG"
