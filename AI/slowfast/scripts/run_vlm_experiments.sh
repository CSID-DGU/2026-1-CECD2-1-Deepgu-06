#!/bin/bash
# VLM 비교 실험 순차 실행 스크립트
# 각 실험 전후로 GPU 메모리를 완전히 해제한 뒤 다음 실험 시작

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GT_JSON="/home/hyrn2/github/archive_extracted/dataset/ground-truth.json"
DATASET_ROOT="/home/hyrn2/github/archive_extracted/dataset"
OUTPUT_BASE="$PROJECT_DIR/outputs/eval/keyframe_vlm"

cd "$PROJECT_DIR"

# GPU 메모리 완전 해제 대기 함수
# 이전 실험 프로세스 종료 후 GPU 1 메모리가 3GB 이하로 내려올 때까지 대기
wait_gpu_clear() {
    local gpu_id=${1:-1}
    local threshold_mib=${2:-3000}
    echo "[cleanup] GPU $gpu_id 메모리 해제 대기 중..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
        echo "[cleanup] GPU $gpu_id 사용 중: ${used}MiB (임계값: ${threshold_mib}MiB)"
        if [ "$used" -le "$threshold_mib" ]; then
            echo "[cleanup] GPU $gpu_id 해제 완료 (${used}MiB)"
            break
        fi
        sleep 5
    done
}

# 실험 실행 함수
run_exp() {
    local name=$1
    local config=$2
    local output_dir="$OUTPUT_BASE/$name"

    mkdir -p "$output_dir"
    echo ""
    echo "======================================================"
    echo "  실험 시작: $name"
    echo "  config:   $config"
    echo "  출력:     $output_dir"
    echo "======================================================"

    python scripts/evaluate_event_batch.py \
        --config "$config" \
        --ground-truth-json "$GT_JSON" \
        --dataset-root "$DATASET_ROOT" \
        --output-json "$output_dir/results.json" \
        --run-prefix "$name" \
        --summary-only \
        2>&1 | tee "$output_dir/run.log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] 실험 $name 실패 (exit $exit_code)"
        return $exit_code
    fi

    echo "[done] 실험 $name 완료"
    wait_gpu_clear 1 3000
    echo ""
}

# ── 실험 목록 ────────────────────────────────────────────────
# 완료된 실험: baseline, internvl8b (exp_A)
# 여기서 실행할 실험 순서: 4B → Qwen2-VL → MiniCPM-V → Phi-3.5

echo "======================================================"
echo "  VLM 비교 실험 배치 시작"
echo "  순서: exp_B (4B) → exp_C (Qwen2VL) → exp_D (MiniCPM) → exp_E (Phi)"
echo "======================================================"

# 시작 전 GPU 상태 확인
echo "[info] 시작 전 GPU 상태:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# Step 1: InternVL2-4B
if [ -d "/home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/models/internvl2_4b" ]; then
    run_exp "internvl4b" "configs/exp_B_internvl4b.yaml"
else
    echo "[skip] internvl2_4b 모델 없음 — 다운로드 완료 후 재실행 필요"
fi

# Step 2: Qwen2-VL-7B
if [ -d "/home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/models/qwen2vl_7b" ]; then
    run_exp "qwen2vl" "configs/exp_C_qwen2vl.yaml"
else
    echo "[skip] qwen2vl_7b 모델 없음"
fi

# Step 3: MiniCPM-V 2.6
if [ -d "/home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/models/minicpmv_2_6" ]; then
    run_exp "minicpmv" "configs/exp_D_minicpmv.yaml"
else
    echo "[skip] minicpmv_2_6 모델 없음 — 다운로드 완료 후 재실행 필요"
fi

# Step 4: Phi-3.5-Vision
if [ -d "/home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/models/phi35_vision" ]; then
    run_exp "phi35vision" "configs/exp_E_phi35vision.yaml"
else
    echo "[skip] phi35_vision 모델 없음 — 다운로드 완료 후 재실행 필요"
fi

echo ""
echo "======================================================"
echo "  모든 실험 완료"
echo "======================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
