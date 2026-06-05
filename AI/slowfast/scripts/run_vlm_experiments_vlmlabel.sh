#!/bin/bash
# InternVL2-4B + VLM pseudo-label BiGRU 실험 배치 (C1~C3)
# VLM: cuda:1 (Phi 재라벨링과 공존), BiGRU/ResNet: cuda:2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GT_JSON="/home/hyrn2/github/archive_extracted/dataset/ground-truth.json"
DATASET_ROOT="/home/hyrn2/github/archive_extracted/dataset"
OUTPUT_BASE="$PROJECT_DIR/outputs/eval/keyframe_vlm"

cd "$PROJECT_DIR"

wait_gpu_clear() {
    local gpu_id=${1:-1}
    local threshold_mib=${2:-10000}
    echo "[cleanup] GPU $gpu_id 메모리 해제 대기 중..."
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
        echo "[cleanup] GPU $gpu_id: ${used}MiB (임계값: ${threshold_mib}MiB)"
        if [ "$used" -le "$threshold_mib" ]; then
            echo "[cleanup] GPU $gpu_id 해제 완료 (${used}MiB)"
            break
        fi
        sleep 5
    done
}

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
    # Phi 재라벨링(~21GB)이 cuda:1에 상시 점유 중이므로 임계값 25000으로 설정
    wait_gpu_clear 1 25000
    echo ""
}

echo "======================================================"
echo "  VLM pseudo-label BiGRU 실험 배치 시작 (C1~C3)"
echo "  VLM: cuda:1 / BiGRU+ResNet: cuda:2"
echo "  기존 출력 디렉토리와 분리됨"
echo "======================================================"
echo "[info] 시작 전 GPU 상태:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

run_exp "internvl4b_vlmlabel_uniform" "configs/exp_C1_internvl4b_uniform.yaml"
run_exp "internvl4b_vlmlabel_4f"      "configs/exp_C2_internvl4b_vlmlabel_4f.yaml"
run_exp "internvl4b_vlmlabel_6f"      "configs/exp_C3_internvl4b_vlmlabel_6f.yaml"

echo ""
echo "======================================================"
echo "  모든 실험 완료"
echo "======================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
