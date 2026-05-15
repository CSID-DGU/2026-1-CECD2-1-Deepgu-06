#!/bin/bash
# PGL-SUM UCF-Crime 학습 파이프라인
#
# 실행:
#   cd PGL-SUM
#   bash model/run_ucf.sh
#
# Step 1: UCF-Crime h5 데이터셋 준비 (ResNet-50 feature 추출)
# Step 2: PGL-SUM 학습 (split 0 기준)
# Step 3: 모델 경로 출력 (AI pipeline evaluate_selector.py에 넘겨주면 됨)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PGLSUM_DIR="$(dirname "$SCRIPT_DIR")"

H5_PATH="${PGLSUM_DIR}/data/UCF/ucf_crime_resnet50.h5"
SPLITS_PATH="${PGLSUM_DIR}/data/splits/ucf_splits.json"
MODEL_PATH="${PGLSUM_DIR}/Summaries/UCF/models/split0/best_model.pth"

# ── 설정 ───────────────────────────────────────────────────────────────

# Step 1 설정
MAX_FRAMES=300          # 비디오당 subsample frame 수
N_PER_ANOMALY=50        # 이상행동 클래스당 비디오 수 (Abuse/Assault/Fighting 각 50개)
N_NORMAL=50             # 정상 비디오 수
BATCH_SIZE_FEAT=32      # ResNet-50 inference batch size

# Step 2 설정
SPLIT_INDEX=0
N_EPOCHS=100
BATCH_SIZE_TRAIN=20     # gradient 누적 비디오 수
LR=5e-5
DEVICE="cuda:0"
SEED=42

# ── Step 1: H5 준비 ────────────────────────────────────────────────────

if [ -f "$H5_PATH" ] && [ -f "$SPLITS_PATH" ]; then
    echo "[Step 1] H5 파일이 이미 존재합니다. 건너뜁니다."
    echo "  H5    : $H5_PATH"
    echo "  Splits: $SPLITS_PATH"
else
    echo "========================================================"
    echo "[Step 1] UCF-Crime h5 데이터셋 준비"
    echo "========================================================"
    python "${SCRIPT_DIR}/prepare_ucf_h5.py" \
        --max_frames    "$MAX_FRAMES" \
        --n_per_anomaly "$N_PER_ANOMALY" \
        --n_normal      "$N_NORMAL" \
        --batch_size    "$BATCH_SIZE_FEAT" \
        --seed          "$SEED"
    echo ""
fi

# ── Step 2: PGL-SUM 학습 ───────────────────────────────────────────────

if [ -f "$MODEL_PATH" ]; then
    echo "[Step 2] 학습된 모델이 이미 존재합니다. 건너뜁니다."
    echo "  Model: $MODEL_PATH"
else
    echo "========================================================"
    echo "[Step 2] PGL-SUM 학습 (UCF-Crime, input_size=2048)"
    echo "========================================================"
    python "${SCRIPT_DIR}/train_ucf.py" \
        --split_index   "$SPLIT_INDEX" \
        --input_size    2048 \
        --n_epochs      "$N_EPOCHS" \
        --batch_size    "$BATCH_SIZE_TRAIN" \
        --lr            "$LR" \
        --device        "$DEVICE" \
        --seed          "$SEED"
    echo ""
fi

# ── 완료 메시지 ────────────────────────────────────────────────────────

echo "========================================================"
echo "PGL-SUM 학습 완료!"
echo "========================================================"
echo ""
echo "학습된 모델 경로:"
echo "  $MODEL_PATH"
echo ""
echo "AI pipeline 3-way 비교 실험 실행 방법:"
echo ""
echo "  cd AI/ai_pipeline"
echo "  python scripts/evaluate_selector.py \\"
echo "    --test_json      outputs/training_data/test.json \\"
echo "    --feature_dir    outputs/training_data/features \\"
echo "    --phase2_model   outputs/frame_selector.pth \\"
echo "    --pglsum_model   ../../PGL-SUM/Summaries/UCF/models/split0/best_model.pth \\"
echo "    --pglsum_input_size 2048"
