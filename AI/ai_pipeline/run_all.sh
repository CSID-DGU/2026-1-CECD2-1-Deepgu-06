#!/bin/bash
# =============================================================================
# Keyframe Selector 전체 학습·평가 파이프라인
#
# 사용법:
#   bash run_all.sh
#
# 실행 전 설정:
#   VIDEO_DIR 에 비디오 파일을 아래 구조로 준비하세요.
#   data/videos/
#   ├── anomaly/   ← 이상행동 영상 (.mp4)  예) UCF-Crime, XD-Violence
#   └── normal/    ← 정상 영상 (.mp4)
#
# 단계:
#   Step 1. 데이터 준비  : clip 분할 + ResNet-50 feature 추출 + InternVL2 pseudo-label 생성
#   Step 2. 학습         : DifferentiableFrameSelector (FrameScorer) 학습
#   Step 3. 평가         : Phase 1 vs Phase 2 (vs PGL-SUM) 성능 비교
#
# PGL-SUM 비교:
#   PGLSUM_MODEL_PATH 를 설정하면 3-way 비교(Phase 1 / Phase 2 / PGL-SUM)를 실행합니다.
#   PGL-SUM 모델은 PGL-SUM/ 디렉토리에서 별도 학습 후 checkpoint 경로를 지정하세요.
#   예) PGLSUM_MODEL_PATH="../../PGL-SUM/outputs/pglsum_resnet50.pth"
# =============================================================================

set -e   # 오류 발생 시 즉시 중단

# -----------------------------------------------
# 설정 (필요에 따라 수정하세요)
# -----------------------------------------------
VIDEO_DIR="data/videos"          # anomaly/, normal/ 하위 디렉토리 포함
                                  # UCF-Crime: anomaly/Fighting***.mp4 등
                                  # XD-Violence: anomaly/A.Beautiful.Mind***.mp4 등
OUTPUT_DIR="outputs"
N_FRAMES=8                        # InternVL2에 넘길 frame 수
CLIP_LEN=16                       # clip 당 frame 수
CLIP_STRIDE=8                     # clip 슬라이딩 간격
EPOCHS=50
BATCH_SIZE=16
LR=0.0001
DEVICE="cuda"                     # cuda / cpu
TEST_SPLIT=0.1                    # test 비율 (비디오 단위)
SKIP_VLM_FOR_NORMAL="--skip_vlm_for_normal"
                                  # normal 비디오 VLM 생략 (속도 향상)
                                  # 모두 VLM 실행: 빈 문자열("")로 변경

# PGL-SUM 3-way 비교 설정 (비워두면 Phase 1 vs Phase 2 만 비교)
PGLSUM_MODEL_PATH=""              # PGL-SUM checkpoint 경로. 없으면 PGL-SUM 비교 skip
PGLSUM_INPUT_SIZE=2048            # PGL-SUM 입력 feature 차원 (ResNet-50=2048)
MIN_GAP=4                         # PGLSumSampler 최소 frame 간격
# -----------------------------------------------

TRAIN_DATA="$OUTPUT_DIR/training_data"
MODEL_PATH="$OUTPUT_DIR/frame_selector.pth"

echo "============================================="
echo " Keyframe Selector 학습·평가 파이프라인 시작"
echo "============================================="
echo "VIDEO_DIR  : $VIDEO_DIR"
echo "OUTPUT_DIR : $OUTPUT_DIR"
echo "N_FRAMES   : $N_FRAMES"
echo "EPOCHS     : $EPOCHS"
echo "DEVICE     : $DEVICE"
if [ -n "$PGLSUM_MODEL_PATH" ]; then
    echo "PGLSUM     : $PGLSUM_MODEL_PATH (3-way 비교)"
else
    echo "PGLSUM     : 없음 (Phase 1 vs Phase 2 만 비교)"
fi
echo ""

# -----------------------------------------------
# Step 1. 데이터 준비
# -----------------------------------------------
echo "---------------------------------------------"
echo "Step 1. 데이터 준비 (feature 추출 + pseudo-label 생성)"
echo "---------------------------------------------"

if [ -f "$TRAIN_DATA/train.json" ] && [ -f "$TRAIN_DATA/test.json" ]; then
    echo "[SKIP] 이미 존재: $TRAIN_DATA/train.json, $TRAIN_DATA/test.json"
    echo "       재생성하려면 해당 파일을 삭제하세요."
else
    python scripts/prepare_data.py \
        --video_dir     "$VIDEO_DIR" \
        --output_dir    "$TRAIN_DATA" \
        --n_frames      "$N_FRAMES" \
        --clip_len      "$CLIP_LEN" \
        --clip_stride   "$CLIP_STRIDE" \
        --test_split    "$TEST_SPLIT" \
        --device        "$DEVICE" \
        $SKIP_VLM_FOR_NORMAL

    echo "Step 1 완료."
fi

echo ""

# -----------------------------------------------
# Step 2. 학습
# -----------------------------------------------
echo "---------------------------------------------"
echo "Step 2. FrameScorer (Phase 2) 학습"
echo "---------------------------------------------"

if [ -f "$MODEL_PATH" ]; then
    echo "[SKIP] 이미 존재: $MODEL_PATH"
    echo "       재학습하려면 해당 파일을 삭제하세요."
else
    python scripts/train_frame_selector.py \
        --label_path    "$TRAIN_DATA/train.json" \
        --save_path     "$MODEL_PATH" \
        --n_frames      "$N_FRAMES" \
        --epochs        "$EPOCHS" \
        --lr            "$LR" \
        --batch_size    "$BATCH_SIZE" \
        --device        "$DEVICE"

    echo "Step 2 완료."
fi

echo ""

# -----------------------------------------------
# Step 3. 평가
# -----------------------------------------------
echo "---------------------------------------------"
if [ -n "$PGLSUM_MODEL_PATH" ]; then
    echo "Step 3. 성능 평가 (Phase 1 vs Phase 2 vs PGL-SUM, 3-way 비교)"
else
    echo "Step 3. 성능 평가 (Phase 1 vs Phase 2)"
fi
echo "---------------------------------------------"

PGLSUM_ARGS=""
if [ -n "$PGLSUM_MODEL_PATH" ]; then
    PGLSUM_ARGS="--pglsum_model_path $PGLSUM_MODEL_PATH \
                 --pglsum_input_size $PGLSUM_INPUT_SIZE \
                 --min_gap $MIN_GAP"
fi

python scripts/evaluate_selector.py \
    --label_path    "$TRAIN_DATA/test.json" \
    --model_path    "$MODEL_PATH" \
    --n_frames      "$N_FRAMES" \
    --device        "$DEVICE" \
    $PGLSUM_ARGS

echo ""
echo "============================================="
echo " 전체 파이프라인 완료"
echo " Phase 2 모델 : $MODEL_PATH"
echo " 평가 결과    : ${MODEL_PATH%.pth}.eval.json"
echo " 학습 로그    : ${MODEL_PATH%.pth}.log.json"
echo "============================================="
echo ""
echo "Phase 2 전환 방법 (main_pipeline.py 등에서):"
echo "  from pipeline.sampler import KeyframeSampler"
echo "  sampler = KeyframeSampler(n_frames=$N_FRAMES, model_path='$MODEL_PATH')"
echo ""
if [ -n "$PGLSUM_MODEL_PATH" ]; then
    echo "PGL-SUM 방식 사용 방법:"
    echo "  from pipeline.sampler import PGLSumSampler"
    echo "  sampler = PGLSumSampler(model_path='$PGLSUM_MODEL_PATH', n_frames=$N_FRAMES)"
fi
