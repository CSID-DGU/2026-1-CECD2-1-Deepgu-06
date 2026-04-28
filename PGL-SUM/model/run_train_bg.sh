#!/bin/bash
# PGL-SUM 학습을 백그라운드로 실행하는 스크립트
#
# 사용법:
#   bash PGL-SUM/model/run_train_bg.sh
#
# 진행 확인:
#   tail -f PGL-SUM/Summaries/UCF/logs/split0/train_progress.log
#
# 실시간 best val loss 요약:
#   python -c "
#   import json
#   log = json.load(open('PGL-SUM/Summaries/UCF/logs/split0/train_log.json'))
#   best = min(log, key=lambda x: x['val_loss'])
#   last = log[-1]
#   print(f'진행: {last[\"epoch\"]+1}epoch  train={last[\"train_loss\"]:.5f}  val={last[\"val_loss\"]:.5f}')
#   print(f'Best: epoch {best[\"epoch\"]+1}  val={best[\"val_loss\"]:.5f}')
#   "

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_FILE="$REPO_DIR/PGL-SUM/Summaries/UCF/logs/split0/train_progress.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "학습 시작: $(date)"
echo "로그 파일: $LOG_FILE"
echo "진행 확인: tail -f $LOG_FILE"
echo ""

nohup python "$REPO_DIR/PGL-SUM/model/train_ucf.py" \
    --split_index 0 \
    --input_size  2048 \
    --n_epochs    200 \
    --batch_size  20 \
    --lr          5e-5 \
    --l2_req      1e-5 \
    --clip        5.0 \
    --n_segments  4 \
    --heads       8 \
    --fusion      add \
    --pos_enc     absolute \
    --init_type   xavier \
    --device      cuda:0 \
    --seed        42 \
    >> "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo $PID > "$REPO_DIR/PGL-SUM/train.pid"
echo "PID 저장: $REPO_DIR/PGL-SUM/train.pid"
echo ""
echo "중단하려면: kill \$(cat $REPO_DIR/PGL-SUM/train.pid)"
