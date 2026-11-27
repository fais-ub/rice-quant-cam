#!/usr/bin/env bash
# Usage: bash scripts/run_matrix.sh splits/paddy.csv
set -euo pipefail
SPLIT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
for ARCH in mnv2 mnv3l effl0 effb0; do
  for SEED in 0 1 2; do
    RUN=runs/${ARCH}_s${SEED}
    [ -f "$RUN/best.pt" ] || python "$HERE/02_train.py" --split "$SPLIT" --arch $ARCH --seed $SEED --out runs --workers ${WORKERS:-12}
    [ -f "$RUN/onnx/int8_qdq.onnx" ] || python "$HERE/03_export_onnx.py" --run "$RUN" --split "$SPLIT"
    [ -f "$RUN/onnx/int8_qdq_pct.onnx" ] || python "$HERE/10_export_int8_pct.py" --run "$RUN" --split "$SPLIT"
    python "$HERE/04_verify_cam.py" --run "$RUN" --split "$SPLIT" --n 50
    python "$HERE/05_eval_onnx.py" --run "$RUN" --split "$SPLIT"
  done
done
