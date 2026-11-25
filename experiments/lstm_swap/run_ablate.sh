#!/bin/bash
# usage: run_ablate.sh STUDENTS_PT OUT_DIR
set -euo pipefail
for only in text_encoder.lstm predictor.text_encoder,predictor.lstm predictor.shared; do
  .venv/bin/python experiments/lstm_swap/eval.py "$GPUC_DATA_DIR/kokoro-lstm-swap" "$1" "$2/$only" "$only" 2>&1 | grep -E "RESULT|Error" | sed "s/^/$only /"
done
