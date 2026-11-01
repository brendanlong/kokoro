#!/bin/bash
# usage: run_train.sh OUT_DIR LOCAL_STEPS CHAIN_STEPS [ARCH_JSON]
set -euo pipefail
out=$1; mkdir -p "$out"
data=$GPUC_DATA_DIR/kokoro-lstm-swap
py=.venv/bin/python
[ -f "$data/val.txt" ] || $py experiments/lstm_swap/prep_data.py "$data"
$py experiments/lstm_swap/train.py "$data" "$out/students.pt" "$2" "$3" "${4:-{\}}"
$py experiments/lstm_swap/eval.py "$data" "$out/students.pt" "$out/eval" 2>&1 | grep -E "RESULT|Error"
$py experiments/lstm_swap/eval.py "$data" "$out/students.pt" "$out/eval-text-encoder-only" text_encoder.lstm 2>&1 | grep -E "RESULT|Error"
bash experiments/lstm_swap/run_bench.sh "$out/bench" "$out/students.pt" original-eager original-compiled seq-conv-compiled
