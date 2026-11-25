#!/bin/bash
# usage: run_te.sh OUT_DIR MODE STEPS [ARCH_JSON]: train_te.py from ckpt/v1.pt, then eval.py on all six students.
set -euo pipefail
out=$1; mkdir -p "$out"
py=.venv/bin/python
$py experiments/lstm_swap/train_te.py ckpt/data ckpt/v1.pt "$out/students.pt" "$2" "$3" "${4:-{\}}"
$py experiments/lstm_swap/eval.py ckpt/data "$out/students.pt" "$out/eval" 2>&1 | grep -E "RESULT|Error"
