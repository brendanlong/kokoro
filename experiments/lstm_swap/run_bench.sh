#!/bin/bash
# usage: run_bench.sh OUT_DIR [STUDENTS_PT] [MODE...]
set -uo pipefail
out=$1; students=${2:-}; shift 2 || shift $#
modes=${*:-original-eager seq-lstm-eager seq-conv-eager original-compiled seq-lstm-compiled seq-conv-compiled}
for mode in $modes; do
  .venv/bin/python experiments/lstm_swap/bench.py "$out/$mode.json" "$mode" $students 2>&1 | grep -E "RESULT|Error|error" | tail -5
done
