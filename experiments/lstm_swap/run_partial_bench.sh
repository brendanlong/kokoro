#!/bin/bash
set -uo pipefail
out=experiments/lstm_swap/out/bench-partial
bash experiments/lstm_swap/run_bench.sh $out "" original-eager
ONLY=predictor.text_encoder,predictor.lstm,predictor.shared bash experiments/lstm_swap/run_bench.sh $out/partial $GPUC_DATA_DIR/kokoro-lstm-swap/v1/students.pt seq-conv-compiled
bash experiments/lstm_swap/run_bench.sh $out/all $GPUC_DATA_DIR/kokoro-lstm-swap/v1/students.pt seq-conv-compiled
