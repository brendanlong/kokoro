#!/bin/bash
# usage: run.sh OUT_DIR LOCAL_STEPS CHAIN_STEPS: train.py from ckpt/te-audio.pt's text side, then eval.py and
# listening samples (original, LSTM-free, and the batch-first student).
set -euo pipefail
out=$1; mkdir -p "$out"
py=.venv/bin/python
d=experiments/batch_first
$py $d/train.py ckpt/data ckpt/te-audio.pt "$out/frame_side.pt" "$2" "$3"
$py $d/eval.py ckpt/data ckpt/te-audio.pt "$out/frame_side.pt" "$out/eval.json" 2>&1 | grep -E "RESULT|Error"
$py experiments/lstm_swap/listen.py "$out/listen" lstmfree=ckpt/te-audio.pt "batchfirst=ckpt/te-audio.pt+$out/frame_side.pt"
