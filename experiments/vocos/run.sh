#!/bin/bash
# usage: run.sh OUT_DIR STEPS: the teacher corpus (once per host, in $GPUC_DATA_DIR), stage-one training,
# eval.py and listening samples.
set -euo pipefail
out=$1; mkdir -p "$out"
py=.venv/bin/python
v=experiments/vocos
corpus=$GPUC_DATA_DIR/kokoro-vocos
[ -f "$corpus/val/index.json" ] || $py $v/make_corpus.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt "$corpus/val" val 60
[ -f "$corpus/train/index.json" ] || $py $v/make_corpus.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt "$corpus/train" train 12000
$py $v/train.py "$corpus/train" "$corpus/val" "$out/decoder.pt" "$2"
$py $v/eval.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt "$out/decoder.pt" "$out/eval.json" 2>&1 | grep -E "RESULT|Error"
$py experiments/lstm_swap/listen.py "$out/listen" lstmfree=ckpt/te-audio.pt \
    "batchfirst=ckpt/te-audio.pt+ckpt/bf-v1.pt" "vocos=ckpt/te-audio.pt+ckpt/bf-v1.pt@$out/decoder.pt"
