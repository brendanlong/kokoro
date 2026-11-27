#!/bin/bash
# usage: run_adv.sh OUT_DIR INIT_PT STEPS [CONFIG_JSON]: adversarial training from a stage-one decoder (run.sh's corpus),
# then eval.py and listening samples.
set -euo pipefail
out=$1; mkdir -p "$out"
py=.venv/bin/python
v=experiments/vocos
corpus=$GPUC_DATA_DIR/kokoro-vocos
$py $v/train_adv.py "$corpus/train" "$corpus/val" "$2" "$out/decoder.pt" "$3" "${4:-{\}}"
$py $v/eval.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt "$out/decoder.pt" "$out/eval.json" 2>&1 | grep -E "RESULT|Error"
$py experiments/lstm_swap/listen.py "$out/listen" "stage1=ckpt/te-audio.pt+ckpt/bf-v1.pt@$2" \
    "vocos=ckpt/te-audio.pt+ckpt/bf-v1.pt@$out/decoder.pt"
