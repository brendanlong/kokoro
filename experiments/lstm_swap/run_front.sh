#!/bin/bash
# usage: run_front.sh OUT_DIR LOCAL CHAIN TE BF_LOCAL BF_CHAIN [CHECK_PER_LANG]: the distilled text side and
# F0/N predictor ("the front") on text in every language (multidata.py): the LSTM students
# (train.py), the text encoder's student fine-tuned on the audio (train_te.py), the batch-first F0/N
# predictor (batch_first/train.py), then vocos/front_check.py on the held-out multilingual chunks for
# the old (English-trained) front and the new one.
set -euo pipefail
out=$1; mkdir -p "$out"
py=.venv/bin/python
d=ckpt/data
$py experiments/lstm_swap/train.py $d "$out/students.pt" "$2" "$3"
$py experiments/lstm_swap/train_te.py $d "$out/students.pt" "$out/te-audio.pt" audio "$4"
$py experiments/batch_first/train.py $d "$out/te-audio.pt" "$out/frame_side.pt" "$5" "$6"
n=${7:-25}
$py experiments/vocos/front_check.py $d/multilingual-val.jsonl ckpt/te-audio.pt ckpt/bf-v1.pt "$out/front-check-old.json" "$n"
$py experiments/vocos/front_check.py $d/multilingual-val.jsonl "$out/te-audio.pt" "$out/frame_side.pt" "$out/front-check-new.json" "$n"
