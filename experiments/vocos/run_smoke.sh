#!/bin/bash
# A tiny corpus and a few training steps: shapes, time per step, memory.
set -euo pipefail
o=experiments/vocos/out/smoke; mkdir -p $o
py=.venv/bin/python
$py experiments/vocos/make_corpus.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt $o/train train 20
$py experiments/vocos/make_corpus.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt $o/val val 4
$py experiments/vocos/train.py $o/train $o/val $o/decoder.pt 200
ls -la $o/train
