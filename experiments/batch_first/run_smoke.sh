#!/bin/bash
# A few steps of each stage, for time per step and memory.
set -euo pipefail
mkdir -p experiments/batch_first/out/smoke
.venv/bin/python experiments/batch_first/train.py ckpt/data ckpt/te-audio.pt experiments/batch_first/out/smoke/frame_side.pt 5 5

