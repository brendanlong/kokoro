#!/bin/bash
# The v1 students again, with eval.py's loudness measure: the five predictor students, and all six.
set -euo pipefail
py=.venv/bin/python
$py experiments/lstm_swap/eval.py ckpt/data ckpt/v1.pt experiments/lstm_swap/out/v1-eval-5 predictor. 2>&1 | grep -E "RESULT|Error"
$py experiments/lstm_swap/eval.py ckpt/data ckpt/v1.pt experiments/lstm_swap/out/v1-eval-6 2>&1 | grep -E "RESULT|Error"
