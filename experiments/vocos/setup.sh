#!/bin/bash
# The LSTM-swap environment plus torchcrepe (metrics.py), with torchaudio pinned to its torch.
set -euo pipefail
bash experiments/lstm_swap/setup.sh
uv pip install --python .venv/bin/python torch==2.8.0 torchaudio==2.8.0 torchcrepe
