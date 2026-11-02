#!/bin/bash
# The lock predates the Python 3.13 bump; torch 2.8 + triton 3.4 matches Kokoro-FastAPI.
set -euo pipefail
uv sync --frozen --python 3.12
uv pip install --python .venv/bin/python torch==2.8.0 triton==3.4.0 soundfile scipy \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
