#!/bin/bash
# usage: sweep.sh OUT_DIR STEPS NAME=CONFIG_JSON ...: stage-one runs of train.py side by side on one
# GPU (each needs under 1 GB), after building the teacher corpus if this host lacks it. Each run
# VAL_NAME names the validation corpus (a host may hold one of another size as "val"). CORPUS_NAME, TRAIN_N and VAL_N make a smaller corpus under another name, for a smoke run. Each run
# writes OUT_DIR/NAME.{pt,log.jsonl,metrics.json} and resumes if the job is restarted.
set -euo pipefail
out=$1; steps=$2; shift 2
mkdir -p "$out"
py=.venv/bin/python
v=experiments/vocos
corpus=$GPUC_DATA_DIR/${CORPUS_NAME:-kokoro-vocos}
val=$corpus/${VAL_NAME:-val}
[ -f "$val/index.json" ] || $py $v/make_corpus.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt "$val" val "${VAL_N:-100}"
[ -f "$corpus/train/index.json" ] || $py $v/make_corpus.py ckpt/data ckpt/te-audio.pt ckpt/bf-v1.pt "$corpus/train" train "${TRAIN_N:-12000}"
pids=()
for run in "$@"; do
    name=${run%%=*}
    [ -f "$out/$name.metrics.json" ] && continue
    $py $v/train.py "$corpus/train" "$val" "$out/$name.pt" "$steps" "${run#*=}" > "$out/$name.out" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
grep -h METRICS "$out"/*.out || true
exit $status
