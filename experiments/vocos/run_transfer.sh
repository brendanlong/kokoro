#!/bin/bash
# The multilingual decoder behind the multilingual front vs behind the original front (b6c1607f's
# suggestion): render the val text behind the front, then metrics.py (each decoder against Kokoro's
# decoder on the same inputs) on that corpus and on the original front's, and listening samples.
# Needs ckpt/front-ml-{te-audio,frame_side}.pt, ckpt/ml-both-sub3{,-d384}.pt, ckpt/data/corpus-val.jsonl.
set -euo pipefail
out=experiments/vocos/out/transfer; mkdir -p "$out"
py=.venv/bin/python
v=experiments/vocos
front=$GPUC_DATA_DIR/kokoro-vocos-front-ml/val
orig=$GPUC_DATA_DIR/kokoro-vocos-orig-ml/val
[ -f "$front/index.json" ] || $py $v/make_corpus.py ckpt/data/corpus-val.jsonl ckpt/front-ml-te-audio.pt ckpt/front-ml-frame_side.pt "$front" 180
for dec in ml-both-sub3 ml-both-sub3-d384; do
    for corpus in front orig; do
        dir=$front; [ $corpus = orig ] && dir=$orig
        $py $v/metrics.py "$dir" ckpt/$dec.pt "$out/$dec-$corpus.json"
    done
done
$py $v/compare.py "$front" "$out/listen" 27 "decoder=ckpt/ml-both-sub3.pt=New decoder (dim 512) behind the multilingual front"
