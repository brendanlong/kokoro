"""Stage one for the Vocos-style decoder (model.py): spectral losses only.

On random crops of the teacher corpus (make_corpus.py), the decoder is given the teacher's inputs
and the harmonic source the teacher used, and matches the teacher's audio with a log-mel L1 loss
and a multi-resolution STFT loss. Crops are exact here: the decoder only looks a few frames to
either side, and the margins, where a crop lacks the context the full sentence has, aren't scored.

usage: train.py CORPUS_DIR VAL_CORPUS_DIR OUT_PT STEPS [INIT_PT]
"""
import json
import math
import random
import sys
import time

import torch
from loguru import logger

import common

logger.remove()
CORPUS, VAL, OUT = sys.argv[1:4]
STEPS = int(sys.argv[4])
INIT = sys.argv[5] if len(sys.argv) > 5 else None
BATCH = 16
LR = 5e-4
dev = "cuda"
torch.manual_seed(0)
rng = random.Random(0)

train, val = common.Corpus(CORPUS, dev), common.Corpus(VAL, dev)
print("train", len(train.index), "sentences,", len(train.long_enough), "long enough for a crop; val", len(val.index), flush=True)
spectral = common.Spectral(dev)
decoder = common.make_decoder(dev, INIT)
params = [p for p in decoder.parameters() if p.requires_grad]
print("decoder params M", round(sum(p.numel() for p in params) / 1e6, 2), flush=True)

log = open(OUT.replace(".pt", ".log.jsonl"), "w")
opt = torch.optim.AdamW(params, lr=LR, betas=(0.8, 0.9), weight_decay=0.01)
warmup = max(1, STEPS // 50)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: min(1, (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(i, STEPS) / STEPS)))
rec = {"step": 0, "val": common.validate(decoder, val, spectral)}
print(json.dumps(rec), flush=True)
log.write(json.dumps(rec) + "\n")
t0, running = time.time(), {}
for step in range(1, STEPS + 1):
    asr, f0, n, audio, source, style = train.crops(rng, BATCH)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = decoder(asr, f0, n, style, source=source)[:, 0]
    losses = spectral(out[:, common.SCORED].float(), audio[:, common.SCORED])
    sum(losses.values()).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    for k, v in losses.items():
        running[k] = running.get(k, 0) + v.item()
    if step % 200 == 0 or step == STEPS:
        rec = {"step": step, "s": round(time.time() - t0), "peak_mib": torch.cuda.max_memory_allocated() >> 20,
               "train": {k: round(v / 200, 5) for k, v in running.items()}}
        running = {}
        if step % 2000 == 0 or step == STEPS:
            rec["val"] = common.validate(decoder, val, spectral)
            common.save(decoder, OUT)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
common.save(decoder, OUT)
