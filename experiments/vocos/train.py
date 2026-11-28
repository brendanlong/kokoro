"""Stage one for the Vocos-style decoder (model.py): losses against the teacher's audio, no critics.

On random crops of the teacher corpus (make_corpus.py), the decoder is given the teacher's inputs
and the harmonic source the teacher used, and matches the teacher's audio under the weighted loss
terms of common.Spectral. Only a crop's middle is scored: the margins lack the context the full
sentence has (the middle matches the whole sentence's to 67 dB, review_cpu.py).

usage: train.py CORPUS_DIR VAL_CORPUS_DIR OUT_PT STEPS [CONFIG_JSON]

CONFIG_JSON: {"model": VocosDecoder's keyword arguments, "loss": {term: weight}, "seed": 0,
"init": a checkpoint to start from, "phase_voiced_only": the phase and cplx terms skip unvoiced stretches,
where a decoder with a noise branch should not be matching the teacher's phase}. The default is the decoder and losses of v2. A run that finds
OUT_PT's ".resume" file carries on from it.
"""
import json
import math
import os
import random
import sys
import time

import torch
from loguru import logger

import common
import metrics

logger.remove()
CORPUS, VAL, OUT = sys.argv[1:4]
STEPS = int(sys.argv[4])
CONFIG = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
BATCH = 16
LR = 5e-4
SAVE_EVERY = 2000
dev = "cuda" if torch.cuda.is_available() else "cpu"
seed = CONFIG.get("seed", 0)
torch.manual_seed(seed)

train, val = common.Corpus(CORPUS, dev), common.Corpus(VAL, dev)
print("train", len(train.index), "sentences,", len(train.long_enough), "long enough for a crop; val", len(val.index), flush=True)
spectral = common.Spectral(dev, CONFIG.get("loss"))
decoder = common.make_decoder(dev, CONFIG.get("init"), **CONFIG.get("model", {}))
params = [p for p in decoder.parameters() if p.requires_grad]
print("config", json.dumps({**CONFIG, "model": decoder.config, "loss": spectral.weights}),
      "decoder params M", round(sum(p.numel() for p in params) / 1e6, 2), flush=True)

opt = torch.optim.AdamW(params, lr=LR, betas=(0.8, 0.9), weight_decay=0.01)
warmup = max(1, STEPS // 50)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: min(1, (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(i, STEPS) / STEPS)))
resume, log_path, first = OUT + ".resume", OUT.replace(".pt", ".log.jsonl"), 1
if os.path.exists(resume):
    ckpt = torch.load(resume, weights_only=True, map_location=dev)
    decoder.load_state_dict(ckpt["state"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    first = ckpt["step"] + 1
    print("resuming at step", first, flush=True)
log = open(log_path, "a" if first > 1 else "w")


def record(rec):
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.flush()


if first == 1:
    record({"step": 0, "val": common.validate(decoder, val, spectral)})
t0, running = time.time(), {}
for step in range(first, STEPS + 1):
    asr, f0, n, audio, source, style, cycles0 = train.crops(random.Random(f"{seed}/{step}"), BATCH)
    with torch.autocast(dev, dtype=torch.bfloat16):
        out = decoder(asr, f0, n, style, source=source, cycles0=cycles0)[:, 0]
    voiced = (f0 > 10).repeat_interleave(300, dim=1)[:, common.SCORED] if CONFIG.get("phase_voiced_only") else None
    losses = spectral(out[:, common.SCORED].float(), audio[:, common.SCORED], voiced=voiced)
    spectral.total(losses).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    for k, v in losses.items():
        running[k] = running.get(k, 0) + v.item()
    if step % 200 == 0 or step == STEPS:
        rec = {"step": step, "s": round(time.time() - t0),
               "train": {k: round(v / 200, 5) for k, v in running.items()}}
        if dev == "cuda":
            rec["peak_mib"] = torch.cuda.max_memory_allocated() >> 20
        running = {}
        if step % SAVE_EVERY == 0 or step == STEPS:
            rec["val"] = common.validate(decoder, val, spectral)
            common.save(decoder, resume, opt=opt.state_dict(), sched=sched.state_dict(), step=step)
        record(rec)
common.save(decoder, OUT)
summary = metrics.evaluate(decoder, val, spectral, dev)
print("METRICS", json.dumps(summary), flush=True)
with open(OUT.replace(".pt", ".metrics.json"), "w") as f:
    json.dump({"config": {**CONFIG, "model": decoder.config, "loss": spectral.weights}, "steps": STEPS, "metrics": summary}, f, indent=1)
os.remove(resume)
