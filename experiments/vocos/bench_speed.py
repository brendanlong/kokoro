"""Speed of the Vocos-style decoder against Kokoro's decoder, in plain PyTorch, fp16 on one GPU.

Both decoders get the same made-up inputs (the decoder's speed doesn't depend on what it says) for
sentence lengths drawn from the teacher corpus. Kokoro's decoder runs one sentence at a time (its
instance norm doesn't batch without the worker's packing); the Vocos-style one also runs batches
padded to the longest sentence, counting only real audio. Each set of lengths runs twice and the
second pass is timed, so cuDNN's per-shape plans are already built: steady state, not the fresh
lengths a server sees (the worker replaces cuDNN for that). First calls are timed separately,
which compiled includes torch.compile.

usage: bench_speed.py CORPUS_DIR DECODER_PT OUT_JSON
"""
import json
import random
import sys
import time

import numpy as np
import torch
from loguru import logger
from torch.nn.utils import parametrize

import common

logger.remove()
CORPUS, DECODER, OUT = sys.argv[1:4]
dev = "cuda"
rng = random.Random(0)
lengths = [e["frames"] for e in rng.sample(json.load(open(f"{CORPUS}/index.json")), 48)]

from kokoro import KModel  # noqa: E402

kokoro = KModel(repo_id="hexgrad/Kokoro-82M").eval()
for m in kokoro.modules():
    if parametrize.is_parametrized(m, "weight"):
        parametrize.remove_parametrizations(m, "weight")
kokoro_dec = kokoro.decoder.half().to(dev).eval()
vocos = common.make_decoder(dev, DECODER).half().eval()
for d in (kokoro_dec, vocos):
    # Compiled fp16, the sine generator gives NaNs for some lengths (lstm_swap/bench.py); keep it eager.
    src = d.generator.m_source if d is kokoro_dec else d.source
    src.forward = torch.compiler.disable(src.forward)


def inputs(ts):
    b, t = len(ts), max(ts)
    asr = torch.randn(b, 512, t, device=dev, dtype=torch.half)
    f0 = (150 + 50 * torch.rand(b, 2 * t, device=dev)) * (torch.rand(b, 2 * t, device=dev) < 0.8)
    n = torch.randn(b, 2 * t, device=dev)
    s = torch.randn(b, 128, device=dev, dtype=torch.half)
    return asr, f0.half(), n.half(), s


def run(fn, batches):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    times = []
    for ts in batches:
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            fn(*inputs(ts))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return time.perf_counter() - t0, times


def measure(name, fn, batch):
    batches = [lengths[i : i + batch] for i in range(0, len(lengths), batch)]
    t0 = time.perf_counter()
    run(fn, batches[:1])
    first = time.perf_counter() - t0
    run(fn, batches)  # every length once, so the timed pass reuses cuDNN's plans
    wall, times = run(fn, batches)
    audio_s = sum(lengths) * 600 / 24000
    r = {"name": name, "batch": batch, "first_call_s": round(first, 1), "x_realtime": round(audio_s / wall, 1),
         "median_ms_per_batch": round(float(np.median(times)) * 1000, 1)}
    print(json.dumps(r), flush=True)
    return r


results = []
for mode in ("eager", "compiled"):
    k = torch.compile(kokoro_dec, dynamic=True) if mode == "compiled" else kokoro_dec
    v = torch.compile(vocos, dynamic=True) if mode == "compiled" else vocos
    results.append(measure(f"kokoro-{mode}", k, 1))
    for b in (1, 8):
        results.append(measure(f"vocos-{mode}", v, b))
results.append({"peak_reserved_mib": torch.cuda.max_memory_reserved() >> 20})
with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
