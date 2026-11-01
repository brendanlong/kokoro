"""Latency/throughput of kokoro with and without the LSTMs, fp16 on one GPU.

usage: bench.py OUT_JSON MODE [STUDENTS_PT]
MODE: {original,seq-lstm,seq-conv}-{eager,compiled}
seq-conv without STUDENTS_PT uses untrained convs with the teacher's durations forced, which
times the architecture but not its quality."""
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from scipy.signal import stft

from kokoro import KModel, KPipeline
import seqfwd
import students as st

logger.remove()
OUT, MODE = sys.argv[1], sys.argv[2]
STUDENTS = sys.argv[3] if len(sys.argv) > 3 else None
ARCH, EXEC = MODE.rsplit("-", 1)
HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"

model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
for mod in model.modules():
    for name in ["weight"]:
        if torch.nn.utils.parametrize.is_parametrized(mod, name):
            torch.nn.utils.parametrize.remove_parametrizations(mod, name)
model = model.half().to(dev)
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
voice = torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"), weights_only=True)
full_ps = pipe.g2p(open(os.path.join(HERE, "time_machine.txt")).read()[:2500])[0]
WARM = range(20, 470, 3)
FRESH = range(53, 260, 6)


def inputs(ps):
    ids = torch.LongTensor([[0, *[model.vocab[p] for p in ps if p in model.vocab], 0]]).to(dev)
    return ids, voice[len(ps) - 1].to(dev).half()


forced = {}
if ARCH == "seq-conv":
    with torch.inference_mode():
        for n in sorted(set(WARM) | set(FRESH) | {150}):
            forced[n] = seqfwd.text_half(model, *inputs(full_ps[:n]))[2]
    studs = st.make_students(model)
    if STUDENTS:
        sd = torch.load(STUDENTS, weights_only=True)
        for path, m in studs.items():
            m.load_state_dict({k[len(path) + 1:]: v for k, v in sd.items() if k.startswith(path + ".")})
        forced = {}
    st.install(model, studs)
    model.half().to(dev)
r = {"mode": MODE, "students": STUDENTS, "torch": torch.__version__,
     "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}

g = model.decoder.generator
th, ah = seqfwd.text_half, seqfwd.acoustic_half
if EXEC == "compiled":
    # compiled fp16 SineGen gives NaNs for some lengths (see the Kokoro-FastAPI latency journal)
    g.m_source.forward = torch.compiler.disable(g.m_source.forward)
    if ARCH == "original":
        model.decoder = torch.compile(model.decoder, dynamic=True)
        model.bert = torch.compile(model.bert, dynamic=True)
        model.text_encoder = torch.compile(model.text_encoder, dynamic=True)
        model.predictor.F0Ntrain = torch.compile(model.predictor.F0Ntrain, dynamic=True)
    else:
        th, ah = torch.compile(th, dynamic=True), torch.compile(ah, dynamic=True)

stage = {"text": [], "acoustic": []}


def run(n, seed=0, timed_stages=False):
    ids, ref = inputs(full_ps[:n])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if ARCH == "original":
            audio = model.forward_with_tokens(ids, ref, 1)[0]
        else:
            d, _, pred_dur = th(model, ids, ref, 1.0)
            if timed_stages:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            audio = ah(model, ids, ref, d, seqfwd.alignment(forced.get(n, pred_dur)))[0]
    audio = audio.float().cpu().numpy()
    t = time.perf_counter() - t0
    if timed_stages:
        stage["text"].append(t1 - t0)
        stage["acoustic"].append(t - (t1 - t0))
    return t, audio


t0 = time.perf_counter()
run(100)
r["first_call_s"] = round(time.perf_counter() - t0, 1)
for n in WARM:
    run(n)
fresh = [run(n) for n in FRESH]
r["median_ms"] = round(float(np.median([t for t, _ in fresh])) * 1000, 1)
r["audio_s_per_s"] = round(sum(len(a) for _, a in fresh) / 24000 / sum(t for t, _ in fresh), 1)
r["repeat_150ph_ms"] = round(float(np.median([run(150)[0] for _ in range(9)])) * 1000, 1)
if ARCH != "original":
    for n in FRESH:
        run(n, timed_stages=True)
    r["text_half_median_ms"] = round(float(np.median(stage["text"])) * 1000, 1)
    r["acoustic_half_median_ms"] = round(float(np.median(stage["acoustic"])) * 1000, 1)
r["nan_lengths"] = sum(int(np.isnan(a).any()) for _, a in fresh)

audio = np.concatenate([a for _, a in fresh[::6]])
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
np.save(OUT.replace(".json", ".npy"), audio)
ref_path = os.path.join(os.path.dirname(OUT), "original-eager.npy")
if os.path.exists(ref_path) and not OUT.endswith("original-eager.json"):
    ref_audio = np.load(ref_path)
    logspec = lambda a: np.log(np.abs(stft(np.nan_to_num(a), nperseg=1024)[2]) + 1e-4)
    r["corr_vs_original_eager"] = (round(float(np.corrcoef(logspec(audio).ravel(), logspec(ref_audio).ravel())[0, 1]), 4)
                                   if len(audio) == len(ref_audio) else f"length {len(audio)} vs {len(ref_audio)}")
r["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
print("RESULT", json.dumps(r), flush=True)
with open(OUT, "w") as f:
    json.dump(r, f, indent=1)
