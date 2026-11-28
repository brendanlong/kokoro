"""Retrain only the text encoder's student, keeping BASE_PT's other students.

Text in every language (multidata.py), each read by its own voices.

usage: train_te.py DATA_DIR BASE_PT OUT_PT MODE STEPS [ARCH_JSON]
MODE "tenc": a fresh student (ARCH_JSON overrides its arch) regressing the LSTM's output, as before.
MODE "audio": BASE_PT's student, fine-tuned so the frozen decoder's audio from its output matches
the audio from the LSTM's (same alignment, F0, N, style and noise), by a multi-resolution STFT loss.
Only text_encoder.lstm changes; the predictor's students in BASE_PT are copied as they are."""
import copy
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from loguru import logger

from kokoro import KModel
import seqfwd
import multidata
import students as st

logger.remove()
DATA, BASE, OUT, MODE = sys.argv[1:5]
STEPS = int(sys.argv[5])
ARCH = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
TE = "text_encoder.lstm"
ACCUM = 16 if MODE == "tenc" else 8
LR = 1e-3 if MODE == "tenc" else 1e-4
# The decoder's backward pass on long sentences doesn't fit next to the desktop on an 8 GB card.
MAX_PHONEMES = 100 if MODE == "audio" else 510
VAL_PER_LANG = 5
dev = "cuda"
torch.manual_seed(0)
random.seed(0)

teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)
for p in teacher.parameters():
    p.requires_grad_(False)
text = multidata.Text(DATA, teacher.vocab, dev, max_ids=MAX_PHONEMES, val_per_lang=VAL_PER_LANG)

base = torch.load(BASE, weights_only=True)
arch = {**base["arch"], TE: {**base["arch"].get(TE, {}), **ARCH}} if MODE == "tenc" else base["arch"]
studs = st.load_students(teacher, BASE)
if MODE == "tenc":
    studs[TE] = st.make_students(teacher, arch)[TE]
student = studs[TE].to(dev).train()
params = list(student.parameters())
print("mode", MODE, "languages", text.langs, "val", len(text.val), "student params M", round(sum(p.numel() for p in params) / 1e6, 2),
      "arch", json.dumps(arch.get(TE, {})), flush=True)
student_te = copy.deepcopy(teacher.text_encoder)
student_te.lstm = student




def stft_loss(a, b):
    total = 0
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        w = torch.hann_window(n_fft, device=a.device)
        A = torch.stft(a, n_fft, hop, window=w, return_complex=True).abs().clamp(min=1e-5)
        B = torch.stft(b, n_fft, hop, window=w, return_complex=True).abs().clamp(min=1e-5)
        total = total + (A.log() - B.log()).abs().mean() + torch.linalg.norm(A - B) / torch.linalg.norm(B)
    return total / 3


def losses(ids, ref):
    with torch.no_grad():
        t_en = seqfwd.text_encoder(teacher.text_encoder, ids)
    s_en = seqfwd.text_encoder(student_te, ids)
    out = {"t_en": F.mse_loss(s_en, t_en) / t_en.var()}
    if MODE == "audio":
        with torch.no_grad():
            d, _, pred_dur = seqfwd.text_half(teacher, ids, ref)
            aln = seqfwd.alignment(pred_dur)
            F0, N = seqfwd.prosody(teacher.predictor, d.transpose(-1, -2) @ aln, ref[:, 128:])
            seed = random.randrange(2**31)
            torch.manual_seed(seed)
            target = teacher.decoder(t_en @ aln, F0, N, ref[:, :128]).squeeze()
        torch.manual_seed(seed)
        audio = teacher.decoder(s_en @ aln, F0, N, ref[:, :128]).squeeze()
        out["stft"] = stft_loss(audio, target)
        out["t_en"] = out["t_en"] * 0.1
    return out


@torch.no_grad()
def validate():
    student.eval()
    tot = {}
    for ids, ref in text.val:
        for k, x in losses(ids, ref).items():
            tot[k] = tot.get(k, 0) + x.item() / len(text.val)
    student.train()
    return {k: round(v, 5) for k, v in tot.items()}


def save():
    st.save_students(studs, arch, OUT + ".tmp")
    os.replace(OUT + ".tmp", OUT)


log = open(OUT.replace(".pt", ".log.jsonl"), "w")
rec = {"step": 0, "val": validate()}
print(json.dumps(rec), flush=True)
log.write(json.dumps(rec) + "\n")
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
warmup = max(1, STEPS // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: min(1, (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(i, STEPS) / STEPS)))
t0 = time.time()
for step in range(1, STEPS + 1):
    running = {}
    for _ in range(ACCUM):
        ls = losses(*text.sample(random))
        (sum(ls.values()) / ACCUM).backward()
        for k, v in ls.items():
            running[k] = running.get(k, 0) + v.item() / ACCUM
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    if step % 100 == 0 or step == STEPS:
        rec = {"step": step, "s": round(time.time() - t0), "train": {k: round(v, 5) for k, v in running.items()}}
        if step % 500 == 0 or step == STEPS:
            rec["val"] = validate()
            save()
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
save()
