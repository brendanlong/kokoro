"""Distill Kokoro's F0/N predictor into a batch-first one (adaln.py).

The teacher is Kokoro with the LSTM-swap students (STUDENTS_PT, what the worker deploys), and the
student is the same model with AdaLN1d in its F0/N predictor, starting from the teacher's weights.
Only the predictor trains; the text side is shared and frozen. (Until 2026-11-28 this also trained
an AdaLN version of Kokoro's decoder; the Vocos-style decoder replaced it. Checkpoints still hold
the whole frame side, adaln.FRAME_SIDE, with the decoder as initialized.)

Stage "local": each predictor block (AdainResBlk1d) is given the teacher block's input and
regresses its output.
Stage "chain": the student's predictor matches the teacher's F0 and N from the teacher's features.

Text in every language (multidata.py), each read by its own voices.

usage: train.py DATA_DIR STUDENTS_PT OUT_PT LOCAL_STEPS CHAIN_STEPS [INIT_PT]
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from loguru import logger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lstm_swap"))

from kokoro import KModel  # noqa: E402
from kokoro.istftnet import AdainResBlk1d  # noqa: E402
import adaln  # noqa: E402
import seqfwd  # noqa: E402
import multidata  # noqa: E402
import students as st  # noqa: E402

logger.remove()
DATA, STUDENTS, OUT = sys.argv[1:4]
LOCAL_STEPS, CHAIN_STEPS = int(sys.argv[4]), int(sys.argv[5])
INIT = sys.argv[6] if len(sys.argv) > 6 else None
ACCUM = 8
LR = {"local": 2e-4, "chain": 1e-4}
MAX_PHONEMES = 510
VAL_PER_LANG = 6
PREDICTOR = [p for p in adaln.FRAME_SIDE if p.startswith("predictor.")]
dev = "cuda"
torch.manual_seed(0)
random.seed(0)

teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)
st.install(teacher, {p: m for p, m in st.load_students(teacher, STUDENTS).items()})
teacher = teacher.to(dev).eval().requires_grad_(False)
student = adaln.make_student(teacher).to(dev)
if INIT:
    adaln.load_frame_side(student, torch.load(INIT, weights_only=True)["state"])
params = []
for p in PREDICTOR:
    params += list(student.get_submodule(p).requires_grad_(True).parameters())
student.train()

text = multidata.Text(DATA, teacher.vocab, dev, max_ids=MAX_PHONEMES, val_per_lang=VAL_PER_LANG)
print("languages", text.langs, "val", len(text.val), "trainable params M", round(sum(p.numel() for p in params) / 1e6, 2), flush=True)

BLOCKS = [name for name, m in teacher.named_modules()
          if isinstance(m, AdainResBlk1d) and name.startswith(("predictor.F0", "predictor.N"))]
captured = {}
capturing = False


def capture(name):
    def hook(mod, args, out):
        if capturing:
            captured[name] = (args[0], args[1], out)
    return hook


for name in BLOCKS:
    teacher.get_submodule(name).register_forward_hook(capture(name))

@torch.no_grad()
def teacher_pass(ids, ref):
    captured.clear()
    d, _, pred_dur = seqfwd.text_half(teacher, ids, ref)
    en = d.transpose(-1, -2) @ seqfwd.alignment(pred_dur)
    F0, N = seqfwd.prosody(teacher.predictor, en, ref[:, 128:])
    return dict(en=en, F0=F0, N=N)


def local_losses(ids, ref, scale=None):
    """With scale, each block's loss is backpropagated (times scale) as soon as it's computed, so
    only one block's activations are held at a time."""
    global capturing
    capturing = True
    teacher_pass(ids, ref)
    capturing = False
    total = 0.0
    for name, (x, s, y) in captured.items():
        loss = F.mse_loss(student.get_submodule(name)(x, s), y) / y.var().clamp(min=1e-4) / len(captured)
        if scale is not None:
            (loss * scale).backward()
        total += loss.item()
    captured.clear()
    return {"blocks": total}


def chain_losses(ids, ref):
    t = teacher_pass(ids, ref)
    F0, N = seqfwd.prosody(student.predictor, t["en"], ref[:, 128:])
    return {"F0": F.smooth_l1_loss(F0 / 50, t["F0"] / 50), "N": F.smooth_l1_loss(N, t["N"])}


@torch.no_grad()
def validate():
    student.eval()
    tot = {}
    for ids, ref in text.val:
        for k, x in {**local_losses(ids, ref), **chain_losses(ids, ref)}.items():
            tot[k] = tot.get(k, 0) + float(x) / len(text.val)
    student.train()
    return {k: round(v, 5) for k, v in tot.items()}


def save():
    torch.save({"state": adaln.frame_side_state(student)}, OUT + ".tmp")
    os.replace(OUT + ".tmp", OUT)


log = open(OUT.replace(".pt", ".log.jsonl"), "w")
rec = {"stage": "start", "val": validate()}
print(json.dumps(rec), flush=True)
log.write(json.dumps(rec) + "\n")
for stage, steps in [("local", LOCAL_STEPS), ("chain", CHAIN_STEPS)]:
    if steps == 0:
        continue
    opt = torch.optim.AdamW(params, lr=LR[stage], weight_decay=0.0)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    t0 = time.time()
    for step in range(1, steps + 1):
        running = {}
        for _ in range(ACCUM):
            args = text.sample(random)
            if stage == "local":
                losses = local_losses(*args, scale=1 / ACCUM)
            else:
                losses = chain_losses(*args)
                (sum(losses.values()) / ACCUM).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0) + float(v) / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 100 == 0 or step == steps:
            rec = {"stage": stage, "step": step, "s": round(time.time() - t0),
                   "peak_mib": torch.cuda.max_memory_allocated() >> 20,
                   "train": {k: round(v, 5) for k, v in running.items()}}
            if step % 500 == 0 or step == steps:
                rec["val"] = validate()
                save()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
save()
