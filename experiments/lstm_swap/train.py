"""Distill each kokoro LSTM into a ConvSeq student.

Stage "local": each student sees the teacher's input to its LSTM and regresses the LSTM's output.
Stage "chain": all students installed at once; match the teacher's continuous durations, F0, N
and text-encoder output, so the students learn to correct each other's errors.

usage: train.py DATA_DIR OUT_PT LOCAL_STEPS CHAIN_STEPS [ARCH_JSON]
ARCH_JSON maps LSTM paths to overrides of students.DEFAULT_ARCH."""
import copy
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

from kokoro import KModel
import seqfwd
import students as st

logger.remove()
DATA, OUT = sys.argv[1], sys.argv[2]
LOCAL_STEPS, CHAIN_STEPS = int(sys.argv[3]), int(sys.argv[4])
ARCH = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
ACCUM, LR = 16, 1e-3
dev = "cuda"
torch.manual_seed(0)
random.seed(0)

teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)
for p in teacher.parameters():
    p.requires_grad_(False)
voice_files = [f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/") and f[7] in "ab"]
voices = [torch.load(hf_hub_download("hexgrad/Kokoro-82M", f), weights_only=True) for f in voice_files]
train = open(os.path.join(DATA, "train.txt")).read().split("\n")[:-1]
val = open(os.path.join(DATA, "val.txt")).read().split("\n")[:-1][:100]
val_voices = [random.randrange(len(voices)) for _ in val]
studs = st.make_students(teacher, ARCH)
for m in studs.values():
    m.to(dev)
params = [p for m in studs.values() for p in m.parameters()]
print("voices", len(voices), "train", len(train), "student params M", round(sum(p.numel() for p in params) / 1e6, 2),
      "lstm params M", round(sum(p.numel() for path in studs for p in teacher.get_submodule(path).parameters()) / 1e6, 2), flush=True)
student_model = copy.deepcopy(teacher)
st.install(student_model, studs)


def inputs(ps, voice_idx):
    ids = torch.LongTensor([[0, *[teacher.vocab[p] for p in ps if p in teacher.vocab], 0]]).to(dev)
    return ids, voices[voice_idx][len(ps) - 1].to(dev)


captured = {}


def capture(path):
    def hook(mod, args, out):
        captured[path] = (args[0], out[0])
    return hook


for path in studs:
    teacher.get_submodule(path).register_forward_hook(capture(path))


@torch.no_grad()
def teacher_pass(ids, ref):
    captured.clear()
    d, duration, pred_dur = seqfwd.text_half(teacher, ids, ref)
    aln = seqfwd.alignment(pred_dur)
    F0, N = seqfwd.prosody(teacher.predictor, d.transpose(-1, -2) @ aln, ref[:, 128:])
    t_en = seqfwd.text_encoder(teacher.text_encoder, ids)
    return dict(duration=duration, aln=aln, F0=F0, N=N, t_en=t_en)


def local_losses(ids, ref):
    teacher_pass(ids, ref)
    return {path: F.mse_loss(studs[path](x), y) / y.var().clamp(min=1e-4) for path, (x, y) in captured.items()}


def chain_losses(ids, ref):
    t = teacher_pass(ids, ref)
    d, duration, _ = seqfwd.text_half(student_model, ids, ref)
    F0, N = seqfwd.prosody(student_model.predictor, d.transpose(-1, -2) @ t["aln"], ref[:, 128:])
    t_en = seqfwd.text_encoder(student_model.text_encoder, ids)
    return {
        "duration": F.smooth_l1_loss(duration, t["duration"]),
        "F0": F.smooth_l1_loss(F0 / 50, t["F0"] / 50),
        "N": F.smooth_l1_loss(N, t["N"]),
        "t_en": F.mse_loss(t_en, t["t_en"]) / t["t_en"].var(),
    }


@torch.no_grad()
def validate():
    tot = {}
    for ps, v in zip(val, val_voices):
        ids, ref = inputs(ps, v)
        for k, x in {**local_losses(ids, ref), **chain_losses(ids, ref)}.items():
            tot[k] = tot.get(k, 0) + x.item() / len(val)
    return {k: round(v, 5) for k, v in tot.items()}


log = open(OUT.replace(".pt", ".log.jsonl"), "w")
for stage, steps in [("local", LOCAL_STEPS), ("chain", CHAIN_STEPS)]:
    if steps == 0:
        continue
    opt = torch.optim.AdamW(params, lr=LR if stage == "local" else LR / 3, weight_decay=0.01)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    fn = local_losses if stage == "local" else chain_losses
    t0 = time.time()
    for step in range(1, steps + 1):
        running = {}
        for _ in range(ACCUM):
            ids, ref = inputs(random.choice(train), random.randrange(len(voices)))
            losses = fn(ids, ref)
            (sum(losses.values()) / ACCUM).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0) + v.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 250 == 0 or step == steps:
            rec = {"stage": stage, "step": step, "s": round(time.time() - t0), "train": {k: round(v, 5) for k, v in running.items()}}
            if step % 1000 == 0 or step == steps:
                rec["val"] = validate()
                st.save_students(studs, ARCH, OUT + ".tmp")
                os.replace(OUT + ".tmp", OUT)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
