"""Distill Kokoro's frame side (F0/N predictor and decoder) into a batch-first one (adaln.py).

The teacher is Kokoro with the LSTM-swap students (STUDENTS_PT, what the worker deploys), and the
student is the same model with AdaLN1d in its frame side, starting from the teacher's weights.
Only the frame side trains; the text side is shared and frozen, so the decoder learns from the
text features it will be given (Paradee, arXiv:2610.06817, found features tuned to one decoder
don't transfer to another).

Stage "local": each residual block (AdainResBlk1d, AdaINResBlock1) is given the teacher block's
input and regresses its output.
Stage "chain": the student's F0/N predictor matches the teacher's F0 and N from the teacher's
features, and the student decoder, given the teacher's features, F0, N, style and random numbers
(the same seed), matches the teacher's audio with a multi-resolution STFT loss.

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
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lstm_swap"))

from kokoro import KModel  # noqa: E402
from kokoro.istftnet import AdaINResBlock1, AdainResBlk1d  # noqa: E402
import adaln  # noqa: E402
import seqfwd  # noqa: E402
import students as st  # noqa: E402

logger.remove()
DATA, STUDENTS, OUT = sys.argv[1:4]
LOCAL_STEPS, CHAIN_STEPS = int(sys.argv[4]), int(sys.argv[5])
INIT = sys.argv[6] if len(sys.argv) > 6 else None
ACCUM = 8
LR = {"local": 2e-4, "chain": 1e-4}
# The decoder's backward pass on longer sentences doesn't fit beside the desktop on the 8 GB card.
MAX_PHONEMES = 100
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
for p in adaln.FRAME_SIDE:
    params += list(student.get_submodule(p).requires_grad_(True).parameters())
student.train()

voice_files = [f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/") and f[7] in "ab"]
voices = [torch.load(hf_hub_download("hexgrad/Kokoro-82M", f), weights_only=True) for f in voice_files]
train = [ps for ps in open(os.path.join(DATA, "train.txt")).read().split("\n")[:-1] if len(ps) <= MAX_PHONEMES]
val = [ps for ps in open(os.path.join(DATA, "val.txt")).read().split("\n")[:-1][:100] if len(ps) <= MAX_PHONEMES][:40]
val_voices = [random.randrange(len(voices)) for _ in val]
print("train", len(train), "val", len(val), "trainable params M", round(sum(p.numel() for p in params) / 1e6, 2), flush=True)

BLOCKS = [name for name, m in teacher.named_modules()
          if isinstance(m, (AdainResBlk1d, AdaINResBlock1)) and name.startswith(("predictor.F0", "predictor.N", "decoder"))]
captured = {}
capturing = False


def capture(name):
    def hook(mod, args, out):
        if capturing:
            captured[name] = (args[0], args[1], out)
    return hook


for name in BLOCKS:
    teacher.get_submodule(name).register_forward_hook(capture(name))


def inputs(ps, voice_idx):
    ids = torch.LongTensor([[0, *[teacher.vocab[p] for p in ps if p in teacher.vocab], 0]]).to(dev)
    return ids, voices[voice_idx][len(ps) - 1].to(dev)


@torch.no_grad()
def teacher_pass(ids, ref, seed):
    captured.clear()
    d, _, pred_dur = seqfwd.text_half(teacher, ids, ref)
    aln = seqfwd.alignment(pred_dur)
    en = d.transpose(-1, -2) @ aln
    F0, N = seqfwd.prosody(teacher.predictor, en, ref[:, 128:])
    asr = seqfwd.text_encoder(teacher.text_encoder, ids) @ aln
    torch.manual_seed(seed)
    audio = teacher.decoder(asr, F0, N, ref[:, :128]).squeeze()
    return dict(en=en, F0=F0, N=N, asr=asr, audio=audio)


def stft_loss(a, b):
    total = 0
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        w = torch.hann_window(n_fft, device=a.device)
        A = torch.stft(a, n_fft, hop, window=w, return_complex=True).abs().clamp(min=1e-5)
        B = torch.stft(b, n_fft, hop, window=w, return_complex=True).abs().clamp(min=1e-5)
        total = total + (A.log() - B.log()).abs().mean() + torch.linalg.norm(A - B) / torch.linalg.norm(B)
    return total / 3


def local_losses(ids, ref):
    global capturing
    capturing = True
    teacher_pass(ids, ref, 0)
    capturing = False
    out = {}
    for name, (x, s, y) in captured.items():
        out[name] = F.mse_loss(student.get_submodule(name)(x, s), y) / y.var().clamp(min=1e-4)
    captured.clear()
    return {"blocks": sum(out.values()) / len(out)}


def chain_losses(ids, ref):
    seed = random.randrange(2**31)
    t = teacher_pass(ids, ref, seed)
    F0, N = seqfwd.prosody(student.predictor, t["en"], ref[:, 128:])
    torch.manual_seed(seed)
    audio = student.decoder(t["asr"], t["F0"], t["N"], ref[:, :128]).squeeze()
    return {
        "F0": F.smooth_l1_loss(F0 / 50, t["F0"] / 50),
        "N": F.smooth_l1_loss(N, t["N"]),
        "stft": stft_loss(audio, t["audio"]),
    }


@torch.no_grad()
def validate():
    student.eval()
    tot = {}
    for ps, v in zip(val, val_voices):
        ids, ref = inputs(ps, v)
        for k, x in {**local_losses(ids, ref), **chain_losses(ids, ref)}.items():
            tot[k] = tot.get(k, 0) + x.item() / len(val)
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
    fn = local_losses if stage == "local" else chain_losses
    t0 = time.time()
    for step in range(1, steps + 1):
        running = {}
        for _ in range(ACCUM):
            losses = fn(*inputs(random.choice(train), random.randrange(len(voices))))
            (sum(losses.values()) / ACCUM).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0) + v.item() / ACCUM
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
