"""Quality of the distilled students vs the teacher on held-out text (The Time Machine), fp32.

usage: eval.py DATA_DIR STUDENTS_PT OUT_DIR [PATH_PREFIX,...]
With prefixes, only the students whose LSTM path starts with one of them are installed."""
import copy
import json
import os
import random
import re
import sys

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger
from scipy.signal import stft
from transformers import pipeline

from kokoro import KModel
import seqfwd
import students as st

logger.remove()
DATA, STUDENTS, OUT = sys.argv[1:4]
ONLY = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("",)
N_UTT, N_WAV = 100, 6
dev = "cuda"
os.makedirs(OUT, exist_ok=True)
random.seed(1)

teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)
student = copy.deepcopy(teacher)
studs = st.make_students(teacher)
sd = torch.load(STUDENTS, weights_only=True)
for path, m in studs.items():
    m.load_state_dict({k[len(path) + 1:]: v for k, v in sd.items() if k.startswith(path + ".")})
st.install(student, {p: m for p, m in studs.items() if p.startswith(ONLY)})
student.to(dev).eval()
voice_files = sorted(f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/") and f[7] in "ab")
voices = [torch.load(hf_hub_download("hexgrad/Kokoro-82M", f), weights_only=True) for f in voice_files]
val = open(os.path.join(DATA, "val.txt")).read().split("\n")[:-1][-N_UTT:]
asr = pipeline("automatic-speech-recognition", "openai/whisper-small.en", device=dev)


def logspec(a):
    return np.log(np.abs(stft(a, nperseg=1024)[2]) + 1e-4).ravel()


def corr(a, b):
    return float(np.corrcoef(logspec(a), logspec(b))[0, 1])


def words(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def synth(m, ids, ref, seed, forced_dur=None):
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = seqfwd.forward(m, ids, ref, forced_dur=forced_dur)
    return {k: (v.float() if v.is_floating_point() else v).cpu() for k, v in out.items()}


rows = []
transcripts = {"teacher": [], "teacher_seed1": [], "student": []}
for i, ps in enumerate(val):
    vi = random.randrange(len(voices))
    ids = torch.LongTensor([[0, *[teacher.vocab[p] for p in ps if p in teacher.vocab], 0]]).to(dev)
    ref = voices[vi][len(ps) - 1].to(dev)
    t0, t1 = synth(teacher, ids, ref, 0), synth(teacher, ids, ref, 1)
    s_free = synth(student, ids, ref, 0)
    s_forced = synth(student, ids, ref, 0, forced_dur=t0["pred_dur"].to(dev))
    voiced = t0["F0"] > 10
    td, sd_ = t0["pred_dur"].float(), s_free["pred_dur"].float()
    rows.append(dict(
        phonemes=len(ps),
        dur_mae_frames=float((td - sd_).abs().mean()),
        dur_exact_frac=float((td == sd_).float().mean()),
        len_ratio=float(sd_.sum() / td.sum()),
        f0_rmse_hz=float(((t0["F0"] - s_forced["F0"])[voiced] ** 2).mean().sqrt()),
        n_mae=float((t0["N"] - s_forced["N"]).abs().mean()),
        corr_student_forced=corr(t0["audio"].numpy(), s_forced["audio"].numpy()),
        corr_teacher_seed1=corr(t0["audio"].numpy(), t1["audio"].numpy()),
    ))
    for name, a in [("teacher", t0), ("teacher_seed1", t1), ("student", s_free)]:
        transcripts[name].append(asr(a["audio"].numpy())["text"])
    if i < N_WAV:
        for name, a in [("teacher", t0), ("student", s_free)]:
            sf.write(os.path.join(OUT, f"{i}-{voice_files[vi][7:-3]}-{name}.wav"), a["audio"].numpy(), 24000)

ref_words = [words(t) for t in transcripts["teacher"]]
n_words = sum(len(w) for w in ref_words)
summary = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
for name in ["teacher_seed1", "student"]:
    summary[f"wer_{name}_vs_teacher"] = round(sum(edits(r, words(h)) for r, h in zip(ref_words, transcripts[name])) / n_words, 4)
summary["n_utterances"] = len(rows)
print("RESULT", json.dumps(summary), flush=True)
json.dump(dict(summary=summary, rows=rows, transcripts=transcripts), open(os.path.join(OUT, "eval.json"), "w"), indent=1)
