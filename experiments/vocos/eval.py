"""Quality of the Vocos-style decoder on held-out text (The Time Machine), fp32.

Variants: the original Kokoro ("original", and at another noise seed, the floor), Kokoro with the
LSTM-swap students ("lstmfree"), the corpus teacher ("teacher": the LSTM-swap text side, the
batch-first F0/N predictor and Kokoro's decoder) and the same with the Vocos-style decoder
("vocos"). Word error rate is Whisper's transcript against its transcript of the original. The
other measures force the timing to the reference's, so they compare sound, not timing: per-token
loudness difference (dB), log-spectrogram correlation and pitch RMSE on voiced frames. Each
variant is compared with the original, and the Vocos decoder also with its teacher (same inputs).

usage: eval.py DATA_DIR STUDENTS_PT FRAME_SIDE_PT DECODER_PT OUT_JSON
"""
import copy
import json
import os
import random
import re
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger
from scipy.signal import stft
from transformers import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lstm_swap"))
sys.path.insert(0, os.path.join(HERE, "..", "batch_first"))

from kokoro import KModel  # noqa: E402
import adaln  # noqa: E402
import model as vm  # noqa: E402
import seqfwd  # noqa: E402
import students as st  # noqa: E402

logger.remove()
DATA, STUDENTS, FRAME_SIDE, DECODER, OUT = sys.argv[1:6]
N_UTT = 100
dev = "cuda"
random.seed(1)

original = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)
lstmfree = KModel(repo_id="hexgrad/Kokoro-82M").eval()
st.install(lstmfree, st.load_students(lstmfree, STUDENTS))
lstmfree = lstmfree.to(dev).eval()
teacher = adaln.make_student(lstmfree)
adaln.load_frame_side(teacher, torch.load(FRAME_SIDE, weights_only=True)["state"])
vocos = copy.deepcopy(teacher)
teacher.decoder = lstmfree.decoder  # Kokoro's decoder behind the batch-first F0/N predictor
vocos.decoder = vm.from_kokoro(lstmfree.decoder)
vocos.decoder.load_state_dict(torch.load(DECODER, weights_only=True)["state"])
teacher, vocos = teacher.to(dev).eval(), vocos.to(dev).eval()
voice_files = sorted(f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/") and f[7] in "ab")
voices = [torch.load(hf_hub_download("hexgrad/Kokoro-82M", f), weights_only=True) for f in voice_files]
val = open(os.path.join(DATA, "val.txt")).read().split("\n")[:-1][-N_UTT:]
asr = pipeline("automatic-speech-recognition", "openai/whisper-small.en", device=dev)


def logspec(a):
    return np.log(np.abs(stft(a, nperseg=1024)[2]) + 1e-4).ravel()


def corr(a, b):
    return float(np.corrcoef(logspec(a), logspec(b))[0, 1])


def loud_db(a, b, dur):
    """Mean |dB| difference per input token's stretch of audio, tokens within 40 dB of the loudest."""
    per_frame = len(a) / int(dur.sum())
    edges = (torch.cat([torch.zeros(1), dur.float().cumsum(0)]) * per_frame).round().long().tolist()
    db = lambda x: np.array([10 * np.log10(np.mean(x[i:j] ** 2) + 1e-10) for i, j in zip(edges, edges[1:]) if j > i])  # noqa: E731
    da, dbb = db(a), db(b)
    return float(np.abs(da - dbb)[da > da.max() - 40].mean())


def compare(ref, other):
    voiced = ref["F0"] > 10
    a, b = ref["audio"].numpy(), other["audio"].numpy()
    return dict(corr=corr(a, b), loud_db=loud_db(a, b, ref["pred_dur"]),
                f0_rmse_hz=float(((ref["F0"] - other["F0"])[voiced] ** 2).mean().sqrt()))


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
names = ["original_seed1", "lstmfree", "teacher", "vocos"]
transcripts = {k: [] for k in ["original", *names]}
for ps in val:
    vi = random.randrange(len(voices))
    ids = torch.LongTensor([[0, *[original.vocab[p] for p in ps if p in original.vocab], 0]]).to(dev)
    ref = voices[vi][len(ps) - 1].to(dev)
    o = synth(original, ids, ref, 0)
    free = {"original": o, "original_seed1": synth(original, ids, ref, 1), "lstmfree": synth(lstmfree, ids, ref, 0),
            "teacher": synth(teacher, ids, ref, 0), "vocos": synth(vocos, ids, ref, 0)}
    forced = lambda m: synth(m, ids, ref, 0, forced_dur=o["pred_dur"].to(dev))  # noqa: E731
    row = {"vs_original": {"original_seed1": compare(o, free["original_seed1"]), "lstmfree": compare(o, forced(lstmfree)),
                           "teacher": compare(o, forced(teacher)), "vocos": compare(o, forced(vocos))},
           # Same text side and F0/N, so the same durations and pitch: what the decoder alone changed.
           "vocos_vs_teacher": compare(free["teacher"], free["vocos"])}
    rows.append(row)
    for k, a in free.items():
        transcripts[k].append(asr(a["audio"].numpy())["text"])

ref_words = [words(t) for t in transcripts["original"]]
n_words = sum(len(w) for w in ref_words)
mean = lambda xs: round(float(np.mean(xs)), 4)  # noqa: E731
summary = {"n_utterances": len(rows), "vs_original": {}}
for k in names:
    summary["vs_original"][k] = {m: mean([r["vs_original"][k][m] for r in rows]) for m in rows[0]["vs_original"][k]}
    summary["vs_original"][k]["wer"] = round(sum(edits(r, words(h)) for r, h in zip(ref_words, transcripts[k])) / n_words, 4)
summary["vocos_vs_teacher"] = {m: mean([r["vocos_vs_teacher"][m] for r in rows]) for m in rows[0]["vocos_vs_teacher"]}
print("RESULT", json.dumps(summary), flush=True)
with open(OUT, "w") as f:
    json.dump(dict(summary=summary, rows=rows, transcripts=transcripts), f, indent=1)
