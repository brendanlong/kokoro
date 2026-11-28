"""The teacher corpus for training a Vocos-style decoder: everything the decoder gets, and the audio
Kokoro's own decoder makes from it.

The text side and F0/N predictor are the ones the new decoder will run behind: the LSTM-swap
students (STUDENTS_PT) and the batch-first F0/N predictor (FRAME_SIDE_PT, experiments/batch_first).
The target audio is Kokoro's original decoder on those inputs, with a seed per sentence, and the
harmonic source it used (its noise included) is saved too, so the student can be given the same
one and its target is deterministic.

usage: make_corpus.py DATA_DIR STUDENTS_PT FRAME_SIDE_PT OUT_DIR train|val N

OUT_DIR gets flat arrays (concatenated sentences) and index.json:
    asr.f16     [frames, 512] phoneme features at 40 frames per second
    f0.f32 n.f32  [2 * frames] F0 (Hz) and N at 80 per second
    audio.f16 source.f16  [600 * frames] samples at 24 kHz
    style.f32   [sentences, 256] the style vector (the decoder takes [:128])
"""
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lstm_swap"))
sys.path.insert(0, os.path.join(HERE, "..", "batch_first"))

from kokoro import KModel  # noqa: E402
import adaln  # noqa: E402
import seqfwd  # noqa: E402
import students as st  # noqa: E402

logger.remove()
DATA, STUDENTS, FRAME_SIDE, OUT, SPLIT, N = sys.argv[1:7]
N = int(N)
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(0 if SPLIT == "train" else 1)

kokoro = KModel(repo_id="hexgrad/Kokoro-82M").eval()
st.install(kokoro, st.load_students(kokoro, STUDENTS))
kokoro = kokoro.to(dev).eval()
front = adaln.make_student(kokoro)  # the text side and batch-first F0/N predictor
adaln.load_frame_side(front, torch.load(FRAME_SIDE, weights_only=True)["state"])
front = front.to(dev).eval()
gen = kokoro.decoder.generator

voice_files = sorted(f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/"))
voices = {f[7:-3]: torch.load(hf_hub_download("hexgrad/Kokoro-82M", f), weights_only=True) for f in voice_files}
names = sorted(v for v in voices if v in os.environ.get("VOICES", v).split(","))  # VOICES=af_heart,... limits them
lines = open(os.path.join(DATA, f"{SPLIT}.txt")).read().split("\n")[:-1]
# eval.py holds out val.txt's last 100 sentences; the val corpus takes its first ones.
lines = lines[:N] if SPLIT == "val" else rng.sample(lines, min(N, len(lines)))

tmp = OUT + ".tmp"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
files = {k: open(os.path.join(tmp, k), "wb") for k in ("asr.f16", "f0.f32", "n.f32", "audio.f16", "source.f16")}
styles, index = [], []
frames_at = 0
for i, ps in enumerate(lines):
    voice = names[rng.randrange(len(names))]
    ids = torch.LongTensor([[0, *[kokoro.vocab[p] for p in ps if p in kokoro.vocab], 0]]).to(dev)
    ref = voices[voice][len(ps) - 1].to(dev)
    seed = rng.randrange(2**31)
    with torch.inference_mode():
        d, _, pred_dur = seqfwd.text_half(front, ids, ref)
        aln = seqfwd.alignment(pred_dur)
        F0, N_ = seqfwd.prosody(front.predictor, d.transpose(-1, -2) @ aln, ref[:, 128:])
        asr = seqfwd.text_encoder(front.text_encoder, ids) @ aln
        torch.manual_seed(seed)
        audio = kokoro.decoder(asr, F0, N_, ref[:, :128]).squeeze()
        # The source the decoder just used: its first random numbers come from this seed.
        torch.manual_seed(seed)
        har = gen.m_source(gen.f0_upsamp(F0[:, None]).transpose(1, 2))[0]
        source = har.transpose(1, 2).squeeze()
    t = asr.shape[-1]
    assert F0.shape[-1] == 2 * t and audio.shape[-1] == 600 * t == source.shape[-1]
    files["asr.f16"].write(asr[0].T.half().cpu().numpy().tobytes())
    files["f0.f32"].write(F0[0].float().cpu().numpy().tobytes())
    files["n.f32"].write(N_[0].float().cpu().numpy().tobytes())
    files["audio.f16"].write(audio.half().cpu().numpy().tobytes())
    files["source.f16"].write(source.half().cpu().numpy().tobytes())
    styles.append(ref[0].float().cpu().numpy())
    index.append({"voice": voice, "seed": seed, "frames": t, "frame_offset": frames_at, "phonemes": ps})
    frames_at += t
    if i % 500 == 0:
        print(i, "sentences,", round(frames_at / 40 / 3600, 2), "hours", flush=True)
for f in files.values():
    f.close()
np.save(os.path.join(tmp, "style.npy"), np.stack(styles))
with open(os.path.join(tmp, "index.json"), "w") as f:
    json.dump(index, f)
shutil.rmtree(OUT, ignore_errors=True)
os.replace(tmp, OUT)
print("wrote", OUT, len(index), "sentences,", round(frames_at / 40 / 3600, 2), "hours", flush=True)
