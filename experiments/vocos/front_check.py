"""Do the distilled text side and F0/N predictor, trained on English only, hold up in Kokoro's other
languages? Durations and pitch from the original model against the LSTM-swap students and the
batch-first predictor, per language, on chunks from phonemize_text.py. No decoder, so CPU is fine.

usage: front_check.py IDS_JSONL STUDENTS_PT FRAME_SIDE_PT OUT_JSON [N_PER_LANG]
"""
import collections
import json
import os
import random
import sys

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
IDS, STUDENTS, FRAME_SIDE, OUT = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 25
torch.set_num_threads(4)
rng = random.Random(0)

original = KModel(repo_id="hexgrad/Kokoro-82M").eval()
lstmfree = KModel(repo_id="hexgrad/Kokoro-82M").eval()
st.install(lstmfree, st.load_students(lstmfree, STUDENTS))
front = adaln.make_student(lstmfree)
adaln.load_frame_side(front, torch.load(FRAME_SIDE, weights_only=True)["state"])
front = front.eval()

by_lang = collections.defaultdict(list)
for line in open(IDS):
    r = json.loads(line)
    if 40 <= len(r["ids"]) <= 200:
        by_lang[r["lang"]].append(r["ids"])
voice_names = sorted(f[7:-3] for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/"))
packs = {}


def front_half(m, ids, ref, forced=None):
    d, duration, pred_dur = seqfwd.text_half(m, ids, ref)
    dur = pred_dur if forced is None else forced
    F0, _ = seqfwd.prosody(m.predictor, d.transpose(-1, -2) @ seqfwd.alignment(dur), ref[:, 128:])
    return duration[0], pred_dur, F0[0]


summary = {}
with torch.inference_mode():
    for lang, chunks in sorted(by_lang.items()):
        rows = collections.defaultdict(list)
        for ids in rng.sample(chunks, N):
            voice = rng.choice([v for v in voice_names if v[0] == lang])
            if voice not in packs:
                packs[voice] = torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{voice}.pt"), weights_only=True)
            x = torch.LongTensor([[0, *ids, 0]])
            ref = packs[voice][len(ids) - 1]
            dur_o, round_o, f0_o = front_half(original, x, ref)
            voiced = f0_o > 10
            for name, m in (("lstmfree", lstmfree), ("batchfirst", front)):
                dur, rounded, _ = front_half(m, x, ref)
                _, _, f0 = front_half(m, x, ref, forced=round_o)  # pitch with the original's timing
                rows[f"{name}_dur_mean_abs_frames"].append((dur - dur_o).abs().mean().item())
                rows[f"{name}_length_ratio"].append(rounded.sum().item() / round_o.sum().item())
                rows[f"{name}_f0_rmse_hz"].append((f0 - f0_o)[voiced].pow(2).mean().sqrt().item())
                rows[f"{name}_voicing_flips"].append(((f0 > 10) != voiced).float().mean().item())
        summary[lang] = {k: round(sum(v) / len(v), 4) for k, v in rows.items()}
        print(lang, json.dumps(summary[lang]), flush=True)
json.dump(summary, open(OUT, "w"), indent=1)
