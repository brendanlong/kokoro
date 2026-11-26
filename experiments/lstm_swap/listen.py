"""Listening samples: the original model against the distilled students, fp32, batch 1.

usage: listen.py OUT_DIR NAME=STUDENTS_PT[:PREFIX,...][+FRAME_SIDE_PT] ...
Each text in listen_texts.json is rendered by the original, the original with another noise seed
(how much two renders of the original differ on their own), and each NAME: the students in
STUDENTS_PT whose LSTM path starts with a PREFIX (all of them without), all with the same seed."""
import copy
import json
import os
import sys

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from kokoro import KModel, KPipeline
import seqfwd
import students as st

logger.remove()
OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)


def with_students(spec):
    spec, _, frame_side = spec.partition("+")
    pt, _, prefixes = spec.partition(":")
    m = copy.deepcopy(teacher)
    studs = st.load_students(teacher, pt)
    st.install(m, {p: s for p, s in studs.items() if p.startswith(tuple(prefixes.split(",")))})
    if frame_side:  # a batch-first frame side (experiments/batch_first)
        sys.path.insert(0, os.path.join(HERE, "..", "batch_first"))
        import adaln

        m = adaln.make_student(m)
        adaln.load_frame_side(m, torch.load(frame_side, weights_only=True)["state"])
    return m.to(dev).eval()


VARIANTS = {"original": (teacher, 0), "original_seed1": (teacher, 1)}
for arg in sys.argv[2:]:
    name, _, spec = arg.partition("=")
    VARIANTS[name] = (with_students(spec), 0)
pipes = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=False) for lang in "ab"}
voices = {}
items = []
for i, item in enumerate(json.load(open(os.path.join(HERE, "listen_texts.json")))):
    v = item["voice"]
    if v not in voices:
        voices[v] = torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{v}.pt"), weights_only=True)
    ps = pipes[v[0]].g2p(item["text"])[0]
    ids = torch.LongTensor([[0, *[teacher.vocab[p] for p in ps if p in teacher.vocab], 0]]).to(dev)
    ref = voices[v][len(ps) - 1].to(dev)
    files = {}
    for name, (m, seed) in VARIANTS.items():
        torch.manual_seed(seed)
        with torch.inference_mode():
            audio = seqfwd.forward(m, ids, ref)["audio"].float().cpu().numpy()
        files[name] = f"{i:02d}-{name}.wav"
        sf.write(os.path.join(OUT, files[name]), audio, 24000)
    items.append({**item, "phonemes": ps, "files": files})
    print(i, item["text"][:40], flush=True)
json.dump(items, open(os.path.join(OUT, "samples.json"), "w"), indent=1, ensure_ascii=False)
