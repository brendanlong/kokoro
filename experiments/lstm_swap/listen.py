"""Listening samples: the original model against the distilled students, fp32, batch 1.

usage: listen.py STUDENTS_PT OUT_DIR
Each text in listen_texts.json is rendered by every variant with the same noise seed, plus the
original with another seed (how much two renders of the original differ on their own)."""
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
STUDENTS, OUT = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

teacher = KModel(repo_id="hexgrad/Kokoro-82M").eval().to(dev)
studs = st.load_students(teacher, STUDENTS)


def with_students(prefixes):
    m = copy.deepcopy(teacher)
    st.install(m, {p: copy.deepcopy(s) for p, s in studs.items() if p.startswith(prefixes)})
    return m.to(dev).eval()


VARIANTS = {
    "original": (teacher, 0),
    "original_seed1": (teacher, 1),
    "swap5": (with_students(("predictor.",)), 0),
    "swap6": (with_students(("",)), 0),
}
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
