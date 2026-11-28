"""Several decoders on the same validation sentences of a teacher corpus, next to the teacher, as
listening samples.

usage: compare.py VAL_CORPUS_DIR OUT_DIR N NAME=DECODER_PT=LABEL ...
Writes WAVs, samples.json (listen_page.py's format) and page.json (labels; the blind pair is the
teacher and the last decoder)."""
import json
import os
import sys

import soundfile as sf
import torch
from loguru import logger

import common
import model as vm

logger.remove()
VAL, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
val = common.Corpus(VAL, dev)
decoders, labels = {}, {"teacher": "Teacher (Kokoro's decoder)"}
for arg in sys.argv[4:]:
    name, path, label = arg.split("=", 2)
    decoders[name], labels[name] = common.make_decoder(dev, path).eval(), label

picks = [i for i, e in enumerate(val.index) if e["frames"] >= 120][:: int(os.environ.get("EVERY", 3))][:N]
items = []
for k, i in enumerate(picks):
    asr, f0, n, audio, source, style, cycles0 = val.sentence(i)
    clips = {"teacher": audio}
    for name, decoder in decoders.items():
        torch.manual_seed(0)
        with torch.no_grad():
            clips[name] = decoder(asr[None], f0[None], n[None], style[None], source=source[None], cycles0=cycles0[None])[0, 0]
    files = {}
    for name, a in clips.items():
        files[name] = f"{k:02d}-{name}.wav"
        sf.write(os.path.join(OUT, files[name]), a.float().cpu().numpy(), vm.SAMPLE_RATE)
    e = val.index[i]
    items.append({"voice": e["voice"], "source": f"Validation sentence {i}", "text": e.get("text") or e["phonemes"], "files": files})
json.dump(items, open(os.path.join(OUT, "samples.json"), "w"), indent=1, ensure_ascii=False)
json.dump({"labels": labels, "blind": ["teacher", list(decoders)[-1]], "intro": ""}, open(os.path.join(OUT, "page.json"), "w"), indent=1)
