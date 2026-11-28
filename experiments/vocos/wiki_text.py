"""Paragraphs of Wikipedia in Kokoro's languages, as text for a multilingual teacher corpus.

usage: wiki_text.py OUT_DIR N LANG ...   (uv run --with pyarrow --with huggingface_hub)

LANG is a Kokoro pipeline code; `a` and `b` both read English Wikipedia, from different articles.
For each, N paragraphs of 150 to 1,200 characters from the first file of the wikimedia/wikipedia
dump of 2023-11-01 are written to OUT_DIR/LANG.txt, one JSON string per line, at most two from
an article.
"""
import json
import os
import random
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

WIKI = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh"}
OUT, N = sys.argv[1], int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
files = list_repo_files("wikimedia/wikipedia", repo_type="dataset")
tables = {}
for code in sys.argv[3:]:
    wiki = WIKI[code]
    if wiki not in tables:
        first = sorted(f for f in files if f.startswith(f"20231101.{wiki}/"))[0]
        tables[wiki] = pq.read_table(hf_hub_download("wikimedia/wikipedia", first, repo_type="dataset"), columns=["text"])
    texts = tables[wiki].column("text")
    # English articles alternate between the two English pipelines.
    rows = list(range(len(texts)))
    if wiki == "en":
        rows = rows[(code == "b") :: 2]
    random.Random(code).shuffle(rows)
    kept = []
    for row in rows:
        paragraphs = [p.strip() for p in texts[row].as_py().split("\n")]
        # Prose: long enough, ending as a sentence does, and not a list or table row.
        prose = [p for p in paragraphs if 150 <= len(p) <= 1200 and p[-1] in ".。!?！？।" and "|" not in p]
        kept += random.Random(f"{code}/{row}").sample(prose, min(2, len(prose)))
        if len(kept) >= N:
            break
    with open(os.path.join(OUT, f"{code}.txt"), "w") as f:
        for p in kept[:N]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(code, wiki, len(kept[:N]), "paragraphs,", sum(map(len, kept[:N])), "characters", flush=True)
