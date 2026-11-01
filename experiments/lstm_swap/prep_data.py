"""Public-domain Gutenberg books -> phoneme strings, one per line. The Time Machine (#35) is held out for eval.

usage: prep_data.py OUT_DIR"""
import os
import random
import re
import sys
import urllib.request

from loguru import logger

from kokoro import KPipeline

logger.remove()
TRAIN_BOOKS = [1342, 1661, 84, 11, 2701, 98, 345, 1400, 36, 1260, 76, 174, 2600, 16328]
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def sentences(book_id):
    path = os.path.join(OUT, f"pg{book_id}.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt", path)
    text = open(path, encoding="utf-8-sig").read()
    start, end = text.find("*** START"), text.find("*** END")
    text = re.sub(r"\s+", " ", text[text.index("\n", start):end])
    return [s for s in re.split(r"(?<=[.!?\"”])\s+(?=[A-Z“\"])", text) if 20 <= len(s) <= 400]


pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
random.seed(0)
for name, books, n in [("train", TRAIN_BOOKS, 30000), ("val", [35], 400)]:
    sents = [s for b in books for s in sentences(b)]
    random.shuffle(sents)
    with open(os.path.join(OUT, f"{name}.tmp"), "w") as f:
        kept = 0
        for s in sents:
            ps = pipe.g2p(s)[0]
            if 5 <= len(ps) <= 500:
                f.write(ps + "\n")
                kept += 1
            if kept == n:
                break
    os.replace(os.path.join(OUT, f"{name}.tmp"), os.path.join(OUT, f"{name}.txt"))
    print(name, kept, "utterances from", len(sents), "sentences", flush=True)
