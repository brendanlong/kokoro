"""The text of a teacher corpus, one JSON line per utterance: {"lang", "ids", "text"}.

usage: corpus_text.py OUT_JSONL SOURCE[:HOURS] ...

A SOURCE is a file of phoneme strings, one per line (experiments/lstm_swap/prep_data.py's
train.txt: American English, lang "a"), or phonemize_text.py's JSONL. With HOURS, a JSONL source
gives about that much speech from each of its languages (at 12.5 ids per second), taken in
order; a phoneme file gives that much in all. The output is shuffled, so its first N lines are a
sample of the mix.
"""
import collections
import json
import random
import sys

from kokoro import KModel

IDS_PER_HOUR = 12.5 * 3600
out_path, sources = sys.argv[1], sys.argv[2:]
vocab = None
rows = []
for source in sources:
    path, _, hours = source.partition(":")
    budget = float(hours) * IDS_PER_HOUR if hours else float("inf")
    used = collections.Counter()
    for line in open(path):
        if path.endswith(".jsonl"):
            row = json.loads(line)
        else:
            vocab = vocab or KModel(repo_id="hexgrad/Kokoro-82M").vocab
            ps = line.rstrip("\n")
            row = {"lang": "a", "ids": [vocab[p] for p in ps if p in vocab], "text": ps}
        if used[row["lang"]] < budget and 10 <= len(row["ids"]) <= 510:
            used[row["lang"]] += len(row["ids"])
            rows.append(row)
    print(path, {k: round(v / IDS_PER_HOUR, 1) for k, v in sorted(used.items())}, "hours", flush=True)
random.Random(0).shuffle(rows)
with open(out_path, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("wrote", out_path, len(rows), "utterances")
