"""Training text in every language Kokoro speaks, each read by its own voices.

Two sources, both turned into Kokoro input ids:
- DATA_DIR/train.txt and val.txt: prep_data.py's English phoneme strings (Gutenberg books), read
  by the American and British voices;
- DATA_DIR/multilingual-{train,val}.jsonl: {"lang": pipeline code, "ids": [...]} lines, chunks of
  Wikipedia text phonemized by optimized-kokoro's own pipeline (experiments/vocos/wiki_text.py and
  phonemize_text.py on brendanlong/kokoro), read by the voices whose name starts with that code.

Training samples a language uniformly (Gutenberg English counts as part of "a"), then a chunk and
one of that language's voices, so no language crowds out the rest.
"""
import json
import os
import random

import torch
from huggingface_hub import hf_hub_download, list_repo_files

ENGLISH = "ab"  # prep_data.py's text: American and British voices


class Text:
    def __init__(self, data_dir, vocab, dev, max_ids=510, val_per_lang=None):
        self.dev = dev
        names = sorted(f[7:-3] for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/"))
        self.voices = {n: torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{n}.pt"), weights_only=True)
                       for n in names}
        self.by_lang = {}
        for n in names:
            self.by_lang.setdefault(n[0], []).append(n)
        english = lambda ps: [vocab[p] for p in ps if p in vocab]  # noqa: E731
        read = lambda name: open(os.path.join(data_dir, name)).read().split("\n")[:-1]  # noqa: E731

        def jsonl(name):
            return [(r["ids"], r["lang"]) for r in map(json.loads, open(os.path.join(data_dir, name)))]

        train = [(english(ps), "en") for ps in read("train.txt")] + jsonl("multilingual-train.jsonl")
        self.train = {}
        for ids, lang in train:
            if 1 <= len(ids) <= max_ids:
                self.train.setdefault("a" if lang == "en" else lang, []).append((ids, lang))
        # Held out: the first of prep_data.py's Time Machine sentences (eval.py scores the last 100)
        # and the multilingual validation chunks.
        val = [(english(ps), "en") for ps in read("val.txt")[:100]] + jsonl("multilingual-val.jsonl")
        rng = random.Random(1)
        per_lang = {}
        for ids, lang in val:
            if 1 <= len(ids) <= max_ids and len(per_lang.setdefault(lang, [])) < (val_per_lang or 10**9):
                per_lang[lang].append(self.inputs(ids, self._voice(lang, rng)))
        self.val = [x for lang in sorted(per_lang) for x in per_lang[lang]]
        self.langs = sorted(self.train)

    def _voice(self, lang, rng):
        return rng.choice([v for c in (ENGLISH if lang == "en" else lang) for v in self.by_lang[c]])

    def inputs(self, ids, voice):
        """(input ids [1, n + 2] with the boundary 0s, style [1, 256]) on the device."""
        return (torch.LongTensor([[0, *ids, 0]]).to(self.dev), self.voices[voice][len(ids) - 1].to(self.dev))

    def sample(self, rng: random.Random):
        ids, lang = rng.choice(self.train[rng.choice(self.langs)])
        return self.inputs(ids, self._voice(lang, rng))
