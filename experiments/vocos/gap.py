"""Why validation loss is twice training loss: the same decoder scored on crops (as in training) and
on whole sentences, for training and validation sentences.

usage: gap.py CORPUS_DIR VAL_CORPUS_DIR DECODER_PT"""
import json
import random
import sys

import torch
from loguru import logger

import common

logger.remove()
dev = "cuda"
train, val = common.Corpus(sys.argv[1], dev), common.Corpus(sys.argv[2], dev)
spectral = common.Spectral(dev)
decoder = common.make_decoder(dev, sys.argv[3]).eval()


@torch.no_grad()
def crops(corpus, n=60):
    rng, tot = random.Random(5), {}
    for _ in range(n // 4):
        asr, f0, nn, audio, source, style, _ = corpus.crops(rng, 4)
        out = decoder(asr, f0, nn, style, source=source)[:, 0]
        for k, v in spectral(out[:, common.SCORED], audio[:, common.SCORED]).items():
            tot[k] = tot.get(k, 0) + v.item() / (n // 4)
    return {k: round(v, 4) for k, v in tot.items()}


class Head:  # the first 60 training sentences, as a Corpus for common.validate
    def __init__(self, c):
        self.c, self.index = c, c.index[:60]

    def sentence(self, i):
        return self.c.sentence(i)


print(json.dumps({"train_crops": crops(train), "val_crops": crops(val),
                  "train_whole": common.validate(decoder, Head(train), spectral),
                  "val_whole": common.validate(decoder, val, spectral)}), flush=True)
