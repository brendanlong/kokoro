"""Stage two for the Vocos-style decoder: adversarial training (discriminators.py) from stage one.

Paradee (arXiv:2610.06817) found a decoder trained with spectral losses alone kept crisp harmonics
where the teacher's turn breathy, which sounded robotic, and that the critics only took effect
once the spectral losses stopped dominating: HiFi-GAN's weight of 45 barely moved it, 10 and then 3
did. So the generator's loss is adversarial + 2 x feature matching + SPECTRAL x (mel + STFT + level),
with SPECTRAL 10 for the first half of the steps and 3 for the second. "level" is the L1 distance
between the log energies of each decoder frame (300 samples): on v1, with only mel and STFT, stage
two came out about 1 dB quieter than the teacher over whole sentences.

usage: train_adv.py CORPUS_DIR VAL_CORPUS_DIR INIT_PT OUT_PT STEPS
"""
import json
import random
import sys
import time

import torch
from loguru import logger

import common
from discriminators import Discriminators, discriminator_loss, generator_losses

logger.remove()
CORPUS, VAL, INIT, OUT = sys.argv[1:5]
STEPS = int(sys.argv[5])
BATCH = 8
LR = 2e-4
FEATURE_MATCHING = 2.0


def level_loss(fake, real):
    energy = lambda a: a.reshape(a.shape[0], -1, 300).pow(2).mean(-1)  # noqa: E731
    return (torch.log(energy(fake) + 1e-6) - torch.log(energy(real) + 1e-6)).abs().mean()


dev = "cuda"
torch.manual_seed(0)
rng = random.Random(1)

train, val = common.Corpus(CORPUS, dev), common.Corpus(VAL, dev)
spectral = common.Spectral(dev)
decoder = common.make_decoder(dev, INIT)
disc = Discriminators().to(dev)
g_params = [p for p in decoder.parameters() if p.requires_grad]
opt_g = torch.optim.AdamW(g_params, lr=LR, betas=(0.8, 0.99), weight_decay=0.01)
opt_d = torch.optim.AdamW(disc.parameters(), lr=LR, betas=(0.8, 0.99), weight_decay=0.01)
print("decoder params M", round(sum(p.numel() for p in g_params) / 1e6, 2),
      "discriminator params M", round(sum(p.numel() for p in disc.parameters()) / 1e6, 2), flush=True)

log = open(OUT.replace(".pt", ".log.jsonl"), "w")
rec = {"step": 0, "val": common.validate(decoder, val, spectral)}
print(json.dumps(rec), flush=True)
log.write(json.dumps(rec) + "\n")
t0, running = time.time(), {}
for step in range(1, STEPS + 1):
    weight = 10.0 if step <= STEPS // 2 else 3.0
    asr, f0, n, audio, source, style = train.crops(rng, BATCH)
    real = audio[:, common.SCORED]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fake = decoder(asr, f0, n, style, source=source)[:, 0, common.SCORED].float()

        d_loss = discriminator_loss(disc(real), disc(fake.detach()))
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
    opt_d.step()
    opt_d.zero_grad(set_to_none=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        adv, fm = generator_losses(disc(real), disc(fake))
    spec = {**spectral(fake, real), "level": level_loss(fake, real)}
    g_loss = adv + FEATURE_MATCHING * fm + weight * (spec["mel"] + spec["stft"] + spec["level"])
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(g_params, 10.0)
    opt_g.step()
    opt_g.zero_grad(set_to_none=True)
    disc.zero_grad(set_to_none=True)

    for k, v in {"d": d_loss, "adv": adv, "fm": fm, **spec}.items():
        running[k] = running.get(k, 0) + v.item()
    if step % 200 == 0 or step == STEPS:
        rec = {"step": step, "s": round(time.time() - t0), "peak_mib": torch.cuda.max_memory_allocated() >> 20,
               "spectral_weight": weight, "train": {k: round(v / 200, 5) for k, v in running.items()}}
        running = {}
        if step % 1000 == 0 or step == STEPS:
            rec["val"] = common.validate(decoder, val, spectral)
            common.save(decoder, OUT)
            common.save(disc, OUT.replace(".pt", ".disc.pt"))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
common.save(decoder, OUT)
