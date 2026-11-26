"""What both training stages share: the teacher corpus, crops of it, and the spectral losses."""
import json
import os
import random

import numpy as np
import torch

from kokoro import KModel
import model as vm

CROP = 128  # scored decoder frames (80 per second): 1.6 s
MARGIN = 16  # unscored frames each side; the decoder's reach is 8 blocks x 3 frames plus the STFT's 2
SPAN = (CROP + 2 * MARGIN) // 2  # a crop with its margins, in 40-per-second frames
SCORED = slice(MARGIN * vm.HOP, (MARGIN + CROP) * vm.HOP)  # a crop's scored samples


class Corpus:
    """make_corpus.py's arrays, memory-mapped."""

    def __init__(self, path, dev):
        self.dev = dev
        self.index = json.load(open(os.path.join(path, "index.json")))
        mm = lambda name, dtype: np.memmap(os.path.join(path, name), dtype=dtype, mode="r")  # noqa: E731
        self.asr = mm("asr.f16", np.float16).reshape(-1, 512)
        self.f0, self.n = mm("f0.f32", np.float32), mm("n.f32", np.float32)
        self.audio, self.source = mm("audio.f16", np.float16), mm("source.f16", np.float16)
        self.style = torch.from_numpy(np.load(os.path.join(path, "style.npy")))[:, :128]
        self.long_enough = [i for i, e in enumerate(self.index) if e["frames"] >= SPAN]
        self.weights = [self.index[i]["frames"] - SPAN + 1 for i in self.long_enough]

    def sentence(self, i, start=0, length=None):
        """Sentence i's frames [start, start + length) at 40 per second:
        (asr [512, T], f0 [2T], n [2T], audio [600 T], source [600 T], style [128]) on the device."""
        e = self.index[i]
        length = length or e["frames"] - start
        f = e["frame_offset"] + start
        cut = lambda a, r: torch.from_numpy(np.array(a[f * r : (f + length) * r])).to(self.dev)  # noqa: E731
        asr = torch.from_numpy(np.array(self.asr[f : f + length])).to(self.dev).T.float()
        return (asr, cut(self.f0, 2), cut(self.n, 2), cut(self.audio, 600).float(), cut(self.source, 600).float(),
                self.style[i].to(self.dev))

    def crops(self, rng: random.Random, n: int):
        """n random crops of SPAN frames, sentences weighted by how many crops they hold, stacked."""
        rows = [self.sentence(i, rng.randrange(self.index[i]["frames"] - SPAN + 1), SPAN)
                for i in rng.choices(self.long_enough, self.weights, k=n)]
        return [torch.stack(x) for x in zip(*rows)]


def make_decoder(dev, init=None):
    kokoro = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    decoder = vm.from_kokoro(kokoro.decoder).to(dev)
    if init:
        decoder.load_state_dict(torch.load(init, weights_only=True)["state"])
    return decoder


def save(module, path):
    torch.save({"state": module.state_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)


class Spectral:
    """A log-mel L1 loss and a multi-resolution STFT loss (log magnitude and spectral convergence)."""

    def __init__(self, dev):
        self.mel = self._mel_filters().to(dev)
        self.windows = {n: torch.hann_window(n, device=dev) for n in (512, 1024, 2048)}

    @staticmethod
    def _mel_filters(n_fft=1024, n_mels=100, sr=24000, fmax=12000):
        mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
        hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
        pts = hz(np.linspace(mel(0), mel(fmax), n_mels + 2))
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
        for m in range(n_mels):
            lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
            fb[m] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        return torch.from_numpy(fb)

    def _mag(self, a, n_fft, hop):
        return torch.stft(a, n_fft, hop, window=self.windows[n_fft], return_complex=True).abs()

    def __call__(self, audio, target):
        logmel = lambda a: torch.log((self.mel @ self._mag(a, 1024, 256)).clamp(min=1e-5))  # noqa: E731
        out = {"mel": (logmel(audio) - logmel(target)).abs().mean()}
        stft = 0
        for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
            A, B = self._mag(audio, n_fft, hop).clamp(min=1e-5), self._mag(target, n_fft, hop).clamp(min=1e-5)
            stft = stft + (A.log() - B.log()).abs().mean() + torch.linalg.norm(A - B) / torch.linalg.norm(B)
        out["stft"] = stft / 3
        return out


@torch.no_grad()
def validate(decoder, val, spectral):
    """Spectral losses over whole validation sentences."""
    was_training = decoder.training
    decoder.eval()
    tot = {}
    for i in range(len(val.index)):
        asr, f0, n, audio, source, style = val.sentence(i)
        out = decoder(asr[None], f0[None], n[None], style[None], source=source[None])[0, 0]
        for k, v in spectral(out, audio).items():
            tot[k] = tot.get(k, 0) + v.item() / len(val.index)
    decoder.train(was_training)
    return {k: round(v, 5) for k, v in tot.items()}
