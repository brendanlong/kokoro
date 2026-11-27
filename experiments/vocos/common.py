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
        # full_band_source's running sum of F0, restarting at each sentence
        cycles = np.cumsum((self.f0.astype(np.float64) / vm.SAMPLE_RATE) % 1)
        starts = np.array([2 * e["frame_offset"] for e in self.index])
        before = np.concatenate([[0.0], cycles])  # before[g]: the sum over frames before g
        self.sentence_start = dict(zip(starts.tolist(), before[starts].tolist()))
        self.cycles_before = before
        self.long_enough = [i for i, e in enumerate(self.index) if e["frames"] >= SPAN]
        self.weights = [self.index[i]["frames"] - SPAN + 1 for i in self.long_enough]

    def sentence(self, i, start=0, length=None):
        """Sentence i's frames [start, start + length) at 40 per second:
        (asr [512, T], f0 [2T], n [2T], audio [600 T], source [600 T], style [128], cycles0 []) on the
        device; cycles0 is full_band_source's running sum before the first frame."""
        e = self.index[i]
        length = length or e["frames"] - start
        f = e["frame_offset"] + start
        cut = lambda a, r: torch.from_numpy(np.array(a[f * r : (f + length) * r])).to(self.dev)  # noqa: E731
        asr = torch.from_numpy(np.array(self.asr[f : f + length])).to(self.dev).T.float()
        cycles0 = self.cycles_before[2 * f] - self.sentence_start[2 * e["frame_offset"]]
        return (asr, cut(self.f0, 2), cut(self.n, 2), cut(self.audio, 600).float(), cut(self.source, 600).float(),
                self.style[i].to(self.dev), torch.tensor(cycles0, dtype=torch.float64, device=self.dev))

    def crops(self, rng: random.Random, n: int):
        """n random crops of SPAN frames, sentences weighted by how many crops they hold, stacked."""
        rows = [self.sentence(i, rng.randrange(self.index[i]["frames"] - SPAN + 1), SPAN)
                for i in rng.choices(self.long_enough, self.weights, k=n)]
        return [torch.stack(x) for x in zip(*rows)]


def make_decoder(dev, init=None, **config):
    kokoro = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    return (vm.load(kokoro.decoder, init) if init else vm.from_kokoro(kokoro.decoder, **config)).to(dev)


def save(module, path, **extra):
    """extra: what a run needs to resume (optimizer state, step)."""
    ckpt = {"state": module.state_dict(), **extra}
    if hasattr(module, "config"):
        ckpt["config"] = module.config
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)


def mel_filters(n_fft, n_mels, sr=24000, fmax=12000):
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = hz(np.linspace(mel(0), mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return torch.from_numpy(fb)


# DAC's multi-scale mel loss (arXiv:2306.06546): window lengths and the mel bands each gets.
MEL_SCALES = ((32, 5), (64, 10), (128, 20), (256, 40), (512, 80), (1024, 160), (2048, 320))
PHASE_FFT, PHASE_HOP = 1024, 256
TERMS = ("mel", "stft", "msmel", "phase", "cplx")


class Spectral:
    """Losses between a decoder's audio and its teacher's, each a term a run can weight:

    mel    log-mel L1 at one resolution
    stft   multi-resolution STFT (log magnitude and spectral convergence)
    msmel  log-mel L1 at seven resolutions, as DAC has it
    phase  how each bin's phase, and its change to the next bin and to the next frame, differ
           from the teacher's, weighted by the teacher's magnitude (the anti-wrapping losses of
           APNet2, arXiv:2311.11545, written on unit phasors so they need no angle)
    cplx   L1 between the complex spectra with magnitudes compressed to the power 0.3

    The last two compare waveforms, not just spectra's shapes. They make sense only because the
    student gets the harmonic source its teacher used, so the teacher's audio is one definite
    target.
    """

    def __init__(self, dev, weights=None):
        self.weights = weights or {"mel": 1.0, "stft": 1.0}
        assert set(self.weights) <= set(TERMS), self.weights
        self.mel = mel_filters(1024, 100).to(dev)
        self.ms_mel = {n: mel_filters(n, m).to(dev) for n, m in MEL_SCALES}
        self.windows = {n: torch.hann_window(n, device=dev) for n in {512, 1024, 2048, *self.ms_mel}}

    def _stft(self, a, n_fft, hop):
        return torch.stft(a, n_fft, hop, window=self.windows[n_fft], return_complex=True)

    def _mag(self, a, n_fft, hop):
        return self._stft(a, n_fft, hop).abs()

    def term(self, name, audio, target):
        if name == "mel":
            logmel = lambda a: torch.log((self.mel @ self._mag(a, 1024, 256)).clamp(min=1e-5))  # noqa: E731
            return (logmel(audio) - logmel(target)).abs().mean()
        if name == "stft":
            stft = 0
            for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
                A, B = self._mag(audio, n_fft, hop).clamp(min=1e-5), self._mag(target, n_fft, hop).clamp(min=1e-5)
                stft = stft + (A.log() - B.log()).abs().mean() + torch.linalg.norm(A - B) / torch.linalg.norm(B)
            return stft / 3
        if name == "msmel":
            total = 0
            for n_fft, fb in self.ms_mel.items():
                logmel = lambda a: torch.log10((fb @ self._mag(a, n_fft, n_fft // 4) ** 2).clamp(min=1e-5))  # noqa: E731
                total = total + (logmel(audio) - logmel(target)).abs().mean()
            return total / len(self.ms_mel)
        X, T = self._stft(audio, PHASE_FFT, PHASE_HOP), self._stft(target, PHASE_FFT, PHASE_HOP)
        if name == "phase":
            return sum(phase_errors(X, T).values()) / 3
        if name == "cplx":
            squash = lambda Z: Z * (Z.abs() + 1e-3) ** -0.7  # noqa: E731
            return (squash(X) - squash(T)).abs().mean()
        raise ValueError(name)

    def __call__(self, audio, target, terms=None):
        return {name: self.term(name, audio, target) for name in (terms or self.weights)}

    def total(self, losses):
        return sum(self.weights[k] * v for k, v in losses.items())


def phase_errors(X, T, weight=None):
    """X, T [..., bins, frames] complex. The distance between unit phasors (2 |sin(difference / 2)|) of
    the phase itself, of its step to the next bin and of its step to the next frame, each averaged
    with `weight` (by default the target's magnitude to the power 0.3)."""
    if weight is None:
        weight = T.abs() ** 0.3
    unit = lambda Z: Z / (Z.abs() + 1e-3)  # noqa: E731
    x, t = unit(X), unit(T)
    avg = lambda d, w: (d.abs() * w).sum() / w.sum().clamp(min=1e-9)  # noqa: E731
    return {
        "phase": avg(x - t, weight),
        "bin_step": avg(x[..., 1:, :] * x[..., :-1, :].conj() - t[..., 1:, :] * t[..., :-1, :].conj(), weight[..., 1:, :]),
        "frame_step": avg(x[..., 1:] * x[..., :-1].conj() - t[..., 1:] * t[..., :-1].conj(), weight[..., 1:]),
    }


@torch.no_grad()
def validate(decoder, val, spectral, terms=("mel", "stft", "msmel")):
    """Spectral losses over whole validation sentences."""
    was_training = decoder.training
    decoder.eval()
    tot = {}
    for i in range(len(val.index)):
        asr, f0, n, audio, source, style, _ = val.sentence(i)
        out = decoder(asr[None], f0[None], n[None], style[None], source=source[None])[0, 0]
        for k, v in spectral(out, audio, terms).items():
            tot[k] = tot.get(k, 0) + v.item() / len(val.index)
    decoder.train(was_training)
    return {k: round(v, 5) for k, v in tot.items()}
