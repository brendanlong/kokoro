"""Paradee's phase-locking filter (arXiv:2610.06817, section 7) on the Vocos-style decoder's output.

The swap test (swap.py) found the decoder's magnitudes fine and its phase what sounds robotic. In
voiced frames between 2 and 8 kHz the filter measures how far each bin's phase drifts from a tone
whose harmonics have steady phase (model.full_band_source: every harmonic of F0, in phase with
Kokoro's source), smooths that drift over time, and resynthesizes with the smoothed phase and the
decoder's own magnitudes. Nothing is trained. Cf. identity phase locking in the phase vocoder.

On validation sentences of the teacher corpus (the decoder gets the teacher's exact inputs and
source, so it lines up with the teacher), writes the teacher, the decoder, the decoder filtered
with each smoothing time, and the decoder with the teacher's phase in the same region (what the
filter could at best reach), plus a phase distance to the teacher in that region for each.

usage: phase_lock.py VAL_CORPUS_DIR DECODER_PT OUT_DIR [N]
"""
import json
import os
import sys

import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger

import common
import model as vm

logger.remove()
VAL, DECODER, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
SMOOTHING_S = (0.1, 0.33)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
val = common.Corpus(VAL, dev)
decoder = common.make_decoder(dev, DECODER).eval()
N_FFT, HOP = 1024, 256
window = torch.hann_window(N_FFT, device=dev)
freqs = torch.arange(N_FFT // 2 + 1, device=dev) * vm.SAMPLE_RATE / N_FFT
band = ((freqs >= 2000) & (freqs < 8000))[:, None]


def stft(a):
    return torch.stft(a, N_FFT, HOP, window=window, return_complex=True)


def unit(Z):
    return Z / (Z.abs() + 1e-9)


def lock(Y, tone, region, seconds):
    """Y's phase in `region` replaced by the tone's plus the drift from it, smoothed over `seconds`."""
    drift = unit(Y) * unit(tone).conj()  # [bins, frames]
    width = max(1, round(seconds * vm.SAMPLE_RATE / HOP)) | 1
    kernel = torch.ones(1, 1, width, device=Y.device) / width
    smooth = lambda x: F.conv1d(x[:, None], kernel, padding=width // 2)[:, 0]  # noqa: E731
    drift = torch.complex(smooth(drift.real), smooth(drift.imag))
    locked = Y.abs() * unit(tone) * unit(drift)
    return torch.where(region, locked, Y)


def lock_harmonics(Y, tone, f0_frames, region, seconds, max_harmonics=160):
    """As lock, but the drift is measured at each harmonic's own frequency (it moves between bins as
    the pitch moves), smoothed over time per harmonic, and applied to every bin around that harmonic
    (each bin takes the drift of its nearest harmonic), as identity phase locking does."""
    df = vm.SAMPLE_RATE / N_FFT
    f0 = f0_frames.clamp(min=50)[None, :]  # [1, frames]
    h = torch.arange(1, max_harmonics + 1, device=Y.device)[:, None]  # [H, 1]
    centre = (h * f0 / df).round().long().clamp(max=Y.shape[0] - 1)  # [H, frames]
    drift = torch.gather(unit(Y), 0, centre) * torch.gather(unit(tone), 0, centre).conj()
    width = max(1, round(seconds * vm.SAMPLE_RATE / HOP)) | 1
    kernel = torch.ones(1, 1, width, device=Y.device) / width
    smooth = lambda x: F.conv1d(x[:, None], kernel, padding=width // 2)[:, 0]  # noqa: E731
    drift = torch.complex(smooth(drift.real), smooth(drift.imag))
    nearest = (freqs[:, None] / f0).round().long().clamp(1, max_harmonics) - 1  # [bins, frames]
    locked = Y.abs() * unit(tone) * unit(torch.gather(drift, 0, nearest))
    return torch.where(region, locked, Y)


def phase_distance(X, T, region):
    """Mean |unit(X) - unit(T)| in region, weighted by T's magnitude^0.3 (common.phase_errors' "phase")."""
    w = T.abs() ** 0.3 * region
    return float(((unit(X) - unit(T)).abs() * w).sum() / w.sum().clamp(min=1e-9))


picks = [i for i, e in enumerate(val.index) if e["frames"] >= 120][::3][:N]
items, distances = [], {}
for k, i in enumerate(picks):
    asr, f0, n, audio, source, style, cycles0 = val.sentence(i)
    with torch.no_grad():
        out = decoder(asr[None], f0[None], n[None], style[None], source=source[None], cycles0=cycles0[None])[0, 0]
        tone = vm.full_band_source(f0[None], cycles0[None])[0]
    Y, T, Ton = stft(out), stft(audio), stft(tone)
    voiced = F.interpolate((f0 > 10).float()[None, None], size=Y.shape[-1], mode="nearest")[0, 0] > 0
    region = band & voiced[None, :]
    istft = lambda Z: torch.istft(Z, N_FFT, HOP, window=window, length=audio.shape[0])  # noqa: E731
    variants = {"teacher": (audio, T), "decoder": (out, Y)}
    f0_frames = F.interpolate(f0[None, None].float(), size=Y.shape[-1], mode="nearest")[0, 0]
    for s in SMOOTHING_S:
        Z = lock(Y, Ton, region, s)
        variants[f"locked_{int(s * 1000)}ms"] = (istft(Z), Z)
        Z = lock_harmonics(Y, Ton, f0_frames, region, s)
        variants[f"harmonic_{int(s * 1000)}ms"] = (istft(Z), Z)
    Z = torch.where(region, torch.polar(Y.abs(), T.angle()), Y)
    variants["teacher_phase_region"] = (istft(Z), Z)
    files = {}
    for name, (a, Z) in variants.items():
        files[name] = f"{k:02d}-{name}.wav"
        sf.write(os.path.join(OUT, files[name]), a.float().cpu().numpy(), vm.SAMPLE_RATE)
        # The resynthesized audio's own STFT, not Z: what a listener gets after overlap-add.
        distances.setdefault(name, []).append(phase_distance(stft(a), T, region))
    e = val.index[i]
    items.append({"voice": e["voice"], "source": f"Validation sentence {i}", "text": e.get("text", e.get("phonemes")), "files": files})
    print(k, i, e["voice"], {name: round(d[-1], 3) for name, d in distances.items()}, flush=True)

summary = {name: round(sum(d) / len(d), 4) for name, d in distances.items()}
print("PHASE_DISTANCE voiced 2-8 kHz", json.dumps(summary), flush=True)
json.dump(items, open(os.path.join(OUT, "samples.json"), "w"), indent=1, ensure_ascii=False)
json.dump({
    "labels": {
        "teacher": "Teacher (Kokoro's decoder)",
        "decoder": "Vocos-style decoder (candidate)",
        **{f"locked_{int(s * 1000)}ms": f"Decoder + phase locking per bin, drift smoothed over {int(s * 1000)} ms"
           for s in SMOOTHING_S},
        **{f"harmonic_{int(s * 1000)}ms": f"Decoder + phase locking per harmonic, drift smoothed over {int(s * 1000)} ms"
           for s in SMOOTHING_S},
        "teacher_phase_region": "Decoder with the teacher's phase in voiced 2-8 kHz (what the filter could at best reach)",
    },
    "blind": ["teacher", f"harmonic_{int(SMOOTHING_S[-1] * 1000)}ms"],
    "intro": (
        "<p>Paradee's phase-locking filter (arXiv:2610.06817, section 7) on the Vocos-style decoder. The swap test "
        "found the decoder's magnitudes fine and its phase what sounds robotic. In voiced frames between 2 and 8 kHz "
        "the filter measures how far each frequency bin's phase drifts from a tone made of every harmonic of the "
        "pitch, smooths that drift over time, and resynthesizes with the smoothed phase and the decoder's own "
        "magnitudes. It has no parameters and needs no training.</p>"
        "<p>Phase distance to the teacher in voiced 2-8 kHz (0 is the teacher's phase; about 1.27 is random), "
        "averaged over these sentences: " + ", ".join(f"{k} {v}" for k, v in summary.items()) + ".</p>"),
}, open(os.path.join(OUT, "page.json"), "w"), indent=1)
