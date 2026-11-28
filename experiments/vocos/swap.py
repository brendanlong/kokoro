"""Where a decoder's robotic sound lives: its magnitude or its phase (Paradee's test, arXiv:2610.06817).

On validation sentences of the teacher corpus, the decoder gets the teacher's inputs and the exact
harmonic source the teacher used, so its audio and the teacher's line up sample for sample. Their
STFTs (n_fft 1024, hop 256) are then recombined: the decoder's magnitudes with the teacher's phase,
the teacher's magnitudes with the decoder's phase, and the decoder with the teacher's phase only
in voiced 2-8 kHz (where Paradee's buzz was). If "decoder magnitude, teacher phase" sounds natural,
the phase is at fault.

usage: swap.py VAL_CORPUS_DIR DECODER_PT OUT_DIR [N]
Writes WAVs, samples.json (listen_page.py's format) and page.json (its labels and text).
"""
import json
import os
import sys

import soundfile as sf
import torch
from loguru import logger

import common
import model as vm

logger.remove()
VAL, DECODER, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
val = common.Corpus(VAL, dev)
decoder = common.make_decoder(dev, DECODER).eval()
N_FFT, HOP = 1024, 256
window = torch.hann_window(N_FFT, device=dev)
freqs = torch.arange(N_FFT // 2 + 1, device=dev) * vm.SAMPLE_RATE / N_FFT
band = ((freqs >= 2000) & (freqs < 8000))[:, None]

# Sentences of 3 s or more, spread over the validation set (and so over voices).
picks = [i for i, e in enumerate(val.index) if e["frames"] >= 120][:: 3][:N]
items = []
for k, i in enumerate(picks):
    asr, f0, n, audio, source, style, cycles0 = val.sentence(i)
    with torch.no_grad():
        out = decoder(asr[None], f0[None], n[None], style[None], source=source[None], cycles0=cycles0[None])[0, 0]
    spec = lambda a: torch.stft(a, N_FFT, HOP, window=window, return_complex=True)  # noqa: E731
    S, T = spec(out), spec(audio)
    # Voiced frames: F0 at 80 per second, resampled to the STFT's frames.
    voiced = torch.nn.functional.interpolate((f0 > 10).float()[None, None], size=S.shape[-1], mode="nearest")[0, 0] > 0
    def wav(Z):
        return torch.istft(Z, N_FFT, HOP, window=window, length=audio.shape[0])
    variants = {
        "teacher": audio,
        "decoder": out,
        "dmag_tphase": wav(torch.polar(S.abs(), T.angle())),
        "tmag_dphase": wav(torch.polar(T.abs(), S.angle())),
        "decoder_tphase_2_8k": wav(torch.where(band & voiced[None, :], torch.polar(S.abs(), T.angle()), S)),
        "decoder_tphase_voiced": wav(torch.where(voiced[None, :], torch.polar(S.abs(), T.angle()), S)),
        "decoder_tphase_voiced_low": wav(torch.where((freqs < 2000)[:, None] & voiced[None, :], torch.polar(S.abs(), T.angle()), S)),
        "decoder_tphase_unvoiced": wav(torch.where(~voiced[None, :], torch.polar(S.abs(), T.angle()), S)),
        # No teacher: the phase of white noise's own STFT where the sentence is unvoiced.
        "decoder_noise_phase_unvoiced": wav(torch.where(
            ~voiced[None, :], torch.polar(S.abs(), spec(torch.randn_like(out)).angle()), S)),
    }
    files = {}
    for name, a in variants.items():
        files[name] = f"{k:02d}-{name}.wav"
        sf.write(os.path.join(OUT, files[name]), a.float().cpu().numpy(), vm.SAMPLE_RATE)
    e = val.index[i]
    items.append({"voice": e["voice"], "source": f"Validation sentence {i}", "text": e["phonemes"], "files": files})
    print(k, i, e["voice"], round(e["frames"] / 40, 1), "s", flush=True)

json.dump(items, open(os.path.join(OUT, "samples.json"), "w"), indent=1, ensure_ascii=False)
json.dump({
    "labels": {
        "teacher": "Teacher (Kokoro's decoder)",
        "decoder": "Vocos-style decoder (candidate)",
        "dmag_tphase": "Decoder's magnitudes, teacher's phase",
        "tmag_dphase": "Teacher's magnitudes, decoder's phase",
        "decoder_tphase_2_8k": "Decoder, with the teacher's phase in voiced 2-8 kHz only",
        "decoder_tphase_voiced": "Decoder, with the teacher's phase in voiced frames only (all frequencies)",
        "decoder_tphase_voiced_low": "Decoder, with the teacher's phase in voiced frames below 2 kHz only",
        "decoder_tphase_unvoiced": "Decoder, with the teacher's phase in unvoiced frames only",
        "decoder_noise_phase_unvoiced": "Decoder, with white noise's phase in unvoiced frames (no teacher)",
    },
    "blind": ["teacher", "dmag_tphase"],
    "intro": (
        "<p>Where the Vocos-style decoder's robotic sound comes from. Each sentence (shown as phonemes) is from the "
        "teacher corpus's validation set: the decoder got the teacher's inputs and the exact harmonic source the "
        "teacher used, so the two line up sample for sample. Their spectrograms (n_fft 1024, hop 256) are then "
        "recombined. If <em>decoder's magnitudes, teacher's phase</em> sounds natural, the phase is at fault; if "
        "<em>teacher's magnitudes, decoder's phase</em> sounds robotic, likewise. The last variant swaps phase only "
        "in voiced 2-8 kHz, where Paradee (arXiv:2610.06817) found its buzz. The ones after it narrow down where the "
        "phase matters: voiced frames at all frequencies, voiced frames below 2 kHz, and unvoiced frames (consonants "
        "and silences). The last needs no teacher: unvoiced frames get the phase of white noise.</p>"
        "<p>Recombining spectrograms isn't perfectly clean (a magnitude and a phase from different signals don't "
        "make a consistent STFT), so a little roughness in the swapped versions is the method, not the decoder.</p>"),
}, open(os.path.join(OUT, "page.json"), "w"), indent=1)
