"""How a Vocos-style decoder's audio differs from its teacher's, over whole validation sentences.

Beyond the spectral losses: phase errors over the whole spectrum and in voiced frames at 2-8 kHz
(where a decoder trained on spectra alone buzzes), the level of voiced and unvoiced frames, and
pitch as CREPE hears it in the audio (periodicity RMSE and voiced/unvoiced F1, as CARGAN defines
them, and pitch RMSE in cents where both are voiced).

usage: metrics.py VAL_CORPUS_DIR DECODER_PT OUT_JSON
"""
import json
import sys

import numpy as np
import torch
import torchcrepe

import common
import model as vm

CREPE_HOP = 240  # 10 ms at 24 kHz


def crepe(audio, dev):
    pitch, periodicity = torchcrepe.predict(audio[None], vm.SAMPLE_RATE, CREPE_HOP, 50.0, 800.0, model="full",
                                            decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                                            batch_size=256, device=dev)
    return pitch[0], periodicity[0]


@torch.no_grad()
def evaluate(decoder, val, spectral, dev):
    decoder.eval()
    window = torch.hann_window(vm.N_FFT, device=dev)
    freqs = torch.arange(vm.BINS, device=dev) * vm.SAMPLE_RATE / vm.N_FFT
    buzz_band = ((freqs >= 2000) & (freqs < 8000))[:, None]
    rows = []
    for i in range(len(val.index)):
        asr, f0, n, audio, source, style, _ = val.sentence(i)
        out = decoder(asr[None], f0[None], n[None], style[None], source=source[None])[0, 0]
        row = {k: v.item() for k, v in spectral(out, audio, common.TERMS).items()}

        X = torch.stft(out, vm.N_FFT, vm.HOP, window=window, return_complex=True)
        T = torch.stft(audio, vm.N_FFT, vm.HOP, window=window, return_complex=True)
        voiced = torch.nn.functional.pad(f0 > 10, (0, 1))[None, :]
        w = T.abs() ** 0.3 * (buzz_band & voiced)
        if w.sum() > 0:
            row.update({f"voiced_2_8k_{k}": v.item() for k, v in common.phase_errors(X, T, w).items()})

        db = lambda Z: 10 * torch.log10(Z.abs().pow(2).sum(0) + 1e-8)  # noqa: E731
        a, b = db(X), db(T)
        heard = b > b.max() - 50
        diff = (a - b)[heard]
        v = voiced[0][heard]
        row["level_abs_db"] = diff.abs().mean().item()
        row["level_bias_voiced_db"] = diff[v].mean().item() if v.any() else float("nan")
        row["level_bias_unvoiced_db"] = diff[~v].mean().item() if (~v).any() else float("nan")
        # Noise has a mean log-magnitude 0.29 below a smooth spectrum of the same energy, so a decoder
        # that plays a smooth sound where the teacher has noise scores about +0.29 here.
        unv = heard & ~voiced[0]
        if unv.any():
            row["unvoiced_logmag_bias"] = (X.abs().clamp(min=1e-5).log() - T.abs().clamp(min=1e-5).log())[:, unv].mean().item()

        (p_out, per_out), (p_ref, per_ref) = crepe(out, dev), crepe(audio, dev)
        row["periodicity_rmse"] = (per_out - per_ref).pow(2).mean().sqrt().item()
        vo, vr = per_out > 0.5, per_ref > 0.5
        row["_vuv"] = [(vo & vr).sum().item(), (vo & ~vr).sum().item(), (~vo & vr).sum().item()]
        both = vo & vr
        row["_cents"] = (1200 * torch.log2(p_out[both] / p_ref[both])).pow(2).tolist()
        row["_lang"] = val.index[i].get("lang", "a")
        rows.append(row)
    tp, fp, fn = (sum(r["_vuv"][j] for r in rows) for j in range(3))
    summary = {k: round(float(np.nanmean([r[k] for r in rows if k in r])), 5) for k in rows[0] if not k.startswith("_")}
    summary["vuv_f1"] = round(2 * tp / max(1, 2 * tp + fp + fn), 5)
    summary["pitch_rmse_cents"] = round(float(np.sqrt(np.mean([c for r in rows for c in r["_cents"]]))), 3)
    langs = sorted({r["_lang"] for r in rows})
    if len(langs) > 1:
        summary["by_lang"] = {lang: {k: round(float(np.nanmean([r[k] for r in rows if r["_lang"] == lang and k in r])), 4)
                                     for k in ("msmel", "stft", "phase", "level_abs_db", "periodicity_rmse")}
                              for lang in langs}
    return summary


if __name__ == "__main__":
    VAL, DECODER, OUT = sys.argv[1:4]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    summary = evaluate(common.make_decoder(dev, DECODER), common.Corpus(VAL, dev), common.Spectral(dev), dev)
    print("METRICS", json.dumps(summary), flush=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=1)
