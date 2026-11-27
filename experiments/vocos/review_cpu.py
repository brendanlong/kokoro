"""CPU checks of the Vocos-style decoder that need no corpus: a few validation sentences are run
through the pipeline the corpus was made with (LSTM-swap text side, batch-first F0/N predictor).

1. Crops: training scores a crop's middle and assumes MARGIN frames each side give it the context
   a whole sentence would. Here the decoder runs on whole sentences and on crops with several
   margins, and the scored middles are compared.
2. Where the time goes per stage with each decoder, on N CPU threads.
3. Where the Vocos decoder's audio differs from its teacher's: log-magnitude error by frequency
   band, for voiced and unvoiced frames, and how much of each band's energy sits between the
   harmonics rather than on them (the teacher's breathiness, which spectral losses average away).

usage: review_cpu.py DATA_DIR STUDENTS_PT FRAME_SIDE_PT DECODER_PT OUT_JSON [THREADS]
"""
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lstm_swap"))
sys.path.insert(0, os.path.join(HERE, "..", "batch_first"))
if os.environ.get("VOCOS_MODEL_DIR"):  # an older model.py, for a checkpoint made with it
    sys.path.insert(0, os.environ["VOCOS_MODEL_DIR"])

from kokoro import KModel  # noqa: E402
import adaln  # noqa: E402
import model as vm  # noqa: E402
import seqfwd  # noqa: E402
import students as st  # noqa: E402

logger.remove()
DATA, STUDENTS, FRAME_SIDE, DECODER, OUT = sys.argv[1:6]
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 4
torch.set_num_threads(THREADS)
N_SENT = 12
CROP = 128
rng = random.Random(0)

kokoro = KModel(repo_id="hexgrad/Kokoro-82M").eval()
st.install(kokoro, st.load_students(kokoro, STUDENTS))
front = adaln.make_student(kokoro)
adaln.load_frame_side(front, torch.load(FRAME_SIDE, weights_only=True)["state"])
front = front.eval()
teacher_dec = kokoro.decoder
gen = teacher_dec.generator
vocos = vm.load(teacher_dec, DECODER).eval()

voices = {v: torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{v}.pt"), weights_only=True)
          for v in ("af_heart", "am_adam", "bf_emma", "bm_george")}
lines = open(os.path.join(DATA, "val.txt")).read().split("\n")[:-1][:60]
lines = sorted(rng.sample(lines, N_SENT), key=len)


def timed(times, key, fn):
    t = time.perf_counter()
    out = fn()
    times[key] = times.get(key, 0) + time.perf_counter() - t
    return out


def logmag(a, n_fft=1200, hop=300):
    w = torch.hann_window(n_fft)
    return torch.stft(a, n_fft, hop, window=w, return_complex=True).abs().clamp(min=1e-5).log()


times, crop_rows, band_rows, audio_s = {}, [], [], 0.0
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
freqs = torch.arange(vm.BINS) * vm.SAMPLE_RATE / vm.N_FFT
with torch.inference_mode():
    for ps in lines:
        voice = rng.choice(sorted(voices))
        ids = torch.LongTensor([[0, *[kokoro.vocab[p] for p in ps if p in kokoro.vocab], 0]])
        ref = voices[voice][len(ps) - 1]
        d, _, pred_dur = timed(times, "bert+durations", lambda: seqfwd.text_half(front, ids, ref))
        aln = seqfwd.alignment(pred_dur)
        F0, N = timed(times, "f0+n", lambda: seqfwd.prosody(front.predictor, d.transpose(-1, -2) @ aln, ref[:, 128:]))
        asr = timed(times, "text encoder", lambda: seqfwd.text_encoder(front.text_encoder, ids) @ aln)
        torch.manual_seed(1)
        source = timed(times, "harmonic source", lambda: vocos.harmonic_source(F0))
        torch.manual_seed(1)
        target = timed(times, "kokoro decoder", lambda: teacher_dec(asr, F0, N, ref[:, :128])).squeeze()
        full = timed(times, "vocos decoder", lambda: vocos(asr, F0, N, ref[:, :128], source=source))[0, 0]
        t = asr.shape[-1]
        audio_s += t * 600 / 24000

        # 1. crops against the whole sentence
        for margin in (8, 16, 24, 32, 48):
            span = (CROP + 2 * margin) // 2
            if t < span + 2:
                continue
            for _ in range(2):
                a = rng.randrange(t - span + 1)
                out = vocos(asr[:, :, a : a + span], F0[:, 2 * a : 2 * (a + span)], N[:, 2 * a : 2 * (a + span)],
                            ref[:, :128], source=source[:, 600 * a : 600 * (a + span)])[0, 0]
                sl = slice(margin * vm.HOP, (margin + CROP) * vm.HOP)
                whole = full[600 * a :][sl]
                err = out[sl] - whole
                crop_rows.append({"margin": margin, "snr_db": float(10 * torch.log10(whole.pow(2).sum() / err.pow(2).sum())),
                                  "logmag_l1": float((logmag(out[sl]) - logmag(whole)).abs().mean())})

        # 3. the decoder against its teacher, by band and voicing
        A, B = logmag(full), logmag(target)  # [bins, 2T + 1]
        voiced = torch.nn.functional.pad(F0[0] > 10, (0, 1))
        f0 = torch.nn.functional.pad(F0[0], (0, 1)).clamp(min=10)
        # a bin is "between harmonics" when it's 0.3-0.7 of the way from one harmonic to the next
        frac = (freqs[:, None] / f0[None, :]) % 1
        between = (frac > 0.3) & (frac < 0.7)
        for lo, hi in BANDS:
            band = ((freqs >= lo) & (freqs < hi))[:, None]
            for name, frames in (("voiced", voiced), ("unvoiced", ~voiced)):
                m = band & frames[None, :]
                if m.sum() == 0:
                    continue
                row = {"band": f"{lo}-{hi}", "frames": name, "n": int(m.sum()),
                       "abs_err": float((A - B)[m].abs().mean()), "bias": float((A - B)[m].mean())}
                if name == "voiced":
                    on, off = m & ~between, m & between
                    # log-magnitude dip between harmonics: bigger means cleaner harmonics, less breath
                    row["dip_teacher"] = float(B[on].mean() - B[off].mean())
                    row["dip_vocos"] = float(A[on].mean() - A[off].mean())
                band_rows.append(row)


def pool(rows, keys, vals):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return [{**dict(zip(keys, k)), **{v: round(float(np.average([r[v] for r in rs if v in r],
                                                                weights=[r.get("n", 1) for r in rs if v in r])), 3)
                                      for v in vals if any(v in r for r in rs)}} for k, rs in out.items()]


summary = {
    "threads": THREADS, "sentences": len(lines), "audio_s": round(audio_s, 1),
    "stage_ms_per_audio_s": {k: round(1000 * v / audio_s, 2) for k, v in times.items()},
    "crops": pool(crop_rows, ["margin"], ["snr_db", "logmag_l1"]),
    "bands": pool(band_rows, ["band", "frames"], ["abs_err", "bias", "dip_teacher", "dip_vocos"]),
    "params_m": {"vocos": round(sum(p.numel() for p in vocos.parameters() if p.requires_grad) / 1e6, 2),
                 "kokoro decoder": round(sum(p.numel() for p in teacher_dec.parameters()) / 1e6, 2)},
}
print(json.dumps(summary, indent=1))
with open(OUT, "w") as f:
    json.dump(summary, f, indent=1)
