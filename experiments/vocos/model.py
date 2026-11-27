"""A Vocos-style decoder for Kokoro: everything at one frame rate, then one inverse STFT.

Kokoro's decoder upsamples its features 300-fold through transposed convolutions and multi-scale
residual blocks, which is nearly all of the model's compute. This one runs a stack of ConvNeXt
blocks at 80 frames per second (300 samples per frame, twice Kokoro's frame rate, the rate of its
F0 and N curves) and predicts each frame's magnitude and phase for an inverse STFT (n_fft 1200,
hop 300), as Vocos (arXiv:2306.00814) does.

Inputs are what Kokoro's decoder gets: the phoneme features stretched to frames (asr), the F0 and
N curves, and the decoder half of the style vector. Like Kokoro's generator, it also gets the
harmonic source (Kokoro's SourceModuleHnNSF, the sines at F0 and 8 overtones plus noise), here as
its STFT at the decoder's own frames, which gives it a phase reference for the harmonics.

Everything but the depthwise convolutions is per frame, so sentences can be batched by padding or
packed end to end, and trained on crops. Each norm is a layer norm over a frame's channels, then a
scale and shift computed from the style and from that frame's F0 and N. A layer norm erases how
loud a frame is, and with N only at the input (v1) the level drifted by up to 2 dB within a
sentence, so every norm gets it again.

The forward pass has Kokoro's Decoder signature, so it drops into KModel.decoder.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 24000
HOP = 300  # samples per decoder frame: 80 frames per second
N_FFT = 4 * HOP
BINS = N_FFT // 2 + 1


FRAME_FEATURES = 3  # log F0 (0 when unvoiced), voiced, N


class AdaLN(nn.Module):
    """Layer norm over channels, scaled and shifted by each frame's conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):  # x [B, T, C], cond [B, T, cond_dim]
        gamma, beta = self.fc(cond).chunk(2, dim=-1)
        return F.layer_norm(x, (x.shape[-1],)) * (1 + gamma) + beta


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, mult: int, cond_dim: int, layer_scale: float):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = AdaLN(dim, cond_dim)
        self.pw1 = nn.Linear(dim, mult * dim)
        self.pw2 = nn.Linear(mult * dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x, cond, keep):  # x [B, T, C], keep [B, T, 1]
        h = self.dw((x * keep).transpose(1, 2)).transpose(1, 2)
        return x + self.scale * self.pw2(F.gelu(self.pw1(self.norm(h, cond))))


FULL_BAND_HZ = 11500  # the full-band source's harmonics stop here
MAX_MAGNITUDE = {"spec": 100.0, "filter": 400.0}  # "spec" as trained before; the teacher's largest bin is ~75
MAX_GAIN = 1e4


def full_band_source(f0, cycles0=None):
    """Every harmonic of F0 up to FULL_BAND_HZ, in phase with the fundamental of Kokoro's source.

    f0 [B, F] (Hz, 80 per second) -> [B, 300 F] samples. Kokoro's source has the fundamental and 8
    overtones; its generator makes the higher harmonics itself, locked to those. This is the sum of
    sin(k phase) for every k with k F0 under FULL_BAND_HZ, where phase is the fundamental's as
    Kokoro's SineGen computes it (a running sum of F0 per frame, interpolated to samples), so a
    filter on it can place harmonics where the teacher has them. `cycles0` [B] is the running sum
    before the first frame, for a crop of a sentence. The sum has a closed form, so the cost
    doesn't grow with the number of harmonics.
    """
    f0 = f0.double()
    cycles = ((f0 / SAMPLE_RATE) % 1).cumsum(1)
    if cycles0 is not None:
        cycles = cycles + cycles0.double()[:, None]
    cycles = F.interpolate(cycles[:, None] * HOP, scale_factor=HOP, mode="linear")[:, 0]
    half = math.pi * (cycles % 1)  # half the fundamental's phase
    f0 = f0.repeat_interleave(HOP, dim=1)
    voiced = f0 > 10
    k = torch.floor(FULL_BAND_HZ / f0.clamp(min=10)) * voiced
    denom = torch.sin(half)
    safe = denom.abs() > 1e-6
    total = torch.sin(k * half) * torch.sin((k + 1) * half) / torch.where(safe, denom, torch.ones_like(denom))
    return (0.1 * total * safe / k.clamp(min=1).sqrt()).float()


class VocosDecoder(nn.Module):
    """head "spec": the network's output is the spectrum's log magnitude and phase.
    head "filter": its output is a gain and a phase shift per frequency bin for each of two source
    spectra, Kokoro's harmonic source and (full_band) every harmonic up to FULL_BAND_HZ, and the
    spectrum is their filtered sum. The noise in Kokoro's source is all the randomness its decoder
    has, and it is too detailed to pass through the network, so a "spec" head can only produce an
    average of what the teacher did with it; a filter passes it through."""

    def __init__(self, source: nn.Module, asr_dim=512, style_dim=128, dim=512, layers=8, mult=3,
                 head="spec", full_band=False):
        super().__init__()
        assert head in MAX_MAGNITUDE and (head == "filter" or not full_band)
        self.config = dict(dim=dim, layers=layers, mult=mult, head=head, full_band=full_band)
        self.head_kind, self.full_band = head, full_band
        sources = 2 if full_band else 1
        # Kokoro's harmonic source, frozen: the sines at F0 and its overtones, mixed by one linear layer.
        self.source = copy.deepcopy(source).requires_grad_(False)
        cond_dim = style_dim + FRAME_FEATURES
        self.embed = nn.Conv1d(asr_dim + FRAME_FEATURES, dim, 7, padding=3)
        self.embed_source = nn.Linear(3 * BINS * sources, dim)
        self.norm = AdaLN(dim, cond_dim)
        self.blocks = nn.ModuleList(ConvNeXtBlock(dim, mult, cond_dim, 1 / layers) for _ in range(layers))
        self.final = AdaLN(dim, cond_dim)
        self.head = nn.Linear(dim, 2 * BINS * (sources if head == "filter" else 1))
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def harmonic_source(self, f0):
        """f0 [B, 2T] (Hz, 80 per second) -> [B, 600 T] samples, as Kokoro's generator makes it."""
        f0 = F.interpolate(f0[:, None], scale_factor=HOP, mode="nearest").transpose(1, 2)
        with torch.no_grad():
            har = self.source(f0)[0]
        return har.transpose(1, 2).squeeze(1)

    def _stft(self, x):
        return torch.stft(x.float(), N_FFT, HOP, window=self.window, center=True, return_complex=True)

    def forward(self, asr, F0_curve, N, s, source=None, cycles0=None):
        """asr [B, 512, T] (40 frames per second), F0_curve and N [B, 2T], s [B, 128] ->
        audio [B, 1, 600 T]. `source` is the harmonic source (computed from F0 if not given);
        `cycles0` is full_band_source's, for crops."""
        if source is None:
            source = self.harmonic_source(F0_curve)
        frames = F0_curve.shape[1]  # 2T; the STFT has one more (center=True)
        specs = [self._stft(source)]
        if self.full_band:
            specs.append(self._stft(full_band_source(F0_curve, cycles0)))
        src = torch.cat([f(sp) for sp in specs for f in (lambda z: torch.log(z.abs() + 1e-5),
                                                         lambda z: torch.cos(z.angle()),
                                                         lambda z: torch.sin(z.angle()))], dim=1)

        voiced = (F0_curve > 10).float()
        frame = torch.stack([torch.log(F0_curve.clamp(min=10)) * voiced, voiced, N.float()], dim=1)
        # 2T + 1 frames, as the STFT has.
        frame = F.pad(frame, (0, 1), mode="replicate")  # [B, 3, 2T + 1]
        asr2 = F.pad(asr.repeat_interleave(2, dim=2), (0, 1), mode="replicate")  # [B, 512, 2T + 1]
        x = self.embed(torch.cat([asr2, frame.to(asr2.dtype)], dim=1)).transpose(1, 2)
        x = x + self.embed_source(src.transpose(1, 2).to(x.dtype))
        frame = frame.transpose(1, 2)
        cond = torch.cat([s[:, None, :].expand(-1, frame.shape[1], -1).to(frame.dtype), frame], dim=-1)
        keep = torch.ones_like(x[..., :1])
        x = self.norm(x, cond)
        for block in self.blocks:
            x = block(x, cond, keep)
        x = self.head(self.final(x, cond)).transpose(1, 2).float()  # [B, 2 * BINS (* sources), 2T + 1]
        if self.head_kind == "spec":
            spec = torch.polar(torch.exp(x[:, :BINS].clamp(max=math.log(MAX_MAGNITUDE["spec"]))), x[:, BINS:])
        else:
            spec = 0
            for i, sp in enumerate(specs):
                gain, shift = x[:, 2 * i * BINS : (2 * i + 1) * BINS], x[:, (2 * i + 1) * BINS : (2 * i + 2) * BINS]
                spec = spec + torch.polar(torch.exp(gain.clamp(max=math.log(MAX_GAIN))), shift) * sp
            mag = spec.abs()
            spec = spec * (mag.clamp(max=MAX_MAGNITUDE["filter"]) / (mag + 1e-9))
        audio = torch.istft(spec, N_FFT, HOP, window=self.window, center=True, length=frames * HOP)
        return audio[:, None]


def from_kokoro(decoder: nn.Module, **kw) -> VocosDecoder:
    """A VocosDecoder with Kokoro decoder's harmonic source (its only reused part)."""
    return VocosDecoder(decoder.generator.m_source, **kw)


def load(decoder: nn.Module, path: str) -> VocosDecoder:
    """The VocosDecoder a checkpoint holds, built as its saved config says."""
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    vocos = from_kokoro(decoder, **ckpt.get("config", {}))
    vocos.load_state_dict(ckpt["state"])
    return vocos
