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


class VocosDecoder(nn.Module):
    def __init__(self, source: nn.Module, asr_dim=512, style_dim=128, dim=512, layers=8, mult=3):
        super().__init__()
        # Kokoro's harmonic source, frozen: the sines at F0 and its overtones, mixed by one linear layer.
        self.source = copy.deepcopy(source).requires_grad_(False)
        cond_dim = style_dim + FRAME_FEATURES
        self.embed = nn.Conv1d(asr_dim + FRAME_FEATURES, dim, 7, padding=3)
        self.embed_source = nn.Linear(3 * BINS, dim)
        self.norm = AdaLN(dim, cond_dim)
        self.blocks = nn.ModuleList(ConvNeXtBlock(dim, mult, cond_dim, 1 / layers) for _ in range(layers))
        self.final = AdaLN(dim, cond_dim)
        self.head = nn.Linear(dim, 2 * BINS)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    def harmonic_source(self, f0):
        """f0 [B, 2T] (Hz, 80 per second) -> [B, 600 T] samples, as Kokoro's generator makes it."""
        f0 = F.interpolate(f0[:, None], scale_factor=HOP, mode="nearest").transpose(1, 2)
        with torch.no_grad():
            har = self.source(f0)[0]
        return har.transpose(1, 2).squeeze(1)

    def forward(self, asr, F0_curve, N, s, source=None):
        """asr [B, 512, T] (40 frames per second), F0_curve and N [B, 2T], s [B, 128] ->
        audio [B, 1, 600 T]. `source` is the harmonic source (computed from F0 if not given)."""
        if source is None:
            source = self.harmonic_source(F0_curve)
        frames = F0_curve.shape[1]  # 2T; the STFT has one more (center=True)
        spec = torch.stft(source.float(), N_FFT, HOP, window=self.window, center=True, return_complex=True)
        src = torch.cat([torch.log(spec.abs() + 1e-5), torch.cos(spec.angle()), torch.sin(spec.angle())], dim=1)

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
        # Contiguous: compiled, torch.complex fails on the transposed layout (a stride assertion in torch 2.8).
        x = self.head(self.final(x, cond)).transpose(1, 2).float().contiguous()  # [B, 2 * BINS, 2T + 1]
        mag = torch.exp(x[:, :BINS].clamp(max=math.log(100.0)))
        phase = x[:, BINS:]
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        audio = torch.istft(spec, N_FFT, HOP, window=self.window, center=True, length=frames * HOP)
        return audio[:, None]


def from_kokoro(decoder: nn.Module, **kw) -> VocosDecoder:
    """A VocosDecoder with Kokoro decoder's harmonic source (its only reused part)."""
    return VocosDecoder(decoder.generator.m_source, **kw)
