"""HiFi-GAN's multi-period discriminator and UnivNet's multi-resolution spectrogram discriminator
(as Vocos and Paradee use), with least-squares GAN losses and feature matching."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    """The waveform folded into columns of `period` samples, then 2-D convolutions along time."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), padding=(2, 0))) for a, b in zip(chans, chans[1:]))
        self.convs.append(weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))))
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):  # x [B, S]
        x = x[:, None]
        pad = (-x.shape[-1]) % self.period
        x = F.pad(x, (0, pad), mode="reflect").view(x.shape[0], 1, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ResolutionDiscriminator(nn.Module):
    """2-D convolutions over a magnitude spectrogram at one resolution."""

    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            *[weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)],
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):  # x [B, S]
        spec = torch.stft(x.float(), self.n_fft, self.hop, window=self.window, return_complex=True).abs()
        x = spec.transpose(1, 2)[:, None]  # [B, 1, frames, bins]
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = nn.ModuleList(PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11))
        self.mrd = nn.ModuleList(ResolutionDiscriminator(n, n // 4) for n in (512, 1024, 2048))

    def forward(self, x):
        return [d(x) for d in [*self.mpd, *self.mrd]]


def discriminator_loss(real, fake):
    """real, fake: Discriminators outputs for real and (detached) generated audio."""
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2) for (r, _), (f, _) in zip(real, fake))


def generator_losses(real, fake):
    """Adversarial loss and feature matching (L1 between each layer's activations, real detached)."""
    adv = sum(torch.mean((1 - f) ** 2) for f, _ in fake)
    fm = sum(F.l1_loss(ff, rf.detach()) for (_, rfs), (_, ffs) in zip(real, fake) for rf, ff in zip(rfs, ffs))
    return adv, fm
