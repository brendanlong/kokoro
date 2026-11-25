"""KModel.forward_with_tokens for batch size 1, rewritten without pack_padded_sequence so each
LSTM can be swapped for any module mapping [B, T, C_in] -> [B, T, 512]."""
import torch
import torch.nn as nn
from kokoro.istftnet import AdaINResBlock1

LSTM_PATHS = [
    "text_encoder.lstm",
    "predictor.text_encoder.lstms.0",
    "predictor.text_encoder.lstms.2",
    "predictor.text_encoder.lstms.4",
    "predictor.lstm",
    "predictor.shared",
]


def adain_resblock_forward(self, x, s):
    # Same as kokoro's, but indexing instead of zip() over a ParameterList, which torch.compile can't trace
    for i in range(len(self.convs1)):
        a1, a2 = self.alpha1[i], self.alpha2[i]
        xt = self.adain1[i](x, s)
        xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)
        xt = self.convs1[i](xt)
        xt = self.adain2[i](xt, s)
        xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
        xt = self.convs2[i](xt)
        x = xt + x
    return x


AdaINResBlock1.forward = adain_resblock_forward


def run_seq(mod, x):
    return mod(x)[0] if isinstance(mod, nn.LSTM) else mod(x)


def text_encoder(te, input_ids):
    x = te.embedding(input_ids).transpose(1, 2)
    for c in te.cnn:
        x = c(x)
    return run_seq(te.lstm, x.transpose(1, 2)).transpose(1, 2)


def duration_encoder(de, x, s):
    st = s[:, None, :].expand(-1, x.shape[2], -1)
    x = torch.cat([x.transpose(1, 2), st], -1)
    for i in range(0, len(de.lstms), 2):
        x = run_seq(de.lstms[i], x)
        x = de.lstms[i + 1](x, s)
        x = torch.cat([x, st], -1)
    return x


def text_half(m, input_ids, ref_s, speed: float = 1.0):
    """Everything up to the durations. Returns (d, continuous duration, rounded duration)."""
    s = ref_s[:, 128:]
    bert = m.bert(input_ids, attention_mask=torch.ones_like(input_ids))
    d = duration_encoder(m.predictor.text_encoder, m.bert_encoder(bert).transpose(-1, -2), s)
    x = run_seq(m.predictor.lstm, d)
    duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / speed
    return d, duration, torch.round(duration).clamp(min=1).long()[0]


def alignment(pred_dur):
    frames = torch.repeat_interleave(torch.arange(pred_dur.shape[0], device=pred_dur.device), pred_dur)
    return (frames[None, :] == torch.arange(pred_dur.shape[0], device=pred_dur.device)[:, None]).float()[None]


def prosody(p, en, s):
    x = run_seq(p.shared, en.transpose(-1, -2)).transpose(-1, -2)
    F0, N = x, x
    for block in p.F0:
        F0 = block(F0, s)
    for block in p.N:
        N = block(N, s)
    return p.F0_proj(F0).squeeze(1), p.N_proj(N).squeeze(1)


def acoustic_half(m, input_ids, ref_s, d, aln):
    """Everything after the durations: returns audio and the decoder's inputs."""
    en = d.transpose(-1, -2) @ aln.to(d.dtype)
    F0, N = prosody(m.predictor, en, ref_s[:, 128:])
    t_en = text_encoder(m.text_encoder, input_ids)
    asr = t_en @ aln.to(t_en.dtype)
    audio = m.decoder(asr, F0, N, ref_s[:, :128]).squeeze()
    return audio, F0, N, t_en


def forward(m, input_ids, ref_s, speed=1.0, forced_dur=None, th=text_half, ah=acoustic_half):
    d, duration, pred_dur = th(m, input_ids, ref_s, speed)
    if forced_dur is not None:
        pred_dur = forced_dur
    audio, F0, N, t_en = ah(m, input_ids, ref_s, d, alignment(pred_dur))
    return dict(audio=audio, pred_dur=pred_dur, duration=duration, F0=F0, N=N, t_en=t_en, d=d)
