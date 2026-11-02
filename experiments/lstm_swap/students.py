import torch
import torch.nn as nn
import torch.nn.functional as F

from seqfwd import LSTM_PATHS


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel, dilation, mult):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, int(mult * dim))
        self.pw2 = nn.Linear(int(mult * dim), dim)

    def forward(self, x):
        h = self.dw(x.transpose(1, 2)).transpose(1, 2)
        return x + self.pw2(F.gelu(self.pw1(self.norm(h))))


class ConvSeq(nn.Module):
    """Drop-in for a bidirectional LSTM: [B, T, d_in] -> [B, T, d_out]."""

    def __init__(self, d_in, d_out, n_blocks, kernel, mult):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.blocks = nn.ModuleList(ConvNeXtBlock(d_out, kernel, 3**i, mult) for i in range(n_blocks))

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return x


DEFAULT_ARCH = dict(n_blocks=2, kernel=7, mult=1.5)


def make_students(model, arch):
    """arch maps an LSTM path to overrides of DEFAULT_ARCH."""
    return {
        path: ConvSeq(lstm.input_size, 2 * lstm.hidden_size, **{**DEFAULT_ARCH, **arch.get(path, {})})
        for path in LSTM_PATHS
        for lstm in [model.get_submodule(path)]
    }


def save_students(studs, arch, file):
    state = {f"{path}.{k}": v for path, m in studs.items() for k, v in m.state_dict().items()}
    torch.save({"arch": arch, "state": state}, file)


def load_students(model, file):
    ckpt = torch.load(file, weights_only=True)
    studs = make_students(model, ckpt["arch"])
    for path, m in studs.items():
        m.load_state_dict({k[len(path) + 1:]: v for k, v in ckpt["state"].items() if k.startswith(path + ".")})
    return studs


def install(model, students):
    for path, mod in students.items():
        parent, name = path.rsplit(".", 1)
        parent = model.get_submodule(parent)
        if isinstance(parent, nn.ModuleList):
            parent[int(name)] = mod
        else:
            setattr(parent, name, mod)
