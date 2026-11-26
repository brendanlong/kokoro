"""Per-frame normalization in place of Kokoro's per-sentence AdaIN, for a batch-first frame side.

AdaIN1d normalizes each channel over the whole sentence (instance norm), so its output at one
frame depends on every frame of the sentence. AdaLN1d normalizes each frame over its channels
(layer norm), so it depends only on that frame: sentences packed end to end, or cut into crops,
compute what they would alone. It keeps AdaIN1d's parameters and their names (norm.weight,
norm.bias, fc), so a student starts from the teacher's weights and loads where the teacher did.
"""

import copy

import torch.nn as nn
import torch.nn.functional as F
from kokoro.istftnet import AdaIN1d


class AdaLN1d(nn.Module):
    def __init__(self, adain: AdaIN1d):
        super().__init__()
        self.norm = copy.deepcopy(adain.norm)  # only for its affine weight and bias
        self.fc = copy.deepcopy(adain.fc)

    def forward(self, x, s):  # x [B, C, T], s [B, style]
        h = self.fc(s)[..., None]
        gamma, beta = h.chunk(2, dim=1)
        y = F.layer_norm(x.transpose(1, 2), (x.shape[1],)).transpose(1, 2)
        if self.norm.affine:
            y = y * self.norm.weight[:, None] + self.norm.bias[:, None]
        return (1 + gamma) * y + beta


def install(module: nn.Module) -> int:
    """Replace every AdaIN1d under module with an AdaLN1d holding its weights; returns how many."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, AdaIN1d):
            setattr(module, name, AdaLN1d(child))
            n += 1
        else:
            n += install(child)
    return n


# The frame side: what changes in a batch-first model, and what its checkpoints hold.
FRAME_SIDE = ("predictor.F0", "predictor.N", "predictor.F0_proj", "predictor.N_proj", "decoder")


def make_student(model: nn.Module) -> nn.Module:
    """A copy of model with AdaLN1d in its F0/N predictor and decoder."""
    student = copy.deepcopy(model)
    n = install(student.predictor.F0) + install(student.predictor.N) + install(student.decoder)
    assert n > 0
    return student


def frame_side_state(model: nn.Module) -> dict:
    return {f"{p}.{k}": v for p in FRAME_SIDE for k, v in model.get_submodule(p).state_dict().items()}


def load_frame_side(model: nn.Module, state: dict) -> None:
    for p in FRAME_SIDE:
        model.get_submodule(p).load_state_dict({k[len(p) + 1:]: v for k, v in state.items() if k.startswith(p + ".")})
