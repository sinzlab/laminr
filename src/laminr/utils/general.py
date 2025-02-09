import torch
from torch import nn


class SingleNeuronModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()


def resolve_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device
