import torch
from torch import nn


class SingleNeuronModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()


def infer_device(module):
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")
