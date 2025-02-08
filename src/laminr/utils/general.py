from torch import nn


class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()
