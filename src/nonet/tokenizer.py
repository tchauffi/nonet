import torch
from torch import nn

class SudokuTokenizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: str) -> torch.Tensor:
        return torch.tensor([int(c) for c in x], dtype=torch.long)
