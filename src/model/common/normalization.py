import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        rmsn = self.w * x / torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        return rmsn