import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2)/torch.pi)*(x+torch.tensor(0.044715)*torch.pow(x, 3))))
