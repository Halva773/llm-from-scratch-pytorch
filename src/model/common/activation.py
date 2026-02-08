import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2)/torch.pi)*(x+torch.tensor(0.044715)*torch.pow(x, 3))))


class SiLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor):
        return x*torch.sigmoid(x)
    

class SwiGLU(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Linear(emb_size, 4 * emb_size)
        self.up = nn.Linear(emb_size, 4 * emb_size)
        self.down = nn.Linear(4 * emb_size, emb_size)
        self.silu = SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        gate = self.silu(self.gate(x))
        up = self.up(x)
        down = self.down(gate * up)
        out = self.dropout(down)
        return out