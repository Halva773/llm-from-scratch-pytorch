import torch.nn as nn
import torch

from model.maskedHeadAttention import MultiHeadAttention
from model.ffn import FeedForward

class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: int = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.ff = FeedForward(emb_size=emb_size, dropout=dropout)
        self.first_norm = nn.LayerNorm(emb_size)
        self.second_norm = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor):
        O = self.mha(x)
        O += x
        Ow = self.first_norm(O)
        Ok = self.ff(Ow)
        Ok += Ow
        Oj = self.second_norm(Ok)
        return Oj
    
if __name__ == "__main__":
    batch_size = 1
    seq_len = 12
    emb_size = 12
    num_heads = 5
    head_size = 8
    max_seq_len = 20
    dropout = 0.1

    

    decoder = Decoder(num_heads, emb_size, head_size, max_seq_len, dropout)

    x = torch.randn(batch_size, seq_len, emb_size)

    O = decoder.forward(x)
    print(O.shape)
