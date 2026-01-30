import torch.nn as nn
import torch


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.Wk = nn.Linear(emb_size, head_size)
        self.Wq = nn.Linear(emb_size, head_size)
        self.Wv = nn.Linear(emb_size, head_size)

        self.mask = torch.tril(torch.randn(max_seq_len, max_seq_len), diagonal=0)


    def forward(self, x: torch.Tensor):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description (batch_size x seq_len x emb_size)
        :type x: torch.Tensor
        """
        Q = self.Wq(x) # Q = x ⋅ Wq.⊤ + b
        K = self.Wk(x)
        V = self.Wv(x)

        attn_weights = Q @ K.transpose(-1, -2) / (K.size(-1) ** 0.5)

        mask = self.mask[:K.size(1), :K.size(1)]

        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1) 
        out = attn_weights @ V
        return out


if __name__ == "__main__":
    emb_size = 128
    head_size = 64
    max_seq_len = 10

    x = torch.randn(2, max_seq_len, emb_size)  # batch_size x seq_len x emb_size

    model = HeadAttention(emb_size=emb_size, head_size=head_size, max_seq_len=max_seq_len)
    out = model.forward(x)
    print(out.shape)  # Expected output shape: (2, max_seq_len, head_size)