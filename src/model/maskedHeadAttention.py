import torch.nn as nn
import torch


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.Wk = nn.Linear(emb_size, head_size)
        self.Wq = nn.Linear(emb_size, head_size)
        self.Wv = nn.Linear(emb_size, head_size)

        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))


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



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        heads_out = [head(x) for head in self.heads]
        O = torch.cat(heads_out, dim=-1)
        O = self.linear(O)
        O = self.dropout(O)
        return O


if __name__ == "__main__":
    emb_size = 128
    head_size = 32
    max_seq_len = 512
    num_heads = 8

    x = torch.randn(2, max_seq_len, emb_size)  # batch_size x seq_len x emb_size

    model = MultiHeadAttention(num_heads=num_heads, emb_size=emb_size, head_size=head_size, max_seq_len=max_seq_len)
    out = model.forward(x)
    print(out.shape)  # Expected output shape: (2, max_seq_len, head_size)