import torch.nn as nn
import torch


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int, rope = None):
        super().__init__()
        self.Wk = nn.Linear(emb_size, head_size)
        self.Wq = nn.Linear(emb_size, head_size)
        self.Wv = nn.Linear(emb_size, head_size)
        self.rope = rope

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )


    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description (batch_size x seq_len x emb_size) 
        :type x: torch.Tensor
        """
        Q = self.Wq(x) # Q = x ⋅ Wq.⊤ + b
        K = self.Wk(x)
        V = self.Wv(x)

        if self.rope is not None:
            seq_len = x.size(1)
            start_pos = cache[0].size(1) if cache is not None else 0
            Q = self.rope(Q, seq_len, start_pos)
            K = self.rope(K, seq_len, start_pos)

        if cache is not None:
            key_cache, value_cache = cache
            K = torch.cat([key_cache, K], dim=1)
            V = torch.cat([value_cache, V], dim=1)


        attn_weights = Q @ K.transpose(-1, -2) / (K.size(-1) ** 0.5)

        if cache is None:
            T = K.size(1)
            mask = self.mask[:T, :T]
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
             
        attn_weights = torch.softmax(attn_weights, dim=-1) 
        out = attn_weights @ V
        if use_cache:
            return out, (K, V)
        else:
            return out, None



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, rope = None, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention(emb_size, head_size, max_seq_len, rope) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None):
        heads_out = [head(x, use_cache=use_cache, cache=cache[i] if cache is not None else None) for i, head in enumerate(self.heads)]
        new_cache = []
        outs = []
        for i, (out, head_cache) in enumerate(heads_out):
            if head_cache is not None:
                new_cache.append(head_cache) 
            outs.append(out)
        O = torch.cat(outs, dim=-1)
        O = self.linear(O)
        O = self.dropout(O)
        if use_cache:
            return O, new_cache
        else:
            return O, None


if __name__ == "__main__":
    emb_size = 128
    head_size = 32
    max_seq_len = 512
    num_heads = 8

    x = torch.randn(2, max_seq_len, emb_size)  # batch_size x seq_len x emb_size

    model = MultiHeadAttention(num_heads=num_heads, emb_size=emb_size, head_size=head_size, max_seq_len=max_seq_len)
    out = model.forward(x, use_cache=False, cache=None)
    print(out.shape)  # Expected output shape: (2, max_seq_len, head_size)
