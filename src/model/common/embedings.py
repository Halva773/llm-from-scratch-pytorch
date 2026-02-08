import torch.nn as nn
import torch


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)
    


class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_size, emb_size):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_size, emb_size)

    def forward(self, seq_len: int, start_pos: int = 0):
        positions = torch.arange(start_pos, start_pos+seq_len, device=self.pos_embedding.weight.device)
        return self.pos_embedding(positions)
        



class RoPE(nn.Module):
    def __init__(self, head_size: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self._build_rotations(head_size, max_seq_len, base)

    def _build_rotations(self, head_size: int, max_seq_len: int, base: int):
        i = torch.arange(0, head_size // 2, dtype=torch.float32)
        O = 1 / (base ** (2 * i / head_size))

        seq_len = torch.arange(0, max_seq_len, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", seq_len, O)
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0):
        _, seq_len, _ = x.shape
        matrix_cos = self.cos[start_pos:start_pos+seq_len].unsqueeze(0)
        matrix_sin = self.sin[start_pos:start_pos+seq_len].unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated_even = x_even * matrix_cos - x_odd * matrix_sin
        x_rotated_odd = x_odd * matrix_cos + x_even * matrix_sin

        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x_rotated_even
        x_out[..., 1::2] = x_rotated_odd
        return x_out





if __name__ == "__main__":
    vocab_size = 1000
    emb_size = 200

    x = torch.tensor([
                        [113, 456, 76, 345],
                        [345, 678, 454, 546]
                    ])

    model = RoPE(max_seq_len=1000, head_size=6)