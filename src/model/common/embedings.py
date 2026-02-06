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
        

        
if __name__ == "__main__":
    vocab_size = 1000
    emb_size = 200

    x = torch.tensor([
                        [113, 456, 76, 345],
                        [345, 678, 454, 546]
                    ])

    model = PositionalEmbeddings(max_seq_size=1000, emb_size=6)
    out = model(seq_len=4)
    print(out, out.shape)
