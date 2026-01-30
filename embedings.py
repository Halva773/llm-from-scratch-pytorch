import torch.nn as nn
import torch


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)

        
if __name__ == "__main__":
    vocab_size = 1000
    emb_size = 200

    x = torch.tensor([
                        [113, 456, 76, 345],
                        [345, 678, 454, 546]
                    ])

    model = TokenEmbeddings(vocab_size=1000, emb_size=128)
    out = model(x)
    print(out.shape)