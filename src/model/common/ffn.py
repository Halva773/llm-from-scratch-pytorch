import torch.nn as nn
import torch



class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: int = 0.1, ffn_norm = nn.ReLU()):
        super().__init__()
        self.first_linear = nn.Linear(emb_size, 4*emb_size)
        self.relu = ffn_norm
        self.second_linear = nn.Linear(4*emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        O = self.first_linear(x)
        O = self.relu(O)
        O = self.second_linear(O)
        O = self.dropout(O)

        return O
    


if __name__ == "__main__":
    emb_size = 128
    ffn = FeedForward(emb_size)
    x = torch.randn(2, 512, emb_size)

    O = ffn.forward(x)
    print(O.shape)