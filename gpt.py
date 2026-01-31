import torch.nn as nn
import torch

from embedings import TokenEmbeddings, PositionalEmbeddings
from decoder import Decoder


class GPT(nn.Module):
    def __init__(self, 
                vocab_size: int, 
                max_seq_len: int, 
                emb_size: int,
                num_heads: int,
                head_size: int,
                num_layers: int,
                dropout: int = 0.1,
                device: str = 'cpu'
                ):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.tokenEmbedings = TokenEmbeddings(vocab_size, emb_size)
        self.positionalEmbeddings = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.decoders = nn.ModuleList([
                Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) 
                for _ in range(num_layers)
            ])
        self.linear = nn.Linear(emb_size, vocab_size)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        
        tokenEmbeddings = self.tokenEmbedings(x)
        posEmbeddings = self.positionalEmbeddings(T)

        embeddings = tokenEmbeddings + posEmbeddings
        out = self.dropout(embeddings)
        for decoder in self.decoders:
            out = decoder(out)
        out = self.linear(out)
        return out
    
    def generate(self, x: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            O = x[:, -self.max_seq_len:]
            logit = self.forward(O)
            # Get the last token's logits
            logits = logit[:, -1, :]
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
        return x






if __name__ == "__main__":
    vocab_size = 10000
    batch_size = 1
    seq_len = 12
    emb_size = 12
    num_heads = 5
    head_size = 8
    max_seq_len = 20
    num_layers = 8
    dropout = 0.1

    device = 'cpu'

    gpt = GPT(vocab_size, max_seq_len, emb_size, num_heads, head_size, num_layers, dropout, device)


    x = torch.tensor([
                    [113, 456, 76, 345],
                    [345, 678, 454, 546]
                ])
    out = gpt.generate(x, max_new_tokens=10)
    print(out, out.shape)