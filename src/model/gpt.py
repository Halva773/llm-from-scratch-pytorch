import torch.nn as nn
import torch

from model.embedings import TokenEmbeddings, PositionalEmbeddings
from model.decoder import Decoder
from model.dataLoader import DataLoader


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
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout_float = dropout
        self.device = device
        
        self.tokenEmbedings = TokenEmbeddings(self.vocab_size, self.emb_size)
        self.positionalEmbeddings = PositionalEmbeddings(self.max_seq_len, self.emb_size)
        self.dropout = nn.Dropout(self.dropout_float)
        self.decoders = nn.ModuleList([
                Decoder(self.num_heads, self.emb_size, self.head_size, self.max_seq_len, self.dropout_float) 
                for _ in range(self.num_layers)
            ])
        self.linear = nn.Linear(self.emb_size, self.vocab_size)

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
    
    def generate(   
                    self, 
                    x: torch.Tensor,
                    max_new_tokens: int,
                    do_sample: bool = False,
                    temperature: float = 1.0,
                    top_k: int = None,
                    top_p: float = None
                ):
        
        
        
        for _ in range(max_new_tokens):
            O = x[:, -self.max_seq_len:]
            logit = self.forward(O)
            # Get the last token's logits
            logits = logit[:, -1, :]
            # Scale logits by temperature
            logits = logits / temperature

            if do_sample and top_k is not None:
                values, _ = torch.topk(logits, top_k, dim=-1)
                min_topk = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_topk,
                    torch.full_like(logits, -float("inf")),
                    logits
                )

            if do_sample and top_p is not None:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                keep = cum_probs <= top_p
                keep[..., 0] = 1 

                mask = torch.full_like(logits, -float("inf"))
                mask.scatter_(
                    dim=-1,
                    index=sorted_idx,
                    src=torch.where(
                        keep,
                        logits.gather(-1, sorted_idx),
                        torch.full_like(sorted_probs, -float("inf"))
                    )
                )
                logits = mask

            probs = torch.softmax(logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            
        return x


    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)
    

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, num_epochs: int, learning_rate: float):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.to(device)        

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.losses = [{'train': [], 'valid': []}]
        cross_entropy = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.train()
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self(x)
                targets = y
                B, T, V = logits.shape
                
                logits_flat = logits.view(B * T, V)
                targets_flat = targets.view(B * T)

                train_loss = cross_entropy(logits_flat, targets_flat)
                self.losses[0]['train'].append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = self(x)
                    targets = y

                    B, T, V = logits.shape
                    logits_flat = logits.view(B * T, V)
                    targets_flat = targets.view(B * T)

                    loss = cross_entropy(logits_flat, targets_flat)
                    self.losses[0]['valid'].append(loss.item())
            print(f"Epoch {epoch + 1}/{num_epochs} completed. Train Loss: {train_loss.item():.4f}, Valid Loss: {loss.item():.4f}")
            self.save('../savepoints/gpt1.pth')


    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model





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
    gpt.save("data/gpt_model.pth")

    gpt = GPT.load("data/gpt_model.pth", device=device)
    print(out, out.shape)