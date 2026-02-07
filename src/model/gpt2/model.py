from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from pathlib import Path
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from model.common.embedings import TokenEmbeddings, PositionalEmbeddings
from model.common.maskedHeadAttention import MultiHeadAttention
from model.common.ffn import FeedForward

class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2)/torch.pi)*(x+torch.tensor(0.044715)*torch.pow(x, 3))))


class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: int = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.ff = FeedForward(emb_size=emb_size, dropout=dropout, ffn_norm=GELU())
        self.first_norm = nn.LayerNorm(emb_size)
        self.second_norm = nn.LayerNorm(emb_size)


    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None):
        O = self.first_norm(x)
        Omha, new_cache = self.mha(O, use_cache=use_cache, cache=cache)
        Omha += x
        outs = self.second_norm(Omha)
        logits = self.ff(outs)
        logits += Omha
        return logits, new_cache if use_cache else None



class GPT2(nn.Module):
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
        self.ln = nn.LayerNorm(self.emb_size)

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None):
        B, T = x.shape
        
        tokenEmbeddings = self.tokenEmbedings(x)
        if cache is None:
            posEmbeddings = self.positionalEmbeddings(seq_len=T, start_pos=0)
        else:
            # берем K первого слоя, первой головы
            prev_len = cache[0][0][0].shape[1]
            posEmbeddings = self.positionalEmbeddings(seq_len=1, start_pos=prev_len)


        embeddings = tokenEmbeddings + posEmbeddings
        out = self.dropout(embeddings)

        new_cache = [] if use_cache else None
        for i, decoder in enumerate(self.decoders):
            layer_cache = cache[i] if cache is not None else None

            out, layer_new_cache = decoder(
                out,
                use_cache=use_cache,
                cache=layer_cache
            )

            if use_cache:
                new_cache.append(layer_new_cache)
        logits = self.ln(out)

        outs = self.linear(logits)
        if use_cache:
            return outs, new_cache
        else:
            return outs, None

    
    def generate(   
                    self, 
                    x: torch.Tensor,
                    max_new_tokens: int,
                    do_sample: bool = False,
                    temperature: float = 1.0,
                    top_k: int = None,
                    top_p: float = None,
                    use_cache: bool = True
                ):
        
        
        cache = None
        for _ in range(max_new_tokens):
            if cache is None:
                logit, cache = self.forward(x, use_cache=use_cache, cache=None)
            else:
                logit, cache = self.forward(x[:, -1:], use_cache=use_cache, cache=cache)
            # Get the last token's logits
            logits = logit[:, -1, :] / temperature
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

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "emb_size": self.emb_size,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_float,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
    
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        checkpoint_path: str = "src/savepoints/gpt1.pth",
        save_every_epochs: int = 1,
    ):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.to(device)
        self.device = device

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.losses = [{'train': [], 'valid': []}]
        cross_entropy = nn.CrossEntropyLoss()
        
        for epoch in tqdm(range(num_epochs)):
            self.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits, _ = self(x, use_cache=False)
                targets = y
                B, T, V = logits.shape
                
                logits_flat = logits.reshape(B * T, V)
                targets_flat = targets.reshape(B * T)

                train_loss = cross_entropy(logits_flat, targets_flat)
                self.losses[0]['train'].append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    logits, _ = self(x, use_cache=False)
                    targets = y

                    B, T, V = logits.shape
                    logits_flat = logits.reshape(B * T, V)
                    targets_flat = targets.reshape(B * T)

                    loss = cross_entropy(logits_flat, targets_flat)
                    self.losses[0]['valid'].append(loss.item())
            print(f"Epoch {epoch + 1}/{num_epochs} completed. Train Loss: {train_loss.item():.4f}, Valid Loss: {loss.item():.4f}")
            if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
                self.save(
                    checkpoint_path,
                    extra={
                        "epoch": epoch + 1,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "losses": self.losses,
                    },
                )

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint.get("dropout", 0.1),
            device=device,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model

if __name__ == "__main__":

    gpt = GPT2(vocab_size=10000,
              max_seq_len=1,
              emb_size=12,
              num_heads=12, 
              head_size=8, 
              num_layers=8,
              dropout=0.1, 
              device='cpu')


    x = torch.tensor([
                    [113, 456, 76, 345],
                    [345, 678, 454, 546]
                ])
    out = gpt.generate(x, max_new_tokens=10)
    gpt.save("data/gpt_model.pth")

    gpt = GPT2.load("data/gpt_model.pth", device='cpu')
    print(out, out.shape)
