import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datetime import datetime

from model.gpt import GPT
from model.bpe import BPE
from model.dataLoader import GetData

def get_text(filepath: str) -> str:
    data = pd.read_csv(filepath)
    data = data.dropna(subset=['text'])
    return "\n".join(data['text'])


def print_text_with_time(text: str) -> None:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {text}")


def main(**params):
    print_text_with_time('Training with parameters:')
    print('\n'.join([f'{k}: {v}' for k, v in params.items()]))

    tokenizer = BPE(params.get("dict_size", 40000))
    text = get_text("dataset/poems.csv")
    tokenizer.fit(text)

    print_text_with_time("Text loaded and tokenizer fitted.")

    tokens_ids = tokenizer.encode(text)
    print(f"Number of tokens: {len(tokens_ids)}")

    print_text_with_time("Preparing data loaders...")

    n = int(0.9*len(tokens_ids)) # 90% train
    train_token_ids = tokens_ids[:n]
    valid_token_ids = tokens_ids[n:]

    print_text_with_time(f"Train tokens: {len(train_token_ids)}, Valid tokens: {len(valid_token_ids)}")

    train_data = GetData(train_token_ids, seq_len=params.get("seq_len", 512), device=params.get("device", "cpu"))
    train_loader = DataLoader(train_data, batch_size=params.get("batch_size", 64))

    valid_data = GetData(valid_token_ids, seq_len=params.get("seq_len", 512), device=params.get("device", "cpu"))
    valid_loader = DataLoader(valid_data, batch_size=params.get("batch_size", 64))

    print_text_with_time("Start fitting...")

    GPT_model = GPT(
        vocab_size=params.get("dict_size", 40000),
        max_seq_len=params.get("seq_len", 512),
        emb_size=params.get("emb_size", 768),
        num_heads=params.get("headAttention", 12),
        head_size=params.get("emb_size", 768)//params.get("headAttention", 12),
        num_layers=params.get("layers", 12),
        dropout=params.get("dropout", 0.1),
        device=params.get("device", "cpu")
    )


    GPT_model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=params.get("num_epoch", 100),
        learning_rate=params.get("learning_rate", 2.4e-4)
    )
    print_text_with_time("Training completed.")

    



if __name__ == "__main__":
    layers = 12
    headAttention = 12
    emb_size = 768
    dict_size = 40000
    dropout = 0.1
    learning_rate = 2.4e-4
    num_epoch = 100
    batch_size = 64
    seq_len = 512
    device = 'cpu'

    main(
        layers=layers,
        headAttention=headAttention,
        emb_size=emb_size,
        dict_size=dict_size,
        dropout=dropout,
        learning_rate=learning_rate,
        num_epoch=num_epoch,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device
    )