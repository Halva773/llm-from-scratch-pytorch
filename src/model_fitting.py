import pandas as pd
from torch.utils.data import DataLoader
import torch
from datetime import datetime
import argparse

from model.gpt import GPT
from model.bpe import BPE
from model.dataLoader import GetData
from pathlib import Path

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

    dict_size = params.get("dict_size", 40000)
    save_dir = Path(params.get("save_dir", "src/savepoints"))
    run_name = params.get("run_name", "gpt1")

    tokenizer = BPE(dict_size)
    text = get_text(params.get("dataset_csv", "dataset/poems.csv"))
    tokenizer.fit(text)

    print_text_with_time("Text loaded and tokenizer fitted.")

    tokenizer_path = save_dir / f"bpe_{dict_size}.dill"
    try:
        tokenizer.save(tokenizer_path)
    except ModuleNotFoundError:
        tokenizer_path = save_dir / f"bpe_{dict_size}.json"
        tokenizer.save_json(tokenizer_path)
    print_text_with_time(f"Tokenizer saved to: {tokenizer_path}")

    tokens_ids = tokenizer.encode(text)
    print(f"Number of tokens: {len(tokens_ids)}")

    print_text_with_time("Preparing data loaders...")

    if len(tokens_ids) < params.get("seq_len", 512) + 2:
        raise ValueError("Text is too short for the selected seq_len.")

    n = int(0.9*len(tokens_ids)) # 90% train
    train_token_ids = tokens_ids[:n]
    valid_token_ids = tokens_ids[n:]

    print_text_with_time(f"Train tokens: {len(train_token_ids)}, Valid tokens: {len(valid_token_ids)}")

    train_data = GetData(train_token_ids, seq_len=params.get("seq_len", 512))
    train_loader = DataLoader(
        train_data,
        batch_size=params.get("batch_size", 64),
        shuffle=True,
        pin_memory=str(params.get("device", "cpu")).startswith("cuda"),
    )

    valid_data = GetData(valid_token_ids, seq_len=params.get("seq_len", 512))
    valid_loader = DataLoader(
        valid_data,
        batch_size=params.get("batch_size", 64),
        shuffle=False,
        pin_memory=str(params.get("device", "cpu")).startswith("cuda"),
    )

    print_text_with_time("Start fitting...")

    GPT_model = GPT(
        vocab_size=dict_size,
        max_seq_len=params.get("seq_len", 512),
        emb_size=params.get("emb_size", 768),
        num_heads=params.get("headAttention", 12),
        head_size=params.get("emb_size", 768)//params.get("headAttention", 12),
        num_layers=params.get("layers", 12),
        dropout=params.get("dropout", 0.1),
        device=params.get("device", "cpu")
    )

    model_path = save_dir / f"{run_name}.pth"

    GPT_model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=params.get("num_epoch", 100),
        learning_rate=params.get("learning_rate", 2.4e-4),
        checkpoint_path=str(model_path),
        save_every_epochs=params.get("save_every_epochs", 1),
    )
    print_text_with_time("Training completed.")

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--headAttention", type=int, default=12)
    parser.add_argument("--emb_size", type=int, default=768)
    parser.add_argument("--dict_size", type=int, default=40000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2.4e-4)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset_csv", type=str, default="dataset/poems.csv")
    parser.add_argument("--save_dir", type=str, default="src/savepoints")
    parser.add_argument("--run_name", type=str, default="gpt1")
    parser.add_argument("--save_every_epochs", type=int, default=1)
    main(**vars(parser.parse_args()))
