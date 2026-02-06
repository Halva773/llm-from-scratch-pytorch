import argparse
import torch

from model.gpt1.model import GPT
from model.common.bpe import BPE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = BPE.load(args.tokenizer)
    model = GPT.load(args.model, device=device)

    token_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

