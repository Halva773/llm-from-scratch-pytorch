import argparse
import torch

from model.common.bpe import BPE


def get_model_cls(model_type: str):
    if model_type == "gpt1":
        from model.gpt1.model import GPT2 as ModelCls
        return ModelCls
    if model_type == "gpt2":
        from model.gpt2.model import GPT2 as ModelCls
        return ModelCls
    raise ValueError(f"Unknown --model_type: {model_type}. Expected: gpt1, gpt2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt1",
        choices=["gpt1", "gpt2"],
        help="Which model implementation to use: gpt1 or gpt2.",
    )
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

    ModelCls = get_model_cls(args.model_type)
    tokenizer = BPE.load(args.tokenizer)
    model = ModelCls.load(args.model, device=device)

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

