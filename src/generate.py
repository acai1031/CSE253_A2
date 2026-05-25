from __future__ import annotations

import argparse
from pathlib import Path

import torch
from miditok import REMI
from miditok.classes import TokSequence

from model import MusicTransformerLM
from utils import load_pickle, read_token_file, set_seed
from markov_utils import sample_markov

def ids_to_midi(tokenizer, ids, out_path):
    valid_ids = set(tokenizer.vocab.values())
    ids = [int(x) for x in ids if int(x) in valid_ids]

    print("after filtering length:", len(ids))
    print("max id:", max(ids), "vocab max:", max(valid_ids))

    seq = TokSequence(ids=ids)
    score = tokenizer.decode([seq])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--method", choices=["transformer", "markov"], default="transformer")
    ap.add_argument("--model_path", default="checkpoints/transformer_best.pt")
    ap.add_argument("--markov_path", default="checkpoints/markov_order1.pkl")
    ap.add_argument("--prompt_file", default=None, help="Optional .npy token file used as prompt")
    ap.add_argument("--prompt_len", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/sample.mid")
    args = ap.parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    tokenizer = REMI(params=data_dir / "tokenizer.json")

    if args.method == "markov":
        model = load_pickle(args.markov_path)

        train_files = sorted((data_dir / "train").glob("*.npy"))
        prompt = read_token_file(train_files[0])[:32].tolist()

        gen = sample_markov(model, length=args.max_new_tokens, seed=args.seed)

        ids = prompt + gen
    else:
        ckpt = torch.load(args.model_path, map_location="cpu")
        margs = ckpt["args"]
        model = MusicTransformerLM(
            ckpt["vocab_size"],
            block_size=margs["block_size"],
            n_layer=margs["n_layer"],
            n_head=margs["n_head"],
            n_embd=margs["n_embd"],
            dropout=margs.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        if args.prompt_file:
            prompt = read_token_file(args.prompt_file)[:args.prompt_len].tolist()
        else:
            train_files = sorted((data_dir / "train").glob("*.npy"))
            prompt = read_token_file(train_files[0])[:args.prompt_len].tolist()
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        ids = y[0].detach().cpu().tolist()

    out_path = ids_to_midi(tokenizer, ids, args.out)
    print(f"Saved generated MIDI to {out_path}")


if __name__ == "__main__":
    main()
