from __future__ import annotations

import random
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

def token_str(tokenizer, token_id):
    return tokenizer[int(token_id)]

def find_first_pitch_index(tokenizer, ids):
    for i, tid in enumerate(ids):
        s = token_str(tokenizer, tid)
        if s.startswith("Pitch_"):
            return i
    return 0

def get_prompt_from_first_note(tokenizer, npy_path, prompt_len=128, context=16):
    ids = read_token_file(npy_path).tolist()
    first_pitch = find_first_pitch_index(tokenizer, ids)
    start = max(0, first_pitch - context)
    return ids[start:start + prompt_len]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--method", choices=["transformer", "markov"], default="transformer")
    ap.add_argument("--model_path", default="checkpoints/transformer_best.pt")
    ap.add_argument("--markov_path", default="checkpoints/markov_order1.pkl")
    ap.add_argument("--prompt_file", default=None, help="Optional .npy token file used as prompt")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=1500)
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
        prompt_source = random.choice(train_files)
        prompt = get_prompt_from_first_note(
            tokenizer,
            prompt_source,
            prompt_len=args.prompt_len,
            context=16
        )
        print(f"Using prompt source: {prompt_source}")

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
            prompt = get_prompt_from_first_note(
                tokenizer,
                args.prompt_file,
                prompt_len=args.prompt_len,
                context=16
            )
        else:
            train_files = sorted((data_dir / "train").glob("*.npy"))
            prompt_source = random.choice(train_files)
            prompt = get_prompt_from_first_note(
                tokenizer,
                prompt_source,
                prompt_len=args.prompt_len,
                context=16
            )
            print(f"Using prompt source: {prompt_source}")
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        ids = y[0].detach().cpu().tolist()

    out_path = ids_to_midi(tokenizer, ids, args.out)
    print(f"Saved generated MIDI to {out_path}")


if __name__ == "__main__":
    main()
