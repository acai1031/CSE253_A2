from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from miditok import REMI

from model import MusicTransformerLM
from data_utils import TokenChunkDataset, evaluate_lm
from utils import read_token_file, load_pickle, save_json


def markov_perplexity(model, files, alpha=1e-3):
    counts = model["counts"]
    unigram = model["unigram"]
    order = model["order"]
    V = max(len(unigram), 1)
    total_nll, total = 0.0, 0
    for f in files:
        ids = read_token_file(f).tolist()
        for i in range(order, len(ids)):
            ctx = tuple(ids[i-order:i])
            tok = ids[i]
            c = counts.get(ctx, Counter())
            prob = (c.get(tok, 0) + alpha) / (sum(c.values()) + alpha * V)
            total_nll -= math.log(prob)
            total += 1
    return math.exp(total_nll / max(total, 1))


def token_distribution(files):
    c = Counter()
    n = 0
    for f in files:
        ids = read_token_file(f).tolist()
        c.update(ids)
        n += len(ids)
    return c, n


def plot_top_tokens(counter, out_path, top_n=30):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = counter.most_common(top_n)
    labels = [str(k) for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=90)
    plt.title("Top token frequencies")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--transformer_ckpt", default="checkpoints/transformer_best.pt")
    ap.add_argument("--markov_path", default="checkpoints/markov_order1.pkl")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted((data_dir / args.split).glob("*.npy"))
    assert files, f"No files for split {args.split}"

    results = {}
    if Path(args.markov_path).exists():
        markov = load_pickle(args.markov_path)
        results["markov_perplexity"] = markov_perplexity(markov, files)

    if Path(args.transformer_ckpt).exists():
        ckpt = torch.load(args.transformer_ckpt, map_location="cpu")
        margs = ckpt["args"]
        model = MusicTransformerLM(ckpt["vocab_size"], margs["block_size"], margs["n_layer"], margs["n_head"], margs["n_embd"], margs.get("dropout", 0.1))
        model.load_state_dict(ckpt["model"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        ds = TokenChunkDataset(files, margs["block_size"], margs.get("stride", margs["block_size"]))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss, ppl = evaluate_lm(model, loader, device)
        results["transformer_loss"] = loss
        results["transformer_perplexity"] = ppl

    c, n = token_distribution(files)
    results["n_tokens"] = n
    results["unique_tokens"] = len(c)
    save_json(results, "outputs/eval_results.json")
    plot_top_tokens(c, "figures/top_token_distribution.png")
    print(results)


if __name__ == "__main__":
    main()
