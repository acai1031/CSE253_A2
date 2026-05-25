from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import math
import random

from tqdm import tqdm

from utils import read_token_file, save_pickle


def train_markov(files, order=1):
    counts = defaultdict(Counter)
    start_contexts = []
    unigram = Counter()
    for f in tqdm(files, desc="Counting n-grams"):
        ids = read_token_file(f).tolist()
        if len(ids) <= order:
            continue
        start_contexts.append(tuple(ids[:order]))
        unigram.update(ids)
        for i in range(order, len(ids)):
            ctx = tuple(ids[i-order:i])
            counts[ctx][ids[i]] += 1
    return {"order": order, "counts": counts, "starts": start_contexts, "unigram": unigram}


def perplexity(model, files, alpha=1e-3):
    counts = model["counts"]
    unigram = model["unigram"]
    order = model["order"]
    vocab = list(unigram.keys())
    V = max(len(vocab), 1)
    total_nll = 0.0
    total = 0
    for f in files:
        ids = read_token_file(f).tolist()
        for i in range(order, len(ids)):
            ctx = tuple(ids[i-order:i])
            token = ids[i]
            c_ctx = counts.get(ctx, Counter())
            denom = sum(c_ctx.values()) + alpha * V
            num = c_ctx.get(token, 0) + alpha
            total_nll -= math.log(num / denom)
            total += 1
    return math.exp(total_nll / max(total, 1))


from markov_utils import sample_markov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    train_files = sorted((data_dir / "train").glob("*.npy"))
    val_files = sorted((data_dir / "val").glob("*.npy"))
    model = train_markov(train_files, order=args.order)
    out = Path(args.out or f"checkpoints/markov_order{args.order}.pkl")
    save_pickle(model, out)
    print(f"Saved Markov model to {out}")
    if val_files:
        print(f"Validation perplexity: {perplexity(model, val_files):.3f}")


if __name__ == "__main__":
    main()
