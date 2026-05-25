from __future__ import annotations

import argparse
from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from miditok import REMI

from model import MusicTransformerLM
from utils import read_token_file, set_seed, load_json


from data_utils import TokenChunkDataset, evaluate_lm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/transformer_best.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_files = sorted((data_dir / "train").glob("*.npy"))
    val_files = sorted((data_dir / "val").glob("*.npy"))
    tokenizer = REMI(params=data_dir / "tokenizer.json")
    vocab_size = len(tokenizer)

    train_ds = TokenChunkDataset(train_files, args.block_size, args.stride)
    val_ds = TokenChunkDataset(val_files, args.block_size, args.stride) if val_files else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if val_ds else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicTransformerLM(vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_val = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device={device}, vocab={vocab_size}, train_chunks={len(train_ds)}, val_chunks={len(val_ds) if val_ds else 0}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(train_loss=np.mean(losses[-20:]))
        train_loss = float(np.mean(losses))
        if val_loader:
            val_loss, val_ppl = evaluate_lm(model, val_loader, device)
        else:
            val_loss, val_ppl = train_loss, math.exp(train_loss)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "vocab_size": vocab_size, "val_loss": val_loss}, out)
            print(f"Saved best checkpoint to {out}")


if __name__ == "__main__":
    main()
