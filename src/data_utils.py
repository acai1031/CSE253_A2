from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_token_file

class TokenChunkDataset(Dataset):
    def __init__(self, files, block_size=512, stride=256):
        self.examples = []
        self.block_size = block_size
        for f in files:
            ids = read_token_file(f)
            if len(ids) < block_size + 1:
                continue
            for start in range(0, len(ids) - block_size - 1, stride):
                chunk = ids[start:start + block_size + 1]
                self.examples.append(chunk.astype(np.int64))
        if not self.examples:
            raise ValueError("No examples. Try reducing block_size.")
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        x = torch.tensor(self.examples[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.examples[idx][1:], dtype=torch.long)
        return x, y

@torch.no_grad()
def evaluate_lm(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    return mean_loss, math.exp(mean_loss)
