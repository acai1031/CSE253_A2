from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_pop909_midis(data_dir: str | Path, use_versions: bool = False) -> List[Path]:
    """Find POP909 midi files. Default uses only canonical 001/001.mid etc.
    If use_versions=True, also includes files in versions/.
    """
    root = Path(data_dir)
    paths: List[Path] = []
    for song_dir in sorted(root.iterdir()):
        if not song_dir.is_dir() or not song_dir.name.isdigit():
            continue
        main_midi = song_dir / f"{song_dir.name}.mid"
        if main_midi.exists():
            paths.append(main_midi)
        if use_versions:
            versions = song_dir / "versions"
            if versions.exists():
                paths.extend(sorted(versions.glob("*.mid")))
    return paths


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_token_file(token_ids: Sequence[int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(token_ids, dtype=np.int32))


def read_token_file(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.int64)


def split_list(items: Sequence, train_ratio=0.8, val_ratio=0.1, seed=42):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]


def top_k_filter(logits: torch.Tensor, k: int | None):
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
