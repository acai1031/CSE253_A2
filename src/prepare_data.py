from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

from miditok import REMI, TokenizerConfig

from utils import find_pop909_midis, save_json, split_list, write_token_file


def encode_file(tokenizer, midi_path: Path):
    """Return a flat list of token ids. MidiTok versions differ slightly, so this is defensive."""
    seq = tokenizer.encode(midi_path)
    # Some tokenizers return a list for multitrack; flatten each track.
    if isinstance(seq, list):
        ids = []
        for s in seq:
            ids.extend(getattr(s, "ids", s))
        return list(map(int, ids))
    return list(map(int, getattr(seq, "ids", seq)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to POP909 root folder")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--use_versions", type=int, default=0, help="0: use only 001.mid; 1: include versions/*.mid")
    ap.add_argument("--vocab_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_files = find_pop909_midis(args.data_dir, bool(args.use_versions))
    assert midi_files, f"No midi files found under {args.data_dir}"
    train_files, val_files, test_files = split_list(midi_files, seed=args.seed)

    # REMI is a strong default for pop music because it explicitly encodes bar/position/duration.
    config = TokenizerConfig(
        num_velocities=8,
        use_chords=False,
        use_programs=False,
        use_tempos=True,
        use_time_signatures=True,
    )
    tokenizer = REMI(config)
    # tokenizer.train(vocab_size=args.vocab_size, files_paths=train_files)
    tokenizer.save(out_dir / "tokenizer.json")

    meta = {
        "data_dir": str(args.data_dir),
        "use_versions": bool(args.use_versions),
        "vocab_size_requested": args.vocab_size,
        "n_files": len(midi_files),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "n_test": len(test_files),
        "splits": {
            "train": [str(p) for p in train_files],
            "val": [str(p) for p in val_files],
            "test": [str(p) for p in test_files],
        },
    }
    save_json(meta, out_dir / "metadata.json")

    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_dir = out_dir / split
        split_dir.mkdir(exist_ok=True)
        lengths = []
        for p in tqdm(files, desc=f"Encoding {split}"):
            try:
                ids = encode_file(tokenizer, p)
                if len(ids) < 32:
                    continue
                write_token_file(ids, split_dir / f"{p.parent.name}_{p.stem}.npy")
                lengths.append(len(ids))
            except Exception as e:
                print(f"[WARN] failed to encode {p}: {e}")
        save_json({"lengths": lengths}, out_dir / f"{split}_stats.json")

    print(f"Done. Tokenized data saved to {out_dir}")


if __name__ == "__main__":
    main()
