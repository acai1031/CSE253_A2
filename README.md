# POP909 Symbolic Unconditioned Generation

Pipeline: POP909 MIDI -> MidiTok REMI tokens -> Markov baseline + small Transformer -> generated MIDI -> evaluation.

## Setup
```bash
pip install -r requirements.txt
```

Put the dataset folder like:
```text
POP909/
  001/001.mid
  002/002.mid
  ...
  909/909.mid
  index.xlsx
```

## Run
```bash
python src/prepare_data.py --data_dir pop909 --out_dir data/processed --use_versions 0

python src/train_markov.py --data_dir data/processed --order 1
python src/generate.py --data_dir data/processed --method markov --markov_path checkpoints/markov_order1.pkl --out outputs/markov_sample.mid --max_new_tokens 1500 --prompt_len 128 --temperature 0.8 --top_k 30

python src/train_transformer.py --data_dir data/processed --epochs 20 --batch_size 16 --block_size 512
python src/generate.py --data_dir data/processed --method transformer --model_path checkpoints/transformer_best.pt --prompt_file data/processed/train/001_001.npy --out outputs/transformer_001_08_30.mid --max_new_tokens 1500 --prompt_len 128 --temperature 0.8 --top_k 30

python src/evaluate.py --data_dir data/processed --transformer_ckpt checkpoints/transformer_best.pt --markov_path checkpoints/markov_order1.pkl
```
