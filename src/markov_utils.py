import random

def sample_markov(model, length=1024, seed=42):
    rng = random.Random(seed)
    order = model["order"]
    counts = model["counts"]
    unigram = model["unigram"]
    if model.get("starts"):
        seq = list(rng.choice(model["starts"]))
    else:
        seq = [unigram.most_common(1)[0][0]] * order
    vocab, weights = zip(*unigram.items())
    for _ in range(length - len(seq)):
        ctx = tuple(seq[-order:])
        options = counts.get(ctx)
        if options:
            toks, ws = zip(*options.items())
            nxt = rng.choices(toks, weights=ws, k=1)[0]
        else:
            nxt = rng.choices(vocab, weights=weights, k=1)[0]
        seq.append(int(nxt))
    return seq
