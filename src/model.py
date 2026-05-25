from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicTransformerLM(nn.Module):
    """Small decoder-only Transformer for next-token prediction.
    Implemented with nn.TransformerEncoder + causal mask for simplicity.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            T = self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=1024, temperature=1.0, top_k=50):
        from utils import top_k_filter
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_filter(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
