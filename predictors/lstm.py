"""
Small LSTM predictor over flow signatures.

Requires: pip install torch --break-system-packages   (see requirements.txt)
Not runnable in this sandbox (no torch here); intended to run on your own
Mininet/Ryu machine alongside the rest of the pipeline. Everything in this
file is otherwise a plain, CPU-friendly model -- no GPU assumed.

Architecture (deliberately small & fixed, per Section 2.2: "no architecture
search ... a fixed, small tuning budget"):
  - signature vocabulary built from the training set (+ an <UNK> token)
  - Embedding -> single-layer LSTM -> Linear -> softmax over vocabulary
  - confidence = softmax probability assigned to the argmax next-token
  - input window = config.INPUT_WINDOW, matching the Markov predictor
"""
from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass

from .base import Prediction

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where torch is absent
    _TORCH_AVAILABLE = False


UNK = "<UNK>"


class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) of token ids
        emb = self.embed(x)
        _, (h_n, _) = self.lstm(emb)
        logits = self.out(h_n[-1])  # (batch, vocab_size)
        return logits


@dataclass
class LSTMConfig:
    input_window: int = 8
    embed_dim: int = 32
    hidden_dim: int = 64
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3


class LSTMPredictor:
    name = "lstm"

    def __init__(self, config: LSTMConfig | None = None):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for LSTMPredictor. "
                "Install it on your Mininet/Ryu machine: "
                "pip install torch --break-system-packages"
            )
        self.config = config or LSTMConfig()
        self.vocab: dict[str, int] = {UNK: 0}
        self.model: _LSTMNet | None = None
        self._fitted = False

    def _tokenize(self, sig: str) -> int:
        return self.vocab.get(sig, 0)  # UNK id = 0

    def _build_vocab(self, signatures: Sequence[str]) -> None:
        for s in signatures:
            if s not in self.vocab:
                self.vocab[s] = len(self.vocab)

    def _windows(self, signatures: Sequence[str]):
        w = self.config.input_window
        xs, ys = [], []
        for i in range(len(signatures) - w):
            xs.append([self._tokenize(s) for s in signatures[i:i + w]])
            ys.append(self._tokenize(signatures[i + w]))
        return xs, ys

    def fit(self, training_signatures: Sequence[str]) -> None:
        self._build_vocab(training_signatures)
        xs, ys = self._windows(training_signatures)
        if not xs:
            raise ValueError("Not enough training signatures for the configured input window.")

        self.model = _LSTMNet(len(self.vocab), self.config.embed_dim, self.config.hidden_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()

        X = torch.tensor(xs, dtype=torch.long)
        Y = torch.tensor(ys, dtype=torch.long)
        n = X.shape[0]
        bs = self.config.batch_size

        self.model.train()
        for _epoch in range(self.config.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                opt.zero_grad()
                logits = self.model(X[idx])
                loss = loss_fn(logits, Y[idx])
                loss.backward()
                opt.step()

        self.model.eval()
        self._fitted = True

    def predict(self, history: Sequence[str]) -> Prediction:
        if not self._fitted or len(history) < 1:
            return Prediction(None, 0.0)
        w = self.config.input_window
        window = list(history[-w:])
        if len(window) < w:
            window = [UNK] * (w - len(window)) + window  # left-pad with UNK

        x = torch.tensor([[self._tokenize(s) for s in window]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            conf, idx = torch.max(probs, dim=0)

        idx = int(idx.item())
        inv_vocab = {v: k for k, v in self.vocab.items()}
        predicted_sig = inv_vocab.get(idx, UNK)
        if predicted_sig == UNK:
            return Prediction(None, float(conf.item()))
        return Prediction(signature=predicted_sig, confidence=float(conf.item()))
