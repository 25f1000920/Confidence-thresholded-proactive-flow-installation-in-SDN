"""
First-order Markov-chain predictor.

confidence = empirical P(next = s' | current = s), i.e. the transition's
own relative frequency out of the current state. This is the "cheaper,
validates the pipeline first" predictor from Section 2.3 Step 6.

Deliberately no smoothing / back-off beyond a flat prior for genuinely
unseen states -- keeping this simple is itself the point of the RQ2
comparison (predictor complexity vs. system-level benefit), and Section 2.2
commits to an equal, small, logged tuning budget for both predictors.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from typing import Sequence
from .base import Prediction


class MarkovPredictor:
    name = "markov"

    def __init__(self):
        # transitions[s] = Counter({s': count, ...})
        self.transitions: dict[str, Counter] = defaultdict(Counter)
        self._fitted = False

    def fit(self, training_signatures: Sequence[str]) -> None:
        for a, b in zip(training_signatures, training_signatures[1:]):
            self.transitions[a][b] += 1
        self._fitted = True

    def predict(self, history: Sequence[str]) -> Prediction:
        if not self._fitted or not history:
            return Prediction(None, 0.0)
        current = history[-1]
        counter = self.transitions.get(current)
        if not counter:
            # current state never observed transitioning anywhere during
            # training -- genuinely nothing to say. This is the desired
            # low-confidence behaviour under cold-start / unstructured traffic.
            return Prediction(None, 0.0)
        total = sum(counter.values())
        best_sig, best_count = counter.most_common(1)[0]
        return Prediction(signature=best_sig, confidence=best_count / total)
