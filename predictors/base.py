"""
Shared predictor interface (Research Plan Section 2.2: "Both predictors
share the same prediction target ... the same input-window definition ...
the architecture is the only thing that differs").

Neither predictor ever sees flow-table state (occupancy, evictions) --
only the flow-signature history -- which is what makes offline replay valid
(Section 2.3 Step 8: a predictor's output must not depend on the
installation policy applied to it afterward).
"""
from __future__ import annotations
from typing import Protocol, Sequence, Optional
from dataclasses import dataclass


@dataclass
class Prediction:
    signature: Optional[str]   # predicted next flow signature's key(), or None if no prediction possible
    confidence: float          # in [0, 1]


class Predictor(Protocol):
    """
    Every predictor:
      - trains once, on generated training traffic only (`fit`), and is
        then reused unchanged across repetitions and tau values
        (Section 2.2: "Both are trained once on a fixed seed and reused
        unchanged").
      - predicts from a *bounded* window of past signatures
        (config.INPUT_WINDOW), never from table state.
    """

    def fit(self, training_signatures: Sequence[str]) -> None:
        ...

    def predict(self, history: Sequence[str]) -> Prediction:
        """
        `history`: the most recent signatures observed on this run so far
        (already truncated to config.INPUT_WINDOW by the caller, or shorter
        at the start of a run). Must not raise on an empty/short history --
        return Prediction(None, 0.0) instead.
        """
        ...
