"""
The tau-gated installation policy (Figure 1 in both PDFs).

This is the ONE shared layer sitting between "a predictor produced
(signature, confidence)" and "a rule gets installed", used identically by
Markov and LSTM (Research Plan Section 2.3 Step 7: "the predictor is the
only difference between the two proactive arms"). The reactive-only arm
never calls this at all -- it only installs reactively, on a table miss.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from predictors.base import Prediction


@dataclass
class PolicyDecision:
    install_proactively: bool
    signature: Optional[str]
    confidence: float


def decide(prediction: Prediction, tau: float) -> PolicyDecision:
    """
    Figure 1's YES/NO branch:
        p = prediction.confidence
        p >= tau  -> "Proactive rule installed"
        p <  tau  -> "No rule"

    A prediction with signature=None (predictor had nothing to say) is
    always treated as "No rule" regardless of tau, since there is nothing
    to install.
    """
    if prediction.signature is None:
        return PolicyDecision(install_proactively=False, signature=None, confidence=prediction.confidence)

    install = prediction.confidence >= tau
    return PolicyDecision(
        install_proactively=install,
        signature=prediction.signature if install else None,
        confidence=prediction.confidence,
    )
