"""Diagnostic metrics derived from per-configuration accuracies."""
from __future__ import annotations

from typing import Dict, List


def accuracy(preds: List[int], golds: List[int]) -> float:
    return sum(int(p == g) for p, g in zip(preds, golds)) / max(1, len(golds))


def modality_contribution(acc_full: float, acc_without: Dict[str, float]) -> Dict[str, float]:
    """Marginal contribution of each modality = acc_full - acc_without_it.

    Sign convention (NOT clamped -- the sign is informative):
      * positive => removing the channel lowered accuracy, i.e. the channel
        *helped* the model;
      * zero     => the channel made no difference (the model ignores it);
      * negative => removing the channel *raised* accuracy, i.e. the channel was
        actively *hurting* the model (a real failure mode on some backends, e.g.
        a noisy/misleading channel an ablated config scores higher without).

    Callers must therefore handle negative contributions; they are reported
    as-is rather than floored at zero.
    """
    return {m: round(acc_full - a, 4) for m, a in acc_without.items()}


def collapse_gap(acc_full: float, blind_score: float) -> float:
    """Gap between full-modality accuracy and the blind language-only prior.

    Sign convention (NOT clamped):
      * large positive => the model genuinely uses the media;
      * near zero      => modality collapse: the model answers from the language
        prior alone and ignores the media;
      * negative       => the media is actively misleading the model (it does
        *worse* than answering blind). Reported as-is, not floored at zero.
    """
    return round(acc_full - blind_score, 4)
