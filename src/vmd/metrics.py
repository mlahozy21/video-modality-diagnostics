"""Diagnostic metrics derived from per-configuration accuracies."""
from __future__ import annotations

from typing import Dict, List


def accuracy(preds: List[int], golds: List[int]) -> float:
    return sum(int(p == g) for p, g in zip(preds, golds)) / max(1, len(golds))


def modality_contribution(acc_full: float, acc_without: Dict[str, float]) -> Dict[str, float]:
    return {m: round(acc_full - a, 4) for m, a in acc_without.items()}


def collapse_gap(acc_full: float, blind_score: float) -> float:
    return round(acc_full - blind_score, 4)
