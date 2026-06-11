"""Run the full modality diagnostic over a dataset and a backend."""
from __future__ import annotations

from typing import Dict

from .config import CONFIG
from .metrics import accuracy, collapse_gap, modality_contribution
from .modalities import make_view


def _eval(backend, items, modalities, corrupt=None, kind="shuffle", seed=0):
    preds, golds = [], []
    for it in items:
        view = make_view(it, modalities, corrupt=corrupt, kind=kind, seed=seed)
        preds.append(backend.answer(it, view))
        golds.append(it.answer_idx)
    return accuracy(preds, golds)


def run_diagnostic(backend, items, config=CONFIG) -> Dict:
    mods = config.modalities
    acc_full = _eval(backend, items, mods)
    acc_without = {}
    for m in mods:
        acc_without[m] = _eval(backend, items, [x for x in mods if x != m])
    acc_single = {m: _eval(backend, items, [m]) for m in mods}
    blind = _eval(backend, items, [])
    robustness = {
        s: _eval(backend, items, mods, corrupt={"vision": s}, kind="shuffle", seed=config.seed)
        for s in config.severities
    }
    return {
        "n": len(items),
        "acc_full": round(acc_full, 4),
        "acc_leave_one_out": {m: round(a, 4) for m, a in acc_without.items()},
        "acc_single_modality": {m: round(a, 4) for m, a in acc_single.items()},
        "blind_language_prior": round(blind, 4),
        "modality_contribution": modality_contribution(acc_full, acc_without),
        "collapse_gap": collapse_gap(acc_full, blind),
        "vision_robustness": {str(k): round(v, 4) for k, v in robustness.items()},
    }
