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


def _eval_avg(backend, items, modalities, corrupt=None, kind="shuffle",
              base_seed=0, n_seeds=1):
    """Mean accuracy over ``n_seeds`` corruption realizations.

    Corruption is one random realization per seed; averaging over a few seeds
    (derived deterministically from ``base_seed``) gives a more stable estimate
    while keeping the whole run reproducible. ``n_seeds=1`` reproduces the
    original single-realization behaviour exactly.
    """
    accs = [
        _eval(backend, items, modalities, corrupt=corrupt, kind=kind,
              seed=base_seed + k)
        for k in range(max(1, n_seeds))
    ]
    return sum(accs) / len(accs)


def run_diagnostic(backend, items, config=CONFIG) -> Dict:
    mods = config.modalities
    acc_full = _eval(backend, items, mods)
    acc_without = {}
    for m in mods:
        acc_without[m] = _eval(backend, items, [x for x in mods if x != m])
    acc_single = {m: _eval(backend, items, [m]) for m in mods}
    blind = _eval(backend, items, [])

    # Robustness probe, swept *per available channel* (not just vision): for each
    # modality we corrupt only that channel across the severity grid and record
    # how accuracy degrades, averaged over ``config.robustness_seeds`` corruption
    # realizations (deterministic, derived from ``config.seed``).
    robustness = {}
    for m in mods:
        robustness[m] = {
            str(s): round(
                _eval_avg(backend, items, mods, corrupt={m: s}, kind="shuffle",
                          base_seed=config.seed, n_seeds=config.robustness_seeds),
                4,
            )
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
        # Per-channel robustness sweep. ``vision_robustness`` is kept as a
        # backwards-compatible alias for the vision channel so existing
        # readers/tests keep working.
        "robustness": robustness,
        "vision_robustness": robustness.get("vision", {}),
    }
