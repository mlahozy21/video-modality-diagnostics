"""Modality availability and corruption operators used in the ablations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModalityView:
    available: Dict[str, str]
    corruption: Dict[str, float]


def _shuffle_tokens(text: str, frac: float, rng: random.Random) -> str:
    toks = text.split()
    n = int(len(toks) * frac)
    if n < 2:
        return text
    idx = rng.sample(range(len(toks)), n)
    vals = [toks[i] for i in idx]
    rng.shuffle(vals)
    for i, v in zip(idx, vals):
        toks[i] = v
    return " ".join(toks)


def make_view(item, modalities: List[str], corrupt: Dict[str, float] | None = None,
              kind: str = "shuffle", seed: int = 0) -> ModalityView:
    rng = random.Random(seed)
    corrupt = corrupt or {}
    available, applied = {}, {}
    for m in modalities:
        sev = corrupt.get(m, 0.0)
        ev = item.evidence(m)
        if kind == "drop" and sev >= 1.0:
            continue
        if sev > 0.0:
            if kind == "noise":
                ev = " ".join(t for t in ev.split() if rng.random() > sev)
            else:
                ev = _shuffle_tokens(ev, sev, rng)
        available[m] = ev
        applied[m] = sev
    return ModalityView(available=available, corruption=applied)
