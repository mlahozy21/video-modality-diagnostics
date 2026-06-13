"""Render a diagnostic result dict as a readable text report."""
from __future__ import annotations


def format_report(r: dict, backend_name: str) -> str:
    L = [f"Modality diagnostic  (backend={backend_name}, n={r['n']})", ""]
    L.append(f"  full accuracy            : {r['acc_full']:.3f}")
    L.append(f"  blind language prior     : {r['blind_language_prior']:.3f}")
    L.append(f"  collapse gap (full-blind): {r['collapse_gap']:.3f}"
             "   <- small => model ignores the media")
    L.append("")
    L.append("  modality contribution (acc drop when removed; sign convention:")
    L.append("    positive => removing the channel HURT, i.e. it helped the model;")
    L.append("    negative => removing it HELPED, i.e. the channel hurt the model):")
    for m, c in sorted(r["modality_contribution"].items(), key=lambda kv: -kv[1]):
        flag = "" if c >= 0 else "   <- negative: channel hurt"
        L.append(f"    {m:9s}: {c:+.3f}{flag}")
    L.append("")
    L.append("  single-modality accuracy:")
    for m, a in r["acc_single_modality"].items():
        L.append(f"    {m:9s}: {a:.3f}")
    L.append("")
    L.append("  robustness sweep (channel corrupted in isolation, per severity):")
    robustness = r.get("robustness") or {"vision": r.get("vision_robustness", {})}
    for m, sweep in robustness.items():
        cells = "  ".join(f"{s}:{a:.3f}" for s, a in sweep.items())
        L.append(f"    {m:9s}: {cells}")
    return "\n".join(L)
