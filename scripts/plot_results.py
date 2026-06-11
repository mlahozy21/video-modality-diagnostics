"""Render the diagnostic results as figures (saved to figures/).

Offline (stub backend, deterministic):
    python scripts/plot_results.py
Real model:
    python scripts/plot_results.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmd.backends import HFChatBackend, OfflineStubBackend
from vmd.data import load_jsonl, load_sample
from vmd.diagnose import run_diagnostic

BLUE, GREY, RED = "#4c72b0", "#999999", "#c44e52"


def plot_accuracies(r, name, path):
    mods = list(r["acc_single_modality"].keys())
    labels = ["full\n(all channels)", "blind\n(no media)"] + [f"only\n{m}" for m in mods]
    vals = [r["acc_full"], r["blind_language_prior"]] + \
           [r["acc_single_modality"][m] for m in mods]
    colors = [BLUE, RED] + [GREY] * len(mods)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Does the model use the media?  collapse gap = "
                 f"{r['collapse_gap']:.2f}  ({name})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(r, name, path):
    sev = [float(s) for s in r["vision_robustness"]]
    acc = list(r["vision_robustness"].values())

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(sev, acc, "o-", color=BLUE)
    ax.axhline(r["blind_language_prior"], ls="--", color=RED,
               label="blind language prior")
    ax.set_xlabel("vision corruption severity (frame shuffling)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness to visual corruption ({name})")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["stub", "hf"], default="stub")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    items = load_jsonl(args.data) if args.data else load_sample()
    backend = HFChatBackend(args.model) if args.backend == "hf" else OfflineStubBackend()
    r = run_diagnostic(backend, items)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_accuracies(r, backend.name, f"{args.out_dir}/accuracies.png")
    plot_robustness(r, backend.name, f"{args.out_dir}/robustness.png")
    print(f"saved {args.out_dir}/accuracies.png and {args.out_dir}/robustness.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
