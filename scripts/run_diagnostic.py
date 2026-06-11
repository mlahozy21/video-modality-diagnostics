"""Run the modality diagnostic on the sample VideoQA set.

Offline (no GPU, deterministic):
    python scripts/run_diagnostic.py

Real model (Colab / GPU, pip install '.[vlm]'):
    python scripts/run_diagnostic.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import sys

from vmd.backends import HFChatBackend, OfflineStubBackend
from vmd.data import load_jsonl, load_sample
from vmd.diagnose import run_diagnostic
from vmd.report import format_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["stub", "hf"], default="stub")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF model name (only with --backend hf)")
    p.add_argument("--data", default=None, help="path to a videoqa .jsonl")
    p.add_argument("--json", action="store_true", help="print raw JSON instead")
    args = p.parse_args()

    items = load_jsonl(args.data) if args.data else load_sample()
    backend = HFChatBackend(args.model) if args.backend == "hf" else OfflineStubBackend()

    result = run_diagnostic(backend, items)
    print(json.dumps(result, indent=2) if args.json else format_report(result, backend.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
