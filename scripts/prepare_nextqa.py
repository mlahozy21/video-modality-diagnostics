"""Build a real-video QA subset from NExT-QA (multiple-choice split).

Downloads the MC annotations and the videos referenced by a sampled subset
from `lmms-lab/NExTQA` (Hugging Face), and writes a `videoqa.jsonl` compatible
with the vmd harness (gold_modality="vision", real `video_path`).

    python scripts/prepare_nextqa.py --n 60 --out-dir data/nextqa

Notes:
- videos.zip is ~6.5 GB; it is downloaded once into the HF cache and only the
  needed members are extracted (a few hundred MB for n=60).
- Run this in Colab (or anywhere with bandwidth); the subset itself is small.
"""
import argparse
import json
import os
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "lmms-lab/NExTQA"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="number of questions")
    ap.add_argument("--out-dir", default="data/nextqa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1) annotations
    parquet = hf_hub_download(REPO, "MC/test-00000-of-00001.parquet",
                              repo_type="dataset")
    df = pd.read_parquet(parquet)
    print(f"MC split: {len(df)} questions, columns: {list(df.columns)}")

    # balanced sample over question types (causal/temporal/descriptive)
    per_type = max(1, args.n // df["type"].nunique())
    sample = (df.groupby("type", group_keys=False)
                .apply(lambda g: g.sample(min(per_type, len(g)),
                                          random_state=args.seed)))
    if len(sample) > args.n:
        sample = sample.sample(args.n, random_state=args.seed)
    print(f"sampled {len(sample)} questions over types: "
          f"{sample['type'].value_counts().to_dict()}")

    # 2) videos (extract only the needed members from videos.zip)
    zip_path = hf_hub_download(REPO, "videos.zip", repo_type="dataset")
    vids = {str(v) for v in sample["video"]}
    out_videos = os.path.join(args.out_dir, "videos")
    os.makedirs(out_videos, exist_ok=True)

    vid_to_path = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            base = os.path.splitext(os.path.basename(name))[0]
            if base in vids and name.endswith(".mp4"):
                target = os.path.join(out_videos, f"{base}.mp4")
                if not os.path.exists(target):
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                vid_to_path[base] = target
    missing = vids - set(vid_to_path)
    if missing:
        print(f"WARNING: {len(missing)} videos not found in the zip: "
              f"{sorted(missing)[:5]}...")

    # 3) jsonl in the vmd QAItem schema
    out_jsonl = os.path.join(args.out_dir, "videoqa.jsonl")
    n_written = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            vid = str(row["video"])
            if vid not in vid_to_path:
                continue
            item = {
                "id": f"nextqa_{row.get('qid', n_written)}",
                "question": str(row["question"]).strip(),
                "options": [str(row[f"a{i}"]).strip() for i in range(5)],
                "answer_idx": int(row["answer"]),
                "gold_modality": "vision",
                "visual_facts": "",
                "audio_tags": "",
                "subtitle": "",
                "video_path": vid_to_path[vid],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"wrote {n_written} items to {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
