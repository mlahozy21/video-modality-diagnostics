"""VideoQA item schema and sample loader."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class QAItem:
    id: str
    question: str
    options: List[str]
    answer_idx: int
    gold_modality: str          # "vision" | "audio" | "subtitle"
    visual_facts: str
    audio_tags: str
    subtitle: str
    video_path: str = ""

    def evidence(self, modality: str) -> str:
        return {"vision": self.visual_facts, "audio": self.audio_tags,
                "subtitle": self.subtitle}[modality]


def load_jsonl(path: str) -> List[QAItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(QAItem(**json.loads(line)))
    return items


def load_sample() -> List[QAItem]:
    here = os.path.dirname(__file__)
    return load_jsonl(os.path.join(here, "..", "..", "data", "sample", "videoqa.jsonl"))
