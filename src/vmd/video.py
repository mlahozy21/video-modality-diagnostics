"""Frame-level video support: real frames, real corruptions, and a VLM backend.

This module turns the textual-proxy harness into an end-to-end *video*
diagnostic: frames are sampled from the actual .mp4, corruptions operate on the
frame sequence itself (temporal shuffling, frame dropping), and the backend is
a video VLM (default: Qwen2.5-VL-3B-Instruct).

Only the backend changes — the metrics, the ablation grid (`run_diagnostic`)
and the reports are reused unchanged.
"""
from __future__ import annotations

import random
import re


# --------------------------------------------------------------------------- #
# Frame sampling and frame-level corruptions
# --------------------------------------------------------------------------- #
def sample_frames(video_path: str, n_frames: int = 8, longest_side: int = 448):
    """Uniformly sample `n_frames` RGB PIL images from a video file."""
    import cv2  # lazy
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"could not read frames from {video_path}")
    idxs = [int(i * (total - 1) / max(1, n_frames - 1)) for i in range(n_frames)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            continue
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max(img.size) > longest_side:
            ratio = longest_side / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        frames.append(img)
    cap.release()
    if not frames:
        raise ValueError(f"no decodable frames in {video_path}")
    return frames


def corrupt_frames(frames: list, severity: float, kind: str = "shuffle",
                   seed: int = 0) -> list:
    """Corrupt a frame sequence.

    - `shuffle`: permute a `severity` fraction of the frames (destroys temporal
      order while preserving content — the video analogue of token shuffling).
    - `drop`: remove a `severity` fraction of the frames (destroys content).
    """
    if severity <= 0 or len(frames) < 2:
        return list(frames)
    rng = random.Random(seed)
    n = len(frames)
    if kind == "drop":
        keep = max(1, round(n * (1 - severity)))
        idx = sorted(rng.sample(range(n), keep))
        return [frames[i] for i in idx]
    # shuffle
    k = max(2, round(n * severity))
    idx = rng.sample(range(n), k)
    vals = [frames[i] for i in idx]
    rng.shuffle(vals)
    out = list(frames)
    for i, v in zip(idx, vals):
        out[i] = v
    return out


# --------------------------------------------------------------------------- #
# VLM backend over real frames
# --------------------------------------------------------------------------- #
class VLMVideoBackend:
    """Video VLM backend: answers from real sampled frames.

    Channel semantics (driven by the ModalityView, like every other backend):
    - "vision" present  -> frames are sampled from `item.video_path`; the
      vision corruption severity is applied to the *frame sequence* with the
      operator chosen at construction (`corruption_kind`).
    - "vision" absent   -> no frames are given (blind / language-prior run).
    - "subtitle" present and non-empty -> appended as text.

    Default model: Qwen/Qwen2.5-VL-3B-Instruct (fits a Colab L4/T4 in fp16).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 n_frames: int = 8, corruption_kind: str = "shuffle",
                 max_new_tokens: int = 8, seed: int = 0):
        self.name = f"{model_name} ({n_frames} frames, corrupt={corruption_kind})"
        self.model_name = model_name
        self.n_frames = n_frames
        self.corruption_kind = corruption_kind
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._m = None
        self._proc = None

    def _load(self):
        if self._m is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # lazy
            self._proc = AutoProcessor.from_pretrained(self.model_name)
            self._m = AutoModelForImageTextToText.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto")
            self._m.eval()

    def _question_block(self, item, view) -> str:
        options = "\n".join(f"{i}. {o}" for i, o in enumerate(item.options))
        sub = view.available.get("subtitle", "")
        sub_block = f"Subtitles: {sub}\n\n" if sub else ""
        return (f"{sub_block}Question: {item.question}\n{options}\n\n"
                "Reply with the number of the correct option and nothing else.")

    def answer(self, item, view) -> int:
        import torch  # lazy
        self._load()

        content = []
        if "vision" in view.available and item.video_path:
            frames = sample_frames(item.video_path, self.n_frames)
            sev = view.corruption.get("vision", 0.0)
            frames = corrupt_frames(frames, sev, kind=self.corruption_kind,
                                    seed=self.seed)
            content += [{"type": "image", "image": f} for f in frames]
            preamble = "These are frames sampled from a video, in order.\n\n"
        else:
            preamble = "No video is available; answer as best you can.\n\n"
        content.append({"type": "text",
                        "text": preamble + self._question_block(item, view)})

        msgs = [{"role": "user", "content": content}]
        inputs = self._proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self._m.device)
        with torch.no_grad():
            out = self._m.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                   do_sample=False)
        text = self._proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        match = re.search(r"\d+", text)
        if match and int(match.group()) < len(item.options):
            return int(match.group())
        return 0  # unparseable -> first option (deterministic fallback)
