"""Model backends.

- OfflineStubBackend: a *simulated* model used to validate the diagnostic
  methodology with no GPU. It answers from the modality channels it is given:
  if the channel holding the cue is present and not too corrupted, it answers
  correctly; otherwise it falls back to a language prior (lexical overlap of the
  question with the options) -- exactly the "modality collapse" failure mode the
  diagnostics expose. Numbers from this backend demonstrate the method, they are
  NOT a measurement of a real model.

- HFChatBackend: a real model (any Hugging Face chat model), loaded lazily;
  run it in Colab (see notebooks/diagnostics_colab.ipynb).
"""
from __future__ import annotations

import re
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


def _bow(text):
    return set(_WORD.findall((text or "").lower()))


class Backend(Protocol):
    name: str
    def answer(self, item, view) -> int: ...


class OfflineStubBackend:
    name = "offline-stub"

    def __init__(self, cue_corruption_tolerance: float = 0.5):
        self.tol = cue_corruption_tolerance

    def _language_prior(self, item) -> int:
        q = _bow(item.question)
        scores = [len(q & _bow(opt)) for opt in item.options]
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def answer(self, item, view) -> int:
        gm = item.gold_modality
        if gm in view.available and view.corruption.get(gm, 0.0) <= self.tol:
            cue = _bow(view.available[gm])
            scores = [len(cue & _bow(opt)) for opt in item.options]
            if max(scores) > 0:
                return int(max(range(len(scores)), key=lambda i: scores[i]))
        return self._language_prior(item)


class HFChatBackend:
    """Real model backend (lazy; install extras: pip install '.[vlm]').

    The diagnostic exposes each modality as a *textual evidence channel*
    (visual facts, audio tags, subtitles), so any instruction-tuned HF chat
    model can act as the "video model": it only sees the channels present in
    the current ModalityView. Default is Qwen2.5-1.5B-Instruct (runs on a free
    Colab T4). For a true end-to-end video VLM you would subclass this and
    feed raw frames/audio instead of the textual channels.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 8):
        self.name = model_name
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._m = None
        self._tok = None

    def _load(self):
        if self._m is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._m = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto")
            self._m.eval()

    def _prompt(self, item, view) -> str:
        channels = "\n".join(f"[{m}] {t}" for m, t in view.available.items()) \
            or "(no media available)"
        options = "\n".join(f"{i}. {o}" for i, o in enumerate(item.options))
        return (
            "You are answering a question about a video. The evidence channels "
            "available to you are listed below; channels that are missing were "
            "not provided.\n\n"
            f"{channels}\n\n"
            f"Question: {item.question}\n{options}\n\n"
            "Reply with the number of the correct option and nothing else."
        )

    def answer(self, item, view) -> int:
        import torch  # lazy
        self._load()
        msgs = [{"role": "user", "content": self._prompt(item, view)}]
        enc = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(self._m.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._m.device)
        with torch.no_grad():
            out = self._m.generate(input_ids, attention_mask=attention_mask,
                                   max_new_tokens=self.max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=self._tok.eos_token_id)
        text = self._tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"\d+", text)
        if match and int(match.group()) < len(item.options):
            return int(match.group())
        # unparseable output -> lexical fallback, same as the stub's prior
        q = _bow(item.question)
        scores = [len(q & _bow(opt)) for opt in item.options]
        return int(max(range(len(scores)), key=lambda i: scores[i]))


# Backwards-compatible alias (the channels are textual, so the default real
# backend is a text chat model rather than the VL variant).
QwenVLBackend = HFChatBackend
