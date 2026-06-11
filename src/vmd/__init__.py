"""Video Modality Diagnostics (vmd).

Tools to diagnose *which modalities a video model actually uses* when answering
questions: modality ablations, a language-prior ("blind") baseline, modality-
contribution scores, and robustness probes (frame shuffling, noise, audio
desync). Model-agnostic via a small backend interface; ships an offline
deterministic stub so the whole methodology runs and is tested without a GPU.
"""
__version__ = "0.1.0"
