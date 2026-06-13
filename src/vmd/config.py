from dataclasses import dataclass, field
from typing import List


@dataclass
class DiagConfig:
    modalities: List[str] = field(default_factory=lambda: ["vision", "audio", "subtitle"])
    severities: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    seed: int = 0
    # Number of corruption realizations averaged per severity in the robustness
    # sweep. Kept small and seeded (base_seed = ``seed``) so runs stay
    # reproducible; 1 reproduces the single-realization behaviour.
    robustness_seeds: int = 3


CONFIG = DiagConfig()
