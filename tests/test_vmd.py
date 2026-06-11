"""Smoke + behaviour tests for vmd. All offline, CPU-only, deterministic."""
import math

from vmd.backends import OfflineStubBackend
from vmd.config import DiagConfig
from vmd.data import load_sample
from vmd.diagnose import run_diagnostic
from vmd.metrics import accuracy, collapse_gap, modality_contribution
from vmd.modalities import make_view
from vmd.report import format_report


def test_metrics_basic():
    assert accuracy([1, 2, 3], [1, 0, 3]) == 2 / 3
    assert accuracy([], []) == 0.0
    contrib = modality_contribution(0.9, {"vision": 0.6, "audio": 0.9})
    assert contrib["vision"] == 0.3 and contrib["audio"] == 0.0
    assert collapse_gap(0.9, 0.25) == 0.65


def test_sample_dataset_loads_and_is_well_formed():
    items = load_sample()
    assert len(items) >= 10
    for it in items:
        assert 0 <= it.answer_idx < len(it.options)
        assert it.gold_modality in {"vision", "audio", "subtitle"}
        assert it.evidence(it.gold_modality)  # gold evidence is non-empty


def test_make_view_drop_and_corruption():
    items = load_sample()
    it = items[0]
    full = make_view(it, ["vision", "audio", "subtitle"])
    assert set(full.available) == {"vision", "audio", "subtitle"}
    # severity 1.0 with kind="drop" removes the channel entirely
    dropped = make_view(it, ["vision", "audio"], corrupt={"vision": 1.0}, kind="drop")
    assert "vision" not in dropped.available and "audio" in dropped.available
    # shuffle corruption preserves the bag of words
    noisy = make_view(it, ["vision"], corrupt={"vision": 0.8}, kind="shuffle", seed=3)
    assert sorted(noisy.available["vision"].split()) == sorted(it.visual_facts.split())


def test_stub_uses_gold_modality_and_falls_back_when_ablated():
    items = load_sample()
    stub = OfflineStubBackend()
    it = next(i for i in items if i.gold_modality == "audio")
    with_audio = make_view(it, ["vision", "audio", "subtitle"])
    without_audio = make_view(it, ["vision", "subtitle"])
    assert stub.answer(it, with_audio) == it.answer_idx
    assert stub.answer(it, without_audio) != it.answer_idx  # collapses to prior


def test_full_diagnostic_shape_and_signal():
    items = load_sample()
    r = run_diagnostic(OfflineStubBackend(), items)
    for key in ("acc_full", "blind_language_prior", "collapse_gap",
                "modality_contribution", "acc_leave_one_out",
                "acc_single_modality", "vision_robustness"):
        assert key in r
    # the stub genuinely uses the media: large collapse gap, full acc high
    assert r["acc_full"] >= 0.9
    assert r["collapse_gap"] >= 0.5
    # every modality contributes on this balanced sample
    assert all(c > 0 for c in r["modality_contribution"].values())
    # vision robustness degrades once corruption exceeds the stub tolerance
    rob = r["vision_robustness"]
    assert rob["0.0"] >= rob["1.0"]


def test_diagnostic_is_deterministic():
    items = load_sample()
    r1 = run_diagnostic(OfflineStubBackend(), items, DiagConfig(seed=0))
    r2 = run_diagnostic(OfflineStubBackend(), items, DiagConfig(seed=0))
    assert r1 == r2


def test_report_renders():
    items = load_sample()
    r = run_diagnostic(OfflineStubBackend(), items)
    text = format_report(r, "offline-stub")
    assert "collapse gap" in text and "modality contribution" in text
    assert not any(math.isnan(v) for v in r["acc_single_modality"].values())
