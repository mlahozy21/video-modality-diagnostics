"""CPU tests for the frame-level pieces (no model, no real video needed)."""
import pytest

PIL = pytest.importorskip("PIL.Image")

from vmd.video import corrupt_frames  # noqa: E402


def _frames(n=8):
    # tiny distinct images so identity can be tracked through corruption
    return [PIL.new("RGB", (4, 4), (i, 0, 0)) for i in range(n)]


def test_severity_zero_is_identity():
    fr = _frames()
    assert corrupt_frames(fr, 0.0) == fr


def test_shuffle_preserves_content_set():
    fr = _frames()
    out = corrupt_frames(fr, 1.0, kind="shuffle", seed=1)
    assert len(out) == len(fr)
    assert {id(f) for f in out} == {id(f) for f in fr}   # same frames
    assert out != fr                                      # different order


def test_drop_removes_content():
    fr = _frames(8)
    out = corrupt_frames(fr, 0.5, kind="drop", seed=0)
    assert len(out) == 4
    # surviving frames keep their original relative order
    pos = [fr.index(f) for f in out]
    assert pos == sorted(pos)


def test_drop_never_empties():
    fr = _frames(4)
    assert len(corrupt_frames(fr, 1.0, kind="drop")) >= 1


def test_deterministic_given_seed():
    fr = _frames()
    a = corrupt_frames(fr, 0.7, kind="shuffle", seed=42)
    b = corrupt_frames(fr, 0.7, kind="shuffle", seed=42)
    assert a == b
