"""
Unit tests for the frequency-averaged (1 channel per spw) self-cal MS
byproduct created by ``AJISAI._make_chanavg_ms``.

These tests do NOT require CASA: ``core.casatasks`` is monkeypatched with a
fake whose ``split`` records the keyword arguments it was called with, so we
can assert the task is invoked with the right output path, datacolumn, and
per-spw averaging width without running the real task.
"""
from __future__ import annotations

import types

from ajisai import core


def _make_stub(tmp_path, derived, verbose=False):
    """A minimal object exposing just what _make_chanavg_ms touches."""
    (tmp_path / "intermediate").mkdir(exist_ok=True)
    return types.SimpleNamespace(
        workdir=tmp_path,
        _derived=derived,
        cfg=types.SimpleNamespace(verbose=verbose),
    )


def test_make_chanavg_ms_averages_every_spw_to_one_channel(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(core, "casatasks",
                        types.SimpleNamespace(split=lambda **kw: calls.update(kw)))
    monkeypatch.setattr(core, "_safe_rmtree_path", lambda p: None)

    stub = _make_stub(tmp_path, {"num_chan_per_spw": [16, 16]})
    src = tmp_path / "intermediate" / "selfcal_1.ms"
    out = core.AJISAI._make_chanavg_ms(stub, src, 1)

    interm = tmp_path / "intermediate"
    # Output lands in intermediate/ with the _avg suffix
    assert out == interm / "selfcal_1_avg.ms"
    assert calls["outputvis"] == str(interm / "selfcal_1_avg.ms")
    # Split reads the calibrated DATA column of the self-cal MS
    assert calls["vis"] == str(src)
    assert calls["datacolumn"] == "data"
    # One width entry per spw -> every spw collapses to a single channel
    assert calls["width"] == [16, 16]


def test_make_chanavg_ms_handles_unequal_channel_counts(monkeypatch, tmp_path):
    """Per-spw width list must mirror differing NUM_CHAN per spw."""
    calls = {}
    monkeypatch.setattr(core, "casatasks",
                        types.SimpleNamespace(split=lambda **kw: calls.update(kw)))
    monkeypatch.setattr(core, "_safe_rmtree_path", lambda p: None)

    stub = _make_stub(tmp_path, {"num_chan_per_spw": [128, 64, 3840]})
    core.AJISAI._make_chanavg_ms(stub, tmp_path / "selfcal_3.ms", 3)
    assert calls["width"] == [128, 64, 3840]


def test_make_chanavg_ms_width_falls_back_to_scalar(monkeypatch, tmp_path):
    """If the per-spw list is missing, fall back to the scalar chavg_width."""
    calls = {}
    monkeypatch.setattr(core, "casatasks",
                        types.SimpleNamespace(split=lambda **kw: calls.update(kw)))
    monkeypatch.setattr(core, "_safe_rmtree_path", lambda p: None)

    stub = _make_stub(tmp_path, {"chavg_width": 8})
    core.AJISAI._make_chanavg_ms(stub, tmp_path / "selfcal_2.ms", 2)
    assert calls["width"] == 8


def test_make_chanavg_ms_cleans_stale_output(monkeypatch, tmp_path):
    """A pre-existing averaged MS is removed before split runs."""
    removed = []
    monkeypatch.setattr(core, "casatasks",
                        types.SimpleNamespace(split=lambda **kw: None))
    monkeypatch.setattr(core, "_safe_rmtree_path", lambda p: removed.append(str(p)))

    stub = _make_stub(tmp_path, {"num_chan_per_spw": [16]})
    core.AJISAI._make_chanavg_ms(stub, tmp_path / "selfcal_1.ms", 1)
    assert str(tmp_path / "intermediate" / "selfcal_1_avg.ms") in removed
