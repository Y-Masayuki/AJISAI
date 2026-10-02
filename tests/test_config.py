"""
Tests for AJISAIConfig / ImagingConfig / GainCalConfig validation.

These tests exercise the immutability, defaults, and validation logic of the
configuration dataclasses. CASA is NOT required; the tests only construct
AJISAI objects and call ``_validate_inputs``, never ``run()``.
"""
from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import pytest

from ajisai import (
    AJISAI,
    AJISAIConfig,
    GainCalConfig,
    ImagingConfig,
    SelfcalSchedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def existing_vis(tmp_path):
    """Create a placeholder file/dir that satisfies the cfg.vis existence check."""
    p = tmp_path / "fake.ms"
    p.mkdir()
    return str(p)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_default_config_smoke(existing_vis):
    """Default AJISAIConfig with only vis works."""
    cfg = AJISAIConfig(vis=existing_vis)
    assert cfg.vis == existing_vis
    assert cfg.phase_shift is False
    assert cfg.phase_center is None
    assert cfg.refant_strategy == "hybrid"
    assert cfg.refant_flag_threshold == 0.25
    assert cfg.rms_method == "sigma_clip_excl"
    assert cfg.rms_target_radius_arcsec is None
    assert cfg.quality_metric == "dynamic_range"
    assert cfg.on_iter_anomaly == "log_only"
    assert cfg.verbose is True
    assert cfg.show_banner is True


def test_default_imaging_config():
    cfg = ImagingConfig()
    assert cfg.robust == 0.5
    assert cfg.weighting == "briggs"
    assert cfg.deconvolver == "multiscale"
    assert cfg.mask_mode == "auto-multithresh"
    assert cfg.cellpix == 10
    assert cfg.uvtaper == ()


def test_default_gaincal_config():
    cfg = GainCalConfig()
    assert cfg.minblperant == 4
    assert cfg.minsnr == 1.5
    assert cfg.gaintype == "T"
    assert cfg.applymode == "calonly"


def test_default_schedule_is_three_phase_one_amp():
    """The locked-in design: 3 phase + 1 amp iterations."""
    sched = SelfcalSchedule()
    assert len(sched.steps) == 4
    assert [s.calmode for s in sched.steps] == ["p", "p", "p", "a"]
    assert sched.steps[3].solnorm is True  # amp step must normalize


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------
def test_config_is_frozen(existing_vis):
    """frozen=True means you cannot mutate the config after construction."""
    cfg = AJISAIConfig(vis=existing_vis)
    with pytest.raises(FrozenInstanceError):
        cfg.vis = "/other.ms"  # type: ignore[misc]


def test_imaging_config_is_frozen():
    cfg = ImagingConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.robust = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation: vis existence
# ---------------------------------------------------------------------------
def test_validate_inputs_missing_vis(tmp_path):
    cfg = AJISAIConfig(vis=str(tmp_path / "does_not_exist.ms"))
    aj = AJISAI(cfg)
    with pytest.raises(FileNotFoundError, match="vis not found"):
        aj._validate_inputs()


# ---------------------------------------------------------------------------
# Validation: refant strategies
# ---------------------------------------------------------------------------
def test_refant_manual_requires_name(existing_vis):
    cfg = AJISAIConfig(vis=existing_vis, refant_strategy="manual", refant_manual=None)
    aj = AJISAI(cfg)
    with pytest.raises(ValueError, match="refant_manual"):
        aj._validate_inputs()


# ---------------------------------------------------------------------------
# Validation: phase_center 3-tuple semantics
# ---------------------------------------------------------------------------
def test_phase_center_icrs_accepted(existing_vis):
    cfg = AJISAIConfig(
        vis=existing_vis,
        phase_shift=True,
        phase_center=("16:25:45.0", "-24:12:23.0", "ICRS"),
    )
    AJISAI(cfg)._validate_inputs()  # should not raise


def test_phase_center_j2000_accepted(existing_vis):
    cfg = AJISAIConfig(
        vis=existing_vis,
        phase_shift=True,
        phase_center=("16:25:45.0", "-24:12:23.0", "J2000"),
    )
    AJISAI(cfg)._validate_inputs()


@pytest.mark.parametrize("bad_center", [
    ("16:25:45.0", "-24:12:23.0", "B1950"),       # wrong frame
    ("16:25:45.0", "-24:12:23.0"),                # only 2 elements
    (1.0, 2.0, "ICRS"),                            # numeric instead of strings
    "not a tuple",                                  # not a tuple
])
def test_phase_center_invalid_raises(existing_vis, bad_center):
    cfg = AJISAIConfig(vis=existing_vis, phase_shift=True, phase_center=bad_center)
    aj = AJISAI(cfg)
    with pytest.raises(ValueError):
        aj._validate_inputs()


def test_phase_center_without_phase_shift_warns(existing_vis):
    """If phase_center is given but phase_shift=False, AJISAI warns."""
    cfg = AJISAIConfig(
        vis=existing_vis,
        phase_shift=False,
        phase_center=("16:25:45.0", "-24:12:23.0", "ICRS"),
    )
    aj = AJISAI(cfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aj._validate_inputs()
    assert any("phase_center is set but phase_shift=False" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Validation: mask_mode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["auto-multithresh", "none"])
def test_mask_mode_valid(existing_vis, mode):
    cfg = AJISAIConfig(vis=existing_vis, imaging=ImagingConfig(mask_mode=mode))
    AJISAI(cfg)._validate_inputs()


def test_mask_mode_invalid_raises(existing_vis):
    cfg = AJISAIConfig(vis=existing_vis, imaging=ImagingConfig(mask_mode="bogus"))
    with pytest.raises(ValueError, match="mask_mode must be one of"):
        AJISAI(cfg)._validate_inputs()


def test_mask_mode_user_requires_user_mask(existing_vis):
    """mask_mode='user' must come with a user_mask path."""
    cfg = AJISAIConfig(
        vis=existing_vis,
        imaging=ImagingConfig(mask_mode="user", user_mask=None),
    )
    with pytest.raises(ValueError, match="user_mask"):
        AJISAI(cfg)._validate_inputs()


def test_mask_mode_user_with_path_ok(existing_vis):
    cfg = AJISAIConfig(
        vis=existing_vis,
        imaging=ImagingConfig(mask_mode="user", user_mask="/path/to/some.mask"),
    )
    AJISAI(cfg)._validate_inputs()


def test_mask_mode_interactive_warns(existing_vis):
    """Interactive mode warns about CASA 6.7+ viewer limitation."""
    cfg = AJISAIConfig(vis=existing_vis, imaging=ImagingConfig(mask_mode="interactive"))
    aj = AJISAI(cfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aj._validate_inputs()
    assert any("CASA <= 6.6" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# Validation: rms_method
# ---------------------------------------------------------------------------
def test_rms_method_annulus_warns(existing_vis):
    """The legacy annulus method emits a deprecation-style warning."""
    cfg = AJISAIConfig(vis=existing_vis, rms_method="annulus", rms_target_radius_arcsec=1.0)
    aj = AJISAI(cfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aj._validate_inputs()
    assert any("annulus" in str(x.message) and "legacy" in str(x.message) for x in w)


def test_rms_method_annulus_requires_target_radius(existing_vis):
    """rms_method='annulus' without rms_target_radius_arcsec is rejected before imaging."""
    cfg = AJISAIConfig(vis=existing_vis, rms_method="annulus")
    with pytest.raises(ValueError, match="rms_target_radius_arcsec"):
        AJISAI(cfg)._validate_inputs()


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_rms_target_radius_must_be_positive(existing_vis, radius):
    cfg = AJISAIConfig(vis=existing_vis, rms_method="annulus", rms_target_radius_arcsec=radius)
    with pytest.raises(ValueError, match="must be positive"):
        AJISAI(cfg)._validate_inputs()


def test_rms_target_radius_without_annulus_warns(existing_vis):
    """A radius given for a method that does not use it is reported, not silently ignored."""
    cfg = AJISAIConfig(vis=existing_vis, rms_target_radius_arcsec=1.0)
    aj = AJISAI(cfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aj._validate_inputs()
    assert any("rms_target_radius_arcsec is set but" in str(x.message) for x in w)


def test_pipeline_passes_target_radius_to_compute_rms(existing_vis, synthetic_fits):
    """Regression: the pipeline must forward rms_target_radius_arcsec to compute_rms.

    Before the field existed, rms_method='annulus' passed validation (with a
    warning) and then raised ValueError at the first image statistics.
    """
    cfg = AJISAIConfig(vis=existing_vis, rms_method="annulus", rms_target_radius_arcsec=1.0)
    stats = AJISAI(cfg).compute_image_stats(synthetic_fits)
    assert stats["rms_info"]["method"] == "annulus"
    assert stats["rms_info"]["justification"]["target_radius_arcsec"] == 1.0
    assert stats["rms_info"]["n_kept"] > 0
    # The 1 arcsec inner radius excludes the central source, so the annulus
    # recovers the 10 uJy/beam noise of the synthetic image.
    assert stats["rms_jy_beam"] == pytest.approx(1e-5, rel=0.05)


# ---------------------------------------------------------------------------
# Config -> dict roundtrip (for justification.json reproducibility)
# ---------------------------------------------------------------------------
def test_config_can_be_dumped_to_json(existing_vis, tmp_path):
    """asdict + json.dump must succeed; this is what justification.json does."""
    import json
    from dataclasses import asdict

    cfg = AJISAIConfig(
        vis=existing_vis,
        imaging=ImagingConfig(uvtaper=("100klambda",)),
    )
    data = asdict(cfg)
    out = tmp_path / "cfg.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    # Round-trip: read back and check vis is preserved
    with open(out) as f:
        roundtrip = json.load(f)
    assert roundtrip["vis"] == existing_vis
    assert roundtrip["imaging"]["uvtaper"] == ["100klambda"]
    assert roundtrip["rms_target_radius_arcsec"] is None
