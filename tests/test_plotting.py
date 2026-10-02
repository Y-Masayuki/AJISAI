"""
Tests for ``ajisai.plotting``: tick labels, the three diagnostic figures,
``plot_from_workdir``, and the ``PlotConfig`` options.

CASA is not required. The FITS images are small synthetic CASA-style images
written into ``tmp_path``; the figures are rendered with the Agg backend.
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from ajisai import (
    AJISAI,
    AJISAIConfig,
    PlotConfig,
    plot_from_workdir,
    plot_refant_selection,
)
from ajisai.plotting import (
    AJISAI_LIGHT_PURPLE,
    format_solint,
    ordinal,
    panels_from_fits_dir,
    panels_from_metrics,
    plot_selfcal_images,
    plot_selfcal_summary,
    round_half_up,
    round_labels,
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------
def _write_fits(path, peak_jy, seed, nx=96, cdelt_arcsec=0.05, bmaj=0.5, bmin=0.4):
    """A small CASA-style FITS image: Gaussian source at the center plus noise."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1e-3, (1, 1, nx, nx))
    yy, xx = np.mgrid[:nx, :nx]
    c = nx // 2
    data[0, 0] += peak_jy * np.exp(-(((xx - c) / 6.0) ** 2 + ((yy - c) / 5.0) ** 2) / 2.0)
    hdr = fits.Header()
    hdr["CDELT1"] = -cdelt_arcsec / 3600.0
    hdr["CDELT2"] = cdelt_arcsec / 3600.0
    hdr["CRPIX1"] = c + 1
    hdr["CRPIX2"] = c + 1
    hdr["BMAJ"] = bmaj / 3600.0
    hdr["BMIN"] = bmin / 3600.0
    hdr["BPA"] = -56.0
    hdr["BUNIT"] = "Jy/beam"
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
    return path


@pytest.fixture
def metrics_df():
    """Five rounds modelled on the TW Hya reference run, with one anomaly."""
    return pd.DataFrame({
        "iteration": [0, 1, 2, 3, 4],
        "label": ["no_selfcal", "phase_inf", "phase_6IT", "phase_3IT", "amp_inf"],
        "calmode": ["-", "p", "p", "p", "a"],
        "solint": ["-", "inf", "36.288s", "18.144s", "inf"],
        "peak_jy_beam": [0.3336, 0.3415, 0.3519, 0.3548, 0.3559],
        "rms_jy_beam": [4.115e-3, 1.939e-3, 1.497e-3, 1.473e-3, 1.444e-3],
        "snr": [81.06, 176.15, 235.13, 240.81, 246.49],
        "dynamic_range": [81.06, 176.15, 235.13, 240.81, 246.49],
        "bmaj_arcsec": [0.5596] * 4 + [0.5558],
        "bmin_arcsec": [0.4164] * 4 + [0.4146],
        "status": ["ok", "ok", "anomaly", "ok", "ok"],
        "fits_path": [None] * 5,
    })


@pytest.fixture
def refant_info():
    """A ``derived["refant"]`` record with three antennas above the threshold."""
    names = ["DA41", "DA42", "DA48", "DV01", "DV07", "DV16", "DV21", "DV23"]
    xs = [-10.0, -15.0, -30.0, -30.0, 20.0, -150.0, 20.0, 25.0]
    ys = [-20.0, 10.0, 0.0, -15.0, 5.0, -40.0, -15.0, 0.0]
    flags = [1.0, 0.02, 0.01, 0.98, 0.95, 0.03, 0.9, 0.05]
    return {
        "refant": "DA42", "strategy": "hybrid", "flag_threshold": 0.25,
        "fallback_used": False, "reason": "hybrid: nearest to center among 5 antennas",
        "antenna_names": names, "antenna_x": xs, "antenna_y": ys,
        "antenna_flag_frac": flags, "antenna_dist_from_center": [0.0] * 8,
        "center_xy": [0.0, 0.0], "chosen_index": 1,
    }


@pytest.fixture
def fits_dir(tmp_path):
    """clean_iter0 + clean_sc1, sc2, sc10 with increasing peaks (sc10 tests sorting)."""
    d = tmp_path / "intermediate"
    d.mkdir()
    for i, peak in [(0, 0.30), (1, 0.33), (2, 0.35), (10, 0.36)]:
        name = "clean_iter0.fits" if i == 0 else f"clean_sc{i}.fits"
        _write_fits(d / name, peak, seed=i)
    return d


@pytest.fixture
def existing_vis(tmp_path):
    p = tmp_path / "fake.ms"
    p.mkdir()
    return str(p)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected", [
    (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"),
    (13, "13th"), (21, "21st"), (22, "22nd"), (101, "101st"), (111, "111th"),
])
def test_ordinal(n, expected):
    assert ordinal(n) == expected


@pytest.mark.parametrize("value, expected", [
    (0.4, 0), (0.5, 1), (1.5, 2), (2.5, 3), (246.49, 246), (240.81, 241), (1443.9, 1444),
])
def test_round_half_up(value, expected):
    assert round_half_up(value) == expected


@pytest.mark.parametrize("solint, expected", [
    ("36.288s", "36 s"), ("18.144s", "18 s"), ("36.5s", "37 s"), ("60s", "60 s"),
    ("0.3s", "0.3 s"), ("inf", "inf"), ("-", "-"), ("6*IT", "6*IT"),
])
def test_format_solint(solint, expected):
    assert format_solint(solint) == expected


def test_round_labels_step(metrics_df):
    assert round_labels(metrics_df, "step") == [
        "no self-cal", "1st: phase\ninf", "2nd: phase\n36 s", "3rd: phase\n18 s", "4th: amp\ninf",
    ]


def test_round_labels_ordinal(metrics_df):
    assert round_labels(metrics_df, "ordinal") == [
        "no self-cal", "1st self-cal", "2nd self-cal", "3rd self-cal", "4th self-cal",
    ]


def test_round_labels_unknown_style(metrics_df):
    with pytest.raises(ValueError, match="unknown style"):
        round_labels(metrics_df, "fancy")


# ---------------------------------------------------------------------------
# selfcal_summary.png
# ---------------------------------------------------------------------------
def test_plot_selfcal_summary_writes_png(tmp_path, metrics_df):
    out = tmp_path / "selfcal_summary.png"
    fig = plot_selfcal_summary(metrics_df, best_iteration=4, projname="test", outpath=out)
    assert out.is_file() and out.stat().st_size > 0
    axes = fig.axes
    assert len(axes) == 4
    # One tick per round, labelled with what the round did.
    assert [t.get_text() for t in axes[-1].get_xticklabels()] == round_labels(metrics_df)
    assert list(axes[-1].get_xticks()) == [0, 1, 2, 3, 4]
    # Legend lists the statuses present and the best round.
    legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert legend_texts == ["ok", "anomaly", "best: 4th: amp, inf"]
    # The best-iteration line is drawn in every panel.
    for ax in axes:
        assert any(line.get_xdata()[0] == 4 and line.get_linestyle() == "--" for line in ax.lines)


def test_plot_selfcal_summary_ordinal_labels_rotated(metrics_df):
    fig = plot_selfcal_summary(metrics_df, xtick_style="ordinal")
    labels = fig.axes[-1].get_xticklabels()
    assert labels[1].get_text() == "1st self-cal"
    assert labels[1].get_rotation() == 90


def test_plot_selfcal_summary_single_round_without_best(metrics_df):
    fig = plot_selfcal_summary(metrics_df.iloc[:1])
    assert fig.axes[-1].get_xlim() == (-0.5, 0.5)
    assert fig.axes[0].get_legend() is not None


def test_plot_selfcal_summary_beam_panel_min_span(metrics_df):
    fig = plot_selfcal_summary(metrics_df, beam_min_rel_span=0.02)
    lo, hi = fig.axes[3].get_ylim()
    # 2 % of ~482 mas: the 2.6 mas beam change must not fill the whole panel.
    assert hi - lo >= 0.02 * 482 * 0.99


def test_plot_functions_do_not_change_global_rcparams(metrics_df, refant_info):
    before = (matplotlib.rcParams["font.family"], matplotlib.rcParams["axes.linewidth"])
    plot_selfcal_summary(metrics_df)
    plot_refant_selection(refant_info)
    after = (matplotlib.rcParams["font.family"], matplotlib.rcParams["axes.linewidth"])
    assert before == after


# ---------------------------------------------------------------------------
# ajisai_refant_selection.png
# ---------------------------------------------------------------------------
def test_plot_refant_selection_writes_square_png(tmp_path, refant_info):
    out = tmp_path / "refant.png"
    fig = plot_refant_selection(refant_info, outpath=out, title="refant test")
    assert out.is_file() and out.stat().st_size > 0
    ax = fig.axes[0]
    # Square frame centred on the geometric centre, same scale on both axes.
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_xlim()[0] == -ax.get_xlim()[1]
    assert ax.get_aspect() == 1.0
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["excluded (flag ≥ 0.25)", "refant = DA42", "XY geometric center"]
    assert ax.get_title() == "refant test"
    # The colour bar is a second axes.
    assert len(fig.axes) == 2


def test_plot_refant_selection_threshold_at_edge_falls_back_to_linear(refant_info):
    info = dict(refant_info, flag_threshold=1.0)
    fig = plot_refant_selection(info)  # TwoSlopeNorm would reject vcenter == vmax
    assert len(fig.axes) == 2


def test_plot_refant_selection_bad_norm_mode(refant_info):
    with pytest.raises(ValueError, match="norm_mode"):
        plot_refant_selection(refant_info, norm_mode="log")


# ---------------------------------------------------------------------------
# selfcal_images.png
# ---------------------------------------------------------------------------
def test_panels_from_fits_dir_sorted_numerically(fits_dir):
    panels = panels_from_fits_dir(fits_dir, exclude_beam_factor=5.0)
    assert [p["iteration"] for p in panels] == [0, 1, 2, 10]
    assert [p["label"] for p in panels] == ["no self-cal", "1st self-cal", "2nd self-cal",
                                             "10th self-cal"]
    assert all(p["fits"].is_file() for p in panels)
    assert all(p["dynamic_range"] > 0 and p["rms_jy_beam"] > 0 for p in panels)


def test_panels_from_metrics_lookup_by_name(metrics_df, fits_dir):
    df = metrics_df.iloc[:3].copy()
    df["fits_path"] = ["/gone/clean_iter0.fits", "/gone/clean_sc1.fits", "/gone/clean_sc2.fits"]
    # Original paths do not exist -> placeholders.
    assert [p["fits"] for p in panels_from_metrics(df)] == [None, None, None]
    # Looked up by file name in fits_dir -> found.
    panels = panels_from_metrics(df, fits_dir)
    assert [p["fits"].name for p in panels] == ["clean_iter0.fits", "clean_sc1.fits",
                                                 "clean_sc2.fits"]
    assert panels[2]["status"] == "anomaly"
    assert panels[1]["label"] == "1st: phase\ninf"


def test_plot_selfcal_images_two_rows(tmp_path, metrics_df, fits_dir):
    df = metrics_df.iloc[:3].copy()
    df["fits_path"] = [str(fits_dir / n) for n in ("clean_iter0.fits", "clean_sc1.fits",
                                                   "clean_sc2.fits")]
    panels = panels_from_metrics(df)
    out = tmp_path / "selfcal_images.png"
    fig = plot_selfcal_images(panels, best_iteration=2, projname="test", outpath=out)
    assert out.is_file() and out.stat().st_size > 0
    # 3 panels x 2 rows; each row's colour bar is an inset of its last panel.
    assert len(fig.axes) == 3 * 2
    top = fig.axes[:3]
    assert [ax.get_title() for ax in top] == round_labels(df)
    # Field of view: 10 beams, east to the left, centred on the phase center.
    beam = np.sqrt(0.5 * 0.4)
    assert top[0].get_xlim() == pytest.approx((10 * beam, -10 * beam))
    assert top[0].get_ylim() == pytest.approx((-10 * beam, 10 * beam))
    # The best round is framed.
    assert top[2].spines["left"].get_edgecolor() == matplotlib.colors.to_rgba(AJISAI_LIGHT_PURPLE)
    assert top[0].spines["left"].get_edgecolor() != matplotlib.colors.to_rgba(AJISAI_LIGHT_PURPLE)
    # Second row has no titles; each row has its own colour bar label.
    assert fig.axes[3].get_title() == ""
    colorbars = [ax.child_axes[0] for ax in (fig.axes[2], fig.axes[5])]
    assert [cb.get_ylabel() for cb in colorbars] == ["mJy/beam (gamma = 1)",
                                                     "mJy/beam (gamma = 0.3)"]
    assert all(len(ax.child_axes) == 0 for ax in fig.axes if ax not in (fig.axes[2], fig.axes[5]))


def test_plot_selfcal_images_single_row_and_placeholder(metrics_df, fits_dir):
    df = metrics_df.iloc[:3].copy()
    df["fits_path"] = [str(fits_dir / "clean_iter0.fits"), None, str(fits_dir / "clean_sc2.fits")]
    panels = panels_from_metrics(df)
    assert panels[1]["fits"] is None
    fig = plot_selfcal_images(panels, gamma2=None, fov_radius_arcsec=1.0)
    assert len(fig.axes) == 3  # one row
    assert len(fig.axes[2].child_axes) == 1  # its colour bar
    # Missing round -> grey placeholder with the status as text.
    placeholder = fig.axes[1]
    assert placeholder.get_facecolor() == pytest.approx(matplotlib.colors.to_rgba("0.85"))
    assert any(t.get_text() == "ok" for t in placeholder.texts)
    assert fig.axes[0].get_xlim() == pytest.approx((1.0, -1.0))


def test_plot_selfcal_images_center_on_peak(fits_dir):
    panels = panels_from_fits_dir(fits_dir)
    fig = plot_selfcal_images(panels, center="peak", gamma2=None)
    x0 = 0.5 * sum(fig.axes[0].get_xlim())
    y0 = 0.5 * sum(fig.axes[0].get_ylim())
    # The synthetic source sits on the reference pixel, so both centres agree.
    assert abs(x0) < 0.06 and abs(y0) < 0.06


def test_plot_selfcal_images_errors(fits_dir):
    panels = panels_from_fits_dir(fits_dir)
    with pytest.raises(ValueError, match="center"):
        plot_selfcal_images(panels, center="corner")
    with pytest.raises(ValueError, match="no FITS image"):
        plot_selfcal_images([dict(p, fits=None) for p in panels])


# ---------------------------------------------------------------------------
# plot_from_workdir
# ---------------------------------------------------------------------------
def test_plot_from_workdir_regenerates_all_three(tmp_path, metrics_df, refant_info, fits_dir):
    workdir = tmp_path / "ajisai_demo"
    workdir.mkdir()
    df = metrics_df.iloc[:3].copy()
    df["fits_path"] = [str(fits_dir / n) for n in ("clean_iter0.fits", "clean_sc1.fits",
                                                   "clean_sc2.fits")]
    df.to_csv(workdir / "metrics.csv", index=False)
    justification = {
        "input_config": {"projname": None},
        "derived": {"refant": refant_info},
        "best": {"iteration": 2, "metric_key": "dynamic_range", "value": 235.13},
    }
    (workdir / "justification.json").write_text(json.dumps(justification))

    written = plot_from_workdir(workdir, gamma2=None)
    assert sorted(written) == ["ajisai_refant_selection.png", "selfcal_images.png",
                               "selfcal_summary.png"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in written.values())
    assert all(p.parent == workdir for p in written.values())

    # out_dir and a moved FITS directory (looked up by name).
    out = tmp_path / "replot"
    written = plot_from_workdir(workdir, fits_dir=fits_dir, out_dir=out, dpi=72)
    assert all(p.parent == out and p.is_file() for p in written.values())
    assert plt.get_fignums() == []  # plot_from_workdir closes its figures


# ---------------------------------------------------------------------------
# PlotConfig
# ---------------------------------------------------------------------------
def test_plot_config_defaults_and_frozen(existing_vis):
    cfg = AJISAIConfig(vis=existing_vis)
    assert cfg.plots == PlotConfig()
    assert cfg.plots.images_cmap == "inferno"
    assert cfg.plots.images_gamma == 1.0
    assert cfg.plots.images_gamma2 == 0.3
    assert cfg.plots.images_fov_beam_factor == 10.0
    assert cfg.plots.images_fov_radius_arcsec is None
    assert cfg.plots.images_center == "phase"
    with pytest.raises(FrozenInstanceError):
        cfg.plots.images_gamma = 0.5  # type: ignore[misc]
    # The options are part of the config dump written to justification.json.
    assert asdict(cfg)["plots"]["images_gamma2"] == 0.3


@pytest.mark.parametrize("plots, match", [
    (PlotConfig(images_center="corner"), "images_center"),
    (PlotConfig(images_gamma=0.0), "images_gamma"),
    (PlotConfig(images_gamma2=-1.0), "images_gamma2"),
])
def test_plot_config_validation(existing_vis, plots, match):
    with pytest.raises(ValueError, match=match):
        AJISAI(AJISAIConfig(vis=existing_vis, plots=plots))._validate_inputs()


def test_plot_config_gamma2_none_is_valid(existing_vis):
    AJISAI(AJISAIConfig(vis=existing_vis, plots=PlotConfig(images_gamma2=None)))._validate_inputs()
