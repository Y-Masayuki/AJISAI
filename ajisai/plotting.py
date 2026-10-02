"""
Diagnostic figures written at the end of an AJISAI run.

Three PNGs are produced in the working directory:

``selfcal_summary.png``
    Dynamic range, peak, off-source RMS, and beam of every self-calibration
    round, one labelled tick per round.
``ajisai_refant_selection.png``
    Antenna positions coloured by flagged fraction, with the excluded
    antennas, the chosen reference antenna, and the array center marked.
``selfcal_images.png``
    The CLEAN image of every round side by side, left to right in the same
    order as the summary plot, on a shared colour scale.

Everything here is CASA-independent: the functions read ``metrics.csv``,
``justification.json``, and the per-iteration FITS images, so an existing
output directory can be re-plotted with :func:`plot_from_workdir` (for
example with a different ``gamma``) without re-running the pipeline.

The figure style (serif font, inward ticks, thick axes) is applied with
``matplotlib.rc_context`` inside each function, so calling them does not
change the global matplotlib configuration of the caller.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

#: Colour of "ok" points, the best-iteration line, and the AJISAI purple theme.
AJISAI_PURPLE = "#280a69"
#: Light purple used for the reference-antenna ring and the "best" image frame.
AJISAI_LIGHT_PURPLE = "#ddccff"

#: Point colour per iteration ``status`` in the summary plot.
STATUS_COLORS = {"ok": AJISAI_PURPLE, "anomaly": "orange", "skipped": "red"}

CALMODE_NAMES = {"p": "phase", "a": "amp", "ap": "amp+phase"}

#: rc parameters applied (via ``rc_context``) while drawing every figure.
#: Times New Roman is used where it is installed (macOS, most workstations);
#: the fallbacks are metric-compatible or similar serif fonts that ship with
#: Linux distributions and matplotlib itself, so no font warning is raised.
PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.top": True,
    "xtick.major.top": True,
    "axes.linewidth": 2.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 4.0,
    "ytick.minor.size": 4.0,
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def ordinal(n: int) -> str:
    """``1 -> '1st'``, ``2 -> '2nd'``, ``3 -> '3rd'``, ``11 -> '11th'``, ..."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def round_half_up(value: float) -> int:
    """Round to the nearest integer with .5 always rounded up (unlike ``round``)."""
    return int(np.floor(float(value) + 0.5))


def format_solint(solint: Any) -> str:
    """Format a resolved CASA solint for a tick label.

    ``'36.288s' -> '36 s'`` (rounded half up). Values that would round to
    zero keep three significant digits (``'0.3s' -> '0.3 s'``). Anything that
    is not a number of seconds, such as ``'inf'`` or ``'-'``, is returned
    unchanged.
    """
    m = re.fullmatch(r"\s*([0-9.]+)\s*s\s*", str(solint))
    if not m:
        return str(solint)
    seconds = float(m.group(1))
    rounded = round_half_up(seconds)
    return f"{rounded} s" if rounded > 0 else f"{seconds:.3g} s"


def round_labels(df: pd.DataFrame, style: str = "step") -> List[str]:
    """One x-axis label per row of the metrics table.

    ``style="step"`` (default)
        ``'no self-cal'``, ``'1st: phase\\ninf'``, ``'2nd: phase\\n36 s'``, ...
        i.e. the calibration mode and solution interval of each round.
    ``style="ordinal"``
        ``'no self-cal'``, ``'1st self-cal'``, ``'2nd self-cal'``, ...
    """
    labels = []
    for _, row in df.iterrows():
        i = int(row["iteration"])
        if i == 0:
            labels.append("no self-cal")
        elif style == "ordinal":
            labels.append(f"{ordinal(i)} self-cal")
        elif style == "step":
            mode = CALMODE_NAMES.get(str(row.get("calmode")), str(row.get("calmode")))
            labels.append(f"{ordinal(i)}: {mode}\n{format_solint(row.get('solint'))}")
        else:
            raise ValueError(f"unknown style {style!r}; use 'step' or 'ordinal'")
    return labels


# ---------------------------------------------------------------------------
# 1. Self-calibration summary
# ---------------------------------------------------------------------------

def plot_selfcal_summary(
    df: pd.DataFrame,
    best_iteration: Optional[int] = None,
    projname: str = "AJISAI",
    xtick_style: str = "step",
    xtick_rotation: Optional[float] = None,
    beam_min_rel_span: Optional[float] = 0.02,
    outpath: Optional[PathLike] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Four-panel summary (DR, peak, RMS, beam) with one labelled tick per round.

    Parameters
    ----------
    df
        Per-iteration metrics (the ``metrics.csv`` table). Needs the columns
        ``iteration``, ``calmode``, ``solint``, ``status``, ``dynamic_range``,
        ``peak_jy_beam``, ``rms_jy_beam``, ``bmaj_arcsec``, ``bmin_arcsec``.
    best_iteration
        Iteration selected as best; drawn as a dashed line in every panel.
    xtick_style
        ``"step"`` labels each round with its calibration mode and solution
        interval on two lines; ``"ordinal"`` uses ``'1st self-cal'`` etc.,
        rotated by 90 degrees unless ``xtick_rotation`` is given.
    beam_min_rel_span
        Minimum y range of the beam panel as a fraction of the median beam,
        so that a sub-percent beam change is not drawn as a large jump.
        ``None`` lets matplotlib autoscale.
    outpath
        If given, the figure is saved there (PNG) at ``dpi``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; the caller is responsible for closing it.
    """
    with plt.rc_context(PLOT_RC):
        x = df["iteration"].to_numpy(dtype=int)
        status = df["status"].fillna("ok").to_numpy()
        colors = [STATUS_COLORS.get(s, "gray") for s in status]
        labels = round_labels(df, style=xtick_style)
        if xtick_rotation is None:
            xtick_rotation = 90 if xtick_style == "ordinal" else 0

        beam_mas = np.sqrt(df["bmaj_arcsec"].astype(float) * df["bmin_arcsec"].astype(float)) * 1e3
        panels = [
            (df["dynamic_range"].astype(float), "Dynamic range\n(peak / RMS_offsrc)"),
            (df["peak_jy_beam"].astype(float) * 1e3, "Peak [mJy/beam]"),
            (df["rms_jy_beam"].astype(float) * 1e6, r"RMS [$\mu$Jy/beam]"),
            (beam_mas, r"Beam $\sqrt{\rm{bmaj}\times\rm{bmin}}$ [mas]"),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
        for ax, (y, ylabel) in zip(axes, panels):
            ax.plot(x, y, "-", color="gray", alpha=0.5, zorder=2)
            ax.scatter(x, y, c=colors, s=80, zorder=3)
            ax.set_ylabel(ylabel)
            ax.grid(True, ls=":", alpha=0.5)
            # Integer tick labels, rounded half up.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{round_half_up(v)}"))
            ax.tick_params(axis="both", which="both", direction="in", right=True)
            if best_iteration is not None:
                ax.axvline(best_iteration, color=AJISAI_PURPLE, ls="--", alpha=0.6, zorder=1)

        if beam_min_rel_span:
            lo, hi = float(np.nanmin(beam_mas)), float(np.nanmax(beam_mas))
            half = max(beam_min_rel_span * float(np.nanmedian(beam_mas)) / 2, 0.6 * (hi - lo))
            axes[3].set_ylim((lo + hi) / 2 - half, (lo + hi) / 2 + half)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=xtick_rotation, ha="center")
        axes[-1].set_xlim(x.min() - 0.5, x.max() + 0.5)

        present = [s for s in STATUS_COLORS if s in set(status)]
        handles = [Patch(color=STATUS_COLORS[s], label=s) for s in present]
        if best_iteration is not None and best_iteration in set(x):
            best_label = labels[list(x).index(best_iteration)].replace("\n", ", ")
            handles.append(Line2D([0], [0], color=AJISAI_PURPLE, ls="--", alpha=0.6,
                                  label=f"best: {best_label}"))
        axes[0].legend(handles=handles, loc="best", fontsize=12)

        fig.suptitle(f"AJISAI self-calibration summary: {projname}", fontsize=18)
        fig.tight_layout()
        if outpath:
            fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Reference-antenna selection
# ---------------------------------------------------------------------------

def draw_refant_selection(
    ax: plt.Axes,
    info: Dict[str, Any],
    cmap: str = "Purples",
    norm_mode: str = "threshold",
    title: Optional[str] = None,
    step: float = 0.25,
):
    """Draw the reference-antenna selection on ``ax``; returns the scatter mappable.

    ``info`` is the ``derived["refant"]`` record of ``justification.json``
    (the return value of :func:`ajisai.select_refant`): antenna names,
    positions, flagged fractions, the flag threshold, the geometric center,
    and the index of the chosen antenna.

    The colour scale is discretised into ``step``-wide bins of flagged
    fraction. With ``norm_mode="threshold"`` the colour map is centred on the
    flag threshold, so antennas on either side of the selection rule get
    clearly different colours; ``norm_mode="linear"`` is a plain 0-1 scale.
    """
    xs = np.asarray(info["antenna_x"], dtype=float)
    ys = np.asarray(info["antenna_y"], dtype=float)
    flags = np.asarray(info["antenna_flag_frac"], dtype=float)
    names = info["antenna_names"]
    cx, cy = info["center_xy"]
    idx = info["chosen_index"]
    threshold = float(info["flag_threshold"])
    dx, dy = xs - cx, ys - cy

    if norm_mode == "threshold" and 0.0 < threshold < 1.0:
        norm = TwoSlopeNorm(vmin=0.0, vcenter=threshold, vmax=1.0)
    elif norm_mode in ("threshold", "linear"):
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        raise ValueError(f"unknown norm_mode {norm_mode!r}; use 'threshold' or 'linear'")
    # Discretise the colour scale into `step`-wide bins, each coloured at its bin centre.
    bounds = np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1)
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    binned_cmap = ListedColormap(plt.get_cmap(cmap)(norm(centers)))
    binned_norm = BoundaryNorm(bounds, binned_cmap.N)

    sc = ax.scatter(dx, dy, c=flags, cmap=binned_cmap, norm=binned_norm, s=120,
                    edgecolors="black", linewidth=0.5, zorder=3)
    excluded = flags >= threshold
    if excluded.any():
        ax.scatter(dx[excluded], dy[excluded], marker="x", s=160, c="grey", linewidth=2.0,
                   zorder=4, label=f"excluded (flag ≥ {threshold:.2f})")
    # Ring around the chosen antenna: black outline under a light-purple ring,
    # drawn behind the points so that the antenna marker stays visible.
    ax.scatter(dx[idx], dy[idx], s=420, facecolors="none", edgecolors="black",
               linewidth=4.5, zorder=0)
    ax.scatter(dx[idx], dy[idx], s=420, facecolors="none", edgecolors=AJISAI_LIGHT_PURPLE,
               linewidth=2.5, zorder=1, label=f"refant = {info['refant']}")
    ax.plot(0, 0, "k+", markersize=18, markeredgewidth=2, zorder=5, label="XY geometric center")
    for x, y, n in zip(dx, dy, names):
        ax.annotate(str(n), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")

    # Square frame centred on the geometric centre, same scale on both axes.
    half = max(1.12 * max(np.abs(dx).max(), np.abs(dy).max()), 1.0)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("X offset from array center [m]")
    ax.set_ylabel("Y offset from array center [m]")
    ax.set_title(title or f"refant = {info['refant']}", fontsize=15)
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9, markerscale=0.5)

    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    cb = ax.figure.colorbar(sc, cax=cax, label="flagged fraction")
    cb.set_ticks(sorted({0.0, threshold, 0.5, 0.75, 1.0}))
    cb.ax.axhline(threshold, color="black", lw=1.5)
    return sc


def plot_refant_selection(
    refant_info: Dict[str, Any],
    outpath: Optional[PathLike] = None,
    title: Optional[str] = None,
    cmap: str = "Purples",
    norm_mode: str = "threshold",
    step: float = 0.25,
    dpi: int = 150,
) -> plt.Figure:
    """Square scatter plot that shows why the reference antenna was chosen.

    - Antennas at their XY positions (meters from the array geometric center),
      coloured by flagged fraction in ``step``-wide bins (``cmap``).
    - Antennas at or above the flag threshold are crossed out in grey.
    - The chosen antenna is ringed in light purple; the geometric center is a
      black ``+``.

    Intended as a justification artifact: one PNG that visually defends the
    refant choice. ``outpath`` saves the figure; the figure is returned and
    the caller is responsible for closing it.
    """
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.tick_params(axis="both", which="both", direction="in", right=True)
        draw_refant_selection(ax, refant_info, cmap=cmap, norm_mode=norm_mode, title=title,
                              step=step)
        if outpath:
            fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Images of every round
# ---------------------------------------------------------------------------

def panels_from_metrics(
    df: pd.DataFrame,
    fits_dir: Optional[PathLike] = None,
    label_style: str = "step",
) -> List[Dict[str, Any]]:
    """One gallery panel per metrics row.

    The FITS image of each round is taken from the ``fits_path`` column. If
    ``fits_dir`` is given, the file is looked up there by name instead, which
    lets an output directory be re-plotted after it has been moved. Rounds
    whose FITS file is missing get ``fits=None`` and are drawn as placeholders.
    """
    panels = []
    for (_, row), label in zip(df.iterrows(), round_labels(df, style=label_style)):
        raw = row.get("fits_path")
        path: Optional[Path] = None
        if isinstance(raw, str) and raw:
            path = Path(fits_dir) / Path(raw).name if fits_dir is not None else Path(raw)
        panels.append({
            "iteration": int(row["iteration"]),
            "label": label,
            "status": row["status"] if isinstance(row.get("status"), str) else "ok",
            "fits": path if path is not None and path.is_file() else None,
            "dynamic_range": (float(row["dynamic_range"])
                              if pd.notna(row.get("dynamic_range")) else None),
            "rms_jy_beam": (float(row["rms_jy_beam"])
                            if pd.notna(row.get("rms_jy_beam")) else None),
        })
    return panels


def panels_from_fits_dir(
    fits_dir: PathLike,
    exclude_beam_factor: float = 5.0,
) -> List[Dict[str, Any]]:
    """Gallery panels from the FITS files alone, without ``metrics.csv``.

    Picks up ``clean_iter0.fits`` and ``clean_sc<N>.fits`` in ``fits_dir``
    (sorted by ``N``) and measures peak and off-source RMS with
    :func:`ajisai.compute_rms`, so that an interrupted run can still be
    plotted. ``exclude_beam_factor`` is the RMS exclusion radius in beams
    (``AJISAIConfig.rms_exclude_beam_factor``).
    """
    from .core import compute_rms, load_fits_image  # local import: core imports this module

    fits_dir = Path(fits_dir)
    found = []
    if (fits_dir / "clean_iter0.fits").is_file():
        found.append((0, fits_dir / "clean_iter0.fits"))
    for p in fits_dir.glob("clean_sc*.fits"):
        m = re.fullmatch(r"clean_sc(\d+)\.fits", p.name)
        if m:
            found.append((int(m.group(1)), p))
    panels = []
    for i, path in sorted(found):
        peak = float(np.nanmax(load_fits_image(str(path))["data2d"]))
        rms = compute_rms(str(path), method="sigma_clip_excl",
                          exclude_factor=exclude_beam_factor)["rms"]
        panels.append({
            "iteration": i,
            "label": "no self-cal" if i == 0 else f"{ordinal(i)} self-cal",
            "status": "ok",
            "fits": path,
            "dynamic_range": peak / rms if rms > 0 else None,
            "rms_jy_beam": rms,
        })
    return panels


def plot_selfcal_images(
    panels: Sequence[Dict[str, Any]],
    fov_radius_arcsec: Optional[float] = None,
    fov_beam_factor: float = 10.0,
    cmap: str = "inferno",
    gamma: float = 1.0,
    gamma2: Optional[float] = 0.3,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    best_iteration: Optional[int] = None,
    center: str = "phase",
    exclusion_radius_arcsec: Optional[float] = None,
    panel_size: float = 2.8,
    projname: str = "AJISAI",
    outpath: Optional[PathLike] = None,
    dpi: int = 150,
) -> plt.Figure:
    """CLEAN image of every round in one row, left to right, on a shared colour scale.

    Parameters
    ----------
    panels
        From :func:`panels_from_metrics` or :func:`panels_from_fits_dir`. The
        number of panels follows the number of rounds, so a run that stopped
        early simply has fewer panels; a panel whose ``fits`` is ``None`` is
        drawn as a grey placeholder.
    fov_radius_arcsec, fov_beam_factor
        Half-width of the field shown. By default ``fov_beam_factor`` times
        the geometric-mean beam of the first available image;
        ``fov_radius_arcsec`` overrides it.
    cmap, gamma, gamma2
        Colour map and power-law stretch (``gamma=1`` is linear; smaller
        values bring up faint emission). With ``gamma2`` a second row with
        that stretch is drawn below the first; ``None`` draws one row only.
        Each row has its own colour bar.
    vmin, vmax
        Colour range in mJy/beam; ``vmax`` defaults to the highest peak in
        the field shown.
    best_iteration
        Its panel is framed and labelled "best".
    center
        ``"phase"`` centres the panels on the phase center (the reference
        pixel), ``"peak"`` on the brightest pixel of the last image.
    exclusion_radius_arcsec
        If given, the off-source exclusion circle used for the RMS is drawn
        as a dashed circle around the phase center.
    outpath
        If given, the figure is saved there (PNG) at ``dpi``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; the caller is responsible for closing it.
    """
    from .core import load_fits_image  # local import: core imports this module

    images = [load_fits_image(str(p["fits"])) if p["fits"] is not None else None for p in panels]
    ref = next((im for im in images if im is not None), None)
    if ref is None:
        raise ValueError("no FITS image found for any round")
    beam = np.sqrt(ref["bmaj_arcsec"] * ref["bmin_arcsec"])
    fov = fov_radius_arcsec or fov_beam_factor * beam
    if center == "phase":
        x0 = y0 = 0.0
    elif center == "peak":
        last = next(im for im in reversed(images) if im is not None)
        iy, ix = np.unravel_index(np.nanargmax(last["data2d"]), last["data2d"].shape)
        x0, y0 = float(last["x_arcsec"][ix]), float(last["y_arcsec"][iy])
    else:
        raise ValueError(f"unknown center {center!r}; use 'phase' or 'peak'")

    cutouts = []
    for im in images:
        if im is None:
            cutouts.append(None)
            continue
        sx = np.abs(im["x_arcsec"] - x0) <= fov + im["cdelt_arcsec"]
        sy = np.abs(im["y_arcsec"] - y0) <= fov + im["cdelt_arcsec"]
        cutouts.append((im["data2d"][np.ix_(sy, sx)] * 1e3,
                        im["x_arcsec"][sx], im["y_arcsec"][sy], im))
    if vmax is None:
        vmax = max(float(np.nanmax(c[0])) for c in cutouts if c is not None)
    gammas = [gamma] if gamma2 is None else [gamma, gamma2]

    with plt.rc_context(PLOT_RC):
        n = len(panels)
        nrows = len(gammas)
        fig, axes_grid = plt.subplots(
            nrows, n, figsize=(panel_size * n + 1.2, panel_size * nrows + 1.1),
            sharex=True, sharey=True, squeeze=False, constrained_layout=True,
        )
        for row, (axes, g) in enumerate(zip(axes_grid, gammas)):
            _draw_selfcal_image_row(
                axes, panels, cutouts, PowerNorm(gamma=g, vmin=vmin, vmax=vmax), g, cmap,
                x0, y0, fov, exclusion_radius_arcsec, best_iteration, show_titles=(row == 0),
            )
        # Shared axes: setting the locators on one panel applies to all of them.
        axes_grid[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5))
        axes_grid[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axes_grid[-1, 0].set_xlabel("Relative RA [arcsec]")
        for axes in axes_grid:
            axes[0].set_ylabel("Relative Dec [arcsec]")
        fig.suptitle(f"AJISAI self-calibration images: {projname}", fontsize=15)
        if outpath:
            fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return fig


def _draw_selfcal_image_row(axes, panels, cutouts, norm, gamma, cmap, x0, y0, fov,
                            exclusion_radius_arcsec, best_iteration, show_titles=True):
    """One gallery row on a shared colour scale, with its colour bar on the right."""
    mappable = None
    for ax, panel, cut in zip(axes, panels, cutouts):
        if show_titles:
            ax.set_title(panel["label"], fontsize=12)
        ax.set_xlim(x0 + fov, x0 - fov)  # east to the left
        ax.set_ylim(y0 - fov, y0 + fov)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", which="both", direction="in", color="white",
                       top=True, right=True)
        if cut is None:
            ax.set_facecolor("0.85")
            ax.text(0.5, 0.5, panel.get("status") or "no image", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            continue
        data, xs, ys, im = cut
        h = im["cdelt_arcsec"] / 2
        mappable = ax.imshow(data, origin="lower", cmap=cmap, norm=norm, interpolation="nearest",
                             extent=[xs[0] + h, xs[-1] - h, ys[0] - h, ys[-1] + h])
        # Beam: BPA is east of north, and x (RA offset) is positive to the east.
        bmaj, bmin = im["bmaj_arcsec"], im["bmin_arcsec"]
        bpa = float(im["header"].get("BPA", 0.0))
        ax.add_patch(Ellipse((x0 + fov - 0.7 * bmaj, y0 - fov + 0.7 * bmaj),
                             width=bmin, height=bmaj, angle=-bpa,
                             facecolor="none", edgecolor="white", lw=1.2))
        if exclusion_radius_arcsec:
            ax.add_patch(Circle((0, 0), exclusion_radius_arcsec, fill=False, ec="white",
                                ls="--", lw=0.8))
        text = []
        if panel.get("dynamic_range") is not None:
            text.append(f"DR = {round_half_up(panel['dynamic_range'])}")
        if panel.get("rms_jy_beam") is not None:
            text.append(f"RMS = {round_half_up(panel['rms_jy_beam'] * 1e6)} μJy/beam")
        ax.text(0.04, 0.96, "\n".join(text), transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=12)
        if best_iteration is not None and panel["iteration"] == best_iteration:
            for spine in ax.spines.values():
                spine.set_edgecolor(AJISAI_LIGHT_PURPLE)
                spine.set_linewidth(3)
            ax.text(0.96, 0.04, "best", transform=ax.transAxes, ha="right", va="bottom",
                    color=AJISAI_LIGHT_PURPLE, fontsize=18, fontweight="bold")
    if mappable is not None:
        # Colour bar attached to the last panel so it spans exactly the image height.
        cax = axes[-1].inset_axes([1.04, 0.0, 0.06, 1.0])
        cb = axes[-1].figure.colorbar(mappable, cax=cax)
        cb.set_label(f"mJy/beam (gamma = {gamma:g})")


# ---------------------------------------------------------------------------
# Re-plot an existing output directory
# ---------------------------------------------------------------------------

def plot_from_workdir(
    workdir: PathLike,
    fits_dir: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    dpi: int = 150,
    **image_options: Any,
) -> Dict[str, Path]:
    """Regenerate the three diagnostic PNGs from an AJISAI output directory.

    Reads ``metrics.csv`` and ``justification.json`` in ``workdir`` and the
    per-iteration FITS images named in ``metrics.csv`` (or looked up by file
    name in ``fits_dir``). No CASA is needed, so this is the way to re-plot a
    finished run with different gallery options, for example::

        from ajisai import plot_from_workdir
        plot_from_workdir("ajisai_myproj", gamma=0.5, fov_beam_factor=15)

    ``image_options`` are passed to :func:`plot_selfcal_images`. The PNGs are
    written to ``out_dir`` (default: ``workdir``). Returns the written paths
    keyed by file name.
    """
    workdir = Path(workdir)
    out_dir = Path(out_dir) if out_dir is not None else workdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(workdir / "metrics.csv")
    with open(workdir / "justification.json") as f:
        justification = json.load(f)
    projname = ((justification.get("input_config") or {}).get("projname")
                or re.sub(r"^ajisai_", "", workdir.resolve().name) or "AJISAI")
    best_iteration = (justification.get("best") or {}).get("iteration")

    written: Dict[str, Path] = {}
    path = out_dir / "selfcal_summary.png"
    plt.close(plot_selfcal_summary(df, best_iteration, projname, outpath=path, dpi=dpi))
    written[path.name] = path

    refant = (justification.get("derived") or {}).get("refant")
    if isinstance(refant, dict) and "antenna_x" in refant:
        path = out_dir / "ajisai_refant_selection.png"
        plt.close(plot_refant_selection(refant, outpath=path,
                                        title=f"AJISAI refant: {projname}", dpi=dpi))
        written[path.name] = path

    panels = panels_from_metrics(df, fits_dir)
    if any(p["fits"] is not None for p in panels):
        path = out_dir / "selfcal_images.png"
        plt.close(plot_selfcal_images(panels, best_iteration=best_iteration, projname=projname,
                                      outpath=path, dpi=dpi, **image_options))
        written[path.name] = path
    return written
