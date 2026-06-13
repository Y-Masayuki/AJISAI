# Tutorial: TW Hya Band 7 continuum

This tutorial walks through a complete AJISAI self-calibration run on the
public ALMA dataset of the protoplanetary disk TW Hya, observed at Band 7
(~372 GHz) with the ALMA 12 m array under
[Project 2011.0.00340.S](https://almascience.eso.org/aq/?result_view=observation).

By the end of this tutorial you will have:

- Downloaded and pre-processed the calibrated MS,
- Run AJISAI end-to-end with default settings,
- Located the final image and the justification log,
- Understood what the per-iteration metrics tell you.

## Prerequisites

- AJISAI installed (see [Installation](../installation.md)).
- ~5 GB of free disk space (the raw MS tarball is ~1.7 GB; intermediate
  AJISAI products consume another 1-2 GB).
- A working CASA environment.

## The one-command demo

The repository ships a self-contained demonstration script,
`examples/run_twhya_demo.py`, that does the download, the pre-processing,
and the AJISAI run in a single command. From the repository root:

```bash
# Inside CASA's Python or with modular CASA installed:
python examples/run_twhya_demo.py

# Or from CASA's interactive prompt:
exec(open("examples/run_twhya_demo.py").read())
```

The script will:

1. Download `twhya_calibrated.ms.tar` from the NRAO casaguides server
   (~1.7 GB; cached locally on subsequent runs).
2. Extract `twhya_calibrated.ms`.
3. Run `split(vis=..., field='5', width=8, datacolumn='data')` to extract
   the TW Hya field and channel-average for continuum imaging. Output:
   `twhya_continuum.ms`.
4. Run AJISAI with default settings.
5. Print the path to the final image and the achieved dynamic range.

Expected total runtime: 10-30 minutes depending on hardware.

## Step-by-step explanation

If you want to understand what the demo script is doing under the hood,
the equivalent manual workflow is below.

### Step 1: Download the data

The calibrated MS is hosted by NRAO at:

```
https://bulk.cv.nrao.edu/almadata/public/casaguides/FirstLook_TWHya_Band7_6.6.1/twhya_calibrated.ms.tar
```

Download and extract:

```bash
wget https://bulk.cv.nrao.edu/almadata/public/casaguides/FirstLook_TWHya_Band7_6.6.1/twhya_calibrated.ms.tar
tar xvf twhya_calibrated.ms.tar
```

This produces the directory `twhya_calibrated.ms`.

### Step 2: Pre-process for continuum

`twhya_calibrated.ms` contains calibrator fields in addition to the science
target, and the full spectral coverage. AJISAI expects a continuum-ready
MS with the target only. From a CASA-aware Python:

```python
from casatasks import split, rmtables

rmtables("twhya_continuum.ms")
split(
    vis        = "twhya_calibrated.ms",
    field      = "5",                  # TW Hya
    width      = 8,                    # average every 8 channels
    outputvis  = "twhya_continuum.ms",
    datacolumn = "data",
)
```

The resulting `twhya_continuum.ms` is a single-field, channel-averaged
measurement set ready for AJISAI.

### Step 3: Run AJISAI

The simplest invocation uses every default:

```python
from ajisai import AJISAI, AJISAIConfig

cfg = AJISAIConfig(
    vis      = "twhya_continuum.ms",
    projname = "TWHya_demo",
)
aj = AJISAI(cfg).run()

print(f"Best image      : {aj.best_image}")
print(f"Best DR         : {aj.best_metric_value:.1f}")
print(f"Working directory: {aj.workdir}")
```

While AJISAI runs, you will see a banner, then progress messages. The
reference run produced the following (exact numbers may vary slightly with
CASA version and the random initialization of the deconvolver, but the
magnitudes match closely):

```
[AJISAI] refant = DA42 (hybrid: nearest to XY geometric center among
                       19 antennas with flag_fraction < 0.25)
[AJISAI] refant selection plot: ajisai_TWHya_demo/ajisai_refant_selection.png
[AJISAI] median frequency       : 372.650 GHz
[AJISAI] integration time       : 6.048 s
[AJISAI] on-source time         : 30.64 min
[AJISAI] cell                   : 0.066 arcsec
[AJISAI] imsize                 : [375, 375]
[AJISAI] MRS                    : 4.51 arcsec
[AJISAI] iter 0 (no selfcal)    : DR= 81.06, peak=333.59 mJy/bm, RMS=4115.18 uJy/bm
[AJISAI] --- iter 1 (phase_inf) calmode=p solint=inf ---
[AJISAI]   -> ok. DR=176.15, peak=341.47 mJy/bm, RMS=1938.51 uJy/bm  (19 solutions)
[AJISAI] --- iter 2 (phase_6IT) calmode=p solint=36.288s ---
[AJISAI]   -> ok. DR=235.13, peak=351.94 mJy/bm, RMS=1496.76 uJy/bm  (1016 solutions)
[AJISAI] --- iter 3 (phase_3IT) calmode=p solint=18.144s ---
[AJISAI]   -> ok. DR=240.81, peak=354.83 mJy/bm, RMS=1473.48 uJy/bm  (1926 solutions)
[AJISAI] --- iter 4 (amp_inf) calmode=a solint=inf ---
[AJISAI]   -> ok. DR=246.49, peak=355.91 mJy/bm, RMS=1443.90 uJy/bm  (19 solutions)
[AJISAI] best iteration: #4, dynamic_range = 246.49
```

### Step 4: Inspect the results

When AJISAI finishes, the working directory contains:

```
ajisai_TWHya_demo/
├── ajisai_refant_selection.png    ← which antenna AJISAI chose
├── metrics.csv                     ← one row per iteration
├── selfcal_summary.png             ← 4-panel diagnostic plot
├── justification.json              ← every parameter choice + reason
├── final.image  / final.fits       ← the best image AJISAI produced
├── intermediate/                   ← all intermediate MS and image products
└── diagnostics/                    ← per-iteration phase/amp gain plots
```

Open `selfcal_summary.png` first: it shows how dynamic range, peak, RMS,
and beam evolved across iterations. You should see DR climbing through the
three phase iterations and a further bump from the amplitude iteration.

The reference run produced the following result on the TW Hya Band 7 demo:

```{image} ../_static/selfcal_summary.png
:alt: TW Hya self-cal summary
:width: 600px
:align: center
```

Full per-iteration metrics:

| iter | label       | calmode | solint   | peak [mJy/bm] | RMS [uJy/bm] | DR       | gain solutions |
| ---- | ----------- | ------- | -------- | ------------- | ------------ | -------- | -------------- |
| 0    | no_selfcal  | -       | -        | 333.59        | 4115.18      |  81.06   | -              |
| 1    | phase_inf   | p       | inf      | 341.47        | 1938.51      | 176.15   | 19             |
| 2    | phase_6IT   | p       | 36.288 s | 351.94        | 1496.76      | 235.13   | 1016           |
| 3    | phase_3IT   | p       | 18.144 s | 354.83        | 1473.48      | 240.81   | 1926           |
| 4    | amp_inf     | a       | inf      | 355.91        | 1443.90      | 246.49   | 19             |

Overall improvement (iter 0 -> iter 4):

| metric                  | change                          |
| ----------------------- | ------------------------------- |
| Dynamic range           | 81.06 -> 246.49 (**3.04x up**)  |
| Peak [mJy/beam]         | 333.59 -> 355.91 (+22.32, +6.7%) |
| RMS  [uJy/beam]         | 4115.18 -> 1443.90 (**2.85x down**) |
| Beam [mas, geom. mean]  | 482.8 -> 480.1 (essentially unchanged) |

All four iterations completed with `status="ok"` (no anomalies). Note how
the amplitude iteration (iter 4) provided a final 2.4% bump in DR after the
phase iterations had nearly plateaued at iter 3 -- this is the typical ALMA
pattern AJISAI's "all iterations complete + best by DR" design is built to
capture.

The reference-antenna selection picked `DA42`, the antenna closest to the
XY geometric center of the array among the 19 antennas (out of 26) that
passed the flag-fraction threshold (<25%). The seven excluded antennas
are marked with red `x`:

```{image} ../_static/ajisai_refant_selection.png
:alt: TW Hya reference antenna selection
:width: 600px
:align: center
```

The exact output files from this run are committed to the repository for
reference, so you can compare your own run byte-by-byte:

- `docs/_static/metrics.csv` -- per-iteration metrics in CSV form
- `docs/_static/justification.json` -- full structured rationale

If you want to know *why* AJISAI made each choice, read
`justification.json`. An abbreviated example (real output is fully valid
JSON; ellipses below are placeholders for omitted fields):

```text
{
  "ajisai_version": "0.1.0",
  "derived": {
    "refant": {
      "value": "DA42",
      "strategy": "hybrid",
      "flag_threshold": 0.25,
      "fallback_used": false,
      "reason": "hybrid: nearest to XY geometric center among 19 antennas
                 with flag_fraction < 0.25",
      ...
    },
    "avg_int_time_sec": {"value":  6.048, ...},
    "onsource_time_sec": {"value": 1838.59, ...},
    "median_freq_hz":    {"value": 3.7265e11, ...},
    "mrs_arcsec":        {"value":  4.51, ...},
    "cellsize":          {"value_arcsec": 0.066, ...},
    "imsize":            {"value": [375, 375], ...},
    "phase_shift": {
      "applied": false,
      "reason": "phase_shift=False (default; no correction applied)"
    },
    ...
  },
  "iterations": [
    {"iteration": 0, "label": "no_selfcal", "dynamic_range":  81.06, ...},
    {"iteration": 1, "label": "phase_inf",  "dynamic_range": 176.15, "status": "ok", ...},
    {"iteration": 2, "label": "phase_6IT",  "dynamic_range": 235.13, "status": "ok", ...},
    {"iteration": 3, "label": "phase_3IT",  "dynamic_range": 240.81, "status": "ok", ...},
    {"iteration": 4, "label": "amp_inf",    "dynamic_range": 246.49, "status": "ok", ...}
  ],
  "best": {"iteration": 4, "metric_key": "dynamic_range", "value": 246.49}
}
```

This file is intended to be cited in publications: it records exactly what
AJISAI did to your data, so the reduction is reproducible by anyone.

### Step 5: Compare images

In CASA, you can compare the pre- and post-selfcal images:

```python
from casatasks import imview

# Before self-cal
imview("ajisai_TWHya_demo/intermediate/clean_iter0.image")

# After self-cal (best iteration)
imview("ajisai_TWHya_demo/final.image")
```

For TW Hya, you should see a dramatic improvement in dynamic range and a
cleaner background (much less side-lobe structure).

## What to do if things look wrong

**The summary plot shows DR decreasing late in the run.** AJISAI always
runs all iterations to completion and picks the best by DR at the end, so
this is fine. The `final.image` will be from the iteration that achieved
peak DR, not the last iteration. The decreasing later iterations are
recorded as `status="anomaly"` in `justification.json` for inspection.

**Gain calibration produced zero solutions in one iteration.** That
iteration is recorded as `status="skipped"`. The next iteration tries
again from the previous good MS state. If this happens on every iteration,
your data may be too faint or too flagged for self-cal at the requested
solution intervals; try lengthening solints or relaxing `minsnr`.

**Actual dirty-image beam differs from predicted by more than 20%.**
This warning appears when AJISAI's predicted effective beam (especially
under uvtaper) is significantly off from what CASA actually produces. The
cell size is based on the prediction, so it might be too coarse or fine.
Override `ImagingConfig.cellpix` (more pixels per beam) or specify cell
and imsize explicitly to fix.

## Next steps

- Try AJISAI on your own data.
- Explore the [Configuration reference](../configuration.md) to customize
  the pipeline.
- Read the [Design](../design.md) page to understand AJISAI's choices.
