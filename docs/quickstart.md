# Quickstart

This page shows the minimum AJISAI usage pattern. For a complete worked
example with public ALMA data, see the [TW Hya tutorial](tutorials/twhya.md).

## Minimal usage

The smallest possible AJISAI invocation:

```python
from ajisai import AJISAI, AJISAIConfig

cfg = AJISAIConfig(vis="/path/to/data.ms")
aj  = AJISAI(cfg).run()
```

That's it. AJISAI takes care of everything else:

- Selects a reference antenna using flag statistics + geometric center.
- Computes the appropriate cell size and image size from the baseline
  distribution.
- Runs an initial CLEAN to set the noise threshold.
- Iterates three rounds of phase self-cal then one round of amplitude
  self-cal (the deterministic schedule from the AJISAI design paper).
- Picks the best image by dynamic range and copies it to
  `<workdir>/final.image` and `<workdir>/final.fits`.

## What you get back

The returned `AJISAI` object exposes everything that was decided and
measured:

```python
print(aj.workdir)               # output directory (./ajisai_<projname>/)
print(aj.best_image)            # path to the best CLEAN image
print(aj.best_fits)             # path to the best FITS
print(aj.best_metric_value)     # achieved dynamic range
print(aj.metrics)               # list of per-iteration dicts
print(aj.justification)         # full structured rationale
```

The same information is also written to disk:

- `<workdir>/metrics.csv` — one row per iteration
- `<workdir>/selfcal_summary.png` — 4-panel diagnostic plot
- `<workdir>/justification.json` — every parameter choice with its reason
- `<workdir>/ajisai_refant_selection.png` — refant choice visualization
- `<workdir>/final.{image,fits}` — the best image

See [Output artifacts](outputs.md) for full details on each file.

## Required vs optional inputs

Only **one** parameter is required: `vis`. Everything else is auto-derived
from the data, but you can override any default:

```python
from ajisai import AJISAI, AJISAIConfig, ImagingConfig, GainCalConfig

cfg = AJISAIConfig(
    vis      = "/path/to/data.ms",
    projname = "my_target",          # default: derived from vis basename
    field    = "0",                  # default: auto-detect single field
    imaging  = ImagingConfig(
        robust  = 0.5,
        cellpix = 10,                # pixels per beam
        uvtaper = ("1arcsec",),      # optional taper
    ),
    gaincal  = GainCalConfig(
        minsnr      = 1.5,
        minblperant = 4,
    ),
)
AJISAI(cfg).run()
```

See [Configuration reference](configuration.md) for every option.

## Multi-field measurement sets

If your MS contains several target fields, loop over them:

```python
from ajisai import AJISAI, AJISAIConfig, list_target_fields

vis = "/path/to/multi_field.ms"

for f in list_target_fields(vis):
    cfg = AJISAIConfig(
        vis      = vis,
        field    = f["id"],                # the FIELD_ID as a string
        projname = f"run_{f['name']}",     # one output dir per field
    )
    AJISAI(cfg).run()
```

Each field gets its own output directory, so results are never mixed.

## Running from inside CASA

AJISAI is normally invoked from a Python script that you run with CASA's
Python. From a CASA prompt:

```python
exec(open("/path/to/my_script.py").read())
```

Or in batch mode from the shell:

```bash
casa --nogui --nologger -c /path/to/my_script.py
```

If you have modular CASA installed (`casatools` and `casatasks` via pip),
you can also run the script with vanilla Python:

```bash
python /path/to/my_script.py
```

## Where to go next

- Walk through a complete example with public ALMA data:
  [TW Hya tutorial](tutorials/twhya.md).
- Understand all the configurable parameters:
  [Configuration reference](configuration.md).
- Understand what every output file contains:
  [Output artifacts](outputs.md).
- Read the design philosophy and how AJISAI differs from `auto_selfcal`:
  [Design](design.md).
