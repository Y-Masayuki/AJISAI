# AJISAI

**Automated Justification-based Imaging and Self-calibration for ALMA Infrastructure**

AJISAI is a fully automated, reproducible, and explainable self-calibration
pipeline for ALMA continuum data, built on top of CASA.

[![CI](https://github.com/Y-Masayuki/AJISAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Y-Masayuki/AJISAI/actions/workflows/ci.yml)
[![Documentation Status](https://app.readthedocs.org/projects/ajisai/badge/?version=latest)](https://ajisai.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Design principles

1. **Fire-and-forget.** Minimum required input is the measurement set path.
   All other parameters have sensible defaults derived from the data itself.
2. **Justification-based.** Every parameter choice (reference antenna,
   cell size, image size, solution intervals, masking strategy, ...) is
   recorded in a structured `justification.json` file with the rationale,
   so the run is auditable and the choices are reproducible in publications.
3. **Deterministic.** A fixed self-calibration schedule (3 phase iterations
   + 1 amplitude iteration by default) runs to completion; the best image is
   selected by dynamic range at the end. No adaptive rollback that would
   make the result depend on run-order.

## Installation

AJISAI requires CASA at runtime (either a monolithic CASA distribution or
modular CASA installed via pip).

For development:
```bash
git clone https://github.com/Y-Masayuki/AJISAI.git
cd AJISAI
pip install -e ".[dev]"
```

Full documentation is hosted on Read the Docs:
**https://ajisai.readthedocs.io**

## Quick start

```python
from ajisai import AJISAI, AJISAIConfig

cfg = AJISAIConfig(vis="/path/to/data.ms")
aj = AJISAI(cfg).run()

print(aj.best_image)         # path to the best CLEAN image
print(aj.best_metric_value)  # the dynamic range achieved
print(aj.metrics)            # per-iteration metrics (DataFrame)
print(aj.justification)      # full structured rationale
```

After the run, the working directory contains:

```
ajisai_<projname>/
  ajisai_refant_selection.png   # which antenna was chosen, and why
  metrics.csv                    # per-iteration peak / RMS / SNR / DR / beam
  selfcal_summary.png            # 4-panel diagnostic plot
  justification.json             # structured rationale for every decision
  final.image / final.fits       # the best CLEAN image
  intermediate/                  # all intermediate MS files and images
  diagnostics/                   # per-iteration gain plots
```

## Multi-field workflow

For measurement sets containing several target fields:

```python
from ajisai import AJISAI, AJISAIConfig, list_target_fields

vis = "/path/to/multi_field.ms"

for f in list_target_fields(vis):
    cfg = AJISAIConfig(
        vis=vis,
        field=f["id"],                # field ID as string
        projname=f"run_{f['name']}",  # one output dir per field
    )
    AJISAI(cfg).run()
```

## Advanced configuration

`AJISAIConfig` exposes sub-configs for imaging, gain calibration, and the
self-cal schedule. All have defaults; override only what you need.

```python
from ajisai import (
    AJISAI, AJISAIConfig, ImagingConfig, GainCalConfig,
    SelfcalSchedule, SelfcalStep,
)

cfg = AJISAIConfig(
    vis="/path/to/data.ms",
    # Optional: shift the source to the phase center first
    phase_shift=True,
    # Optional: supply phase center manually (frame is required)
    phase_center=("16:25:45.0", "-24:12:23.0", "ICRS"),
    # Sub-configs
    imaging=ImagingConfig(robust=0.5, mask_mode="auto-multithresh"),
    gaincal=GainCalConfig(minsnr=1.5, gaintype="T"),
    schedule=SelfcalSchedule(steps=(
        SelfcalStep("p", "inf",  label="phase_inf"),
        SelfcalStep("p", "6*IT", label="phase_6IT"),
        SelfcalStep("p", "3*IT", label="phase_3IT"),
        SelfcalStep("a", "inf",  solnorm=True, label="amp_inf"),
    )),
)
AJISAI(cfg).run()
```

## Tools

The `tools/` directory contains standalone diagnostic utilities:

* `sigma_clip_rms_test.py` — compares five off-source RMS estimators on a
  CASA-exported FITS image to validate the sigma-clipping-based default
  used in AJISAI.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If AJISAI helps your work, please cite [`Yamaguchi et al. 2026`](https://arxiv.org/abs/2605.11486)  and the software itself via the Zenodo DOI (to be assigned at first GitHub release).

## Author

Masayuki Yamaguchi (Kyushu Univ. / NAOJ)

