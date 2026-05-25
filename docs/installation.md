# Installation

AJISAI requires CASA at runtime. There are two supported installation paths
depending on whether you already have a CASA distribution.

## Requirements

- **Python 3.8 or newer**
- **CASA 6.5 or newer.** Either monolithic CASA (the standard distribution
  from NRAO) or modular CASA (`casatools` and `casatasks` installed via pip).
- Python dependencies: `numpy`, `pandas`, `matplotlib`, `astropy`.
  These are typically already present in any CASA-compatible Python
  environment, and `pip` will install them automatically if not.

:::{note}
**CASA viewer limitation.** AJISAI's `mask_mode="interactive"` option uses
the CASA viewer GUI to let you draw a mask interactively. The viewer was
removed in **CASA 6.7+**, so `mask_mode="interactive"` only works on
**CASA <= 6.6**. Users on CASA 6.7+ should use the default
`mask_mode="auto-multithresh"` or provide a pre-made mask file via
`mask_mode="user"`. See [Configuration reference](configuration.md) for
details.
:::

## Option A: Install into an existing CASA Python environment

If you already have CASA (monolithic or modular), install AJISAI directly:

```bash
pip install ajisai
```

If you are using a monolithic CASA distribution, you may need to point at
its internal `pip`:

```bash
/path/to/casa/lib/py/bin/pip install ajisai
```

After installation, verify that AJISAI imports cleanly from inside CASA:

```python
# from inside CASA's Python prompt
import ajisai
print(ajisai.__version__)
```

## Option B: Self-contained install with modular CASA

If you do not have CASA, you can install both AJISAI and modular CASA in
one go:

```bash
pip install "ajisai[casa]"
```

This pulls in `casatools` and `casatasks`. Note that modular CASA has
specific Python version constraints; see the
[CASA documentation](https://casadocs.readthedocs.io/) for the current
compatibility matrix.

## Option C: Development install from source

For contributors or users who want to track the latest changes:

```bash
git clone https://github.com/masayuki-yamaguchi/AJISAI.git
cd AJISAI
pip install -e ".[dev]"
```

This installs AJISAI in editable mode along with development tools (`pytest`,
`ruff`). To also build the documentation locally:

```bash
pip install -e ".[dev,docs]"
cd docs
make html
open _build/html/index.html      # or your preferred browser command
```

## Verifying the installation

Run the math-only self-check (no CASA required):

```python
from ajisai.ms_utils import _self_check_math
_self_check_math()
# Expected output: "ALL PASS"
```

If you have CASA available and a target MS file, you can additionally run
the end-to-end demo on the public TW Hya dataset:

```bash
python examples/run_twhya_demo.py
```

See the [TW Hya tutorial](tutorials/twhya.md) for a full walkthrough.

## Troubleshooting

**`ModuleNotFoundError: No module named 'casatools'`**
:   AJISAI is being imported outside a CASA-compatible Python environment.
    Install modular CASA (`pip install casatools casatasks`) or run AJISAI
    from within monolithic CASA.

**Warning: `mask_mode='interactive' requires CASA <= 6.6`**
:   You are on CASA 6.7 or newer where the viewer was removed.
    Switch `mask_mode` to `"auto-multithresh"` (default) or `"user"`.

**Warning: `actual dirty-image beam differs from predicted ... by N%`**
:   When `uvtaper` is set, AJISAI's predicted effective beam is based on an
    empirical approximation. Mismatches up to 20-30% are normal for strong
    tapers. For tighter pixel sampling, override `ImagingConfig.cellpix` or
    set `cell` manually. See [Configuration reference](configuration.md).
