# Installation

AJISAI requires CASA at runtime. The current install method is **from the
GitHub source**; a PyPI release is planned for a future version (see the
section at the bottom of this page).

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

## Install from GitHub (current method)

Install the latest released tag directly with `pip`, which fetches the
package from the GitHub repository:

```bash
pip install "git+https://github.com/Y-Masayuki/AJISAI.git@v0.1.0"
```

Or to track the `main` branch (latest development snapshot):

```bash
pip install "git+https://github.com/Y-Masayuki/AJISAI.git@main"
```

If you are using a monolithic CASA distribution, you may need to point at
its internal `pip`:

```bash
/path/to/casa/lib/py/bin/pip install "git+https://github.com/Y-Masayuki/AJISAI.git@v0.1.0"
```

After installation, verify that AJISAI imports cleanly from inside CASA:

```python
# from inside CASA's Python prompt
import ajisai
print(ajisai.__version__)
```

## Development install (editable, with tests and docs)

For contributors or users who want to modify AJISAI:

```bash
git clone https://github.com/Y-Masayuki/AJISAI.git
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

## Coming soon: PyPI release

`pip install ajisai` and `pip install "ajisai[casa]"` are **not yet supported**
because AJISAI has not been uploaded to PyPI. A PyPI release is planned for
a future version once the end-to-end behaviour has been validated against
real ALMA data. Until then, please use the GitHub install method above.

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
