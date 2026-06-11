#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
examples/run_twhya_demo.py
==========================
End-to-end AJISAI demonstration on the public TW Hya Band 7 dataset
(ALMA Project 2011.0.00340.S).

What this script does (one command, from start to finish):

    1. Downloads ``twhya_calibrated.ms.tar`` from the NRAO casaguides server
       if it is not already present locally.
    2. Extracts the tarball to obtain ``twhya_calibrated.ms``.
    3. Pre-processes for continuum imaging: splits out the science target
       (TW Hya, field='5') and averages 8 channels together. The output
       is ``twhya_continuum.ms``.
    4. Runs AJISAI on the prepared continuum MS with default settings
       (3 phase + 1 amp self-cal iterations, dynamic-range best selection,
       sigma-clip RMS, hybrid refant).
    5. Prints the paths to the resulting image, FITS, summary plot, and
       justification log.

Usage:

    From a CASA prompt (monolithic CASA distribution):
        execfile('examples/run_twhya_demo.py')
        # or
        exec(open('examples/run_twhya_demo.py').read())

    From a Python environment with modular CASA installed:
        python examples/run_twhya_demo.py

    From a shell using CASA's batch mode:
        casa --nogui --nologger -c examples/run_twhya_demo.py

Total runtime is typically 10-30 minutes depending on hardware. The data
download is ~1.7 GB, so the first run requires good network bandwidth.

Reference:
    https://casaguides.nrao.edu/index.php?title=First_Look_at_Imaging_CASA_6
"""
from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
import ssl
try:
    import certifi
    _CERTIFI_AVAILABLE = True
except ImportError:
    _CERTIFI_AVAILABLE = False


# ============================================================================
# User-tweakable configuration
# ============================================================================
DATA_DIR = "./twhya_demo_data"
TAR_URL = (
    "https://bulk.cv.nrao.edu/almadata/public/casaguides/"
    "FirstLook_TWHya_Band7_6.6.1/twhya_calibrated.ms.tar"
)
TAR_NAME = "twhya_calibrated.ms.tar"
RAW_MS_NAME = "twhya_calibrated.ms"
CONT_MS_NAME = "twhya_continuum.ms"
TARGET_FIELD = "5"          # field ID of TW Hya in the calibrated MS
CHANNEL_AVG_WIDTH = 8       # width passed to split() for channel averaging
PROJ_NAME = "TWHya_demo"    # AJISAI projname (also names the output directory)


# ============================================================================
# Make sure the ajisai package can be imported
# ============================================================================
def _ensure_ajisai_on_path():
    """Locate the ajisai package and add it to sys.path if necessary.

    Tries (in order):
        1. ajisai already importable (e.g., installed via `pip install -e .`)
        2. ajisai package directory next to this script's parent directory
           (i.e., running from a fresh git clone without `pip install`)
    """
    try:
        import ajisai  # noqa: F401
        return
    except ImportError:
        pass

    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    repo_root = os.path.dirname(here)
    if os.path.isdir(os.path.join(repo_root, "ajisai")):
        sys.path.insert(0, repo_root)
        return

    raise ImportError(
        "Cannot find the 'ajisai' package. Install it with "
        "'pip install -e .' from the repo root, or run this script "
        "from inside a checkout of the AJISAI repository."
    )


# ============================================================================
# Step 1-2: Download and extract the calibrated MS
# ============================================================================
def _download_with_progress(url: str, dst: str) -> None:
    """Download ``url`` to ``dst`` with a simple percentage progress bar.

    Uses certifi's CA bundle when available. Some monolithic CASA
    distributions ship a Python whose default certificate path points to a
    non-existent build-time location (e.g. /root/local/cert.pem), which makes
    SSL verification fail with CERTIFICATE_VERIFY_FAILED. Pointing explicitly
    at certifi's bundle avoids this.
    """
    if _CERTIFI_AVAILABLE:
        ctx = ssl.create_default_context(cafile=certifi.where())
    else:
        ctx = ssl.create_default_context()

    block_size = 1024 * 256
    with urllib.request.urlopen(url, timeout=60, context=ctx) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dst, "wb") as out:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(100.0, 100.0 * downloaded / total_size)
                    bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
                    mb_done = downloaded / (1024 ** 2)
                    mb_total = total_size / (1024 ** 2)
                    msg = f"\r  [{bar}] {pct:5.1f}%  ({mb_done:7.1f} / {mb_total:.1f} MB)"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
    print()  # newline after the progress bar



# ============================================================================
# Step 3: Pre-process - split TW Hya and channel-average for continuum
# ============================================================================
def split_continuum(raw_ms: str) -> str:
    """Split out the TW Hya field and channel-average; return the new MS path."""
    from casatasks import split, rmtables  # type: ignore

    cont_ms = os.path.join(DATA_DIR, CONT_MS_NAME)
    if os.path.isdir(cont_ms):
        print(f"[twhya_demo] Found existing {cont_ms} - skipping split.")
        return cont_ms

    print(f"[twhya_demo] Splitting field={TARGET_FIELD} from {raw_ms}")
    print(f"[twhya_demo]   width={CHANNEL_AVG_WIDTH}, output={cont_ms}")
    rmtables(cont_ms)
    split(
        vis=raw_ms,
        field=TARGET_FIELD,
        width=CHANNEL_AVG_WIDTH,
        outputvis=cont_ms,
        datacolumn="data",
    )
    if not os.path.isdir(cont_ms):
        raise RuntimeError(f"split() did not produce {cont_ms}")
    return cont_ms


# ============================================================================
# Step 4: Run AJISAI on the prepared continuum MS
# ============================================================================
def run_ajisai(cont_ms: str):
    """Run the AJISAI pipeline with default settings on the prepared MS."""
    from ajisai import AJISAI, AJISAIConfig

    cfg = AJISAIConfig(
        vis=cont_ms,
        projname=PROJ_NAME,
        # All other parameters use their defaults:
        #   phase_shift=False  (TW Hya is already at the phase center)
        #   rms_method='sigma_clip_excl'
        #   refant_strategy='hybrid'
        #   quality_metric='dynamic_range'
        #   schedule = 3 phase iterations + 1 amp iteration
        verbose=True,
    )

    print()
    print("=" * 72)
    print("  Running AJISAI on TW Hya")
    print(f"  Input MS  : {cont_ms}")
    print(f"  Projname  : {PROJ_NAME}")
    print(f"  Schedule  : {[s.label for s in cfg.schedule.steps]}")
    print("=" * 72)

    aj = AJISAI(cfg).run()

    print()
    print("=" * 72)
    print("  AJISAI completed successfully")
    print("=" * 72)
    print(f"  Working directory : {aj.workdir}")
    print(f"  Best image        : {aj.best_image}")
    print(f"  Best FITS         : {aj.best_fits}")
    print(f"  Best dynamic range: {aj.best_metric_value:.1f}"
          if aj.best_metric_value is not None
          else "  Best dynamic range: (n/a)")
    print()
    print(f"  Metrics CSV       : {os.path.join(str(aj.workdir), 'metrics.csv')}")
    print(f"  Summary plot      : {os.path.join(str(aj.workdir), 'selfcal_summary.png')}")
    print(f"  Justification log : {os.path.join(str(aj.workdir), 'justification.json')}")
    print(f"  Refant plot       : {os.path.join(str(aj.workdir), 'ajisai_refant_selection.png')}")
    return aj


# ============================================================================
# Main
# ============================================================================
def main():
    print()
    print("=" * 72)
    print("  AJISAI demonstration: TW Hya Band 7 continuum self-calibration")
    print("  Source: ALMA Project 2011.0.00340.S (FirstLook_TWHya casaguide)")
    print("=" * 72)

    _ensure_ajisai_on_path()
    raw_ms = prepare_data()
    cont_ms = split_continuum(raw_ms)
    aj = run_ajisai(cont_ms)
    return aj


if __name__ == "__main__":
    main()
