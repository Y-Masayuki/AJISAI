"""
Shared pytest fixtures and configuration for the AJISAI test suite.

Test layering
-------------
* CASA-free tests (the majority): import only the parts of ``ajisai`` that
  do not need ``casatools``/``casatasks``. These run anywhere with
  numpy + astropy + matplotlib.
* CASA-required tests: marked with ``@pytest.mark.casa``. These need a
  working CASA environment (either monolithic or modular) and a real MS.
  They are skipped automatically when CASA is not importable.
* End-to-end tests (Phase 1.8): full ``AJISAI(cfg).run()`` invocations
  on a tiny test MS. Not yet implemented.

Fixtures here are CASA-free helpers used by the unit tests.
"""
from __future__ import annotations


import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture(scope="session")
def synthetic_fits(tmp_path_factory):
    """A small synthetic CASA-style FITS image.

    Resembles an ALMA continuum image with:
      - 512 x 512 pixels at 0.012 arcsec/pix
      - A single Gaussian source at the center, peak ~5 mJy/beam
      - Gaussian noise with sigma = 10 uJy/beam
      - BMAJ = 0.12 arcsec, BMIN = 0.10 arcsec

    Used by RMS tests, FITS loader tests, etc.
    """
    rng = np.random.default_rng(seed=42)
    nx, ny = 512, 512
    sigma_noise = 1e-5  # 10 uJy/beam
    data = rng.normal(0.0, sigma_noise, (1, 1, ny, nx))

    # Add a Gaussian source at the center
    cx, cy = nx // 2, ny // 2
    yy, xx = np.mgrid[:ny, :nx]
    r2 = ((xx - cx) / 15.0) ** 2 + ((yy - cy) / 12.0) ** 2
    source = 5e-3 * np.exp(-r2 / 2.0)  # 5 mJy/beam peak
    data[0, 0] += source

    hdr = fits.Header()
    hdr["NAXIS"] = 4
    hdr["NAXIS1"], hdr["NAXIS2"], hdr["NAXIS3"], hdr["NAXIS4"] = nx, ny, 1, 1
    hdr["CDELT1"] = -0.012 / 3600.0      # 0.012 arcsec/pix
    hdr["CDELT2"] = 0.012 / 3600.0
    hdr["CRPIX1"] = nx // 2 + 1
    hdr["CRPIX2"] = ny // 2 + 1
    hdr["BMAJ"] = 0.12 / 3600.0          # 0.12 arcsec
    hdr["BMIN"] = 0.10 / 3600.0          # 0.10 arcsec
    hdr["BPA"] = 0.0
    hdr["BUNIT"] = "Jy/beam"

    out = tmp_path_factory.mktemp("synthetic_fits") / "synth.fits"
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(out, overwrite=True)
    return str(out)


@pytest.fixture(scope="session")
def synthetic_fits_no_source(tmp_path_factory):
    """Same shape as ``synthetic_fits`` but pure-noise (no source).

    Useful for testing RMS estimators in the limit where all pixels are noise.
    """
    rng = np.random.default_rng(seed=43)
    nx, ny = 512, 512
    sigma_noise = 1e-5
    data = rng.normal(0.0, sigma_noise, (1, 1, ny, nx))

    hdr = fits.Header()
    hdr["NAXIS"] = 4
    hdr["NAXIS1"], hdr["NAXIS2"], hdr["NAXIS3"], hdr["NAXIS4"] = nx, ny, 1, 1
    hdr["CDELT1"] = -0.012 / 3600.0
    hdr["CDELT2"] = 0.012 / 3600.0
    hdr["CRPIX1"] = nx // 2 + 1
    hdr["CRPIX2"] = ny // 2 + 1
    hdr["BMAJ"] = 0.12 / 3600.0
    hdr["BMIN"] = 0.10 / 3600.0
    hdr["BPA"] = 0.0
    hdr["BUNIT"] = "Jy/beam"

    out = tmp_path_factory.mktemp("synthetic_fits_no_source") / "noise.fits"
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(out, overwrite=True)
    return str(out)


def has_casa() -> bool:
    """Return True if casatools/casatasks are importable."""
    try:
        import casatools  # noqa: F401
        import casatasks  # noqa: F401
        return True
    except ImportError:
        return False


# Auto-applied marker: pytest.mark.casa tests are skipped when CASA is absent.
def pytest_collection_modifyitems(config, items):
    skip_casa = pytest.mark.skip(reason="CASA not available in this environment")
    if has_casa():
        return
    for item in items:
        if "casa" in item.keywords:
            item.add_marker(skip_casa)
