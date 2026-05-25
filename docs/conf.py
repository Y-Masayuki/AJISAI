"""
Sphinx configuration for the AJISAI documentation.

Build locally with:
    cd docs && make html
    open _build/html/index.html
"""
from __future__ import annotations

import os
import sys

# Add the repository root to sys.path so that ``import ajisai`` works during
# autodoc generation, regardless of whether the package has been pip-installed.
sys.path.insert(0, os.path.abspath(".."))

import ajisai  # noqa: E402  (placed after sys.path mutation)


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "AJISAI"
author = "Masayuki Yamaguchi and the AJISAI development team"
copyright = "2026, Masayuki Yamaguchi"
release = ajisai.__version__
version = ".".join(release.split(".")[:2])  # short X.Y version


# ---------------------------------------------------------------------------
# General Sphinx configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",         # pull docstrings into the API reference
    "sphinx.ext.autosummary",     # tables of module/class contents
    "sphinx.ext.napoleon",        # parse NumPy/Google-style docstrings
    "sphinx.ext.intersphinx",     # cross-link to numpy/casa/astropy docs
    "sphinx.ext.viewcode",        # source code links in API pages
    "sphinx.ext.mathjax",         # render LaTeX math
    "myst_parser",                # Markdown sources via MyST-Parser
]

# Files Sphinx will pick up as documents.
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

master_doc = "index"
language = "en"

# Build settings
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]


# ---------------------------------------------------------------------------
# Autodoc / autosummary configuration
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",   # follow source code order, not alphabetical
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_typehints = "description"  # render type hints in the description block
autosummary_generate = True
autoclass_content = "both"         # combine __init__ docstring with class doc


# ---------------------------------------------------------------------------
# Napoleon (NumPy-style docstring parsing)
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True


# ---------------------------------------------------------------------------
# MyST-Parser (Markdown) options
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",      # ::: code blocks (alternative to triple-backtick)
    "deflist",          # definition lists
    "fieldlist",        # field lists
    "substitution",     # variable substitution
    "tasklist",         # GitHub-style task lists
]
# 'linkify' (auto-convert URLs to links) requires the linkify-it-py package.
# Enable only if that package is importable, so docs build cleanly on minimal
# environments.
try:
    import linkify_it  # noqa: F401
    myst_enable_extensions.append("linkify")
except ImportError:
    pass
myst_heading_anchors = 3


# ---------------------------------------------------------------------------
# Intersphinx mapping
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3/",                       None),
    "numpy":      ("https://numpy.org/doc/stable/",                    None),
    "pandas":     ("https://pandas.pydata.org/docs/",                  None),
    "astropy":    ("https://docs.astropy.org/en/stable/",              None),
    "matplotlib": ("https://matplotlib.org/stable/",                   None),
}


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"AJISAI {release}"
html_short_title = "AJISAI"
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "titles_only": False,
    "logo_only": False,
}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# Show "Edit on GitHub" link
html_context = {
    "display_github": True,
    "github_user": "Y-Masayuki",
    "github_repo": "AJISAI",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Suppress warnings for missing references in CASA-only code paths
nitpicky = False
suppress_warnings = ["autodoc.import_object"]
