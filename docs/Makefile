# Minimal Sphinx Makefile for AJISAI documentation.
#
# Usage:
#     make html        - build HTML docs into _build/html
#     make clean       - remove the _build directory
#     make linkcheck   - validate external links
#     make help        - list all sphinx-build targets
#

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     := .
BUILDDIR      := _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean html livehtml linkcheck Makefile

clean:
	rm -rf "$(BUILDDIR)"

# Catch-all target: route any unrecognized target to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Live-reload server for editing the docs interactively.
# Requires:  pip install sphinx-autobuild
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
