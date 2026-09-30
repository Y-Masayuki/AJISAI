# Contributing to AJISAI

Thank you for your interest in AJISAI. This document explains how to set up a
development environment, run the checks, and submit changes. Please also read
the [design policy](#design-policy) below before proposing new features.

## Reporting bugs and asking questions

Please use [GitHub Issues](https://github.com/Y-Masayuki/AJISAI/issues). For
bug reports, use the bug report template and attach the `justification.json`
from the affected run; it lets us reproduce the pipeline's decisions without
access to your data.

## Development setup

```bash
git clone https://github.com/Y-Masayuki/AJISAI.git
cd AJISAI
pip install -e ".[dev]"
```

The `dev` extra installs `pytest`, `pytest-cov`, and `ruff`. Python 3.8 or
newer is required by the package; CI runs on Python 3.10, 3.11, and 3.12.

### Is CASA required?

- **For development and the unit tests: no.** The test suite is CASA-free.
  Tests marked `@pytest.mark.casa` are skipped automatically when `casatools`
  and `casatasks` are not importable, which is how CI runs them.
- **For running the pipeline on real data: yes.** AJISAI needs CASA at
  runtime, either a monolithic CASA distribution or modular CASA
  (`pip install -e ".[casa]"`, which installs `casatools` and `casatasks`).
  If your change touches code that calls CASA tasks, please run the relevant
  part of the pipeline in a CASA environment (for example the TW Hya demo in
  `examples/run_twhya_demo.py`) and mention the CASA version in your pull
  request.

## Running the tests

```bash
pytest
```

Useful variations:

```bash
pytest tests/test_<module>.py     # a single file
pytest -m "not slow"              # skip slow tests
pytest --cov=ajisai               # with coverage
```

Available markers are `casa`, `slow`, and `integration` (see `pyproject.toml`).
New behavior should come with tests, and bug fixes should come with a
regression test where practical.

## Linting

We use [ruff](https://docs.astral.sh/ruff/); the configuration lives in
`pyproject.toml` (line length 100, pyflakes rules, target Python 3.8).

```bash
ruff check .
```

Please make sure this passes before opening a pull request. The `legacy/` and
`analysis_scripts/` directories are excluded from linting.

## Branches and pull requests

1. Fork the repository and create a topic branch from `main`
   (for example `fix/phaseshift-dec-parsing` or `feat/short-description`).
2. Make focused changes; keep unrelated refactoring out of the same PR.
3. Run `pytest` and `ruff check .` locally.
4. Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change,
   and update the documentation in `docs/` if behavior or options change.
5. Open a pull request against `main` and fill in the PR template.
   CI (pytest and ruff) must pass before review.

## Commit messages

Follow the style of the existing history (`git log --oneline`): a short,
imperative summary line, optionally prefixed with a
[Conventional Commits](https://www.conventionalcommits.org/)-style type and
scope. Examples from the log:

```
feat(selfcal): also emit channel-averaged selfcal_<idx>_avg.ms (1 ch/spw)
fix(phaseshift): parse CASA dec format in icrs_to_j2000
docs: incorporate TW Hya demo results into README, tutorial, CHANGELOG
fix(lint): clean up ruff F401/F541/F841 issues
```

Common types are `feat`, `fix`, `docs`, `test`, and `chore`. Put the detail
and rationale in the commit body when the summary is not enough.

## Design policy

AJISAI's first principles are a **deterministic, fixed self-calibration
schedule** and **explainability through `justification.json`**. Every
parameter choice is recorded with its rationale, and the same input must
produce the same sequence of steps, independent of run order.

For this reason, **we do not accept proposals that add adaptive rollback or
heuristic decision-making**, such as the adaptive behavior of `auto_selfcal`
(for example, reverting an iteration or changing the schedule based on
intermediate results). Pull requests and feature requests of this kind will be
closed.

Contributions that fit the design are welcome, for example:

- Bug fixes and robustness improvements.
- Better data-derived defaults, provided the derivation is deterministic and
  recorded in `justification.json`.
- Additional diagnostics, documentation, tests, and tutorials.
- Support for additional data layouts that the fixed schedule can handle.

AI-assisted contributions are welcome, provided the contributor has reviewed
and tested the code and discloses the assistance in the PR description. The
contributor remains responsible for the correctness and design fit of every
change submitted.

If you are unsure whether an idea fits, please open an issue to discuss it
before investing time in an implementation.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers this project (see [LICENSE](LICENSE)).
