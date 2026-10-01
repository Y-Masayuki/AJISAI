---
title: 'AJISAI: An auditable, deterministic self-calibration pipeline for ALMA continuum data'
tags:
  - Python
  - astronomy
  - radio astronomy
  - interferometry
  - ALMA
  - self-calibration
  - CASA
authors:
  - name: Masayuki Yamaguchi
    orcid: 0000-0002-8185-9882
    affiliation: "1, 2"
affiliations:
  - name: Department of Physics, Kyushu University, Japan
    index: 1
  - name: National Astronomical Observatory of Japan (NAOJ), Japan
    index: 2
date: 6 July 2026
bibliography: paper.bib
---

# Summary

Radio telescopes such as the Atacama Large Millimeter/submillimeter Array
[ALMA, @wootten2009] make images by combining the signals recorded by many
separate antennas. The atmosphere and the instruments themselves alter the
timing and strength of the signal reaching each antenna, which blurs the image
and adds spurious features. Self-calibration corrects these errors by using the
image itself as a reference: an initial image is used to estimate how each
antenna's signal was distorted, the data are corrected, a sharper image is
made, and the cycle is repeated. The technique is standard, but every run
involves many choices, such as which antenna serves as the reference, how often
the corrections are updated, and how large the image should be. These choices
are usually made by hand in custom scripts and rarely reported in full, so it
is often difficult to reproduce exactly how a published image was made.

`AJISAI` (*Automated Justification-based Imaging and Self-calibration for ALMA
Infrastructure*) is a Python package that self-calibrates ALMA continuum
(broadband) data. It relies on CASA [@casa2022], the standard software for
processing ALMA data, for the underlying calculations, and its only required
input is the location of the data. `AJISAI` chooses its settings from measured
properties of the data and always runs the same sequence of correction rounds,
regardless of how the intermediate images turn out; by default, three rounds
correct the timing of the signals and a fourth corrects their strength. It then
selects, from all rounds, the image with the highest dynamic range, the ratio
of the source's peak brightness to the background noise. Its distinguishing
feature is that each choice is written, together with the reason and the
numbers behind it, to a single machine-readable file, `justification.json`.
Publishing this file alongside a result lets other researchers check, and
reproduce, how the image was made.

# Statement of need

`AJISAI` is intended for ALMA users who need a self-calibration procedure that
can be documented, repeated, and independently checked. It addresses three
needs.

First, **auditability**. Self-calibration choices are usually recorded, if at
all, in hand-written scripts or verbose CASA logs, so reconstructing *why* a
particular reference antenna, solution interval, or masking choice was made for
a published image is laborious. `AJISAI` emits a single structured
`justification.json` capturing every parameter and the quantitative reason it
was chosen. A reader with the same measurement set and that file can trace and
reproduce the entire chain of decisions, which is valuable for the
reproducibility standards increasingly expected of published reductions.

Second, **deterministic, homogeneous processing**. `AJISAI`'s schedule is
fixed and independent of intermediate image quality: a given measurement set
and configuration always yield the same number of iterations, the same gain
tables, and the same final image. Anomalies (e.g. a large dynamic-range drop,
or an interval returning no solutions) are detected and logged but never alter
the control flow. This makes `AJISAI` well suited to large, homogeneous
samples — surveys or archival studies — where every source should be processed
identically regardless of its individual signal-to-noise ratio, and where a
data-dependent number of iterations would be an uncontrolled variable across
the sample.

Third, **transparency and pedagogy**. `AJISAI`'s codebase is small
(~3000 lines) and single-purpose. Together with the justification log, this
makes it readable as a concrete, end-to-end reference for how ALMA continuum
self-calibration is actually carried out — useful for teaching and for
newcomers who find full pipeline installations opaque.

`AJISAI` targets the cases where an explicit audit trail, deterministic
homogeneous processing, or educational transparency matter more than squeezing
out the last decibel of dynamic range on marginal data. It assumes a
continuum-ready measurement set and does not perform flagging or continuum–line
separation.

# State of the field

ALMA self-calibration is assembled from general-purpose CASA tasks for solving
antenna gains (`gaincal`), applying them (`applycal`), and imaging (`tclean`).
In practice it is most often carried out with hand-written scripts adapted from
community tutorials such as the CASA Guides. Such scripts are flexible, but
they differ between groups and datasets, are rarely published, and record their
choices only implicitly in code. Automating these steps reduces the effort but
does not by itself solve the provenance problem: when a procedure decides at
run time whether to keep or discard each step, the number and nature of the
steps vary from dataset to dataset, and the reasons must be recovered from
logs. `AJISAI` instead fixes the schedule in advance and records every
data-dependent choice in a structured file. We built it as a separate package
on top of CASA rather than contributing it to CASA itself, because CASA
provides general-purpose tasks and leaves the workflow to the user, whereas the
value of `AJISAI` lies in one specific workflow (a fixed schedule, data-derived
defaults, and the justification record) layered on those tasks. The scholarly
contribution of `AJISAI` is therefore not a better image-quality optimizer but
a self-calibration procedure whose every decision is machine-readable and
reproducible.

# Software design

`AJISAI` is organized around three decisions, each with a cost that we accepted
deliberately.

**A fixed schedule, run to completion.** The default schedule is three
phase-only iterations (solution intervals `inf`, `6*IT`, and `3*IT`, where `IT`
is the average integration time) followed by one amplitude iteration. All
iterations always run, and the best image is selected afterwards by dynamic
range (peak divided by off-source RMS). The cost is that `AJISAI` can spend
effort on iterations that do not help, and on marginal data it can reach a
lower dynamic range than a schedule tuned to the individual dataset. The
benefit is that a given measurement set and configuration always produce the
same number of iterations and the same gain tables. A rollback rule would also
stop runs prematurely in a common ALMA pattern, where phase solutions plateau
before amplitude calibration recovers the image. Anomalies, such as a large
dynamic-range drop or an interval with no solutions, are detected and written
to the log but never change the control flow.

**A structured justification record.** Each parameter choice is stored in
`justification.json` together with its rationale and the quantities behind it.
For the reference antenna, for example, the record contains the chosen antenna,
the strategy, the flag-fraction threshold, the full candidate list with
positions and flag fractions, and the array's geometric center. We chose a
single structured file over free-form logs so that the decisions can be read,
compared across runs, and published alongside a result.

**Data-derived defaults.** To make the measurement-set path the only required
input, defaults are computed from the data: the reference antenna combines a
flag-fraction filter with proximity to the array's geometric center, the cell
and image sizes follow from the baseline distribution and the primary beam, and
the off-source noise is estimated by sigma clipping with the source region
excluded. The design assumes a clean, continuum-ready measurement set and does
not perform flagging or continuum-line separation, which keeps the scope and
the code base small enough to read and test. Imaging and
calibration are delegated to CASA: deconvolution uses multi-scale,
multi-frequency-synthesis `tclean` [@cornwell2008; @rau2011] with Briggs
weighting [@briggs1995], and the cleaning threshold is set from the noise of
the preceding image, starting with an uncleaned (dirty) image. `AJISAI`
therefore adds orchestration, parameter derivation, and the justification
record rather than new numerical algorithms.

# Research impact statement

`AJISAI` was developed for, and used in, the self-calibration of ALMA
observations of the protoplanetary disk V1094 Sco, published in @yamaguchi2026,
where the procedure is described in Appendix B. The package has since been
validated end-to-end on the public ALMA TW Hya Band 7 dataset (Project
2011.0.00340.S): with default settings, all four iterations completed without
anomalies and the dynamic range rose from 81 to 246, a factor of three. The
reference outputs, including `justification.json` and `metrics.csv`, are
committed to the repository and the demo can be reproduced with
`examples/run_twhya_demo.py`. The software is openly licensed (MIT), archived
on Zenodo with a versioned DOI, documented on Read the Docs, and covered by a
continuous-integration test suite. We are not aware of use by other groups at
the time of writing; the near-term significance we claim is that the audit
trail and fixed schedule give archival and survey studies a documented,
homogeneous self-calibration procedure.

# AI usage disclosure

The core of `AJISAI`, including its algorithms and their implementation, was
written by the author before any generative AI tool was used in this project.
Generative AI (Anthropic Claude, accessed via Claude Code; the Claude Opus,
Fable, and Sonnet model families, including Claude Sonnet 5.5 and Claude Opus
5.5 for the work named below; exact versions of the other models were not
recorded) was used during 2026 for two purposes. First, it helped prepare the
existing code for public release on GitHub. This included drafting and
revising repository documentation (`CONTRIBUTING.md`, the README support
section, the issue and pull-request templates, and the online documentation)
and may have included other release-preparation tasks, which were not
systematically logged; it was not used to design or write the pipeline from
scratch. Second, it drafted parts of this paper (the Summary, State of the
field, Software design, and Research impact statement) from the author's
documentation and design notes and edited the other sections; the author
reviewed and revised all text. The scientific design of the pipeline (the
fixed self-calibration schedule, the parameter choices and their
justification, the reference-antenna selection criterion, and the decision to
record machine-readable justifications for every step) was made by the author.
After the code was published, the author tested it, including comparison
against reference values from `analysisUtils`, and confirmed that it behaves
correctly. The author takes full responsibility for the software and this
paper.

# Acknowledgements

We acknowledge the developers of CASA, on which this package is built. This
work was supported by JSPS KAKENHI Grant Numbers JP26K17220 and JP26K00741 and
by the NAOJ ALMA Scientific Research Grant Code 2022-22B.

This paper makes use of the following ALMA data: ADS/JAO.ALMA#2011.0.00340.S.
ALMA is a partnership of ESO (representing its member states), NSF (USA) and
NINS (Japan), together with NRC (Canada), NSTC and ASIAA (Taiwan), and KASI
(Republic of Korea), in cooperation with the Republic of Chile. The Joint ALMA
Observatory is operated by ESO, AUI/NRAO and NAOJ.

# References
