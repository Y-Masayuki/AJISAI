---
title: 'AJISAI: Automated Justification-based Imaging and Self-calibration for ALMA Infrastructure'
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
date: 15 October 2026
bibliography: paper.bib
---

# Summary

Radio telescopes such as the Atacama Large Millimeter/submillimeter Array
[ALMA, @wootten2009] make images by combining the signals recorded by many
separate antennas. The atmosphere and the instruments alter the phase and
strength of the signal reaching each antenna, which blurs the image and adds
spurious features. Self-calibration [@schwab1980; @cornwell1981] corrects these
errors by using the image itself as a reference: a model of the sky is used to
estimate each antenna's errors, the data are corrected, a better image is made,
and the cycle is repeated. Every run involves choices, above all how often the
corrections are updated (the solution interval), and these are usually made by
hand and rarely reported, so it is often difficult to reproduce how a published
image was made, or to know whether a correction helped or merely fitted noise.

`AJISAI` (*Automated Justification-based Imaging and Self-calibration for ALMA
Infrastructure*) is a Python package that self-calibrates ALMA continuum data
on top of CASA [@casa2022]. Its only required input is the location of the
data. It derives every setting from measured properties of the data, decides
each self-calibration stage by testing it on data that were not used to solve
it, never modifies its input, and writes every choice, with the reason and the
numbers behind it, to one machine-readable file, `justification.json`.

# Statement of need

`AJISAI` is intended for ALMA users who need self-calibration that is both
correct on faint data and fully documented. It addresses three needs.

First, **protection against noise fitting**. Gain solutions are fitted to the
noise of the data they are solved from, so any quality measure computed on the
same data, such as the peak signal-to-noise ratio of the self-calibrated image,
improves as the solution interval shortens, even when the corrections are
mostly noise [@martividal2008]. On faint sources this produces images that look
better but are biased. `AJISAI` judges each candidate interval on held-out
data, so a stage is accepted only if it improves data it was not fitted to.

Second, **auditability**. A reader with the same measurement set and the
`justification.json` file can trace every decision: the measured noise, signal
rates and phase structure, the candidate intervals and the reasons each was
accepted or rejected, the gain tables, and the final imaging parameters.

Third, **use as an engine**. `AJISAI` can be driven by other software. The
data-combination tool `ASEBI`, which aligns and flux-scales ALMA execution
blocks before they are combined, calls `AJISAI` for its self-calibration and
passes its own imaging geometry and masks, so that the same rules apply whether
`AJISAI` is run alone or inside a larger reduction.

`AJISAI` assumes a continuum-ready measurement set and does not perform flagging
or continuum-line separation.

# State of the field

ALMA self-calibration is assembled from general-purpose CASA tasks for solving
antenna gains (`gaincal`), applying them (`applycal`) and imaging (`tclean`),
most often in hand-written scripts that follow a common ladder of solution
intervals (one solution per execution, then progressively shorter intervals,
then an amplitude stage) and judge each stage by the image peak
signal-to-noise ratio. Such scripts differ between groups and record their
choices only implicitly. The trade-off behind the interval is well understood:
a shorter interval follows the atmosphere better but each solution is noisier,
and the optimum follows from the phase structure function and the solution
signal-to-noise ratio, as derived for VLBI fringe fitting by @blackburn2019 and
@pesce2024. What `AJISAI` adds is to measure these quantities for each dataset,
to confirm the predicted choice on held-out data, and to record the whole
chain. We built it on CASA rather than inside it because CASA provides
general-purpose tasks and leaves the workflow to the user, whereas the value of
`AJISAI` lies in one specific, recorded workflow layered on those tasks.

# Software design

**Measured quantities instead of a fixed schedule.** Before each stage,
`AJISAI` measures, without making images, the noise scale of each spectral
window from differences of adjacent integrations, the rate at which each
antenna's solution signal-to-noise ratio grows with the interval (from the
model visibilities and the weights), and the phase structure function from the
lag correlation of model-weighted visibilities, which is free of thermal-noise
bias. From these it predicts the residual phase variance
$V(T) = a\,(T/\tau_0)^{\beta} + \kappa/(r^{2}T)$ for each candidate interval
$T$, where the first term is the atmosphere left within an interval and the
second the noise of the solutions, and chooses candidates near the minimum.
Candidates are whole multiples of the integration time that divide the scan,
plus one solution per scan and per execution.

**Gates and held-out validation.** A candidate is admitted only where the
predicted true solution signal-to-noise ratio is at least 5 for each antenna
(gate G1, about 4% coherence loss); antennas below it keep the previous
correction. The remaining candidates are tested by gate G2: within each scan the
integrations are split into two halves, gains are solved on one half and
applied to the other, and the coherence of the held-out half is measured with a
delete-one-antenna jackknife error. The held-out result is corrected for the
known loss of half-data solutions and compared with the next shorter candidate;
a stage is accepted only if its corrected improvement exceeds one standard
error and an absolute floor. The first stage solves once per execution; at most
three phase stages follow, and an amplitude stage is attempted only when the
data support it and a short-baseline flux check (gate G3) passes. The number of
stages therefore depends on the data, but the rule does not: the same data and
configuration always give the same result. The final image is the one made
after the last accepted stage.

**The input is never modified.** Each stage solves and applies its gains on a
temporary copy of the previous stage's data and writes a new, immutable
measurement set; the product is a single data-only measurement set of the
selected stage. Every intermediate state therefore remains available for
inspection, and the input can be shared between tools.

**Imaging rules from the data.** The pixel size is one fifth of the finest
fringe spacing of the unflagged data, the image covers the primary beam of the
smallest dish, the number of Taylor terms and the scales of multi-scale,
multi-frequency deconvolution [@cornwell2008; @rau2011] follow from the data,
and masks can be supplied as text region files. Inside the loop the image is
cleaned to three times a noise level measured on a source-free polarization
image; the final image is cleaned in steps until the flux left in the residual,
counted in the units of the dirty beam [@jorsater1995], no longer changes the
corrected total flux, and the remaining bias of the restored image is
recorded. Weighting is Briggs [@briggs1995].

**One structured record.** Each measured quantity, candidate, gate result,
decision and reason is stored in `justification.json`, together with the
per-stage solution settings and checks that every image and model prediction
covered the data. `AJISAI` adds orchestration, measurement and decision rules on
top of CASA rather than new imaging algorithms.

# Research impact statement

Version 0.2 of `AJISAI`, with a fixed schedule of solution intervals, was used
for the self-calibration of ALMA observations of the protoplanetary disk V1094
Sco [@yamaguchi2026]. Reprocessing those data showed that the fixed ladder
drove some solution signal-to-noise ratios into the range where noise fitting
biases the image by several per cent, which motivated the data-driven rules of
version 0.3 described here. Version 0.3 is the self-calibration engine of
`ASEBI` and is being validated on the V1094 Sco data and on the multi-epoch
ALMA observations of DG Tau B. The public TW Hya demonstration
(`examples/run_twhya_demo.py`, project 2011.0.00340.S) reproduces a full run
with committed reference outputs. <!-- TODO(v0.3): replace with the v0.3 TW Hya and V1094 Sco numbers once the validation runs are done. -->
The software is openly licensed (MIT), archived on Zenodo with versioned DOIs,
documented on Read the Docs, and tested by continuous integration, including
CASA tests on simulated data that check that the input is left byte-identical.

# AI usage disclosure

The core of `AJISAI`, including its algorithms and their implementation, was
written by the author before any generative AI tool was used in this project.
The scientific design of the pipeline (the fixed self-calibration schedule,
which remains available as an option, the parameter choices and their
justification, the reference-antenna selection criterion, and the decision to
record machine-readable justifications for every step) was made by the author.
Generative AI (Anthropic Claude via Claude Code; the Claude Opus, Fable and
Sonnet model families, including Claude Opus 5.5 and Claude Fable 5.1) was used
during 2026 to help prepare the code for public release (documentation,
templates and release tasks), and the code of version 0.3 was written by the
author together with generative AI. Generative AI also drafted this paper from
the author's documentation and design notes; the author reviewed and revised
all text. The
author takes full responsibility for the software and this paper.

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
