# Chios project state

Last updated: 2026-10-31. This file is the source of truth after a crash.
Update it after every step. Do not use or refresh Graphify in this project phase.

## Current checkpoint

- Checkpoints A and B: complete.
- SAR null test v2: complete (see results below).
- Baseline suite for Checkpoint C preparation: `.venv-qgis/bin/python -m
  pytest -p no:cacheprovider -q` passed (140 tests) under
  `ulimit -v 8000000` on 2026-10-31.
- Pre-commit amendment verification: config JSON is valid, `git diff --check`
  passed, and the full `.venv-qgis` test suite passed again (140 tests) under
  `ulimit -v 8000000`. No scoring was run.
- Checkpoint C STEP 1 complete: `configs/checkpoint_c.json` and the
  "Checkpoint C pre-registration" section in `docs/DESIGN.md` are ready.
  Pre-commit amendments specify the primary outcome, expected result,
  reliability combination and valid-land rule, temporal discount distinction,
  bootstrap ratio calculation, provenance fields, and expanded label-informed
  disclosure. Scoring has NOT started.
- Decision taken (user): proceed with option A. Primary SAR candidate is
  VV with burn_direction=positive; VH positive is a disclosed variant.
- NEXT: Wait for the user to commit the frozen config. Proceed to STEP 2 only
  after the user says "config committed"; no scoring has started.
- Checkpoint C scoring has NOT started. No accuracy metrics exist yet.

## Rules and constraints

- No commits by the agent; the user commits.
- The SAR sign was examined against the reference direction, so the choice of
  VV positive is label-informed and must be disclosed. Both polarizations are
  reported. Never pick a sign or parameter by whichever scores better.
- After scoring starts, no parameter, breakpoint multiple, sign or rule may
  change. Later changes are new runs labelled EXPLORATORY and never replace
  the primary result.
- Memory discipline: AOI window or block reads only, compute masks and evidence
  once, keep scalars per replicate, test new code under `ulimit -v 8000000`
  before full runs. Earlier null runs once used ~20 GB and froze the machine.
- Keep dst_fusion/core free of qgis and osgeo imports.
- If a step repeats more than twice, stop and report the blocker.

## Event and data

- Chios 2024 fire, started ~1 July, ~36 h (fire_end ~3 July).
- S2 L2A (T35SMC, R050, N0510): pre 2024-06-28, post 2024-07-08; B8A, B12, SCL
  at 20 m. Reflectance = (DN + BOA_ADD_OFFSET)/quantification
  (offset -1000, quantification 10000, read from MTD_MSIL2A.xml); DN=0 is NoData.
- S1 HyP3 RTC gamma0 dB, 20 m, EPSG:32635, no speckle filtering, VH and VV:
  pre 2024-06-29, post 2024-07-11; relative orbit 36, descending (derived from
  absolute orbits 054533 and 054708). S2 post is 3 days earlier than S1 post.
- Reference: data/chios2024/reference/fire_reference.gpkg (delineation, grading
  with damage_grade, EMSN198_AOI). The reference is optical-derived (EMS grading
  vs our dNBR: Spearman 0.90), so it favours optical-only and understates SAR.
- Grid facts: full S2/S1 intersection 5,490 x 4,170 px; AOI window on the S2
  tile (191, 1994, 702, 757); fire delineation 28,090 px (28,069 valid).
- Far-water raw DN is higher than expected for open sea (B12 p1/p50/p99
  1610/1899/2353 pre, 1610/1844/2297 post). Cause unconfirmed; non-blocking
  because water is masked.

## Verified design facts

- k=2 asymmetry Burned floor = 1 - exp(-6/14) = 0.3486. The asymmetry is gated
  on fire_end; on Chios both post images are after fire_end, so k=1 applies.
- Temporal discount (dt=3 d, tau=14): alpha_time = 0.8071, uniform floor 0.1929.
- Optical alpha from SCL: classes 3, 8, 9, 10 -> 0, else 1; SAR alpha 0 where
  ls_map layover/shadow bits are set (~0.2% of pixels).
- Median centering of the SAR change does not change a difference of means.

## Pipeline variants run so far (counts only, no accuracy metrics)

Common settings: centered SAR, 5x5 linear-power boxcar, temporal on, robust
label-free SAR breakpoints, 327,519 valid land pixels in the AOI window.

- VH negative, USGS dNBR (0.10, 0.27): 8,885 Burned / 291,604 Unburned /
  27,030 Undecided; mean conflict 0.0852.
- VH positive, USGS dNBR: 9,094 / 295,747 / 22,678; mean conflict 0.0740.
- Negative with Otsu dNBR (0.193888, 0.19858): 8,762 / 288,752 / 30,005;
  mean conflict 0.0872.
- Outputs under outputs/chios/ and outputs/chios/variants/.

## SAR null test v2 (Option A, full-intersection placement)

- The earlier AOI-restricted null accepted 0/200 placements. That was a design
  flaw (fire mask larger than the usable AOI land), not evidence about SAR.
- v2: 466/500 placements accepted (86,013 offsets tried, cap 500 per replicate),
  44 s, peak RSS ~655 MB. Placements must be on land, >=500 m from coast and
  from the fire and grading polygons, optically unburned (dNBR < 0.10, alpha 1),
  with a pre-fire SCL class 4/5 mix within +/-0.10, low layover/shadow.
- Real contrast vs null (two-sided p; 0.004283 is the resolution floor):
  - VH raw +0.129 dB, p=0.030, z=+2.4
  - VH 5x5 +0.148 dB, p=0.021, z=+3.4
  - VV raw +0.588 dB, p=0.004, z=+22.2
  - VV 5x5 +0.595 dB, p=0.004, z=+23.0
- VV is positive and in the null tail in all four severity classes.
- VH sign flips by severity: High -0.57 dB and Moderate high -0.47 dB
  (negative); Moderate low +0.32 dB and Low +0.53 dB (positive). The overall VH
  contrast is a net of opposing effects, so a single fixed VH sign is unreliable.
  A physical explanation is plausible but unverified.
- Limits: placements overlap and are not independent, so p-values are descriptive
  and not exact significance tests. The test shows the contrast is unusual, not
  that SAR detects burned pixels well. Null median is ~+0.05 dB (small regional
  offset between dates).
- Files: outputs/chios/null_test_v2.csv, null_test_v2_summary.txt, four
  histograms, null_test_v2.log, null_memory_test_v2.log.

## Environment

- Use .venv-qgis (Python 3.12.3, system GDAL 3.8.4 with JP2OpenJPEG):
  `.venv-qgis/bin/python -m pytest -q`. Last result: 140 passed.
- Tests under QGIS's own bundled Python have not been run yet.

## Not done

- Checkpoint C (validation), including the pre-registered config, scoring,
  block-bootstrap CIs, artificial-cloud ablation and conflict analysis.
- Final burn_direction decision beyond the disclosed primary/variant plan.
- QGIS Processing algorithm wrapper, styling, packaging, documentation,
  repository submission.
