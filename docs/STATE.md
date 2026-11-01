# Chios project state

Last updated: 2026-11-1. This file is the source of truth after a crash.
Update it after every step. Do not use or refresh Graphify in this project phase.

## Current checkpoint

- Checkpoints A and B: complete.
- SAR null test v2: complete (see results below).
- Baseline suite for Checkpoint C preparation: `.venv-qgis/bin/python -m
  pytest -p no:cacheprovider -q` passed (140 tests) under
  `ulimit -v 8000000` on 2026-10-31.
- Pre-commit amendment verification: config JSON is valid, `git diff --check`
  passed, and the full `.venv-qgis` test suite passed again (140 tests) under
  `ulimit -v 8000000`; scoring had not started at that point.
- Checkpoint C STEP 2 preflight: passed. Working config SHA-256, tag
  `checkpoint-c-config`, and `HEAD` all match
  `8074f639c2bdd2cd8433e9e67ec50987082b02f1830ba2035a9cbfd47bd5a648`;
  both tag and HEAD resolve to
  `4e1b1619065c3756806e8878bb39ba599637c925`.
- Checkpoint C metric/bootstrap prerequisite: added NumPy-only
  `dst_fusion/core/validation.py` (no QGIS/GDAL imports) and tiny synthetic
  tests. Under `ulimit -v 8000000`, `.venv-qgis/bin/python -m pytest
  -p no:cacheprovider -q tests/test_validation.py
  tests/test_checkpoint_c_rasterization.py` passed (5 tests), including 2x2
  pixel-center and polygon-difference rasterization cases. The scorer compiles
  and `git diff --check` passes.
- Initial scorer launch stopped during grid construction: `RasterGrid` accepts
  four fields, not the extra description argument. No AOI raster window was
  read, no scoring metrics were computed, and no output files were written.
  The call was corrected before the successful single scoring run.
- Checkpoint C STEP 1 complete: `configs/checkpoint_c.json` and the
  "Checkpoint C pre-registration" section in `docs/DESIGN.md` are ready.
  Pre-commit amendments specify the primary outcome, expected result,
  reliability combination and valid-land rule, temporal discount distinction,
  bootstrap ratio calculation, provenance fields, and expanded label-informed
  disclosure.
- Checkpoint C STEP 2 complete: all registered variants, including the
  `EXPLORATORY` VH two-sided variant, were scored once. Results are in
  `outputs/chios/checkpoint_c/`; see `checkpoint_c_summary.txt` and
  `checkpoint_c_results.csv`. The common AOI mask has 111,191 pixels.
  Against `firedelineation`, primary fused IoU is 0.4962 (95% block CI
  0.4328–0.5551), optical-only IoU is 0.8233 (0.7696–0.8737), and VV
  SAR-only IoU is 0.3777 (0.3219–0.4281). Primary-minus-optical IoU is
  -0.3271 (paired CI -0.3994 to -0.2622); primary-minus-SAR IoU is +0.1184
  (+0.0955 to +0.1456). The config SHA-256 is
  `8074f639c2bdd2cd8433e9e67ec50987082b02f1830ba2035a9cbfd47bd5a648`;
  config commit is `4e1b1619065c3756806e8878bb39ba599637c925`; seed 20261031.
  `vh_positive` is typed as SAR and duplicates the `sar_only_vh` settings;
  both registered IDs and outputs are retained. The 5 focused synthetic
  metric/bootstrap/rasterization tests passed under the 8 GB limit.
- STEP 3 evidence-cache preparation attempt found a reference rasterization
  mismatch: the scorer's saved FireDelineation raster is the first source
  feature, while the supporting helper initially rasterized every fire-layer
  feature. No cache was saved and no supporting metrics were computed. Updated
  prep to reuse the exact two scored reference rasters and derive the common
  mask from their saved NoData masks; will continue with the same frozen
  predictions and grid.
- Second cache-preparation attempt stopped before writing a cache due to a
  local mask-ordering error (`valid` was referenced before assignment). No
  supporting metrics were computed; corrected the validity expression.
- STEP 3 preparation complete: built one AOI-window evidence cache under the
  8 GB virtual-memory limit, verified against the scored reference rasters and
  common mask (111,191 pixels). Cache provenance matches the frozen config;
  SAR breakpoint values match STEP 2. Subsequent analyses reuse this cache.
- STEP 3 item 1 complete: the three registered 25x25 block positions were
  selected by fire overlap descending (row/column tie-break), each 625/625
  pixels overlapped FireDelineation. In all three blocks, optical-only under
  synthetic alpha=0 predicted zero burned pixels; VV SAR-only and adaptive
  fused each predicted 229, 135, and 247 burned pixels, respectively. Pooled
  within-block IoU was 0.0000 for synthetic optical-only and 0.3259 for VV
  SAR-only/adaptive fusion (paired 500 m block bootstrap CI 0.2160–0.3952).
  Per-block outside estimates and CIs are in
  `outputs/chios/checkpoint_c/synthetic_cloud_ablation.txt` and `.csv`;
  per-block inside CIs are omitted as each patch is one spatial unit.
- STEP 3 item 2 complete: descriptive conflict associations are saved in
  `outputs/chios/checkpoint_c/conflict_associations.txt` and `.csv`. Mean
  conflict was 0.025 within 100 m of SCL-6 water, 0.124 at 100–500 m, 0.222
  at 500–1,000 m, and 0.101 beyond 1,000 m (Spearman with water distance
  -0.147). Mean conflict by post SCL: class 2=0.377, 4=0.087, 5=0.118,
  7=0.000, 8=0.000. By reference damage grade it decreases from High=0.414
  and Moderate high=0.409 to Moderate low=0.315 and Low=0.233. No DEM was
  found, so slope association is unavailable. The water-distance estimate is
  only a proxy, not surveyed coastline distance.
- STEP 3 item 3 complete: `outputs/chios/checkpoint_c/posthoc_diagnostics.txt`
  and `.csv` record 0 adaptive-vs-conjunctive class differences across the
  111,191 common pixels (hence identical confusion counts and metrics);
  optical-detected/reference-burned pixels missed by primary are predominantly
  undecided for High (100%), Moderate high (99.91%), and Moderate low
  (98.54%), but mostly called unburned for Low (69.05%). POST-HOC
  temporal-discount-off minus primary is +0.02705 IoU (paired 95% block CI
  +0.02389 to +0.03025) and +0.02374 F1 (+0.02107 to +0.02675).
- STEP 3 item 4 complete: explicit POST-HOC optical + VH-positive adaptive
  fusion is saved in `posthoc_vh_fusion.txt` and `.csv`, with rasters and
  reference comparison maps. Against FireDelineation IoU=0.3144
  (95% block CI 0.2565–0.3680), F1=0.4784 (0.4083–0.5380), precision=0.9923,
  recall=0.3152. It is supplementary and does not replace the primary.
- STEP 3 item 5 complete: `docs/DESIGN.md` now records the primary and paired
  results, the duplicate registered `vh_positive` note and separate
  POST-HOC VH-fused run, optical-reference bias, label-informed sign choice,
  one-event limitation, conflict associations, synthetic ablation, post-hoc
  diagnostics, and the unverified relatively flat VV grade-recall observation.
- Final validation: the full `.venv-qgis` test suite passed (146 tests) under
  `ulimit -v 8000000`; scorer/supporting scripts compile and `git diff --check`
  passes. Config SHA-256 remains
  `8074f639c2bdd2cd8433e9e67ec50987082b02f1830ba2035a9cbfd47bd5a648`.
- Decision taken (user): proceed with option A. Primary SAR candidate is
  VV with burn_direction=positive; VH positive is a disclosed variant.
- NEXT: Stop. STEP 3 is complete; wait for the user.
- Checkpoint C scoring is complete; do not change the primary result.

## Milestone 6 — QGIS Processing plugin

- STEP 1 design only: the full `.venv-qgis` suite passed (146 tests) under
  `ulimit -v 8000000` on 2026-10-31. User requested proposal revisions before
  approval; awaiting approval before implementation. No code, Checkpoint C
  setting, or core math default changed.
- Revised algorithm `dst_fusion:fuse_burned_area`: four required S2 B8A/B12
  layers; `S2_INPUT_MODE` reflectance or DN-with-offset (offset `-1000`,
  quantization `10000`, DN 0 NoData); optional SCL pair and optional binary
  land mask; required same-polarization SAR pre/post dB layers; four
  acquisition dates; optional paired fire dates; `tau_days=14`; explicit
  `BURN_DIRECTION` enum with a first “choose explicitly” entry and validation
  error if unselected; filter size 5 (0 off, otherwise positive odd), median
  centering on, SAR reliability 1.0, optional `ls_map`; dNBR thresholds
  0.10/0.27; robust SAR thresholds from median plus k=1/2 scaled MAD;
  adaptive/conjunctive/disjunctive fusion; auto/always/never temporal mode;
  AOI extent; necessity threshold 0; optional undecided mask. Class code 2
  always remains undecided. Outputs include class, necessity, conflict,
  optional Burned/Unburned possibility bands, and optional undecided mask.
- The strict Chios regression must use `TEMPORAL_MODE=always` (not auto),
  primary frozen settings, and AOI window extent `(403780, 4245000, 417820,
  4260140)` in EPSG:32635 (delineation bbox plus 5 km). Expected primary class
  counts on the 111,191 common valid pixels are 13,961 Burned, 85,382
  Unburned, and 11,848 Undecided. This reproduces the uniform temporal
  discount despite both post acquisitions following fire end; it does not
  change the `auto` mode design.
- Median/MAD/centering are computed over valid pixels inside AOI_EXTENT.
  Proposal: exact order statistics through 2,000,000 valid pixels; above
  that, emit a feedback warning and use deterministic bounded-histogram
  estimates, recording exact/approximate mode in GeoTIFF metadata. SCL and
  LAND_MASK together refine valid land; if neither is supplied, warn that
  water is not masked. `LAND_MASK` proposal: finite nonzero means land.
- Input checks include DN=0 NoData and reflectance-mode unscaled warning
  above 1.5; check_temporal_plausibility messages become Processing feedback
  warnings; validate dNBR low < high, MAD k1 < k2, and odd nonzero filter
  sizes; warn (without converting) when SAR values fail a broad dB-range
  plausibility check. Write algorithm parameters/effective thresholds and
  statistics to output GeoTIFF metadata; write QML sidecars for class and
  continuous outputs so headless `qgis_process` products are styled.
- Proposed layout reuses existing plugin shell, core and IO; add
  `dst_fusion/algorithms/fuse_burned_area.py` and focused tests. The wrapper
  owns QGIS parameters, warnings, streamed orchestration, metadata and QML;
  numerical work stays in `core/`, and alignment/block IO stays in `io/`.
- STEP 1 design approved (with the revisions above). STEP 2 built: dst_fusion:fuse_burned_area.
  152 tests pass in .venv-qgis (system QGIS Python, offscreen), including the frozen Chios
  regression on the 111,191-pixel scoring mask and a synthetic cloud-recovery test.
- Undecided mask: the in-loop write is correct in the current code (verified by commenting out
  the rebuild step: 152 tests still passed). The earlier all-255 outputs most likely came from a
  half-edited intermediate version; this was not verified. _rebuild_undecided_mask is kept as a
  redundant safeguard and can be simplified later. ComputeStatistics failures are non-fatal.
- Mask/class consistency checks were added to both processing tests.
- NEXT: manual test in the QGIS interface (plugin loads, algorithm appears in the Processing
  Toolbox, runs on real layers). No packaging yet.

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
  `.venv-qgis/bin/python -m pytest -q`. Last result: 152 passed under the
  8 GB virtual-memory limit.
- Tests run in .venv-qgis, which uses the system QGIS Python (qgis and osgeo import fine). A run under a separately installed QGIS app's bundled Python is not done.

## Not done

- Final burn_direction decision beyond the disclosed primary/variant plan.
- Manual QGIS interface test, packaging, user documentation, repository submission.

