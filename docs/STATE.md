# Chios project state

## Current checkpoint

- Checkpoint A: done.
- Checkpoint B: done.
- Checkpoint C: not started; do not start it during this work.
- Current activity: read-only robustness diagnostics requested for Checkpoint B.

## Decisions and constraints

- The previous burn-direction decision remains open; the answer was option 4
  (custom). Do not change `burn_direction` or select a final sign in this work.
- Sign diagnostics are one-time, label-informed comparisons against the
  delineation/reference direction, not tuning. Any eventual final sign choice
  will therefore be label-informed.
- Do not calculate accuracy metrics.
- Keep all work uncommitted.

## Baseline

- Existing test suite: **125 passed** using
  `.venv-qgis/bin/python -m pytest -q`.
- Checkpoint mapping in `docs/DESIGN.md`: A and B done, C not started.
- Existing `outputs/chios/` inventory: **93 files** (90 GeoTIFFs, 3 PNGs).
  Existing outputs will be preserved; new comparisons will use a separate
  labeled subdirectory.
- Chios source directories exist: `data/chios2024/s1/` and `data/chios2024/s2/`;
  the EMSN198 fire reference GeoPackage is available.
- At baseline, the only reported worktree change was an unrelated untracked
  `.vscode/` directory; do not alter it.

## Completed steps

- Read `docs/DESIGN.md` and inspected the Checkpoint B scripts/core helpers.
- Verified the full baseline suite: 125 tests passed.
- Found that the existing null-test outside sample includes land beyond the EMS
  AOI and the real delineation. The null comparator must instead use valid land
  inside EMSN198_AOI and outside the delineation; fix before interpreting it.
- The existing far-water helper already reads raw DN for B8A/B12 and both dates,
  using a >1 km SCL-land proximity mask; its corrected values still need to be
  rerun and reported.
- Implemented the null-test sample correction: both observed and null
  inside-minus-outside comparisons now use valid EMSN198_AOI land outside the
  delineation, excluding the candidate polygon from each null control sample.
- Added switchable AOI-land SAR median centering to `run_chios.py`, changed its
  primary dNBR breakpoints to USGS (0.10, 0.27), retained a named Otsu option,
  and made median-plus-1/2-scaled-MAD SAR breakpoints the label-free default.
  The existing uncentered and explicit Otsu options remain available.
- Added focused regressions for the EMS-only null contrast, placement
  constraints, and robust SAR breakpoints.
- Focused validation passed: 32 tests passed across SAR diagnostics,
  preprocessing, and memberships.
- Verified `1 - exp(-6/14) = 0.3485609424689444` (rounds to 0.3486).

## Planned read-only diagnostics

1. SAR VH/VV sign contrast, filtered/unfiltered, land only, with inside/outside
   summaries and per-damage-grade summaries.
2. Approximately 200 same-shape/area null placements on unburned land inside
   EMSN198 AOI and outside the delineation; report the real contrast's empirical
   rank/p-value and whether it is distinguishable from background variation.
3. Outside-fire post-minus-pre global offset; optional AOI-land median centering
   with switchable behavior; centered/uncentered histograms and an assessment
   of Otsu suitability plus a label-free robust-scale alternative.
4. USGS optical dNBR breakpoints (0.10, 0.27) as primary and existing Otsu
   thresholds (0.194, 0.199) as a labelled variant.
5. Two labelled previews only: negative and positive burn direction, both
   median-centered, filtered, temporal-on, with USGS optical breakpoints.
   Report burned/unburned/undecided counts and mean conflict; no accuracy scores.
6. Corrected far-water S2 DN percentiles (1/50/99) for B8A and B12 over SCL
   class 6 pixels more than about 1 km from land, for both dates; correct any
   earlier reflectance/DN mislabelling.
7. Numerically verify the Checkpoint B k=2 Burned floor:
   `1 - exp(-6/14) = 0.3486` (approximately), not 0.3497.

## Open questions

- What is the empirical SAR null-test result and is the observed contrast
  distinguishable from background variation?
- Which SAR direction, if either, has stronger one-time sign evidence? This
  does not close the final burn-direction decision.
- Is the SAR change histogram unimodal, and what label-free robust-scale
  membership description is appropriate?
- What are the corrected far-water DN percentiles?
- Are any required inputs or geometry details insufficient for the planned
  diagnostics?

## Not done

- Checkpoint C has not started.
- No final burn direction has been chosen or applied.
- The requested diagnostic analyses, corrected water check, and paired
  comparison previews/counts are not yet completed.
- `docs/DESIGN.md` has not yet been updated with the final sign-evidence caveat
  and revised diagnostics findings.
