# Chios project state

## Current checkpoint

- Checkpoints A and B are complete; Checkpoint C (validation) has **not**
  started and must not start during this work.
- Current activity: document verified SAR diagnostics and pipeline variants,
  run the final full-suite check, and refresh ignored Graphify outputs. All
  requested work is complete; stop without starting Checkpoint C.
- Do not choose or commit a final `burn_direction`.

## Decisions and constraints

- Compare SAR signs to the delineation only as a one-time diagnostic. Since the
  reference direction is examined, disclose that any eventual sign choice is
  label-informed; do not tune a final sign from these results.
- Do not calculate accuracy metrics. Keep all work uncommitted.
- Before any large null run, verify the memory-safe implementation with five
  replicates under the requested 8 GB virtual-memory limit and record peak RSS.
- Reuse AOI-window data and repaired/rasterized geometries; avoid retaining
  per-replicate arrays or repeatedly rasterizing polygons. Hard-cap placement
  attempts and report successful placement count.
- Preserve existing `outputs/chios/` files; put newly labelled outputs alongside
  them.

## Rebuilt baseline (2026-10-26)

- Full suite: **128 passed** with
  `.venv-qgis/bin/python -m pytest -q` (2.37 seconds).
- `outputs/chios/` contains **94 files**: 90 GeoTIFFs, 3 PNGs, and
  `null_test.log`.
- Worktree status at inspection: unrelated untracked `.vscode/`; do not alter.
- Re-read the design/state and inspected the diagnostic, audit, pipeline, and
  focused tests. The old null implementation was not rerun as-is.

## Completed implementation visible on disk

- Existing code includes AOI-land SAR median centering, USGS dNBR breakpoints
  as the default with an Otsu option, and robust median/MAD SAR breakpoints.
- Tests cover null contrast masking/placement helpers, contrast summaries,
  histogram modality, and robust SAR breakpoints. The full suite passes.
- These are code-state observations, not evidence that the scene diagnostics or
  paired variants have been run.
- Previously recorded `k=2` Burned floor is `1 - exp(-6/14) = 0.3485609425`
  (0.3486).

## Pre-run review findings (historical snapshot)

- The current diagnostic script reads AOI windows rather than full source scenes,
  but its null sampler materializes the full offset grid and retains every
  full-window candidate mask; it also accumulates null distributions before
  summarizing them. It has no per-replicate attempt cap or incremental CSV.
- Grade polygons are repaired before rasterization, but delineation/AOI
  geometries are not. Grade masks are rasterized one time per grade, but they
  are not returned as a cached scene object.
- The diagnostic uses one outside sample for both sign comparison and null
  controls. The sign summary should expose the AOI-window outside-fire land
  sample and the null comparison must use only EMSN198 AOI land.
- The script currently prints compact per-grade summaries that omit outside
  median/std, so it does not yet satisfy the requested complete statistics.

## Memory-safe diagnostic rewrite and constrained trial

- The null sampler now draws random integer-pixel shifts with at most 200
  attempts per requested replicate. It uses the cropped fire-mask bounding box,
  checks the candidate against cached EMS and valid-land masks, rejects
  delineation overlap, and yields one candidate at a time instead of retaining
  full-window masks or materializing the full offset grid.
- Null output is written and flushed incrementally to `outputs/chios/null_test.csv`;
  only scalar contrasts per successful replicate are retained for the final
  empirical summaries. A progress line is emitted every 10 attempted
  replicates, and successful placements are counted explicitly.
- Delineation and EMS geometries are repaired once; damage-grade geometries are
  repaired and rasterized once, then their masks are cached in the scene.
- Sign and grade summaries now include means, medians, standard deviations,
  pooled standard deviation, and effect size. The all-outside-fire AOI-window
  comparison is distinct from the EMSN198-only observed contrast used for the
  null test.
- Focused diagnostics tests pass: **9 passed**. A five-replicate run under
  `ulimit -v 8000000` completed in 3.65 s with peak RSS **206,168 KB**, well
  below the limit.
- The first trial accepted **0/5** placements. The cause was a misread of OGR
  `GetEnvelope()` ordering; it was fixed and is covered by the focused tests.
- Geometry is now passed through MakeValid once before rasterization; the
  repeated `Ring Self-intersection` warnings are gone in the second trial.
- The corrected five-replicate trial still accepted **0/5**. A bounded
  feasibility diagnostic showed candidate shifts are within the correct EMS
  envelope but sampled candidates overlap the real delineation. The valid-land
  EMS mask has 110,848 pixels; the candidate has 28,090 pixels.
- Second trial peak RSS: **205,496 KB**, exit status 0; no self-intersection
  warnings.
- Verified descriptive stats from the final run: VH raw mean contrast +0.15484
  dB over all outside-fire land (+0.14614 dB for the EMSN null comparator);
  filtered +0.17516 (+0.17314 EMSN); VV raw +0.64407 (+0.62824 EMSN);
  filtered +0.65049 (+0.62652 EMSN). All four global histograms were counted
  as one mode by the descriptive detector.

## Null-test result

- The required timed 200-replicate attempt completed within the timeout and
  under the same 8 GB virtual-memory limit. It attempted up to 200 offsets per
  replicate, wrote **200 rows** incrementally, and accepted **0/200** valid
  placements. No empirical p-value or null interval exists; this is a placement
  feasibility failure, not evidence that SAR is or is not distinguishable from
  background.
- The EMSN198 AOI envelope permits shifts smaller than both the fire mask's
  rasterized bounding-box dimensions; the candidate remained intersecting the
  real fire in all tested positions. The task's required same-shape,
  no-overlap null distribution therefore cannot be estimated from this
  placement scheme/available AOI. Do not reinterpret the sign contrast as
  statistically distinguishable or non-distinguishable.
- The two-sided null p-values are **unavailable** for VH and VV, filtered and
  unfiltered. `outputs/chios/null_test.csv` and
  `outputs/chios/null_test_summary.txt` preserve the honest zero-success
  outcome; `null_test_200.log` captures all 200-replicate output.

## Verified pipeline variants

- Both sign runs used centered SAR, 5x5 filter, temporal discount on, USGS
  dNBR `(0.10, 0.27)`, and label-free robust SAR breakpoints; the one-time
  reference diagnostic was skipped in these repeat pipeline runs.
- Negative variant: 8,885 Burned / 291,604 Unburned / 27,030 Undecided;
  mean conflict 0.08518554 across 327,519 valid pixels.
- Positive variant: 9,094 Burned / 295,747 Unburned / 22,678 Undecided;
  mean conflict 0.07395738 across 327,519 valid pixels.
- Labelled negative/Otsu dNBR variant used `(0.193888, 0.19858)` and produced
  8,762 / 288,752 / 30,005 with mean conflict 0.08723.
- Side-by-side PNG and a counts summary are saved under
  `outputs/chios/variants/`. No accuracy metrics were calculated and no sign
  was selected.

## Final validation and output inventory

- Full suite after all code changes: **129 passed** in 2.06 seconds with
  `.venv-qgis/bin/python -m pytest -q`; `git diff --check` passes.
- Ignored Graphify outputs were refreshed: **508 nodes**, **1,319 edges**,
  **22 communities**; the prior health check found no dangling, missing, or
  collapsed edges. The incremental manifest records 31 files. The report
  labels token usage unavailable because the semantic-agent usage was not
  exposed; the cost tracker records `usage_available: false`.
- Graphify intermediate sidecars and temporary backups were removed. The
  refreshed report is `graphify-out/GRAPH_REPORT.md`; the HTML graph is
  `graphify-out/graph.html`.
- Current `outputs/chios/` root contains **100 files**; the labelled
  `outputs/chios/variants/` subtree contains **101 files**.
- Required records include `null_memory_test.log`, `null_test_200.log`,
  `null_test.csv` (200 failed-placement records), `null_test_summary.txt`,
  `chios_sar_histograms_centering.png`, per-variant run logs and rasters, and
  `variants/sign_comparison.png` with `sign_comparison_summary.txt`.
- Tracked changes are limited to `docs/DESIGN.md`, `docs/STATE.md`,
  `examples/chios_sar_diagnostics.py`, `examples/run_chios.py`, and
  `tests/test_chios_sar_diagnostics.py`. The unrelated untracked `.vscode/`
  remains untouched.

## Next actions

1. Stop without beginning Checkpoint C; no further analysis or implementation
   is authorized in this task.

## Open questions

- Null distinguishability remains unresolved because no same-shape,
  non-overlapping placement could be accepted within the specified EMS AOI and
  200-attempt limits.
- No final burn direction is selected; any later sign decision must disclose
  the one-time comparison against the reference direction.

## Not done

- No empirical null p-value can be calculated from the zero-success placement
  sample.
- No final burn direction selected; no accuracy metrics computed.
- Checkpoint C has not started.
- Graphify outputs are ignored and refreshed; no tracked Graphify files were
  added.
