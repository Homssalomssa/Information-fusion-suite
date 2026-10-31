# Chios project state

## Current checkpoint

- Checkpoints A and B are complete; Checkpoint C (validation) has **not**
  started and must not start during this work.
- Current activity (2026-10-31): redesign and validate the SAR null test under
  the instructions in the current request. The old AOI-restricted sampler's
  `0/200` result is a failed placement design, not an inferential result. Do
  not start Checkpoint C or select/change `burn_direction`.
- Do not choose or commit a final `burn_direction`.

## Decisions and constraints

- Compare SAR signs to the delineation only as a one-time diagnostic. Since the
  reference direction is examined, disclose that any eventual sign choice is
  label-informed; do not tune a final sign from these results.
- Do not calculate accuracy metrics. Keep all work uncommitted.
- Context check completed: read the on-disk `docs/STATE.md` and `docs/DESIGN.md`;
  the previous state records an old sampler with 0/200 accepted placements,
  contrasts VH +0.15484 dB and VV +0.64407 dB, and a 129-pass full-suite
  baseline. The fresh baseline command `.venv-qgis/bin/python -m pytest -q`
  passed **129 tests in 3.58 seconds**. Graphify was queried for navigation
  only; it points to `examples/chios_sar_diagnostics.py` and
  `tests/test_chios_sar_diagnostics.py`. This was the initial context-check
  state, before the implementation below.
- Source/test inspection confirms the current sampler is still bounded to the
  EMSN198 AOI and the tests only cover that prior placement behavior. It builds
  its scene from the buffered AOI window, so the new all-intersection-land
  domain needs a separate memory-bounded scene read before tests or
  implementation proceed.
- Verified input dimensions with the selected `.venv-qgis` interpreter:
  the full S2/S1 intersection is **5,490 × 4,170 (22,893,300 pixels)**, far
  larger than the prior 702 × 757 AOI window. The new implementation therefore
  must use block reads and disk-backed intermediates, not scene-sized NumPy
  copies. Pylance references confirm the old sampler is used only by its
  runner and focused tests.
- Test-first placement checks have been implemented for clean acceptance,
  direct rejection of a placement on the fire exclusion, and rejection of
  failures in land, coast distance, grading exclusion, optical evidence/alpha,
  pre-fire SCL mix, and layover/shadow tolerance. The focused diagnostics suite
  now passes **18 tests**; this exercises the acceptance logic and capped
  full-intersection offset sampler, but not yet the real raster preparation.
- Verified the project optical reliability rule before implementing the null
  mask: SCL classes 3, 8, 9, and 10 receive alpha 0; other valid SCL classes
  receive alpha 1, with class 0 remaining NoData and water excluded separately.
  The new candidate test requires optical alpha exactly 1 throughout.
- Implemented the shared placement acceptance helper and capped whole-pixel
  sampler, and began the block-streamed full-intersection preparation using
  disk-backed GDAL rasters. Syntax compilation and the focused placement suite
  pass (**19 tests**, including the exact two-sided +1 p-value tails). Pylance
  diagnostics report no errors; only private test helpers are unused within the
  production file. Full-scene preparation/output behavior remains to be
  validated before running the constrained memory trial.
- The first constrained five-replicate invocation stopped before full-scene
  preparation because the sampler expected a reader callable in the live
  scene; the recorded process peak was 659,000 KB, but this is not a valid
  memory-test result. The reader now uses the shared disk-backed layer reader;
  the required memory trial must be rerun and pass before the full run.
- After that correction, the focused diagnostics suite passes again
  (**19 tests**); the sampler test covers the same lazy window-reader path
  used by the full-scene runner. No full null statistics are available yet.
- Required five-replicate constrained-memory trial now completes: **5/5
  accepted**, 351 total offsets, 36.31 seconds, maximum RSS **689,152 KB**
  under `ulimit -v 8000000`. It produced the v2 CSV, incremental summary,
  and four marked histograms. Before the full run, a discrepancy needs
  resolution: the new scene reports pre-fire SCL class-4/5 shares
  16.34%/83.66%, unlike the previously audited 23.86%/76.13%.
- The reference mix is now computed directly from pre-fire SCL class-4/5
  pixels within the fire mask (without additionally conditioning the reference
  composition on SAR or layover validity). A fresh full-scene check confirms
  the mix is 23.86%/76.14%, with 6,702/21,386 pixels, matching the prior SCL
  audit. The corrected five-replicate constrained-memory trial accepts **5/5**
  after 712 offsets; peak RSS is **685,004 KB** and elapsed time is **35.90 s**.
- `docs/DESIGN.md` now documents the superseded AOI failure and Option A's
  full-intersection validity filters, exact statistic, median-centering
  invariance, p/z/percentile definitions, spatial-dependence limits,
  label-informed sign disclosure, and new output paths. The 500-replicate
  production run is pending.
- Added and passed a focused block-halo equivalence test: filtered SAR changes
  computed per tile match a full-array 5×5 linear-power filter, including
  scene-edge and NoData behavior. The updated focused diagnostics suite passes
  **20 tests**.
- Full suite after the implementation and test additions passes: **140
  passed** in 3.35 seconds; `py_compile` and `git diff --check` also pass. The
  500-replicate run under the requested 30-minute timeout is now authorized.
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

## Historical AOI-only null-test result (superseded)

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

## Prior final validation and output inventory (2026-10-26)

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

## Full-intersection null-v2 results (2026-10-31)

- Requested 500 placements with a 500-offset hard cap per replicate. **466/500
  were accepted**, after 86,013 offsets total; 34 replicates hit their cap
  without acceptance. Progress was flushed every 10 replicates. The run
  completed in 44.34 seconds under the 30-minute timeout, with peak RSS
  **654,852 KB**. The corrected five-replicate run under
  `ulimit -v 8000000` accepted **5/5**, used 712 offsets, took 35.90 seconds,
  and peaked at **685,004 KB**.
- The full-domain real contrasts are VH raw **+0.12892 dB**, VH 5×5
  **+0.14825 dB**, VV raw **+0.58780 dB**, and VV 5×5 **+0.59471 dB**.
  These differ from the previously reported AOI-window values because the
  comparator is now the full-intersection valid land outside the real-fire
  500 m buffer.
- All four observed contrasts fall in the empirical placement-null tails:
  VH raw `p=0.029979`, `z=+2.4259`, percentile 98.712%;
  VH 5×5 `p=0.021413`, `z=+3.4049`, percentile 99.142%;
  VV raw `p=0.004283`, `z=+22.2348`, percentile 100%;
  VV 5×5 `p=0.004283`, `z=+23.0041`, percentile 100%.
  Their null p2.5/median/p97.5 intervals are recorded in
  `outputs/chios/null_test_v2_summary.txt`. All damage-grade contrasts are
  reported against those same null distributions; each is in a tail at the
  two-sided resolution floor `p=0.004283`.
- These are empirical tail findings for this placement process, not a sign
  decision. Placements overlap each other and are not independent, so the
  p-values are not exact independent-replicate significance tests. The
  reference direction was examined; any later sign selection is label-informed
  and must report both `burn_direction` variants.
- New deliverables are `outputs/chios/null_test_v2.csv`,
  `outputs/chios/null_test_v2_summary.txt`, four null histograms with the real
  values marked, `outputs/chios/null_test_v2.log`, and
  `outputs/chios/null_memory_test_v2.log`. Existing Chios outputs were left in
  place.
- The final full suite passes **140 tests**; syntax compilation and
  `git diff --check` pass. The worktree has only the requested documentation,
  diagnostic, and focused-test modifications plus the pre-existing unrelated
  untracked `.vscode/`.
- The measured overall and grade-level results, interpretation, and the
  466-placement finite-sample/dependence limits are now recorded in
  `docs/DESIGN.md`. Graphify remains pending so it can be refreshed only at the
  end as requested.
- Final revalidation after the documentation edits: `.venv-qgis/bin/python -m
  pytest -q` passes **140 tests in 3.40 seconds**; `git diff --check` passes.
  `graphify-out/graph.json`, the saved Graphify interpreter marker, and the
  `graphify` executable are present, so the incremental refresh can proceed.
- The normal `graphify . --update` path correctly detected 3 changed code files
  and 2 changed documents, but stopped before extracting because no supported
  LLM API key is configured. Per Graphify's documented fallback, continue with
  host-agent semantic extraction for the two changed documents; do not request
  or add an API key.
- Reproduced the incremental scan state from disk: 2 changed code files,
  2 changed documents (`docs/DESIGN.md` and `docs/STATE.md`), no deletions, and
  31 corpus files total. The CLI's third code entry was a manifest-stamped
  code file that the CLI re-queued because it has no nodes in the current graph.
- The changed-code AST extraction completed for the two changed files: **69
  nodes and 220 edges**. Document semantic extraction is running separately;
  the graph merge has since completed; final report/visualization regeneration
  remains pending.
- Both changed documents miss Graphify's semantic cache. The existing graph
  also confirms the CLI's requeued code source is `.vscode/settings.json`; it
  remains unmodified and will only be included in local AST extraction so the
  graph update does not leave that already-manifested source unresolved.
- Re-ran the AST pass with the two changed source files plus the requeued
  `.vscode/settings.json`. The local extractor produced **69 nodes and 220
  edges**; the settings file itself contributed no AST nodes.
- Rejected the first host-agent semantic extraction as incomplete: it had read
  297 of the then-current 307 lines of this state file while progress entries
  were being appended. None of its partial state-file nodes are being used. The
  state file is now stable for the final complete extraction.
- The complete manual semantic fallback validated 241 lines of `DESIGN.md` and
  311 lines of `STATE.md`, yielding 30 document nodes and 24 edges; Graphify's
  token usage was not exposed, so token totals are recorded as unavailable.
- Incremental `build_merge` replaced 91 nodes from the re-extracted sources and
  removed 2 source-less nodes, leaving **509 nodes and 1,354 edges**. The graph
  diff reports 61 new nodes/146 new edges and 60 removed nodes/111 removed
  edges. Integrity diagnostics report no dangling or missing endpoints,
  self-loops, or collapsed edges.

## Next actions

1. Complete the final Graphify synchronization of this status snapshot, then
   stop; do not start Checkpoint C.

## Open questions

- Empirical contrasts are in the placement-null tails, but interpretation is
  limited by overlapping, dependent placements and is not an exact
  independent-replicate significance test.
- No final burn direction is selected; any later sign decision must disclose
  the one-time comparison against the reference direction.

## Not done

- No final burn direction selected; no accuracy metrics computed.
- Checkpoint C has not started.
- Graphify outputs remain ignored; no tracked Graphify files have been added.
- No commit made; the user will commit.

## Graphify finalization correction

- The first `graphify cluster-only .` invocation loaded the prior 508-node,
  1,319-edge `graph.json`. `build_merge` had prepared the updated
  512-node/1,320-edge extraction but does not itself export that extraction to
  `graph.json`; therefore that first report/HTML regeneration did not include
  the current task's final graph changes.
- The curated-graph export guard identified eight prior semantic nodes from
  this state file that the concise fresh extraction omitted. Each still
  describes a fact present in the current state; retain those nodes and their
  existing edges in the final export rather than dropping valid graph content.
- A subsequent export attempt identified six additional prior state nodes not
  included in the fresh extraction. Those concepts are also supported by the
  current state and will be retained alongside the original eight.
- The corrected export included the complete merged extraction, the current
  state semantics, and the eight still-supported prior state nodes/edges. It
  reported **517 nodes, 1,329 edges, and 23 communities**; the regenerated
  report and HTML exist, and the post-build integrity check found no missing,
  dangling, self-loop, or collapsed edges.
- This status update is the final documentation edit; the final Graphify
  synchronization will include it. No Checkpoint C or burn-direction change
  is involved.
