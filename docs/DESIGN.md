# Design decisions

## Project goal

Produce burned/unburned evidence by possibility-theory fusion of pre/post Sentinel-2 optical and Sentinel-1 SAR change inputs. The frame of discernment is `{Burned, Unburned}`. This project uses possibility theory (including the Dubois-Prade adaptive rule), not Dempster-Shafer evidence theory.

## Input and preprocessing assumptions

- Sentinel-2 inputs are L2A B8A and B12 on their native 20 m grid. Core index functions consume surface reflectance, scaled from SAFE DNs by `(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE` after treating DN 0 as NoData. For baseline N0510, the offset is `-1000` and quantification value is `10000`. The index implementation warns when values look unscaled (for example, a maximum above 1.5); it does not silently rescale inputs.
- `NBR = (B8A - B12) / (B8A + B12)` and `dNBR = NBR_pre - NBR_post`; larger positive dNBR indicates more burn evidence.
- Sentinel-1 inputs are calibrated, terrain-corrected backscatter in dB. The SAR change sign is a parameter, `burn_direction` (`negative` or `positive`), with `negative` as the provisional VH default because a burned surface commonly has a backscatter drop. This default is a placeholder to calibrate against the study scene.
- Pre/post SAR inputs are assumed to use the same relative orbit, pass direction, and polarization.
- If SAR is finer than 20 m, resample in linear power by averaging, then convert the average back to dB. Nearest-neighbor resampling is not appropriate for this continuous backscatter input.
- All four acquisition dates (S2 pre/post and S1 pre/post) are required. Fire start/end dates are used for plausibility warnings and to determine whether post-fire temporal asymmetry is eligible.
- Optional reliability rasters use `1` for fully reliable and `0` for unusable. A cloud-probability raster is inverted (`alpha = 1 - cloud_probability`) before discounting. If no reliability raster is supplied, scalar `alpha = 1` is used. When both pre- and post-fire reliability values are present, combine them with `min`.
- NoData in any required input propagates to NoData in every output. NoData is never interpreted as zero or as total ignorance.

## Temporal and reliability decisions

- Temporal reliability is `alpha_time = exp(-abs(dt) / tau_days)`, where `tau_days` is positive.
- Post-fire asymmetry applies only to the earlier source in the post-fire pair and only when that acquisition predates `fire_end`; absent either date, or when the earlier acquisition is on/after fire end, the source gets uniform `k=1` discount. When eligible, exponent `k >= 1` sets the earlier source's Burned possibility floor to `1 - alpha_sensor * alpha_time**k`, while its Unburned possibility floor is `1 - alpha_sensor` because burned scars persist. The later source receives uniform sensor/time discount. Pre-fire pairs never use this asymmetric adjustment.
- Sensor and temporal reliability combine by configurable minimum or product; product is the default. Reliability rasters use 1 for fully reliable and 0 for unusable.
- Pre-fire acquisition-date differences warn above a configurable threshold because of possible phenological/seasonal change. Warn when post-fire acquisitions predate fire end or image/fire dates are otherwise implausible.
- A fire-progression/FIRMS-derived date raster remains a future extension point only.

## Architecture

- `dst_fusion/core/` is pure Python and NumPy-only for numerical work; it must not import QGIS. Core functions will be tested independently with pytest.
- `dst_fusion/io/` owns raster reads/writes, NoData handling, reprojection, the footprint intersection, 20 m grid alignment, and block-wise processing using GDAL/QGIS APIs.
- `dst_fusion/algorithms/` contains the QGIS Processing algorithms and parameter wiring.
- `tests/` contains unit tests. `pytest.ini` configures the repository's test discovery without adding runtime dependencies.
- The plugin shell registers a Processing provider; no algorithms or fusion math are implemented in this milestone.
- QGIS 3.28 is the initial minimum version declared in plugin metadata; the available local QGIS is 3.44.7.

## Possibility core — milestone 1

- `core/possibility.py` accepts two-class possibility arrays shaped `(2, H, W)` in `[Burned, Unburned]` order. Values and outputs are `float32`; valid possibility distributions have per-pixel maximum 1. `normalize_possibility` converts bounded class memberships into that normalized representation, preserving `(1, 1)` as total ignorance and rejecting an all-zero valid pixel.
- Reliability discounting is `pi'[c] = max(pi[c], 1 - alpha)`. Alpha can be a scalar or `(H, W)` float reliability raster in `[0, 1]`; it is not a probability of unreliability. Thus alpha 0 produces total ignorance and alpha 1 leaves the distribution unchanged.
- Conjunctive fusion starts with per-class `min(pi1, pi2)`, then normalizes by `h = max_c min(pi1[c], pi2[c])`. Disjunctive fusion is per-class `max(pi1, pi2)`. Both return the shared conflict metric `1 - h`.
- Adaptive fusion implements `max(min(pi1, pi2) / h, min(max(pi1, pi2), 1 - h))` per class. The named module constant `H_EPSILON = 1e-12` guards near-zero `h`: the undefined division is omitted and the result uses disjunction, the limiting normalized rule. Conjunctive fusion uses the same disjunctive fallback when `h <= H_EPSILON` because an all-zero or numerically negligible conjunction cannot be normalized. This preserves the invariant that valid output possibility distributions have a maximum of 1 rather than returning zero possibilities. Conflict remains `1 - h` and reports this high-conflict case.
- Necessity is `1 - runner_up_possibility`. A tied top pair or necessity below configurable `necessity_threshold` (default 0) returns int8 class code `2` (undecided); otherwise the selected class is `0` (Burned) or `1` (Unburned). NoData uses class code `-1`.
- NaN is the NoData representation in the numerical core. If either class in a source pixel is NaN, the complete output pixel (both possibility classes, conflict or necessity, and class decision) is NoData; NoData is never treated as zero or ignorance. Infinite values, out-of-range values, mismatched grids, and non-normalized fusion inputs raise `ValueError`.
- Tests cover the hand-computed adaptive case `(1, 0.3)` fused with `(0.5, 1)` (`h=0.5`, adaptive result `(1, 0.6)`, conflict `0.5`), agreement, conflict, total ignorance, reliability endpoints, near-zero intersection, NoData, and output normalization for all three rules.

## Synthetic validation — milestone 2

- `examples/synthetic_scene.py` creates a deterministic 100x100 circular burned reference with independent noisy optical and SAR burn-score fields. Optical evidence is unreliable beneath a rectangular cloud/smoke mask and becomes total ignorance after alpha discounting.
- The synthetic dates differ by four days: the optical acquisition is earlier than SAR. `core/temporal.discount_for_date_offset` applies the date-offset discount, and the output explicitly reports the offset and earlier source.
- Optical-only, SAR-only, and adaptive fused classifications use the shared core decision function. Undecided pixels count as not predicted burned for IoU/F1, and reference/predicted burned-pixel counts and undecided fraction are reported separately inside and outside the cloud.
- The seeded 20-scene run reports whole-scene mean and population standard deviation of IoU, plus mean/std IoU and F1 and burned counts inside and outside the cloud. Tests require fused classifications to equal SAR-only under total optical ignorance, and fused outside-cloud IoU to be at least the better single source for every tested seed.
- A growing-fire variant expands the reference blob between the earlier optical and later SAR dates. It reports overall and growth-ring fused IoU for temporal asymmetry exponents `k=1` and `k=2`, with mean/std across the same seeds. For the current fixed setup, k=2 improves both overall IoU (`0.6940` to `0.9104`) and growth-ring IoU (`0.2192` to `0.8944`); the script derives and prints those comparisons rather than assuming improvement.
- `outputs/synthetic/` is gitignored and contains the evidence plot, classification comparison, growing-fire comparison, and a JSON record of seeds, dates, temporal reliability, regional counts/metrics, and k comparisons. Run with `python -m examples.synthetic_scene`.

## Indices, memberships, and temporal reliability — milestone 3

- `core/indices.py` expects aligned 2D B8A/B12 bands in scaled Sentinel-2 L2A surface reflectance; the `/10000` scaling and processing-baseline BOA `-1000` offset are assumed already applied. It warns if any finite value exceeds `1.5` as a likely unscaled-input indicator; it never rescales bands implicitly.
- `NBR = (B8A - B12) / (B8A + B12)` and `dNBR = NBR_pre - NBR_post`. Any input NoData (NaN) propagates, and a denominator whose absolute value is at most `1e-6` becomes NoData rather than producing an unstable ratio.
- Trapezoids first produce a bounded burned-membership score; its complement is the unburned score. The two scores are then normalized by their per-pixel maximum to form valid possibilities in `(2,H,W)` order `[Burned, Unburned]`. Thus the larger class is exactly 1; intermediate values are relative possibilities, not probabilities. Trapezoids support infinite outer breakpoints to express shoulders.
- The default dNBR right-shoulder rise starts at `0.10` and reaches full burned possibility at `0.27`, using the USGS dNBR unburned/low and low/moderate severity boundaries as a binary burned-evidence transition; this is not a severity-class classifier. Default SAR breakpoints are oriented change magnitudes `(0.5, 1.5, inf, inf)` dB, provisional placeholders to calibrate per scene. `burn_direction` is `negative` by default for VH and can be `positive`.
- Otsu suggestion uses a histogram between-class-variance split, then recommends a right-shoulder ramp one histogram bin wide below the selected threshold. NaN values are excluded; empty or constant inputs raise `ValueError`.
- Temporal reliability is `alpha_time = exp(-abs(dt) / tau_days)`, with positive `tau_days`. Sensor and temporal reliability combine by configurable `min` or `product` (default `product`).
- Post-fire asymmetry is enabled by default only when the earlier source's acquisition date is before `fire_end`; otherwise it receives the uniform `k=1` temporal discount. The caller must provide both `source_date` and `fire_end_date`; with neither date supplied, asymmetry is gated off, and supplying only one is an error. The configurable exponent defaults to `k=2.0` and must be at least 1. For an eligible earlier source, `alpha_B = alpha_sensor * alpha_time**k` and `alpha_U = alpha_sensor`, giving floors `pi'(Burned) = max(pi(Burned), 1 - alpha_sensor * alpha_time**k)` and `pi'(Unburned) = max(pi(Unburned), 1 - alpha_sensor)`. Raising the Burned possibility floor weakens the evidence for Unburned during an active fire; that extra discount is a design rule, not tuned against the reference. The later source receives uniform sensor/time discount using the configured combination. At `dt=0`, neither source is earlier and no temporal discount is applied beyond sensor reliability. Pre-fire pairs never use this asymmetric rule.
- The pre-fire S1/S2 date gap warns above a configurable threshold (default 30 days) because of possible phenological change. Date validation also warns for pre/post order errors, pre-fire acquisitions on/after fire start, post-fire acquisitions before fire start/end, and invalid fire chronology.
- `firms_temporal_reliability` is an explicit `NotImplementedError` extension point. Per-pixel date decoding and date-raster NoData conventions remain unselected and are not simulated as completed behavior.

## Raster IO and alignment — milestone 4

- `dst_fusion/io/raster.py` is the GDAL-only raster boundary; pure numerical code under `core/` remains free of QGIS/GDAL imports.
- The Sentinel-2 reference raster defines the target CRS, origin, and 20 m square-pixel grid. All raster footprints are transformed to that CRS; their extent intersection is snapped inward to complete reference pixels so resampling does not expand coverage beyond the common footprint.
- Continuous inputs are warped in bounded output windows using GDAL average resampling. When source and target have the same CRS, resolution, and integer-pixel origin shift, the IO layer reads/copies matching source windows directly without resampling. SAR dB inputs use that same exact-copy path when grids match; otherwise they are converted to linear power in source blocks, averaged/reprojected in linear power, then converted back to dB per output block. Nearest-neighbor is never used for SAR.
- `read_raster_window(path, (xoff, yoff, width, height))` reads native-type pixel windows directly. Callers reading Sentinel-2 JP2s use the tile frame's offsets, not offsets from a clipped intersection grid.
- Aligned outputs are Float32 GeoTIFFs with NaN NoData. `iter_aligned_blocks` requires identical dimensions, geotransforms, and CRS, and applies a shared invalid-pixel mask: if any required input is NoData, every array in that output block is NaN at that pixel.
- Source conversion, warp, block reading, and output writes are all block/window based to bound memory use. Rotated grids and non-meter/non-projected reference CRSs are rejected rather than silently misaligned.
- Generated GeoTIFF tests live in `tests/test_io.py`. They use `pytest.importorskip` for GDAL and OSR so environments without `osgeo` skip the integration module cleanly. The unified `.venv-qgis` development environment uses `/usr/bin/python3` 3.12.3 with system site packages (QGIS 3.44.7, GDAL 3.8.4, NumPy 1.26.4, and Matplotlib 3.6.3) plus pytest. On this host, system Python has no `ensurepip`; the exact setup and test commands are recorded in `requirements-dev.txt`.
- `align_rasters` bypasses GDAL warp for same-lattice integer-pixel shifts and writes source-aligned windows block by block. Sub-pixel shifts or differing resolutions continue through average resampling; SAR uses linear power for that resampling path.

## Module map — milestone 4

- `dst_fusion/__init__.py` is QGIS's `classFactory` entry point.
- `dst_fusion/plugin.py` manages plugin lifecycle and registers `DstFusionProvider`.
- `dst_fusion/provider.py` is the Processing provider shell; algorithm registration is deferred.
- `dst_fusion/core/possibility.py` implements normalized two-class possibility arrays, reliability discounting, three fusion rules, conflict, and class/necessity/undecided decisions.
- `dst_fusion/core/indices.py` computes NBR/dNBR from reflectance bands and handles scale warnings and NoData.
- `dst_fusion/core/memberships.py` implements trapezoid memberships, burned/unburned possibilities for dNBR and SAR change, and Otsu breakpoint suggestions.
- `dst_fusion/core/temporal.py` computes temporal reliability, combines sensor reliability, applies the post-fire asymmetric date-offset rule, reports date plausibility warnings, and reserves the FIRMS date-raster hook.
- `dst_fusion/io/raster.py` inspects rasters, intersects footprints, aligns/resamples inputs, handles SAR in linear power, and reads aligned blocks with shared NoData propagation.
- `dst_fusion/algorithms/` reserves QGIS Processing algorithm wrappers.
- `examples/synthetic_scene.py` compares optical-only, SAR-only, and fused decisions over 20 seeds, reports regional burned counts and metrics, and compares k=1/k=2 in a growing-fire scenario.
- `examples/audit_chios.py` prints raster/vector metadata, checks the delineation-bbox-plus-5-km and delineation windows, compares EMS dNBR classes, and saves an ignored post-fire RGB preview. Raster reads are AOI-windowed; source data is not modified.
- `examples/run_chios.py` reads only the delineation-bbox-plus-5-km AOI window, builds the SCL/layover-shadow reliability masks, optionally boxcar-filters SAR in linear power, selects Otsu or USGS dNBR breakpoints and AOI-only Otsu SAR breakpoints, compares temporal/speckle/fusion-rule settings, and writes GeoTIFFs plus previews. It does not score or tune against reference labels.
- `tests/` covers the core math, synthetic scenarios, and generated GeoTIFF alignment; GDAL tests skip if system bindings are unavailable.

## Chios 2024 real-data audit — Checkpoint A

- Source metadata is read directly from the R20m Sentinel-2 JP2s and the two HyP3 RTC folders under the gitignored `data/chios2024/`. The system/QGIS Python has GDAL 3.8.4 with `JP2OpenJPEG`; Miniforge Python has no `osgeo` module. Both JP2 scenes use EPSG:32635, 5490x5490 pixels, 20 m spacing, and origin `(399960, 4300020)`. Their MTD_MSIL2A.xml files specify `BOA_ADD_OFFSET=-1000` and `BOA_QUANTIFICATION_VALUE=10000`; DN 0 is set to NoData before applying `(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE`. Negative values are preserved and counted, not clipped. In the corrected AOI audit no valid B8A/B12 pixels were negative.
- The common four-science-input footprint (S2 pre/post B8A and S1 pre/post VH) is a 5490x4170 EPSG:32635 grid at origin `(399960, 4273620)`. It contains the FireDelineation geometry. The rectangular fire-bbox-plus-5-km AOI window is `(xoff=191, yoff=674, width=702, height=757)`, extent `(403780, 4245000, 417820, 4260140)` in EPSG:32635.
- All listed 20 m science, auxiliary, aligned-copy, and EMS dNBR rasters have origin offsets congruent to the S2 grid modulo 20 m. S1 pre/post differ by exactly four columns (`post` origin is 80 m west) and zero rows; they occupy the same lattice. The layover/shadow masks are Byte rasters. HyP3's README defines bit 4 as layover and bit 16 as shadow; the AOI contains codes 1, 5, and 17, with either flag on about 0.21% of window pixels. No incidence-angle raster is present.
- S1 raster names and source granules provide the dates, VH/VV polarizations, and absolute orbit identifiers 054533 (pre) and 054708 (post). Relative orbit 36 for each is derived from the absolute orbit with the supplied S1A formula `(absolute - 73) mod 175 + 1`; descending pass is derived from the approximately 04:15 UTC acquisition time. These are derived values, not fields read from a supplied manifest. Both acquisitions have the same VH/VV polarization set.
- The hand-aligned July VH/VV copies were compared against the IO layer's exact integer-window copy path over 529,723 mutually valid AOI pixels per band. VH and VV both have zero differing pixels and maximum absolute difference `0 dB`; this uses the same direct-window behavior as the newly added IO fast path.
- For Checkpoint B, `*_ls_map.tif` is the planned SAR reliability mask: set `alpha_sar=0` when `(mask & (4 | 16)) != 0` (layover or shadow), and otherwise use the configurable scalar SAR reliability, default `1.0`. No incidence-angle input is available.
- The AOI has two coordinate frames: `(191, 674, 702, 757)` on the clipped intersection grid and `(191, 1994, 702, 757)` on the full S2 tile. The audit now reads JP2s using the latter tile offsets. The previous audit mistakenly used intersection-grid row offsets as tile offsets; that sampled a different, sea-heavy area and produced the erroneous 93% class-6 values. Corrected SCL fractions over the whole buffered rectangle are pre class 3 0%, 6 33.6579%, 8 5.4449%, 9 0.0006%, 10 0%; post class 3 0%, 6 36.3515%, 8 2.7408%, 9 0%, 10 0%. The remainder of the rectangle is largely land and other SCL classes, so water is still a substantial fraction but not 93%.
- In the delineation polygon rasterized with GDAL's pixel-center rule (28,090 pixels), pre SCL fractions are class 4 23.8590%, 5 76.1339%, 6 water 0.0036%, 8 0.0036%; post fractions are class 4 0.4165%, 5 96.7497%, 6 water 0.0036%, with 0% for classes 3, 8, 9, and 10. This confirms water is effectively absent inside the delineation. The post-fire B04/B03/B02 10 m RGB preview with yellow delineation outline is saved to the gitignored `outputs/chios/chios_s2_post_delineation_rgb.png` for visual inspection; the audit does not alter any mask based on the preview.
- The full inside-delineation SCL breakdown accounts for the prior 97.2% post-fire class-4/5 total: class 2 (dark area pixels) is 2.8266%, class 4 is 0.4165%, class 5 is 96.7497%, class 6 is 0.0036%, and class 7 is 0.0036%; all other classes are 0%. Pre-fire is class 4 23.8590%, class 5 76.1339%, class 6 0.0036%, class 8 0.0036%, and all other classes are 0%. The approximately 93% class-6 result from the earlier audit was caused by using intersection-grid row offsets to read the full S2 tile; it sampled a different, sea-heavy window. The corrected tile-coordinate window is `yoff=1994`. The audit reports the complete class fractions and does not change the water mask.
- Raw class-6 water **DN** percentiles (not reflectance) are not close to 1000. More than 1 km from classified land, B12 p1/p50/p99 are 1610/1899/2353 pre and 1610/1844/2297 post; B8A values are 1695/2048/2595 pre and 1705/1985/2523 post. After applying `(DN - 1000) / 10000`, the respective median reflectances are B12 0.0899/0.0844 and B8A 0.1048/0.0985 (pre/post). The RGB preview shows strong spatial streaking/reflection over the sea, compatible with possible sunglint, but the values and preview do not establish the cause. Water remains masked; no water-mask change is made.
- Correctly scaled AOI B8A/B12 ranges are pre 0.0465–0.5056 / 0.0501–0.5499 and post 0.0627–0.5094 / 0.0526–0.5571. There were no DN=0 pixels, negative values, or values above 1.5 in these windows. Mean class-6 water vs class-4/5 land reflectance was pre B8A 0.11564 vs 0.26352, B12 0.09985 vs 0.19599; post B8A 0.11156 vs 0.25483, B12 0.09610 vs 0.19852. These low water-band means are a sanity check on the corrected scaling.
- The EMSN198 dNBR GeoTIFF is a Byte graded/classification product with no NoData tag, not a continuous dNBR measurement. Comparing our dNBR ranks against its class codes over only its 202x257 extent gives Spearman rho `0.901240` over 51,914 valid pixels. Our mean dNBR by EMS class: 0 `0.004610` (n=23,824), 1 `0.086490` (944), 2 `0.202734` (8,924), 3 `0.327040` (9,487), 4 `0.470076` (6,938), and 5 `0.676202` (1,797). This ordered class trend supports the reflectance sign and date-pairing sanity check; the Byte classes are not treated as continuous values.
- Dates from the input names are S2 pre 2024-06-28 09:06:01, S1 pre 2024-06-29 04:15:46, S2 post 2024-07-08 09:06:01, and S1 post 2024-07-11 04:15:46. The pre-fire gap is one calendar day; S2 post is three calendar days earlier than S1 post. The user-specified fire dates are 2024-07-01 to 2024-07-03.

## Chios AOI pipeline — Checkpoint B

- Run with `.venv-qgis/bin/python -m examples.run_chios` from the repository root. The script reads 702x757 pixels (intersection-grid offsets `(191,674)`, S2 tile offsets `(191,1994)`) from each source; it does not load full Sentinel scenes. Common valid land requires all four science inputs and both SCL rasters to be valid, excludes SCL class 6 water and class 0, and propagates NoData to every output. Optical alpha is the minimum of pre/post SCL reliability; classes 3, 8, 9, and 10 receive configurable alpha (default 0). SAR alpha is the configured scalar (default 1) except where either HyP3 mask has bit 4 or 16 set; auxiliary NoData propagates.
- SAR boxcar filtering (default 5x5) averages dB-converted linear power with NaN-aware local means, then converts back to dB before calculating post-minus-pre VH change. Both filtered and unfiltered cases run by default. Temporal discount likewise runs on/off by default; `alpha_time=exp(-3/14)=0.807118`, giving a uniform sensor-alpha-1 floor of 0.192882 when enabled and 0 when disabled. S2 is the earlier post-fire source, but 2024-07-08 is after `fire_end=2024-07-03`; the fire-end gate therefore disables extra k=2 asymmetry. The k=1 and k=2 outputs are identical here by design, not because k=2 was omitted or tuned.
- USGS dNBR breakpoints `(0.10, 0.27, inf, inf)` are the primary membership thresholds. The labelled AOI Otsu dNBR variant is `(0.193888, 0.198580, inf, inf)`. The default SAR membership option is label-free robust scale: median plus one and two scaled MAD standard deviations, after optional AOI-land median centering and sign orientation. Otsu thresholds are not used for SAR because its AOI-land change histograms are descriptively unimodal. No fire/reference labels select either membership breakpoint set.
- The one-time SAR sign assessment below compares with the delineation and damage-grade reference, so it is label-informed evidence. This comparison does not select or tune a final `burn_direction`; any later sign decision must disclose that it is label-informed. The raw/unfiltered and 5x5-linear-power-filtered statistics use common valid land in the buffered AOI window (28,069 inside; 299,093 outside). Effect size is the mean difference divided by the pooled standard deviation. dB units apply to every value except counts and the dimensionless effect size.

| Polarization/mode | Inside mean / median / SD (dB) | Outside mean / median / SD (dB) | Inside − outside (dB) | Effect |
|---|---:|---:|---:|---:|
| VH unfiltered | +0.34094 / +0.43036 / 2.49012 | +0.18610 / +0.22109 / 2.07344 | +0.15484 | +0.0733 |
| VH 5x5 | +0.36963 / +0.55049 / 1.12696 | +0.19447 / +0.22800 / 0.66806 | +0.17516 | +0.2436 |
| VV unfiltered | +0.73810 / +0.75435 / 2.20220 | +0.09404 / +0.15795 / 1.79404 | +0.64407 | +0.3514 |
| VV 5x5 | +0.74907 / +0.83192 / 0.90593 | +0.09858 / +0.17939 / 0.83362 | +0.65049 | +0.7743 |

- By grade, each entry gives inside mean/median/SD, followed by difference and pooled effect size against the same all-outside-fire land sample. Since the outside sample is identical across grades, its mean/median/SD for each polarization/filter are those in the preceding overall table.

| Grade | Pol./mode | n | Inside mean / median / SD (dB) | Difference (dB) | Effect |
|---|---|---:|---:|---:|---:|
| High | VH unfiltered | 1,797 | -0.35621 / -0.25833 / 2.69625 | -0.54230 | -0.2610 |
| High | VH 5x5 | 1,797 | -0.17437 / +0.00550 / 1.23817 | -0.36884 | -0.5481 |
| High | VV unfiltered | 1,797 | +0.56716 / +0.62165 / 2.48421 | +0.47312 | +0.2630 |
| High | VV 5x5 | 1,797 | +0.65722 / +0.81844 / 1.02688 | +0.55863 | +0.6691 |
| Moderate high | VH unfiltered | 6,938 | -0.26202 / -0.16724 / 2.84857 | -0.44811 | -0.2140 |
| Moderate high | VH 5x5 | 6,938 | -0.13865 / +0.06936 / 1.37991 | -0.33312 | -0.4811 |
| Moderate high | VV unfiltered | 6,938 | +0.46481 / +0.54471 / 2.41475 | +0.37077 | +0.2048 |
| Moderate high | VV 5x5 | 6,938 | +0.50832 / +0.65584 / 1.09297 | +0.40974 | +0.4876 |
| Moderate low | VH unfiltered | 9,478 | +0.52856 / +0.59226 / 2.41021 | +0.34246 | +0.1643 |
| Moderate low | VH 5x5 | 9,478 | +0.52364 / +0.65623 / 1.03653 | +0.32917 | +0.4824 |
| Moderate low | VV unfiltered | 9,478 | +0.91656 / +0.90182 / 2.18216 | +0.82252 | +0.4551 |
| Moderate low | VV 5x5 | 9,478 | +0.91090 / +0.98133 / 0.86385 | +0.81231 | +0.9733 |
| Low | VH unfiltered | 8,912 | +0.74462 / +0.74901 / 2.13795 | +0.55853 | +0.2691 |
| Low | VH 5x5 | 8,912 | +0.70705 / +0.74583 / 0.77711 | +0.51259 | +0.7634 |
| Low | VV unfiltered | 8,912 | +0.83629 / +0.81469 / 1.98534 | +0.74226 | +0.4124 |
| Low | VV 5x5 | 8,912 | +0.81984 / +0.83503 / 0.71310 | +0.72125 | +0.8686 |

- Outside-fire global offsets (mean/median, dB) are VH unfiltered `+0.18610/+0.22109`, VH 5x5 `+0.19447/+0.22800`, VV unfiltered `+0.09404/+0.15795`, and VV 5x5 `+0.09858/+0.17939`. AOI-valid-land median centering offsets are respectively `+0.23305`, `+0.24335`, `+0.19278`, and `+0.20453 dB`. Centering is optional and switchable (`--center-sar`); it is label-free and occurs before SAR membership construction. All four AOI-land histograms returned one prominent mode with the documented smoothed-histogram diagnostic; this is descriptive, not a formal test. For a unimodal SAR distribution, Otsu is not meaningful; a transparent label-free alternative is breakpoints at one and two robust standard deviations (`1.4826 × MAD`) from the AOI-land median.
- The previous AOI-only null sampler accepted **0/200** placements because the EMSN198 AOI was too small for a non-overlapping same-shape mask. This was a placement-feasibility failure, not evidence about distinguishability. That AOI-restricted design is superseded by the full-intersection design below.
- Two non-scored pipeline variants use AOI-land centering, 5x5 filtering, temporal discount on, USGS dNBR, and robust SAR breakpoints. The pipeline's own VH-valid-land median centering offset is `+0.24349 dB` for the filtered change (its valid mask differs slightly from the stricter shared VH/VV diagnostic mask above). On 327,519 valid pixels, negative direction gives 8,885 Burned / 291,604 Unburned / 27,030 Undecided and mean conflict 0.08519; positive gives 9,094 / 295,747 / 22,678 and mean conflict 0.07396. These are map summaries, not validation or accuracy metrics. A separate negative-direction Otsu dNBR variant uses `(0.193888, 0.198580)` and gives 8,762 / 288,752 / 30,005 with mean conflict 0.08723.
- Side-by-side sign preview: `outputs/chios/variants/sign_comparison.png`. Per-variant rasters, previews, logs, counts, and the labelled Otsu variant are under `outputs/chios/variants/`. The SAR diagnostics, histogram, memory test, CSV, and null summary are saved under `outputs/chios/`.
- `check_temporal_plausibility` reports no warnings for the supplied acquisition and fire dates. AOI-only valid land count is 327,519 of 531,414 window pixels. Under the default boxcar + temporal-on run, the adaptive output has 12,509 Burned, 182,094 Unburned, and 132,916 Undecided pixels; mean conflict is 0.33367. Temporal-off changes those counts to 12,591 / 184,029 / 130,899 and mean conflict 0.41139. With no speckle filter, temporal-on gives 14,074 / 165,015 / 148,430 and mean conflict 0.37294; temporal-off gives 14,195 / 167,232 / 146,092 and mean conflict 0.45970. These are descriptive map totals, not accuracy metrics. The disjunctive rule is more conservative in these outputs; no rule was chosen using reference labels.
- AOI evidence histograms are saved as `outputs/chios/chios_aoi_histograms.png`; the seven-panel original preview is `outputs/chios/chios_fusion_preview.png`. GeoTIFFs include adaptive, conjunctive, and disjunctive classes, optical-only and SAR-only classes, necessity, conflict, and two-band Burned/Unburned possibilities. Class codes are 0 Burned, 1 Unburned, 2 Undecided, and -1 NoData. All generated outputs remain in gitignored `outputs/chios/`.

## Full-intersection SAR placement null — Option A

- The null domain is every valid land pixel in the 5,490 × 4,170 full S2/S1
  intersection, not the EMSN198 AOI. Inputs and evidence are computed once in
  256 × 256 blocks and written to temporary disk-backed GeoTIFFs; no full-scene
  NumPy copies are retained. The fire mask is rasterized once and cropped to
  its bounding box. Placements use whole-pixel offsets with a hard
  500-attempt cap per replicate.
- Candidate acceptance requires valid non-water/non-NoData land, at least
  500 m from SCL water/NoData, no overlap with the real delineation or any EMS
  grading polygon after a 500 m exclusion buffer, clear optical alpha exactly
  1, mean dNBR below 0.10, fewer than 5% of pixels with dNBR ≥ 0.10, pre-fire
  SCL class-4/5 shares within a configurable absolute tolerance (default 0.10)
  of the real fire's class-4/5 mix, and no more than 1% layover/shadow flags.
  The reference SCL mix is calculated from pre-fire SCL within the rasterized
  fire mask, independent of SAR quality masks.
- The statistic is defined identically for each real or placed mask and each
  SAR variant: the mean post-minus-pre SAR change inside the mask minus the
  mean over the remaining valid land outside the real-fire 500 m buffer.
  For a null placement, that placement's pixels are removed from its own
  outside sample. The four variants are VH/VV, each unfiltered and 5×5
  linear-power boxcar filtered. Subtracting the same median from both groups
  leaves a difference of means unchanged; median centering therefore does not
  affect this statistic.
- The null table and text summary are flushed incrementally. Each row stores
  scalar placement metadata and four null contrasts; only those four scalar
  distributions are retained for inference. Progress is reported every ten
  replicates, with both accepted replicates and total offset attempts. The
  requested target is 500 replicates; if fewer than about 100 are accepted,
  all estimates use and report the smaller sample.
- Each observed overall and damage-grade contrast is compared with the same
  four placement distributions. The two-sided empirical p-value is
  `min(1, 2 × min(p_lower, p_upper))`, where each one-sided tail is
  `(1 + number of null values at least as extreme in that tail) / (N + 1)`.
  The z-score is `(observed − null mean) / null sample SD`; the reported
  percentile is `100 × P(null ≤ observed)`, alongside null 2.5/50/97.5
  percentiles. Values in the bulk are described plainly as not distinguishable
  from background at this scale and event. Tail values identify the
  polarization/filter only; they do not choose a burn sign.
- Placements are allowed to overlap each other. They are consequently not
  independent replicates: the empirical p-value is a descriptive tail
  position for this placement process, not an exact independent-replicate
  significance test. The one-sided tail resolution is `1/(N+1)` and the
  minimum two-sided value under the definition above is `2/(N+1)`; spatial
  dependence further limits inferential interpretation.
- The one-time SAR sign comparison against the delineation/reference was
  examined as part of this analysis. Any eventual sign selection is therefore
  label-informed and must be reported for both variants; no final
  `burn_direction` is selected here.
- Outputs are `outputs/chios/null_test_v2.csv`,
  `outputs/chios/null_test_v2_summary.txt`, four
  `outputs/chios/null_test_v2_<variant>.png` histograms with the real value
  marked, and `outputs/chios/null_memory_test_v2.log` for the constrained
  five-replicate memory trial.

### Option A run (2026-10-31)

- The constrained-memory run accepted **5/5** placements with peak RSS
  **685,004 KB** under the 8 GB virtual-memory limit. The full run completed
  **500** requested replicates, accepting **466** after 86,013 total offset
  attempts. All 466 successful placements contribute one scalar to each
  variant's null.
- Overall full-domain observed contrast and null results:

| Variant | Real contrast (dB) | Two-sided p | z | Percentile | Null p2.5 / p50 / p97.5 (dB) |
|---|---:|---:|---:|---:|---:|
| VH unfiltered | +0.12892 | 0.029979 | +2.4259 | 98.712% | -0.01393 / +0.05342 / +0.09261 |
| VH 5×5 | +0.14825 | 0.021413 | +3.4049 | 99.142% | -0.02148 / +0.04282 / +0.08288 |
| VV unfiltered | +0.58780 | 0.004283 | +22.2348 | 100.000% | +0.01766 / +0.06365 / +0.10103 |
| VV 5×5 | +0.59471 | 0.004283 | +23.0041 | 100.000% | +0.01462 / +0.05949 / +0.09433 |

- Each overall real contrast is in the empirical null tail. The damage-grade
  contrasts also fall in the tails (`p=0.004283` for each listed grade/variant);
  the grade-specific contrasts, z-scores, percentiles, and null intervals are
  recorded in `outputs/chios/null_test_v2_summary.txt`. The full-domain raw
  VH/VV contrasts differ from the prior AOI-window values because their
  comparator is now the valid land outside the real-fire buffer over the
  complete intersection.
- Thus, under this event- and scale-specific placement null, the observed SAR
  changes are distinguishable from the sampled background contrasts for VH
  and VV, both unfiltered and 5×5 filtered. This does **not** make a sign
  decision: the reference direction was examined, so any eventual sign choice
  remains label-informed and both variants must be reported. Spatially
  overlapping placements make these p-values descriptive rather than exact
  independent-replicate significance tests.

## Synthetic temporal-refactor metric change

- The earlier synthetic 20-seed mean fused IoU of `0.940962` (reported as 0.941) is reproducible from the pre-refactor code path: optical possibilities received only the cloud reliability discount, while SAR possibilities received the uniform `alpha_time=exp(-4/14)` discount.
- The refactored 20-seed path yields fused IoU `0.913180 ± 0.006259` (population standard deviation). A direct apples-to-apples 20-seed comparison with the former discount calls reproduced `0.940962 ± 0.005935` before the date-offset adjustment and `0.913180 ± 0.006259` after it. The 0.027782 mean decrease is explained by the newly applied earlier-source Burned floor `1 - alpha_time**2 = 0.4353` on optical evidence; later SAR discount is unchanged.
- The earlier explanation that the optical Unburned possibility array itself changed was imprecise. For a normalized two-class distribution, raising the earlier source's Burned floor weakens its support for Unburned as a decision, even when the Unburned possibility value remains numerically unchanged. Across all 20x10,000 pixels, optical Unburned values were unchanged and Burned values changed in 152,428 cells. Fused classifications changed in 893 pixels: 13 true-burned and 880 reference-unburned pixels became newly classified burned. Those 880 false positives are the expected cost of applying the asymmetry to a non-growing synthetic fire; this effect is a design-rule tradeoff, not evidence about real-fire performance. The synthetic caller now explicitly supplies its source and fire-end dates so the asymmetry is exercised only while the simulated fire is still active.

## Checkpoint C pre-registration

Checkpoint C was frozen in [`configs/checkpoint_c.json`](../configs/checkpoint_c.json) before scoring. The primary is VV with positive burn direction, a 5x5 linear-power boxcar, AOI-valid-land median centering, and label-free breakpoints at the oriented median plus one and two scaled MADs (`1.4826 × MAD`). Optical evidence uses USGS dNBR breakpoints `(0.10, 0.27)`. Optical alpha is the minimum of pre- and post-date SCL alpha; classes 3, 8, 9, and 10 set alpha to zero. Valid land means not SCL class 6 and not NoData on either date. `ls_map` bits 4 and 16 set SAR alpha to zero. Fusion uses the Dubois-Prade adaptive rule. The uniform temporal discount with `tau=14` days is applied; the asymmetric `k=2` discount is not applied because both post-fire images are after fire end.

The primary outcome is all-pixel burned-class IoU against `firedelineation`, counting undecided as not predicted burned. The two primary comparisons are primary fused minus optical-only and primary fused minus VV SAR-only, using paired spatial block bootstrap. All other results are secondary or descriptive. The pre-registered expectation is that optical-only will match or exceed fused on real clear-sky data because the reference is dNBR-derived; fusion's value is judged mainly through the `SYNTHETIC` cloud ablation and fused versus SAR-only.

The reference-informed sign choice is disclosed: VV positive was selected after examining its direction against the reference, VH positive is also reported, and VH changes sign by damage severity. Class counts for pipeline variants and SAR null test v2 were seen before freezing this config; no accuracy metrics were seen. The frozen variants include optical-only, VV and VH SAR-only, conjunctive and disjunctive fusion, temporal discount off, and the recorded Otsu dNBR breakpoints. A VH two-sided run using the absolute centered change is explicitly `EXPLORATORY`; it cannot replace the primary.

Scoring uses pixels inside EMSN198_AOI that are valid land and valid in every registered output and both reference variants. The references are the fire delineation and that delineation minus repaired `Unburned or Regrowth` grading polygons; geometry repair is done once. Rasterization uses pixel-center inclusion on the fixed 20 m grid. Metrics, undecided handling, prevalence baselines, grade-specific recall, and paired 500 m spatial-block bootstrap intervals (1,000 resamples) are specified in the config. For each bootstrap replicate, sum TP, FP, and FN over the sampled blocks first, then compute each metric ratio. Use paired blocks for comparisons; pixel bootstrap is prohibited. Each results file must record the SHA-256 of the exact frozen config bytes, the git commit hash containing that config, and the seed. No parameters, breakpoints, signs, rules, masks, or rasterization choices may change once scoring starts. Poor results will be reported as observed; any later change is a separately labelled exploratory run and never replaces the primary.

The reference is optical-derived (EMS grading versus dNBR Spearman `0.90`), so optical-only is favoured and SAR-only may be understated. The SAR null test shows that the SAR contrast is unusual relative to its sampled background, not that SAR detects burned pixels well. VH sign instability by severity is retained as a limitation, not resolved by selecting a different sign after scoring.

## Checkpoint C STEP 2 and STEP 3 results

The frozen config SHA-256 was `8074f639c2bdd2cd8433e9e67ec50987082b02f1830ba2035a9cbfd47bd5a648`, recorded commit `4e1b1619065c3756806e8878bb39ba599637c925`, and bootstrap seed `20261031`. The all-output, valid-land, AOI mask contains 111,191 pixels; against FireDelineation, burned prevalence is 25.244%. Primary fused VV IoU is `0.4962` (95% 500 m block-bootstrap CI `0.4328–0.5551`), versus optical-only `0.8233` (`0.7696–0.8737`) and VV SAR-only `0.3777` (`0.3219–0.4281`). Paired primary-minus-optical IoU is `-0.3271` (`-0.3994` to `-0.2622`); primary-minus-VV-SAR IoU is `+0.1184` (`+0.0955` to `+0.1456`). The second reference, FireDelineation minus repaired Unburned or Regrowth polygons, yields primary IoU `0.5124` and optical-only IoU `0.8519`. Full outputs, per-variant results, and provenance are in [`checkpoint_c_summary.txt`](../outputs/chios/checkpoint_c/checkpoint_c_summary.txt) and [`checkpoint_c_results.csv`](../outputs/chios/checkpoint_c/checkpoint_c_results.csv).

The real clear-sky result agrees with the pre-registered expectation: optical-only substantially outperforms fused against this optical-derived reference. This comparison is reference-dependent, not independent evidence that optical is universally superior; the EMS grading is closely related to dNBR (Spearman `0.90`), which favours optical and can understate SAR. The tested positive SAR sign was label-informed: class counts for pipeline variants and null test v2 had been examined before freezing, and the reference direction had been inspected. The null test establishes an unusual SAR contrast relative to sampled background, not reliable burned-pixel detection. This is one fire event and one acquisition pairing; generalization across events, sensors, seasons, and fire conditions remains untested. VV SAR-only grade recalls were `0.454` High, `0.401` Moderate high, `0.530` Moderate low, and `0.449` Low. They appear relatively flat compared with the larger VH sign reversals, but that pattern is unverified and is not a severity-calibration claim.

The registered synthetic cloud ablation set optical alpha to zero in three non-overlapping 25x25 blocks selected by fire overlap, with row/column tie-breaking: upper-left raster pixels `(275,350)`, `(300,300)`, and `(300,350)`. All three blocks have 625/625 FireDelineation pixels. Under synthetic alpha zero, optical-only classified no pixels as burned in the pooled 1,875-pixel region (IoU `0.0000`); VV SAR-only and adaptive fusion each classified 611 burned (IoU `0.3259`, paired block-bootstrap 95% CI `0.2160–0.3952`). Per-block inside results are descriptive (one spatial unit, so no meaningful within-block CI); outside-block metrics and CIs are reported in [`synthetic_cloud_ablation.txt`](../outputs/chios/checkpoint_c/synthetic_cloud_ablation.txt).

Descriptive conflict analysis uses nearest post-date SCL class 6 distance as a water/coast proximity proxy, not surveyed shoreline distance. Mean conflict was `0.025` within 100 m, `0.124` at 100–500 m, `0.222` at 500–1,000 m, and `0.101` beyond 1,000 m; Spearman correlation with distance was `-0.147`. This non-monotone pattern does not establish a coastal effect. Mean conflict by post SCL class was `0.377` for class 2, `0.087` for class 4, `0.118` for class 5, and zero for classes 7 and 8 in the common valid mask. By reference damage grade, mean conflict declined from `0.414` High and `0.409` Moderate high to `0.315` Moderate low and `0.233` Low; this is descriptive and uses the EMS-derived grading. No DEM/elevation raster is available in the project inputs, so slope association could not be evaluated. Tables are in [`conflict_associations.txt`](../outputs/chios/checkpoint_c/conflict_associations.txt) and its CSV.

Post-hoc decision diagnostics found zero class-decision differences between adaptive and conjunctive fusion on the common mask, explaining their identical confusion counts and classification metrics for this event; this does not assert equality of the possibility-valued outputs. Of reference-burned pixels detected by optical-only but missed by primary, primary was undecided for `100.0%` of High, `99.91%` of Moderate high, and `98.54%` of Moderate low misses, while `69.05%` of Low misses were called Unburned. POST-HOC temporal-discount-off minus primary was `+0.02705` IoU (paired 95% block CI `+0.02389–+0.03025`) and `+0.02374` F1 (`+0.02107–+0.02675`); these are secondary diagnostics, not the registered primary comparison.

The config's `vh_positive` variant was typed as SAR-only, so the registered output duplicates `sar_only_vh`; it was not fused. To address the likely intended interpretation, a separate **POST-HOC supplementary** optical-plus-VH-positive adaptive fusion was run and kept distinct. Against FireDelineation it obtained IoU `0.3144` (95% block CI `0.2565–0.3680`), F1 `0.4784` (`0.4083–0.5380`), precision `0.9923`, and recall `0.3152`. It does not replace or alter the frozen primary. Results are in [`posthoc_vh_fusion.txt`](../outputs/chios/checkpoint_c/posthoc_vh_fusion.txt); post-hoc diagnostics are in [`posthoc_diagnostics.txt`](../outputs/chios/checkpoint_c/posthoc_diagnostics.txt).
