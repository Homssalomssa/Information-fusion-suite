# Design decisions

## Project goal

Produce burned/unburned evidence by possibility-theory fusion of pre/post Sentinel-2 optical and Sentinel-1 SAR change inputs. The frame of discernment is `{Burned, Unburned}`. This project uses possibility theory (including the Dubois-Prade adaptive rule), not Dempster-Shafer evidence theory.

## Input and preprocessing assumptions

- Sentinel-2 inputs are L2A B8A and B12 on their native 20 m grid. Inputs are assumed to be surface reflectance with the `/10000` scaling and processing-baseline BOA `-1000` offset already applied. The index implementation will warn when values look unscaled (for example, a maximum above 1.5); it will not silently rescale input data.
- `NBR = (B8A - B12) / (B8A + B12)` and `dNBR = NBR_pre - NBR_post`; larger positive dNBR indicates more burn evidence.
- Sentinel-1 inputs are calibrated, terrain-corrected backscatter in dB. The SAR change sign is a parameter, `burn_direction` (`negative` or `positive`), with `negative` as the provisional VH default because a burned surface commonly has a backscatter drop. This default is a placeholder to calibrate against the study scene.
- Pre/post SAR inputs are assumed to use the same relative orbit, pass direction, and polarization.
- If SAR is finer than 20 m, resample in linear power by averaging, then convert the average back to dB. Nearest-neighbor resampling is not appropriate for this continuous backscatter input.
- All four acquisition dates (S2 pre/post and S1 pre/post) are required. Fire start/end dates are optional and are used for plausibility warnings.
- Optional reliability rasters use `1` for fully reliable and `0` for unusable. A cloud-probability raster is inverted (`alpha = 1 - cloud_probability`) before discounting. If no reliability raster is supplied, scalar `alpha = 1` is used. When both pre- and post-fire reliability values are present, combine them with `min`.
- NoData in any required input propagates to NoData in every output. NoData is never interpreted as zero or as total ignorance.

## Temporal and reliability decisions

- Temporal reliability is planned as `alpha_time = exp(-abs(dt) / tau)`, where `tau` is specified in days.
- The post-fire temporal-asymmetry adjustment applies only to the post-fire S1/S2 pair: for a source acquired earlier than the other, discount its Unburned evidence more strongly while retaining its Burned evidence. This is enabled by default.
- Pre-fire acquisition-date differences do not use the asymmetry adjustment. They produce a warning when they exceed a configurable threshold because of possible phenological/seasonal change.
- Warn when a post-fire acquisition predates the fire end date, or when image/fire dates are otherwise implausibly ordered.
- The combination of temporal reliability and sensor reliability (minimum versus product) remains configurable and will be made explicit when temporal discounting is implemented.
- A fire-progression/FIRMS-derived date raster is a future extension point only; it is not part of the initial implementation.

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
- The synthetic dates differ by four days: the optical acquisition is earlier than SAR. The demo computes `exp(-4/14)` and discounts SAR evidence; this local calculation illustrates the offset effect but is not a replacement for `core/temporal.py`.
- Optical-only, SAR-only, and adaptive fused classifications use the shared core decision function. Undecided pixels count as not predicted burned for IoU/F1, and undecided fraction is reported separately.
- The seeded 20-scene run reports whole-scene mean and population standard deviation of IoU, plus mean/std IoU and F1 and undecided fraction inside and outside the cloud. Tests require fused classifications to equal SAR-only under total optical ignorance, and fused outside-cloud IoU to be at least the better single source for every tested seed.
- `outputs/synthetic/` is gitignored and contains the first-seed evidence plot, classification comparison, and a JSON record of seeds, dates, temporal reliability, regional metrics, and whole-scene IoU summaries. Run with `python -m examples.synthetic_scene`.

## Indices, memberships, and temporal reliability — milestone 3

- `core/indices.py` expects aligned 2D B8A/B12 bands in scaled Sentinel-2 L2A surface reflectance; the `/10000` scaling and processing-baseline BOA `-1000` offset are assumed already applied. It warns if any finite value exceeds `1.5` as a likely unscaled-input indicator; it never rescales bands implicitly.
- `NBR = (B8A - B12) / (B8A + B12)` and `dNBR = NBR_pre - NBR_post`. Any input NoData (NaN) propagates, and a denominator whose absolute value is at most `1e-6` becomes NoData rather than producing an unstable ratio.
- Memberships are two-class normalized possibilities in `(2,H,W)` order `[Burned, Unburned]`, with the unburned membership defined as the complement of the burned membership. Trapezoids support infinite outer breakpoints to express shoulders.
- The default dNBR right-shoulder rise starts at `0.10` and reaches full burned possibility at `0.27`, using the USGS dNBR unburned/low and low/moderate severity boundaries as a binary burned-evidence transition; this is not a severity-class classifier. Default SAR breakpoints are oriented change magnitudes `(0.5, 1.5, inf, inf)` dB, provisional placeholders to calibrate per scene. `burn_direction` is `negative` by default for VH and can be `positive`.
- Otsu suggestion uses a histogram between-class-variance split, then recommends a right-shoulder ramp one histogram bin wide below the selected threshold. NaN values are excluded; empty or constant inputs raise `ValueError`.
- Temporal reliability is `alpha_time = exp(-abs(dt) / tau_days)`, with positive `tau_days`. Sensor and temporal reliability combine by configurable `min` or `product` (default `product`).
- Post-fire asymmetry is enabled by default and applies only to whichever source is earlier in the post-fire S1/S2 pair. The configurable exponent defaults to `k=2.0` and must be at least 1. For the earlier source, `alpha_B = alpha_sensor * alpha_time**k` and `alpha_U = alpha_sensor`, giving floors `pi'(Burned) = max(pi(Burned), 1 - alpha_sensor * alpha_time**k)` and `pi'(Unburned) = max(pi(Unburned), 1 - alpha_sensor)`. This raises a weak earlier Burned possibility to account for fire spread while preserving the earlier Unburned claim from temporal decay; an already-high Burned possibility is unchanged. The later source receives uniform sensor/time discount using the configured combination. At `dt=0`, neither source is earlier and no temporal discount is applied beyond sensor reliability. Pre-fire pairs never use this asymmetric rule.
- The pre-fire S1/S2 date gap warns above a configurable threshold (default 30 days) because of possible phenological change. Date validation also warns for pre/post order errors, pre-fire acquisitions on/after fire start, post-fire acquisitions before fire start/end, and invalid fire chronology.
- `firms_temporal_reliability` is an explicit `NotImplementedError` extension point. Per-pixel date decoding and date-raster NoData conventions remain unselected and are not simulated as completed behavior.

## Module map — milestone 3

- `dst_fusion/__init__.py` is QGIS's `classFactory` entry point.
- `dst_fusion/plugin.py` manages plugin lifecycle and registers `DstFusionProvider`.
- `dst_fusion/provider.py` is the Processing provider shell; algorithm registration is deferred.
- `dst_fusion/core/possibility.py` implements normalized two-class possibility arrays, reliability discounting, three fusion rules, conflict, and class/necessity/undecided decisions.
- `dst_fusion/core/indices.py` computes NBR/dNBR from reflectance bands and handles scale warnings and NoData.
- `dst_fusion/core/memberships.py` implements trapezoid memberships, burned/unburned possibilities for dNBR and SAR change, and Otsu breakpoint suggestions.
- `dst_fusion/core/temporal.py` computes temporal reliability, combines sensor reliability, applies the post-fire asymmetric date-offset rule, reports date plausibility warnings, and reserves the FIRMS date-raster hook.
- `dst_fusion/io/` reserves the GDAL/QGIS raster boundary.
- `dst_fusion/algorithms/` reserves QGIS Processing algorithm wrappers.
- `examples/synthetic_scene.py` compares optical-only, SAR-only, and fused decisions over 20 seeded scenes and reports regional and whole-scene metrics.
- `tests/` covers scaffold imports, possibility fusion, index/membership/temporal behavior, and the synthetic cloud/fusion invariants.
