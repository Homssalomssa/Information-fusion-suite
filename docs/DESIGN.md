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
- The synthetic dates differ by four days. The demo computes `exp(-4/14)` and discounts SAR evidence to illustrate the reliability effect; this local demonstration is not a substitute for `core/temporal.py`, which remains for milestone 3.
- Optical-only, SAR-only, and adaptive fused classifications use the shared core decision function. Undecided pixels count as not predicted burned for the printed burned-class IoU and F1 scores. The comparison therefore penalizes abstention over known burned pixels rather than hiding it.
- `outputs/synthetic/` is gitignored and contains the generated evidence plot, classification comparison, and a JSON record of the fixed seed, dates, temporal reliability, undecided counts, and metrics. Run with `python -m examples.synthetic_scene`.

## Module map — milestone 2

- `dst_fusion/__init__.py` is QGIS's `classFactory` entry point.
- `dst_fusion/plugin.py` manages plugin lifecycle and registers `DstFusionProvider`.
- `dst_fusion/provider.py` is the Processing provider shell; algorithm registration is deferred.
- `dst_fusion/core/possibility.py` implements normalized two-class possibility arrays, reliability discounting, three fusion rules, conflict, and class/necessity/undecided decisions. `indices.py`, `memberships.py`, and `temporal.py` remain reserved pure numerical modules for milestone 3.
- `dst_fusion/io/` reserves the GDAL/QGIS raster boundary.
- `dst_fusion/algorithms/` reserves QGIS Processing algorithm wrappers.
- `examples/synthetic_scene.py` compares optical-only, SAR-only, and fused decisions on a seeded scene with a cloud mask and acquisition-date offset.
- `tests/test_scaffold.py` verifies that the core module scaffold imports without a QGIS dependency; `tests/test_possibility.py` covers the possibility core and its undecided decision behavior.
