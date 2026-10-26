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

## Module map — milestone 0

- `dst_fusion/__init__.py` is QGIS's `classFactory` entry point.
- `dst_fusion/plugin.py` manages plugin lifecycle and registers `DstFusionProvider`.
- `dst_fusion/provider.py` is the Processing provider shell; algorithm registration is deferred.
- `dst_fusion/core/{indices,memberships,possibility,temporal}.py` reserve the pure numerical modules for later milestones.
- `dst_fusion/io/` reserves the GDAL/QGIS raster boundary.
- `dst_fusion/algorithms/` reserves QGIS Processing algorithm wrappers.
- `tests/test_scaffold.py` verifies that the core module scaffold imports without a QGIS dependency.
