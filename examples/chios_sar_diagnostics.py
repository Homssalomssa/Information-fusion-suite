"""Read-only Sentinel-1 sign and background-contrast diagnostics for Chios.

Run from the repository root with:
    .venv-qgis/bin/python -m examples.chios_sar_diagnostics

This compares the mapped fire to damage-grade polygons and label-free,
same-shape placements across the full valid-land S2/S1 intersection. It reports
sign evidence only; it does not select a classifier sign or compute accuracy.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np
from osgeo import gdal, ogr, osr

from possibility_fusion.core.indices import dnbr
from possibility_fusion.core.preprocessing import (
    boxcar_filter,
    db_to_linear_power,
    linear_power_to_db,
    scl_optical_reliability,
)
from possibility_fusion.io.raster import compute_intersection_grid
from examples.audit_chios import (
    AOI_BUFFER_METERS,
    REFERENCE_PATH,
    _find_inputs,
    _get_reference_geometry,
    _pixel_window,
    _read_boa_metadata,
    _rasterize_geometry,
    _read_window,
    _scale_reflectance,
    _window_for_grid,
)
from examples.run_chios import _read_float_window

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "chios"
PIXEL_SIZE = 20.0
FIRE_GRADE_EXCLUSION_METERS = 500.0
POLARIZATIONS = ("VH", "VV")
GRADE_LABELS = (
    "High severity",
    "Moderate high severity",
    "Moderate low severity",
    "Low severity",
)
LS_FLAG_MASK = 4 | 16
NULL_BLOCK_SIZE = 256
NULL_LAYER_BANDS = {
    "VH_unfiltered": 1,
    "VH_boxcar_5x5": 2,
    "VV_unfiltered": 3,
    "VV_boxcar_5x5": 4,
    "dnbr_values": 5,
    "optical_alpha": 6,
}
MASK_LAYER_BANDS = {
    "land": 1,
    "pre_scl": 2,
    "layover_shadow": 3,
    "fire_exclusion": 4,
    "grading_exclusion": 5,
    "placeable": 6,
    "coast_seed": 7,
    "ls_valid": 8,
}
gdal.UseExceptions()
ogr.UseExceptions()


def _contrast_summary(
    inside_values: np.ndarray,
    outside_values: np.ndarray,
) -> dict[str, float | int]:
    """Summarize samples and their pooled-standard-deviation effect size."""
    inside = np.asarray(inside_values, dtype=np.float64)
    outside = np.asarray(outside_values, dtype=np.float64)
    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]
    if inside.size == 0 or outside.size == 0:
        raise ValueError("At least one finite value is required in each sample.")
    inside_std = float(np.std(inside, ddof=1)) if inside.size > 1 else 0.0
    outside_std = float(np.std(outside, ddof=1)) if outside.size > 1 else 0.0
    degrees = inside.size + outside.size - 2
    pooled_variance = (
        ((inside.size - 1) * inside_std**2 + (outside.size - 1) * outside_std**2)
        / degrees
        if degrees > 0
        else 0.0
    )
    pooled_std = math.sqrt(pooled_variance)
    difference = float(np.mean(inside) - np.mean(outside))
    effect_size = difference / pooled_std if pooled_std > 0.0 else math.nan
    return {
        "inside_count": int(inside.size),
        "inside_mean": float(np.mean(inside)),
        "inside_median": float(np.median(inside)),
        "inside_std": inside_std,
        "outside_count": int(outside.size),
        "outside_mean": float(np.mean(outside)),
        "outside_median": float(np.median(outside)),
        "outside_std": outside_std,
        "difference": difference,
        "pooled_std": pooled_std,
        "effect_size": effect_size,
    }


def _empirical_upper_tail_p(null_values: np.ndarray, observed: float) -> float:
    """One-sided empirical p-value with the finite-sample correction."""
    values = np.asarray(null_values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(observed):
        raise ValueError("Null samples and observed difference must be finite.")
    return float((1 + np.count_nonzero(finite >= observed)) / (finite.size + 1))


def _empirical_two_sided_p(null_values: np.ndarray, observed: float) -> float:
    """Two-sided empirical tail probability with the +1 finite-sample correction."""
    values = np.asarray(null_values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(observed):
        raise ValueError("Null samples and observed difference must be finite.")
    lower = (1 + np.count_nonzero(finite <= observed)) / (finite.size + 1)
    upper = (1 + np.count_nonzero(finite >= observed)) / (finite.size + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def _null_difference_from_sums(
    placed_sum: float,
    placed_count: int,
    background_sum: float,
    background_count: int,
) -> float:
    """Contrast a placement with the remaining valid-land background."""
    outside_count = background_count - placed_count
    if placed_count < 1 or outside_count < 1:
        raise ValueError("Null placement requires non-empty inside and outside samples.")
    return float(
        placed_sum / placed_count
        - (background_sum - placed_sum) / outside_count
    )


def _translate_mask(
    mask: np.ndarray,
    *,
    column_shift: int,
    row_shift: int,
) -> np.ndarray:
    """Translate a boolean raster by whole pixels without wraparound."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    if any(isinstance(shift, bool) or not isinstance(shift, (int, np.integer))
           for shift in (column_shift, row_shift)):
        raise ValueError("pixel shifts must be integers.")
    result = np.zeros(source.shape, dtype=bool)
    source_y0 = max(0, -row_shift)
    source_y1 = min(source.shape[0], source.shape[0] - row_shift)
    source_x0 = max(0, -column_shift)
    source_x1 = min(source.shape[1], source.shape[1] - column_shift)
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        return result
    result[
        source_y0 + row_shift : source_y1 + row_shift,
        source_x0 + column_shift : source_x1 + column_shift,
    ] = source[source_y0:source_y1, source_x0:source_x1]
    return result


def _placement_is_accepted(
    candidate: np.ndarray,
    *,
    land: np.ndarray,
    coast_distance: np.ndarray,
    fire_exclusion: np.ndarray,
    grading_exclusion: np.ndarray,
    dnbr_values: np.ndarray,
    optical_alpha: np.ndarray,
    pre_scl: np.ndarray,
    layover_shadow: np.ndarray,
    reference_scl_mix: tuple[float, float],
    coast_buffer_m: float = 500.0,
    scl_mix_tolerance: float = 0.10,
    max_layover_fraction: float = 0.01,
) -> bool:
    """Apply the label-free land, optical, SCL-mix, and SAR-quality filters."""
    mask = np.asarray(candidate, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return False
    arrays = (
        land,
        coast_distance,
        fire_exclusion,
        grading_exclusion,
        dnbr_values,
        optical_alpha,
        pre_scl,
        layover_shadow,
    )
    if any(np.asarray(array).shape != mask.shape for array in arrays):
        raise ValueError("Placement evidence windows must match the candidate mask.")
    if coast_buffer_m < 0.0 or scl_mix_tolerance < 0.0:
        raise ValueError("Coast buffer and SCL mix tolerance must be non-negative.")
    if not 0.0 <= max_layover_fraction <= 1.0:
        raise ValueError("max_layover_fraction must be in [0, 1].")
    reference_mix = np.asarray(reference_scl_mix, dtype=np.float64)
    if (
        reference_mix.shape != (2,)
        or np.any(~np.isfinite(reference_mix))
        or np.any(reference_mix < 0.0)
        or not np.isclose(reference_mix.sum(), 1.0)
    ):
        raise ValueError("reference_scl_mix must contain class-4/5 shares summing to one.")

    land_values = np.asarray(land, dtype=bool)
    coast_values = np.asarray(coast_distance, dtype=np.float64)
    fire_buffer = np.asarray(fire_exclusion, dtype=bool)
    grade_buffer = np.asarray(grading_exclusion, dtype=bool)
    optical = np.asarray(optical_alpha, dtype=np.float64)
    scl = np.asarray(pre_scl)
    flagged = np.asarray(layover_shadow, dtype=bool)
    selected_dnbr = np.asarray(dnbr_values, dtype=np.float64)[mask]
    if (
        not np.all(land_values[mask])
        or not np.all(np.isfinite(coast_values[mask]))
        or np.any(coast_values[mask] < coast_buffer_m)
        or np.any(fire_buffer[mask])
        or np.any(grade_buffer[mask])
        or not np.all(np.isfinite(optical[mask]) & (optical[mask] == 1.0))
        or not np.all(np.isfinite(selected_dnbr))
        or float(np.mean(selected_dnbr)) >= 0.10
        or float(np.mean(selected_dnbr >= 0.10)) >= 0.05
        or float(np.mean(flagged[mask])) > max_layover_fraction
    ):
        return False

    candidate_scl = scl[mask]
    class4 = int(np.count_nonzero(candidate_scl == 4))
    class5 = int(np.count_nonzero(candidate_scl == 5))
    clear_count = class4 + class5
    if clear_count == 0:
        return False
    candidate_mix = np.array([class4, class5], dtype=np.float64) / clear_count
    return bool(np.all(np.abs(candidate_mix - reference_mix) <= scl_mix_tolerance))


def _iter_random_null_placements(
    scene: dict[str, Any],
    *,
    requested: int,
    seed: int,
    max_attempts: int = 500,
    coast_buffer_m: float = 500.0,
    scl_mix_tolerance: float = 0.10,
    max_layover_fraction: float = 0.01,
) -> Iterator[
    tuple[int, int, tuple[int, int, tuple[slice, slice], np.ndarray] | None]
]:
    """Yield bounded, full-intersection placement attempts without scene copies."""
    if requested < 1:
        raise ValueError("requested null replicate count must be positive.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    template = np.asarray(scene["template_mask"], dtype=bool)
    if template.ndim != 2 or not np.any(template):
        raise ValueError("The cropped rasterized fire mask is empty or invalid.")
    height, width = scene["height"], scene["width"]
    if template.shape[0] > height or template.shape[1] > width:
        raise ValueError("The fire mask does not fit inside the full intersection grid.")

    rng = np.random.default_rng(seed)
    reference_mix = scene["reference_scl_mix"]
    seen_offsets: set[tuple[int, int]] = set()
    for replicate in range(1, requested + 1):
        placement = None
        attempts = 0
        for attempts in range(1, max_attempts + 1):
            x0 = int(rng.integers(0, width - template.shape[1] + 1))
            y0 = int(rng.integers(0, height - template.shape[0] + 1))
            offset = (x0, y0)
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            window = (slice(y0, y0 + template.shape[0]), slice(x0, x0 + template.shape[1]))
            placeable = np.asarray(
                _read_null_layer(scene, "placeable", window), dtype=bool
            )
            if not np.all(placeable[template]):
                continue
            evidence = {
                name: _read_null_layer(scene, name, window)
                for name in (
                    "land",
                    "coast_distance",
                    "fire_exclusion",
                    "grading_exclusion",
                    "dnbr_values",
                    "optical_alpha",
                    "pre_scl",
                    "layover_shadow",
                )
            }
            if not _placement_is_accepted(
                template,
                **evidence,
                reference_scl_mix=reference_mix,
                coast_buffer_m=coast_buffer_m,
                scl_mix_tolerance=scl_mix_tolerance,
                max_layover_fraction=max_layover_fraction,
            ):
                continue
            placement = (
                x0 - scene["template_x"],
                y0 - scene["template_y"],
                window,
                template,
            )
            break
        yield replicate, attempts, placement


def _significant_histogram_modes(
    values: np.ndarray,
    *,
    bins: int = 256,
    min_relative_height: float = 0.1,
) -> int:
    """Count prominent modes after a short moving-average histogram smooth.

    The range is clipped to the 0.5th–99.5th percentiles so extreme tails do
    not collapse the central histogram resolution. This is a descriptive
    modality check, not a formal statistical test of unimodality.
    """
    sample = np.asarray(values, dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise ValueError("At least one finite value is required for a histogram.")
    if bins < 3 or not 0.0 < min_relative_height <= 1.0:
        raise ValueError("bins must be at least 3 and min_relative_height in (0, 1].")
    low, high = np.percentile(finite, (0.5, 99.5))
    if low == high:
        return 1
    histogram, _ = np.histogram(finite, bins=bins, range=(low, high))
    smoothing_width = 31
    smoothed = np.convolve(
        histogram.astype(np.float64),
        np.ones(smoothing_width) / smoothing_width,
        mode="same",
    )
    threshold = float(np.max(smoothed)) * min_relative_height
    candidates = [
        index
        for index in range(1, smoothed.size - 1)
        if smoothed[index] >= smoothed[index - 1]
        and smoothed[index] > smoothed[index + 1]
        and smoothed[index] >= threshold
    ]
    peaks: list[int] = []
    for index in sorted(candidates, key=lambda item: smoothed[item], reverse=True):
        if all(abs(index - peak) >= smoothing_width for peak in peaks):
            peaks.append(index)
    return max(1, len(peaks))


def _open_reference_geometry(layer_name: str) -> tuple[Any, Any]:
    dataset = ogr.Open(str(REFERENCE_PATH), 0)
    if dataset is None:
        raise RuntimeError(f"Could not open reference GeoPackage: {REFERENCE_PATH}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Reference layer {layer_name!r} is missing.")
    feature = layer.GetNextFeature()
    if feature is None:
        raise RuntimeError(f"Reference layer {layer_name!r} is empty.")
    geometry = feature.GetGeometryRef().Clone()
    spatial_reference = layer.GetSpatialRef().Clone()
    dataset = layer = feature = None
    return geometry, spatial_reference


def _transform_geometry(geometry: Any, source_srs: Any, target_srs: Any) -> Any:
    result = geometry.Clone()
    if not source_srs.IsSame(target_srs):
        result.Transform(osr.CoordinateTransformation(source_srs, target_srs))
    return result


def _repair_geometry(geometry: Any) -> tuple[Any, bool]:
    original_wkb = geometry.ExportToWkb()
    result = geometry.MakeValid()
    if result is None or not result.IsValid():
        raise ValueError("Geometry remains invalid after MakeValid().")
    return result, original_wkb != result.ExportToWkb()


def _rasterize_geometries(
    geometries: list[Any],
    spatial_reference: Any,
    geotransform: tuple[float, ...],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize many polygon geometries using pixel-center inclusion."""
    driver = gdal.GetDriverByName("MEM")
    raster = driver.Create("", width, height, 1, gdal.GDT_Byte)
    raster.SetGeoTransform(geotransform)
    raster.SetProjection(spatial_reference.ExportToWkt())
    vector_driver = ogr.GetDriverByName("Memory")
    vector = vector_driver.CreateDataSource("")
    layer = vector.CreateLayer(
        "polygons",
        srs=spatial_reference,
        geom_type=ogr.wkbUnknown,
    )
    definition = layer.GetLayerDefn()
    for geometry in geometries:
        feature = ogr.Feature(definition)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        feature = None
    error = gdal.RasterizeLayer(
        raster,
        [1],
        layer,
        burn_values=[1],
        options=["ALL_TOUCHED=FALSE"],
    )
    if error != gdal.CE_None:
        raise RuntimeError("GDAL failed rasterizing the diagnostic polygons.")
    values = raster.GetRasterBand(1).ReadAsArray()
    layer = vector = raster = None
    return np.asarray(values, dtype=bool)


def _prepare_scene() -> dict[str, Any]:
    """Read only the buffered AOI windows needed for sign diagnostics."""
    s2, s1, _ = _find_inputs()
    grid = compute_intersection_grid(
        {
            "s2_pre": s2["pre_B8A"],
            "s2_post": s2["post_B8A"],
            "s1_pre": s1["pre_VH"],
            "s1_post": s1["post_VH"],
        },
        reference_key="s2_pre",
    )
    fire_geometry, fire_srs = _get_reference_geometry()[0]
    reference_srs = osr.SpatialReference()
    reference_srs.ImportFromWkt(grid.projection_wkt)
    reference_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    fire_geometry = _transform_geometry(fire_geometry, fire_srs, reference_srs)
    fire_geometry, fire_repaired = _repair_geometry(fire_geometry)
    fire_envelope = fire_geometry.GetEnvelope()
    fire_bbox = (
        fire_envelope[0],
        fire_envelope[2],
        fire_envelope[1],
        fire_envelope[3],
    )
    grid_window = _pixel_window(
        grid.geotransform, grid.width, grid.height, fire_bbox, AOI_BUFFER_METERS
    )
    tile_ds = gdal.OpenEx(str(s2["pre_B8A"]), gdal.OF_RASTER | gdal.OF_READONLY)
    if tile_ds is None:
        raise RuntimeError("Could not open the Sentinel-2 reference grid.")
    tile_gt = tile_ds.GetGeoTransform()
    tile_window = _pixel_window(
        tile_gt,
        tile_ds.RasterXSize,
        tile_ds.RasterYSize,
        fire_bbox,
        AOI_BUFFER_METERS,
    )
    tile_ds = None
    x_offset, y_offset, width, height = grid_window
    output_gt = (
        grid.geotransform[0] + x_offset * grid.geotransform[1],
        grid.geotransform[1],
        0.0,
        grid.geotransform[3] + y_offset * grid.geotransform[5],
        0.0,
        grid.geotransform[5],
    )

    scl = {
        period: _read_window(s2[f"{period}_SCL"], tile_window)
        for period in ("pre", "post")
    }
    land = np.ones((height, width), dtype=bool)
    for values in scl.values():
        land &= np.isfinite(values) & (values != 0) & (values != 6)

    changes: dict[str, dict[str, np.ndarray]] = {}
    for polarization in POLARIZATIONS:
        periods = {}
        for period in ("pre", "post"):
            source_path = s1[f"{period}_{polarization}"]
            source_window = _window_for_grid(source_path, grid.geotransform, grid_window)
            periods[period] = _read_float_window(source_path, source_window)
        valid_pair = np.isfinite(periods["pre"]) & np.isfinite(periods["post"])
        land &= valid_pair
        raw_change = periods["post"] - periods["pre"]
        filtered_pre = linear_power_to_db(
            boxcar_filter(db_to_linear_power(periods["pre"]), window_size=5)
        )
        filtered_post = linear_power_to_db(
            boxcar_filter(db_to_linear_power(periods["post"]), window_size=5)
        )
        filtered_change = filtered_post - filtered_pre
        changes[polarization] = {
            "unfiltered": raw_change,
            "boxcar_5x5": filtered_change,
        }

    for period in ("pre", "post"):
        source_path = s1[f"{period}_ls_map"]
        source_window = _window_for_grid(source_path, grid.geotransform, grid_window)
        mask_values = _read_float_window(source_path, source_window)
        safe_values = np.where(np.isfinite(mask_values), mask_values, 0).astype(np.uint8)
        land &= np.isfinite(mask_values) & ((safe_values & LS_FLAG_MASK) == 0)

    for polarization in POLARIZATIONS:
        for values in changes[polarization].values():
            land &= np.isfinite(values)
    for polarization in POLARIZATIONS:
        for mode in ("unfiltered", "boxcar_5x5"):
            changes[polarization][mode][~land] = np.nan

    ems_geometry, ems_srs = _open_reference_geometry("EMSN198_AOI")
    ems_geometry = _transform_geometry(ems_geometry, ems_srs, reference_srs)
    ems_geometry, ems_repaired = _repair_geometry(ems_geometry)
    ems_mask = _rasterize_geometry(
        ems_geometry, reference_srs, output_gt, width, height
    )
    fire_mask = _rasterize_geometry(
        fire_geometry, reference_srs, output_gt, width, height
    )
    scene = {
        "s1": s1,
        "s2": s2,
        "grid": grid,
        "grid_window": grid_window,
        "output_gt": output_gt,
        "width": width,
        "height": height,
        "reference_srs": reference_srs,
        "fire_geometry": fire_geometry,
        "fire_bbox": fire_bbox,
        "fire_geometry_repaired": fire_repaired,
        "ems_geometry": ems_geometry,
        "ems_geometry_repaired": ems_repaired,
        "ems_mask": ems_mask,
        "fire_mask": fire_mask,
        "land": land,
        "changes": changes,
    }
    grade_masks, repaired_grades = _damage_grade_masks(scene)
    scene["grade_masks"] = grade_masks
    scene["repaired_grade_geometries"] = repaired_grades
    return scene


def _damage_grade_masks(scene: dict[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    dataset = ogr.Open(str(REFERENCE_PATH), 0)
    if dataset is None:
        raise RuntimeError(f"Could not open reference GeoPackage: {REFERENCE_PATH}")
    layer = dataset.GetLayerByName("P01_FDG_FireGrading")
    if layer is None:
        raise RuntimeError("P01_FDG_FireGrading layer is missing.")
    field_names = {
        layer.GetLayerDefn().GetFieldDefn(index).GetName()
        for index in range(layer.GetLayerDefn().GetFieldCount())
    }
    if "damage_grade" not in field_names:
        raise RuntimeError("P01_FDG_FireGrading has no damage_grade field.")
    source_srs = layer.GetSpatialRef()
    grouped: dict[str, list[Any]] = {label: [] for label in GRADE_LABELS}
    repaired = 0
    for feature in layer:
        grade = feature.GetField("damage_grade")
        if grade not in grouped:
            continue
        geometry = feature.GetGeometryRef().Clone()
        if not source_srs.IsSame(scene["reference_srs"]):
            geometry.Transform(
                osr.CoordinateTransformation(source_srs, scene["reference_srs"])
            )
        geometry, was_repaired = _repair_geometry(geometry)
        if was_repaired:
            repaired += 1
        grouped[grade].append(geometry)
        feature = None
    dataset = layer = None
    masks = {
        grade: _rasterize_geometries(
            geometries,
            scene["reference_srs"],
            scene["output_gt"],
            scene["width"],
            scene["height"],
        )
        for grade, geometries in grouped.items()
        if geometries
    }
    return masks, repaired


def _create_scratch_raster(
    path: Path,
    grid,
    *,
    bands: int,
    data_type: int,
    fill_value: float | None = None,
):
    options = [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "BIGTIFF=IF_SAFER",
        f"BLOCKXSIZE={NULL_BLOCK_SIZE}",
        f"BLOCKYSIZE={NULL_BLOCK_SIZE}",
    ]
    if data_type == gdal.GDT_Float32:
        options.append("PREDICTOR=3")
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), grid.width, grid.height, bands, data_type, options=options
    )
    if dataset is None:
        raise RuntimeError(f"Could not create disk-backed scratch raster: {path}")
    dataset.SetGeoTransform(grid.geotransform)
    dataset.SetProjection(grid.projection_wkt)
    for band_index in range(1, bands + 1):
        band = dataset.GetRasterBand(band_index)
        if data_type == gdal.GDT_Float32:
            band.SetNoDataValue(float("nan"))
        if fill_value is not None:
            band.Fill(fill_value)
    return dataset


def _grid_source_offset(dataset, target_geotransform: tuple[float, ...]) -> tuple[int, int]:
    source_gt = dataset.GetGeoTransform()
    column = (target_geotransform[0] - source_gt[0]) / source_gt[1]
    row = (target_geotransform[3] - source_gt[3]) / source_gt[5]
    if (
        not math.isclose(source_gt[1], target_geotransform[1], abs_tol=1e-9)
        or not math.isclose(source_gt[5], target_geotransform[5], abs_tol=1e-9)
        or not math.isclose(column, round(column), abs_tol=1e-8)
        or not math.isclose(row, round(row), abs_tol=1e-8)
    ):
        raise ValueError("Null-test source rasters must share the 20 m target lattice.")
    return int(round(column)), int(round(row))


def _read_source_block(
    dataset,
    source_offset: tuple[int, int],
    target_window: tuple[int, int, int, int],
    *,
    floating: bool,
) -> np.ndarray:
    x_offset, y_offset = source_offset
    x, y, width, height = target_window
    values = dataset.GetRasterBand(1).ReadAsArray(
        x_offset + x, y_offset + y, width, height
    )
    if values is None:
        raise RuntimeError("GDAL failed reading a null-test raster block.")
    if not floating:
        return np.asarray(values)
    band = dataset.GetRasterBand(1)
    result = np.asarray(values, dtype=np.float32)
    invalid = ~np.isfinite(result)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        invalid |= result == np.float32(nodata)
    invalid |= result == np.float32(-3.4028235e38)
    result[invalid] = np.nan
    return result


def _filtered_sar_change(
    pre_extended: np.ndarray,
    post_extended: np.ndarray,
    core_window: tuple[slice, slice],
) -> np.ndarray:
    pre_filtered = linear_power_to_db(
        boxcar_filter(db_to_linear_power(pre_extended), window_size=5)
    )
    post_filtered = linear_power_to_db(
        boxcar_filter(db_to_linear_power(post_extended), window_size=5)
    )
    return post_filtered[core_window] - pre_filtered[core_window]


def _rasterize_geometries_to_band(
    dataset,
    band_index: int,
    geometries: list[Any],
    spatial_reference: Any,
) -> None:
    if not geometries:
        return
    vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = vector.CreateLayer(
        "polygons", srs=spatial_reference, geom_type=ogr.wkbUnknown
    )
    definition = layer.GetLayerDefn()
    for geometry in geometries:
        feature = ogr.Feature(definition)
        feature.SetGeometry(geometry)
        if layer.CreateFeature(feature) != 0:
            raise RuntimeError("Could not stage a geometry for null-test rasterization.")
        feature = None
    error = gdal.RasterizeLayer(
        dataset,
        [band_index],
        layer,
        burn_values=[1],
        options=["ALL_TOUCHED=FALSE"],
    )
    layer = vector = None
    if error != gdal.CE_None:
        raise RuntimeError("GDAL failed rasterizing a full-intersection null mask.")


def _full_reference_geometries(
    reference_srs: Any,
) -> tuple[Any, dict[str, list[Any]], int]:
    fire_geometry, fire_srs = _get_reference_geometry()[0]
    fire_geometry = _transform_geometry(fire_geometry, fire_srs, reference_srs)
    fire_geometry, fire_repaired = _repair_geometry(fire_geometry)
    dataset = ogr.Open(str(REFERENCE_PATH), 0)
    if dataset is None:
        raise RuntimeError(f"Could not open reference GeoPackage: {REFERENCE_PATH}")
    layer = dataset.GetLayerByName("P01_FDG_FireGrading")
    if layer is None:
        raise RuntimeError("P01_FDG_FireGrading layer is missing.")
    definition = layer.GetLayerDefn()
    fields = {
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    }
    if "damage_grade" not in fields:
        raise RuntimeError("P01_FDG_FireGrading has no damage_grade field.")
    source_srs = layer.GetSpatialRef()
    grades: dict[str, list[Any]] = {label: [] for label in GRADE_LABELS}
    repaired_count = 0
    for feature in layer:
        grade = feature.GetField("damage_grade")
        if grade not in grades:
            continue
        geometry = _transform_geometry(
            feature.GetGeometryRef(), source_srs, reference_srs
        )
        geometry, was_repaired = _repair_geometry(geometry)
        repaired_count += int(was_repaired)
        grades[grade].append(geometry)
        feature = None
    dataset = layer = None
    return fire_geometry, grades, repaired_count + int(fire_repaired)


def _prepare_full_null_scene(
    temp_dir: str,
    *,
    coast_buffer_m: float,
    exclusion_buffer_m: float = FIRE_GRADE_EXCLUSION_METERS,
) -> dict[str, Any]:
    """Build full-intersection evidence in bounded blocks and disk-backed rasters."""
    gdal.SetCacheMax(256 * 1024 * 1024)
    s2, s1, _ = _find_inputs()
    if any(f"{period}_ls_map" not in s1 for period in ("pre", "post")):
        raise FileNotFoundError("Both pre- and post-fire Sentinel-1 ls_map rasters are required.")
    grid = compute_intersection_grid(
        {
            "s2_pre": s2["pre_B8A"],
            "s2_post": s2["post_B8A"],
            "s1_pre": s1["pre_VH"],
            "s1_post": s1["post_VH"],
        },
        reference_key="s2_pre",
    )
    reference_srs = osr.SpatialReference()
    reference_srs.ImportFromWkt(grid.projection_wkt)
    reference_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    fire_geometry, grade_geometries, repaired_grades = _full_reference_geometries(
        reference_srs
    )
    fire_bbox_envelope = fire_geometry.GetEnvelope()
    fire_bbox = (
        fire_bbox_envelope[0],
        fire_bbox_envelope[2],
        fire_bbox_envelope[1],
        fire_bbox_envelope[3],
    )
    template_x, template_y, template_width, template_height = _pixel_window(
        grid.geotransform,
        grid.width,
        grid.height,
        fire_bbox,
    )
    template_gt = (
        grid.geotransform[0] + template_x * grid.geotransform[1],
        grid.geotransform[1],
        0.0,
        grid.geotransform[3] + template_y * grid.geotransform[5],
        0.0,
        grid.geotransform[5],
    )
    template = _rasterize_geometries(
        [fire_geometry],
        reference_srs,
        template_gt,
        template_width,
        template_height,
    )
    if not np.any(template):
        raise ValueError("The rasterized fire delineation is empty on the full intersection grid.")

    signal_ds = _create_scratch_raster(
        Path(temp_dir) / "signals.tif",
        grid,
        bands=len(NULL_LAYER_BANDS),
        data_type=gdal.GDT_Float32,
    )
    mask_ds = _create_scratch_raster(
        Path(temp_dir) / "masks.tif",
        grid,
        bands=len(MASK_LAYER_BANDS),
        data_type=gdal.GDT_Byte,
    )
    grade_ds = _create_scratch_raster(
        Path(temp_dir) / "grades.tif",
        grid,
        bands=len(GRADE_LABELS),
        data_type=gdal.GDT_Byte,
    )
    distance_ds = _create_scratch_raster(
        Path(temp_dir) / "coast_distance.tif",
        grid,
        bands=1,
        data_type=gdal.GDT_Float32,
    )

    fire_buffer = fire_geometry.Buffer(exclusion_buffer_m)
    if fire_buffer is None or fire_buffer.IsEmpty():
        raise RuntimeError("Could not buffer the fire delineation.")
    _rasterize_geometries_to_band(
        mask_ds,
        MASK_LAYER_BANDS["fire_exclusion"],
        [fire_buffer],
        reference_srs,
    )
    grade_exclusion = [
        geometry.Buffer(exclusion_buffer_m)
        for geometries in grade_geometries.values()
        for geometry in geometries
    ]
    if any(geometry is None or geometry.IsEmpty() for geometry in grade_exclusion):
        raise RuntimeError("Could not buffer every EMS damage-grading geometry.")
    _rasterize_geometries_to_band(
        mask_ds,
        MASK_LAYER_BANDS["grading_exclusion"],
        grade_exclusion,
        reference_srs,
    )
    for grade_index, grade in enumerate(GRADE_LABELS, start=1):
        _rasterize_geometries_to_band(
            grade_ds, grade_index, grade_geometries[grade], reference_srs
        )

    source_names = (
        *(f"s2_{period}_{band}" for period in ("pre", "post") for band in ("B8A", "B12", "SCL")),
        *(f"s1_{period}_{band}" for period in ("pre", "post") for band in ("VH", "VV", "ls_map")),
    )
    paths = {
        name: s2[name.removeprefix("s2_")] if name.startswith("s2_")
        else s1[name.removeprefix("s1_")]
        for name in source_names
    }
    sources = {
        name: gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
        for name, path in paths.items()
        if name.startswith(("s2_pre_", "s2_post_", "s1_pre_", "s1_post_"))
    }
    if any(dataset is None for dataset in sources.values()):
        raise RuntimeError("Could not open every source raster for the null test.")
    offsets = {
        name: _grid_source_offset(dataset, grid.geotransform)
        for name, dataset in sources.items()
    }
    boa = _read_boa_metadata(s2)
    radius = 2
    for y in range(0, grid.height, NULL_BLOCK_SIZE):
        height = min(NULL_BLOCK_SIZE, grid.height - y)
        for x in range(0, grid.width, NULL_BLOCK_SIZE):
            width = min(NULL_BLOCK_SIZE, grid.width - x)
            core = (x, y, width, height)
            ext_x = max(0, x - radius)
            ext_y = max(0, y - radius)
            ext_x1 = min(grid.width, x + width + radius)
            ext_y1 = min(grid.height, y + height + radius)
            extended = (ext_x, ext_y, ext_x1 - ext_x, ext_y1 - ext_y)
            crop_y = y - ext_y
            crop_x = x - ext_x
            crop = (slice(crop_y, crop_y + height), slice(crop_x, crop_x + width))

            scl_pre = _read_source_block(
                sources["s2_pre_SCL"], offsets["s2_pre_SCL"], core, floating=False
            )
            scl_post = _read_source_block(
                sources["s2_post_SCL"], offsets["s2_post_SCL"], core, floating=False
            )
            reflectance: dict[str, np.ndarray] = {}
            for period in ("pre", "post"):
                for band_name in ("B8A", "B12"):
                    raw = _read_source_block(
                        sources[f"s2_{period}_{band_name}"],
                        offsets[f"s2_{period}_{band_name}"],
                        core,
                        floating=False,
                    )
                    offset, quantification = boa[period][band_name]
                    reflectance[f"{period}_{band_name}"] = _scale_reflectance(
                        raw,
                        boa_add_offset=offset,
                        boa_quantification_value=quantification,
                    ).astype(np.float32)
            block_dnbr = dnbr(
                reflectance["pre_B8A"],
                reflectance["pre_B12"],
                reflectance["post_B8A"],
                reflectance["post_B12"],
            )
            optical_alpha = np.minimum(
                scl_optical_reliability(scl_pre, unreliable_classes=(3, 8, 9, 10)),
                scl_optical_reliability(scl_post, unreliable_classes=(3, 8, 9, 10)),
            )

            changes: dict[str, np.ndarray] = {}
            for polarization in POLARIZATIONS:
                sar: dict[str, np.ndarray] = {}
                for period in ("pre", "post"):
                    sar[period] = _read_source_block(
                        sources[f"s1_{period}_{polarization}"],
                        offsets[f"s1_{period}_{polarization}"],
                        extended,
                        floating=True,
                    )
                unfiltered = sar["post"][crop] - sar["pre"][crop]
                changes[f"{polarization}_unfiltered"] = unfiltered
                changes[f"{polarization}_boxcar_5x5"] = _filtered_sar_change(
                    sar["pre"], sar["post"], crop
                )

            layover_values: list[np.ndarray] = []
            ls_valid = np.ones((height, width), dtype=bool)
            for period in ("pre", "post"):
                values = _read_source_block(
                    sources[f"s1_{period}_ls_map"],
                    offsets[f"s1_{period}_ls_map"],
                    core,
                    floating=True,
                )
                valid = np.isfinite(values)
                ls_valid &= valid
                safe = np.where(valid, values, 0).astype(np.uint8)
                layover_values.append(~valid | ((safe & LS_FLAG_MASK) != 0))
            layover_shadow = layover_values[0] | layover_values[1]
            land = (
                np.isfinite(scl_pre)
                & np.isfinite(scl_post)
                & (scl_pre != 0)
                & (scl_post != 0)
                & (scl_pre != 6)
                & (scl_post != 6)
            )
            for values in changes.values():
                land &= np.isfinite(values)
            coast_seed = (
                ~np.isfinite(scl_pre)
                | ~np.isfinite(scl_post)
                | (scl_pre == 0)
                | (scl_post == 0)
                | (scl_pre == 6)
                | (scl_post == 6)
            )
            for layer, values in changes.items():
                signal_ds.GetRasterBand(NULL_LAYER_BANDS[layer]).WriteArray(values, x, y)
            signal_ds.GetRasterBand(NULL_LAYER_BANDS["dnbr_values"]).WriteArray(
                block_dnbr, x, y
            )
            signal_ds.GetRasterBand(NULL_LAYER_BANDS["optical_alpha"]).WriteArray(
                optical_alpha, x, y
            )
            mask_ds.GetRasterBand(MASK_LAYER_BANDS["land"]).WriteArray(
                land.astype(np.uint8), x, y
            )
            mask_ds.GetRasterBand(MASK_LAYER_BANDS["pre_scl"]).WriteArray(
                np.asarray(scl_pre, dtype=np.uint8), x, y
            )
            mask_ds.GetRasterBand(MASK_LAYER_BANDS["layover_shadow"]).WriteArray(
                layover_shadow.astype(np.uint8), x, y
            )
            mask_ds.GetRasterBand(MASK_LAYER_BANDS["coast_seed"]).WriteArray(
                coast_seed.astype(np.uint8), x, y
            )
            mask_ds.GetRasterBand(MASK_LAYER_BANDS["ls_valid"]).WriteArray(
                ls_valid.astype(np.uint8), x, y
            )
    sources.clear()
    signal_ds.FlushCache()
    mask_ds.FlushCache()
    grade_ds.FlushCache()
    error = gdal.ComputeProximity(
        mask_ds.GetRasterBand(MASK_LAYER_BANDS["coast_seed"]),
        distance_ds.GetRasterBand(1),
        options=["VALUES=1", "DISTUNITS=GEO"],
    )
    if error != gdal.CE_None:
        raise RuntimeError("GDAL failed computing full-intersection coast distances.")

    for y in range(0, grid.height, NULL_BLOCK_SIZE):
        height = min(NULL_BLOCK_SIZE, grid.height - y)
        for x in range(0, grid.width, NULL_BLOCK_SIZE):
            width = min(NULL_BLOCK_SIZE, grid.width - x)
            land = mask_ds.GetRasterBand(MASK_LAYER_BANDS["land"]).ReadAsArray(
                x, y, width, height
            ).astype(bool)
            coast_distance = distance_ds.GetRasterBand(1).ReadAsArray(
                x, y, width, height
            )
            alpha = signal_ds.GetRasterBand(NULL_LAYER_BANDS["optical_alpha"]).ReadAsArray(
                x, y, width, height
            )
            dnbr_block = signal_ds.GetRasterBand(
                NULL_LAYER_BANDS["dnbr_values"]
            ).ReadAsArray(x, y, width, height)
            ls_valid = mask_ds.GetRasterBand(MASK_LAYER_BANDS["ls_valid"]).ReadAsArray(
                x, y, width, height
            ).astype(bool)
            fire_exclusion = mask_ds.GetRasterBand(
                MASK_LAYER_BANDS["fire_exclusion"]
            ).ReadAsArray(x, y, width, height).astype(bool)
            grading_exclusion = mask_ds.GetRasterBand(
                MASK_LAYER_BANDS["grading_exclusion"]
            ).ReadAsArray(x, y, width, height).astype(bool)
            valid_signals = land.copy()
            for signal_band in range(1, 5):
                valid_signals &= np.isfinite(
                    signal_ds.GetRasterBand(signal_band).ReadAsArray(
                        x, y, width, height
                    )
                )
            placeable = (
                land
                & np.isfinite(coast_distance)
                & (coast_distance >= coast_buffer_m)
                & np.isfinite(alpha)
                & (alpha == 1.0)
                & np.isfinite(dnbr_block)
                & ls_valid
                & valid_signals
                & ~fire_exclusion
                & ~grading_exclusion
            )
            mask_ds.GetRasterBand(MASK_LAYER_BANDS["placeable"]).WriteArray(
                placeable.astype(np.uint8), x, y
            )
    mask_ds.FlushCache()
    distance_ds.FlushCache()
    sources.clear()

    scene = {
        "grid": grid,
        "width": grid.width,
        "height": grid.height,
        "template_mask": template,
        "template_x": template_x,
        "template_y": template_y,
        "signal_ds": signal_ds,
        "mask_ds": mask_ds,
        "grade_ds": grade_ds,
        "distance_ds": distance_ds,
        "reference_scl_mix": (0.0, 0.0),
        "repaired_geometries": repaired_grades,
    }
    fire_scl = _read_null_layer(scene, "pre_scl", (
        slice(template_y, template_y + template_height),
        slice(template_x, template_x + template_width),
    ))
    class4 = int(np.count_nonzero(template & (fire_scl == 4)))
    class5 = int(np.count_nonzero(template & (fire_scl == 5)))
    if class4 + class5 == 0:
        raise ValueError("The real fire has no pre-fire SCL class-4/5 land pixels.")
    scene["reference_scl_counts"] = (class4, class5)
    scene["reference_scl_mix"] = (
        class4 / (class4 + class5),
        class5 / (class4 + class5),
    )
    return scene


def _read_null_layer(
    scene: dict[str, Any],
    name: str,
    window: tuple[slice, slice],
) -> np.ndarray:
    y0, y1 = window[0].start, window[0].stop
    x0, x1 = window[1].start, window[1].stop
    if "read_layer" in scene:
        return scene["read_layer"](name, window)
    width, height = x1 - x0, y1 - y0
    if name in NULL_LAYER_BANDS:
        band = scene["signal_ds"].GetRasterBand(NULL_LAYER_BANDS[name])
    elif name in MASK_LAYER_BANDS:
        band = scene["mask_ds"].GetRasterBand(MASK_LAYER_BANDS[name])
    elif name.startswith("grade:"):
        grade = name.partition(":")[2]
        band = scene["grade_ds"].GetRasterBand(GRADE_LABELS.index(grade) + 1)
    elif name == "coast_distance":
        band = scene["distance_ds"].GetRasterBand(1)
    else:
        raise KeyError(f"Unknown null-scene layer: {name}")
    values = band.ReadAsArray(x0, y0, width, height)
    if values is None:
        raise RuntimeError(f"GDAL failed reading null-scene layer {name!r}.")
    return np.asarray(values)


def _close_null_scene(scene: dict[str, Any]) -> None:
    for name in ("signal_ds", "mask_ds", "grade_ds", "distance_ds"):
        scene[name] = None


def _iter_null_blocks(scene: dict[str, Any]) -> Iterator[tuple[int, int, int, int]]:
    for y in range(0, scene["height"], NULL_BLOCK_SIZE):
        height = min(NULL_BLOCK_SIZE, scene["height"] - y)
        for x in range(0, scene["width"], NULL_BLOCK_SIZE):
            width = min(NULL_BLOCK_SIZE, scene["width"] - x)
            yield x, y, width, height


def _compute_null_reference(scene: dict[str, Any]) -> dict[str, Any]:
    key_by_band = {
        band: key
        for key, band in NULL_LAYER_BANDS.items()
        if key.endswith(("unfiltered", "boxcar_5x5"))
    }
    background_sums = {key: 0.0 for key in key_by_band.values()}
    background_counts = {key: 0 for key in key_by_band.values()}
    grade_sums = {
        grade: {key: 0.0 for key in key_by_band.values()} for grade in GRADE_LABELS
    }
    grade_counts = {
        grade: {key: 0 for key in key_by_band.values()} for grade in GRADE_LABELS
    }
    for x, y, width, height in _iter_null_blocks(scene):
        window = (slice(y, y + height), slice(x, x + width))
        land = _read_null_layer(scene, "land", window).astype(bool)
        fire_exclusion = _read_null_layer(scene, "fire_exclusion", window).astype(bool)
        background = land & ~fire_exclusion
        grade_masks = {
            grade: _read_null_layer(scene, f"grade:{grade}", window).astype(bool)
            & land
            for grade in GRADE_LABELS
        }
        for key in key_by_band.values():
            values = _read_null_layer(scene, key, window).astype(np.float64)
            selected = values[background & np.isfinite(values)]
            background_sums[key] += float(np.sum(selected, dtype=np.float64))
            background_counts[key] += int(selected.size)
            for grade in GRADE_LABELS:
                grade_values = values[grade_masks[grade] & np.isfinite(values)]
                grade_sums[grade][key] += float(
                    np.sum(grade_values, dtype=np.float64)
                )
                grade_counts[grade][key] += int(grade_values.size)

    template_window = (
        slice(scene["template_y"], scene["template_y"] + scene["template_mask"].shape[0]),
        slice(scene["template_x"], scene["template_x"] + scene["template_mask"].shape[1]),
    )
    fire_land = (
        scene["template_mask"]
        & _read_null_layer(scene, "land", template_window).astype(bool)
    )
    fire_counts = int(fire_land.sum())
    if fire_counts == 0:
        raise ValueError("The fire delineation contains no valid land in the intersection.")
    observed: dict[str, float] = {}
    grades: dict[str, dict[str, float]] = {}
    for key in key_by_band.values():
        background_count = background_counts[key]
        if background_count <= fire_counts:
            raise ValueError("Too few valid land pixels remain outside the fire buffer.")
        values = _read_null_layer(scene, key, template_window).astype(np.float64)
        inside_values = values[fire_land]
        if not np.all(np.isfinite(inside_values)):
            raise RuntimeError("The real fire sample contains missing SAR changes.")
        background_mean = background_sums[key] / background_count
        observed[key] = float(np.mean(inside_values) - background_mean)
        for grade in GRADE_LABELS:
            count = grade_counts[grade][key]
            if count:
                grades.setdefault(grade, {})[key] = float(
                    grade_sums[grade][key] / count - background_mean
                )
    return {
        "background_sums": background_sums,
        "background_counts": background_counts,
        "observed": observed,
        "grades": grades,
        "fire_count": fire_counts,
    }


def _null_summary_statistics(
    distribution: np.ndarray,
    observed: float,
) -> dict[str, float]:
    values = np.asarray(distribution, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "p_two_sided": math.nan,
            "z_score": math.nan,
            "percentile": math.nan,
            "p2_5": math.nan,
            "p50": math.nan,
            "p97_5": math.nan,
            "null_mean": math.nan,
            "null_std": math.nan,
        }
    null_mean = float(np.mean(finite))
    null_std = float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan
    z_score = (observed - null_mean) / null_std if null_std > 0 else math.nan
    return {
        "p_two_sided": _empirical_two_sided_p(finite, observed),
        "z_score": float(z_score),
        "percentile": 100.0 * float(np.mean(finite <= observed)),
        "p2_5": float(np.percentile(finite, 2.5)),
        "p50": float(np.percentile(finite, 50.0)),
        "p97_5": float(np.percentile(finite, 97.5)),
        "null_mean": null_mean,
        "null_std": null_std,
    }


def _save_null_histograms(
    null_results: dict[str, list[float]],
    observed: dict[str, float],
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for key, results in null_results.items():
        distribution = np.asarray(results, dtype=np.float64)
        if distribution.size == 0:
            continue
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.hist(distribution, bins="auto", color="#4472C4", alpha=0.8)
        axis.axvline(
            observed[key],
            color="#C00000",
            linewidth=2,
            linestyle="--",
            label=f"real contrast {observed[key]:+.4f} dB",
        )
        axis.axvline(
            float(np.median(distribution)),
            color="black",
            linewidth=1.5,
            label=f"null median {np.median(distribution):+.4f} dB",
        )
        axis.set_title(f"Chios SAR placement null: {key}")
        axis.set_xlabel("Inside minus remaining-background SAR change (dB)")
        axis.set_ylabel("Accepted placements")
        axis.legend()
        path = OUTPUT_DIR / f"null_test_v2_{key}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _run_v2_null_test(
    *,
    null_replicates: int,
    seed: int,
    max_attempts: int,
    coast_buffer_m: float,
    scl_mix_tolerance: float,
    max_layover_fraction: float,
) -> dict[str, Any]:
    if null_replicates < 1:
        raise ValueError("null_replicates must be positive.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    if coast_buffer_m < 0:
        raise ValueError("coast_buffer_m must be non-negative.")
    if scl_mix_tolerance < 0:
        raise ValueError("scl_mix_tolerance must be non-negative.")
    if not 0 <= max_layover_fraction <= 1:
        raise ValueError("max_layover_fraction must be in [0, 1].")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "null_test_v2.csv"
    summary_path = OUTPUT_DIR / "null_test_v2_summary.txt"
    keys = ("VH_unfiltered", "VH_boxcar_5x5", "VV_unfiltered", "VV_boxcar_5x5")
    fieldnames = [
        "replicate",
        "attempts",
        "accepted",
        "column_shift_px",
        "row_shift_px",
        "candidate_dnbr_mean",
        "candidate_dnbr_ge_0_10_fraction",
        "candidate_scl4_share",
        "candidate_scl5_share",
        "candidate_layover_shadow_fraction",
        *keys,
    ]
    null_results: dict[str, list[float]] = {key: [] for key in keys}
    accepted = 0
    offset_attempts = 0
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(
            "Option A full-intersection placement null; null statistics are "
            "written to null_test_v2.csv as replicates finish.\n"
            f"Requested replicates: {null_replicates}\n"
            f"Hard offset-attempt cap per replicate: {max_attempts}\n"
            f"Coast exclusion distance: {coast_buffer_m:g} m\n"
            f"Fire and EMS-grade exclusion buffer: "
            f"{FIRE_GRADE_EXCLUSION_METERS:g} m\n"
            f"Pre-fire class-4/5 share tolerance: +/-{scl_mix_tolerance:.4f}\n"
            f"Maximum layover/shadow fraction: {max_layover_fraction:.4f}\n"
            f"Null background: all valid land outside the real-fire "
            f"{FIRE_GRADE_EXCLUSION_METERS:g} m buffer; "
            "each placed mask is removed from its own outside comparison.\n"
        )
        summary_file.flush()
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()
            with tempfile.TemporaryDirectory(prefix="chios-null-v2-") as temp_dir:
                scene = _prepare_full_null_scene(
                    temp_dir, coast_buffer_m=coast_buffer_m
                )
                reference = _compute_null_reference(scene)
                summary_file.write(
                    f"Full intersection: {scene['width']} x {scene['height']} "
                    f"pixels; valid real-fire pixels: {reference['fire_count']}; "
                    f"pre-fire SCL class-4/5 shares: "
                    f"{scene['reference_scl_mix'][0]:.6f}/"
                    f"{scene['reference_scl_mix'][1]:.6f}; counts="
                    f"{scene['reference_scl_counts'][0]}/"
                    f"{scene['reference_scl_counts'][1]}.\n"
                )
                summary_file.flush()
                for replicate, attempts, placement in _iter_random_null_placements(
                    scene,
                    requested=null_replicates,
                    seed=seed,
                    max_attempts=max_attempts,
                    coast_buffer_m=coast_buffer_m,
                    scl_mix_tolerance=scl_mix_tolerance,
                    max_layover_fraction=max_layover_fraction,
                ):
                    offset_attempts += attempts
                    row: dict[str, Any] = {
                        "replicate": replicate,
                        "attempts": attempts,
                        "accepted": placement is not None,
                        "column_shift_px": "",
                        "row_shift_px": "",
                        "candidate_dnbr_mean": "",
                        "candidate_dnbr_ge_0_10_fraction": "",
                        "candidate_scl4_share": "",
                        "candidate_scl5_share": "",
                        "candidate_layover_shadow_fraction": "",
                    }
                    if placement is not None:
                        accepted += 1
                        column_shift, row_shift, window, candidate = placement
                        row["column_shift_px"] = column_shift
                        row["row_shift_px"] = row_shift
                        candidate_count = int(candidate.sum())
                        dnbr_values = _read_null_layer(
                            scene, "dnbr_values", window
                        )[candidate]
                        scl_values = _read_null_layer(scene, "pre_scl", window)[candidate]
                        layover_values = _read_null_layer(
                            scene, "layover_shadow", window
                        )[candidate].astype(bool)
                        class4_count = int(np.count_nonzero(scl_values == 4))
                        class5_count = int(np.count_nonzero(scl_values == 5))
                        clear_count = class4_count + class5_count
                        row["candidate_dnbr_mean"] = float(np.mean(dnbr_values))
                        row["candidate_dnbr_ge_0_10_fraction"] = float(
                            np.mean(dnbr_values >= 0.10)
                        )
                        row["candidate_scl4_share"] = class4_count / clear_count
                        row["candidate_scl5_share"] = class5_count / clear_count
                        row["candidate_layover_shadow_fraction"] = float(
                            np.mean(layover_values)
                        )
                        for key in keys:
                            placed_values = _read_null_layer(
                                scene, key, window
                            )[candidate]
                            placed_sum = float(
                                np.sum(placed_values, dtype=np.float64)
                            )
                            difference = _null_difference_from_sums(
                                placed_sum,
                                candidate_count,
                                reference["background_sums"][key],
                                reference["background_counts"][key],
                            )
                            null_results[key].append(difference)
                            row[key] = f"{difference:.8f}"
                    writer.writerow(row)
                    csv_file.flush()
                    if replicate % 10 == 0 or replicate == null_replicates:
                        summary_file.write(
                            f"Progress: {replicate}/{null_replicates} replicates; "
                            f"accepted={accepted}/{replicate}; "
                            f"offsets_attempted={offset_attempts}.\n"
                        )
                        summary_file.flush()
                        print(
                            f"Null-v2 progress: {replicate}/{null_replicates}; "
                            f"accepted={accepted}/{replicate}; "
                            f"offsets attempted={offset_attempts}",
                            flush=True,
                        )
                _close_null_scene(scene)

                summary_file.write(
                    f"\nSuccessful placements: {accepted}/{null_replicates} "
                    f"replicates; {offset_attempts} total whole-pixel offsets "
                    f"attempted (cap {max_attempts} per replicate).\n"
                )
                if accepted < min(100, null_replicates):
                    summary_file.write(
                        "WARNING: fewer than approximately 100 placements were "
                        "accepted; all reported statistics use the smaller sample.\n"
                    )
                summary_file.write(
                    "Two-sided p uses 2*min(lower-tail, upper-tail), each tail "
                    "with the +1 finite-sample correction. z=(observed-null mean)"
                    "/null sample SD. Percentile is 100*P(null<=observed).\n"
                )
                summary_file.write(
                    "Placements may overlap each other and are not independent; "
                    "the empirical p-value describes this placement process, "
                    "has resolution at least 1/(accepted+1), and is not an exact "
                    "independent-replicate significance test.\n"
                )
                if accepted == 0:
                    summary_file.write(
                        "No null distribution was available; no p-value, z-score, "
                        "percentile, or interval can be estimated.\n"
                    )
                for key in keys:
                    stats = _null_summary_statistics(
                        np.asarray(null_results[key], dtype=np.float64),
                        reference["observed"][key],
                    )
                    if accepted:
                        is_tail = stats["p_two_sided"] < 0.05
                        conclusion = (
                            "in the empirical null tail; this does not select a sign"
                            if is_tail
                            else "not distinguishable from background at this scale and event"
                        )
                        summary_file.write(
                            f"\n{key}: real contrast={reference['observed'][key]:+.8f} dB; "
                            f"p_two_sided={stats['p_two_sided']:.6f}; "
                            f"z={stats['z_score']:+.6f}; "
                            f"percentile={stats['percentile']:.3f}%; "
                            f"null p2.5/p50/p97.5="
                            f"{stats['p2_5']:+.8f}/{stats['p50']:+.8f}/"
                            f"{stats['p97_5']:+.8f} dB; {conclusion}.\n"
                        )
                    else:
                        summary_file.write(
                            f"\n{key}: real contrast={reference['observed'][key]:+.8f} dB; "
                            "no accepted placements; null p-value, z-score, "
                            "percentile, and interval unavailable.\n"
                        )
                    for grade in GRADE_LABELS:
                        grade_observed = reference["grades"].get(grade, {}).get(key)
                        if grade_observed is None:
                            continue
                        if not accepted:
                            summary_file.write(
                                f"  damage_grade={grade}: real contrast="
                                f"{grade_observed:+.8f} dB; null distribution "
                                "unavailable.\n"
                            )
                            continue
                        grade_stats = _null_summary_statistics(
                            np.asarray(null_results[key], dtype=np.float64),
                            grade_observed,
                        )
                        grade_conclusion = (
                            "in the empirical null tail; no sign decision"
                            if grade_stats["p_two_sided"] < 0.05
                            else "not distinguishable from background at this scale and event"
                        )
                        summary_file.write(
                            f"  damage_grade={grade}: real contrast="
                            f"{grade_observed:+.8f} dB; "
                            f"p_two_sided={grade_stats['p_two_sided']:.6f}; "
                            f"z={grade_stats['z_score']:+.6f}; "
                            f"percentile={grade_stats['percentile']:.3f}%; "
                            f"null p2.5/p50/p97.5="
                            f"{grade_stats['p2_5']:+.8f}/"
                            f"{grade_stats['p50']:+.8f}/"
                            f"{grade_stats['p97_5']:+.8f} dB; {grade_conclusion}.\n"
                        )
                summary_file.flush()

    histogram_paths = _save_null_histograms(null_results, reference["observed"])
    print(
        f"Null-v2 placements accepted: {accepted}/{null_replicates}; "
        f"offsets attempted={offset_attempts}."
    )
    print(f"Saved null-v2 replicate table: {csv_path}")
    print(f"Saved null-v2 summary: {summary_path}")
    for path in histogram_paths:
        print(f"Saved null-v2 histogram: {path}")
    return {
        "null_results": null_results,
        "observed": reference["observed"],
        "grade_observed": reference["grades"],
        "accepted": accepted,
        "attempted_replicates": null_replicates,
        "offset_attempts": offset_attempts,
        "null_csv_path": csv_path,
        "null_summary_path": summary_path,
        "histogram_paths": histogram_paths,
    }


def _print_summary(label: str, summary: dict[str, float | int]) -> None:
    print(
        f"{label}: n_in={summary['inside_count']:,}, "
        f"mean_in={summary['inside_mean']:+.5f}, "
        f"median_in={summary['inside_median']:+.5f}, "
        f"sd_in={summary['inside_std']:.5f}; "
        f"n_out={summary['outside_count']:,}, "
        f"mean_out={summary['outside_mean']:+.5f}, "
        f"median_out={summary['outside_median']:+.5f}, "
        f"sd_out={summary['outside_std']:.5f}; "
        f"inside-outside={summary['difference']:+.5f} dB; "
        f"pooled_sd={summary['pooled_std']:.5f}; "
        f"effect={summary['effect_size']:+.4f}"
    )


def run_diagnostics(
    *,
    null_replicates: int = 500,
    seed: int = 2024,
    max_attempts: int = 500,
    coast_buffer_m: float = 500.0,
    scl_mix_tolerance: float = 0.10,
    max_layover_fraction: float = 0.01,
) -> dict[str, Any]:
    scene = _prepare_scene()
    land = scene["land"]
    ems_land = land & scene["ems_mask"]
    outside_land = land & ~scene["fire_mask"]
    print(
        "ONE-TIME SAR SIGN DIAGNOSTIC (reference comparison only; not a tuning step)"
    )
    print(
        f"AOI window={scene['grid_window']}; valid land={int(land.sum()):,}; "
        f"valid land inside EMSN198_AOI={int(ems_land.sum()):,}; "
        f"fire land={int((land & scene['fire_mask']).sum()):,}; "
        f"outside-fire land={int(outside_land.sum()):,}; "
        f"terrain-flagged/NoData pixels excluded from land={int((~land).sum()):,}."
    )
    print(
        "Geometry repair cache: "
        f"delineation={scene['fire_geometry_repaired']}; "
        f"EMSN198_AOI={scene['ems_geometry_repaired']}; "
        f"damage-grade geometries={scene['repaired_grade_geometries']}."
    )
    offsets: dict[str, float] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    for polarization in POLARIZATIONS:
        for mode, values in scene["changes"][polarization].items():
            key = f"{polarization}_{mode}"
            land_values = values[land]
            outside_values = values[outside_land]
            summary = _contrast_summary(values[land & scene["fire_mask"]], outside_values)
            summaries[key] = summary
            offsets[key] = float(np.median(land_values))
            print(f"\n{key}; median centering offset={offsets[key]:+.5f} dB")
            _print_summary("  delineation versus all outside-fire land", summary)
            print(
                f"  outside-fire AOI-window land offset: "
                f"mean={summary['outside_mean']:+.5f} dB; "
                f"median={summary['outside_median']:+.5f} dB"
            )
            print(
                f"  AOI-land histogram significant modes="
                f"{_significant_histogram_modes(land_values)} "
                "(256-bin histogram, 31-bin smooth, 10% peak-height cutoff)."
            )

    grade_masks = scene["grade_masks"]
    print("\nDAMAGE-GRADE SIGN BREAKDOWN (cached, MakeValid-repaired masks)")
    for grade in GRADE_LABELS:
        if grade not in grade_masks:
            print(f"  {grade}: no polygons")
            continue
        grade_land = land & grade_masks[grade]
        if not np.any(grade_land):
            print(f"  {grade}: no valid land pixels in the AOI window")
            continue
        print(f"  {grade}:")
        for polarization in POLARIZATIONS:
            for mode, values in scene["changes"][polarization].items():
                summary = _contrast_summary(values[grade_land], values[outside_land])
                _print_summary(f"    {polarization} {mode}", summary)

    print(
        "\nOption A null: whole-pixel placements are sampled across all valid land "
        "in the full S2/S1 intersection, not the EMSN198 AOI. The fire/reference "
        "direction is examined here, so any eventual sign selection remains "
        "label-informed; no sign is selected by this diagnostic."
    )
    null_result = _run_v2_null_test(
        null_replicates=null_replicates,
        seed=seed,
        max_attempts=max_attempts,
        coast_buffer_m=coast_buffer_m,
        scl_mix_tolerance=scl_mix_tolerance,
        max_layover_fraction=max_layover_fraction,
    )
    return {
        "scene": scene,
        "summaries": summaries,
        **null_result,
        "offsets": offsets,
        "grade_masks": scene["grade_masks"],
        "land": land,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-replicates", type=int, default=500)
    parser.add_argument("--attempt-cap", type=int, default=500)
    parser.add_argument("--coast-buffer-m", type=float, default=500.0)
    parser.add_argument("--scl-mix-tolerance", type=float, default=0.10)
    parser.add_argument("--max-layover-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)
    arguments = parser.parse_args()
    run_diagnostics(
        null_replicates=arguments.null_replicates,
        seed=arguments.seed,
        max_attempts=arguments.attempt_cap,
        coast_buffer_m=arguments.coast_buffer_m,
        scl_mix_tolerance=arguments.scl_mix_tolerance,
        max_layover_fraction=arguments.max_layover_fraction,
    )


if __name__ == "__main__":
    main()
