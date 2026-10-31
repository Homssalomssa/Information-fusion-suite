"""Read-only Sentinel-1 sign and background-contrast diagnostics for Chios.

Run from the repository root with:
    .venv-qgis/bin/python -m examples.chios_sar_diagnostics

This compares the mapped fire to damage-grade polygons and randomized
same-shape translations inside the EMSN198 AOI. It reports sign evidence only;
it does not select a classifier sign or compute accuracy scores.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from osgeo import gdal, ogr, osr

from dst_fusion.core.preprocessing import (
    boxcar_filter,
    db_to_linear_power,
    linear_power_to_db,
)
from dst_fusion.io.raster import compute_intersection_grid
from examples.audit_chios import (
    AOI_BUFFER_METERS,
    REFERENCE_PATH,
    _find_inputs,
    _get_reference_geometry,
    _pixel_window,
    _rasterize_geometry,
    _read_window,
    _window_for_grid,
)
from examples.run_chios import _read_float_window

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "chios"
PIXEL_SIZE = 20.0
POLARIZATIONS = ("VH", "VV")
GRADE_LABELS = (
    "High severity",
    "Moderate high severity",
    "Moderate low severity",
    "Low severity",
)
LS_FLAG_MASK = 4 | 16
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
    """Two-sided empirical p-value based on absolute contrast magnitude."""
    values = np.asarray(null_values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(observed):
        raise ValueError("Null samples and observed difference must be finite.")
    return float(
        (1 + np.count_nonzero(np.abs(finite) >= abs(observed))) / (finite.size + 1)
    )


def _null_placement_difference(
    values: np.ndarray,
    candidate: np.ndarray,
    candidate_window: tuple[slice, slice],
    land: np.ndarray,
    ems: np.ndarray,
    fire: np.ndarray,
) -> float:
    """Compare a cropped null mask with remaining valid, unburned EMS land."""
    rows, columns = candidate_window
    candidate_values = values[rows, columns][candidate]
    candidate_values = candidate_values[np.isfinite(candidate_values)]
    outside_values = values[land & ems & ~fire]
    outside_values = outside_values[np.isfinite(outside_values)]
    outside_sum = float(np.sum(outside_values, dtype=np.float64))
    candidate_sum = float(np.sum(candidate_values, dtype=np.float64))
    outside_count = outside_values.size - candidate_values.size
    if candidate_values.size == 0 or outside_count <= 0:
        raise ValueError("Null placement requires finite inside and outside samples.")
    return float(
        np.mean(candidate_values)
        - (outside_sum - candidate_sum) / outside_count
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


def _iter_random_null_placements(
    scene: dict[str, Any],
    *,
    requested: int,
    seed: int,
    max_attempts: int = 200,
) -> Iterator[
    tuple[
        int,
        int,
        tuple[int, int, tuple[slice, slice], np.ndarray] | None,
    ]
]:
    """Yield one bounded placement attempt result per requested null replicate."""
    if requested < 1:
        raise ValueError("requested null replicate count must be positive.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    fire_min_x, fire_min_y, fire_max_x, fire_max_y = scene["fire_bbox"]
    ems_min_x, ems_max_x, ems_min_y, ems_max_y = scene["ems_geometry"].GetEnvelope()
    column_min = math.ceil((ems_min_x - fire_min_x) / PIXEL_SIZE)
    column_max = math.floor((ems_max_x - fire_max_x) / PIXEL_SIZE)
    row_min = math.ceil((fire_max_y - ems_max_y) / PIXEL_SIZE)
    row_max = math.floor((fire_min_y - ems_min_y) / PIXEL_SIZE)
    if column_min > column_max or row_min > row_max:
        raise ValueError("No same-shape placements fit inside the EMSN198 AOI bounds.")

    fire_mask = scene["fire_mask"]
    fire_y, fire_x = np.nonzero(fire_mask)
    if fire_y.size == 0:
        raise ValueError("The rasterized fire delineation is empty.")
    fire_y0, fire_y1 = int(fire_y.min()), int(fire_y.max()) + 1
    fire_x0, fire_x1 = int(fire_x.min()), int(fire_x.max()) + 1
    cropped_fire = fire_mask[fire_y0:fire_y1, fire_x0:fire_x1]
    fire_area = int(cropped_fire.sum())
    height, width = fire_mask.shape
    rng = np.random.default_rng(seed)
    seen_offsets: set[tuple[int, int]] = set()

    for replicate in range(1, requested + 1):
        placement = None
        attempt = 0
        for attempt in range(1, max_attempts + 1):
            column_shift = int(rng.integers(column_min, column_max + 1))
            row_shift = int(rng.integers(row_min, row_max + 1))
            offset = (column_shift, row_shift)
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            y0, x0 = fire_y0 + row_shift, fire_x0 + column_shift
            y1, x1 = y0 + cropped_fire.shape[0], x0 + cropped_fire.shape[1]
            if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
                continue
            window = (slice(y0, y1), slice(x0, x1))
            if int(cropped_fire.sum()) != fire_area:
                raise RuntimeError("Cropped fire-mask area changed unexpectedly.")
            if np.any(cropped_fire & fire_mask[window]):
                continue
            if not np.all(scene["ems_mask"][window][cropped_fire]):
                continue
            if not np.all(scene["land"][window][cropped_fire]):
                continue
            placement = (
                column_shift,
                row_shift,
                window,
                cropped_fire.copy(),
            )
            break
        yield replicate, attempt, placement


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


def _save_histogram_plot(
    changes: dict[str, dict[str, np.ndarray]],
    land: np.ndarray,
    offsets: dict[str, float],
) -> Path:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row, polarization in enumerate(POLARIZATIONS):
        for column, mode in enumerate(("unfiltered", "boxcar_5x5")):
            axis = axes[row, column]
            values = changes[polarization][mode][land]
            centered = values - offsets[f"{polarization}_{mode}"]
            low, high = np.percentile(values, (0.5, 99.5))
            axis.hist(
                values,
                bins=180,
                range=(low, high),
                alpha=0.55,
                label="uncentered",
            )
            axis.hist(
                centered,
                bins=180,
                range=(low - offsets[f"{polarization}_{mode}"], high - offsets[f"{polarization}_{mode}"]),
                alpha=0.55,
                label="median-centered",
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1, label="zero")
            axis.set_title(f"{polarization} {mode}: median={offsets[f'{polarization}_{mode}']:+.4f} dB")
            axis.set_xlabel("Post-minus-pre change (dB)")
            axis.set_ylabel("Land-pixel count")
            axis.legend()
    path = OUTPUT_DIR / "chios_sar_histograms_centering.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def run_diagnostics(*, null_replicates: int = 200, seed: int = 2024) -> dict[str, Any]:
    scene = _prepare_scene()
    land = scene["land"]
    ems_land = land & scene["ems_mask"]
    fire_land = ems_land & scene["fire_mask"]
    outside_land = land & ~scene["fire_mask"]
    ems_outside_land = ems_land & ~scene["fire_mask"]
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
    null_observed: dict[str, dict[str, float | int]] = {}
    for polarization in POLARIZATIONS:
        for mode, values in scene["changes"][polarization].items():
            key = f"{polarization}_{mode}"
            land_values = values[land]
            outside_values = values[outside_land]
            summary = _contrast_summary(values[land & scene["fire_mask"]], outside_values)
            summaries[key] = summary
            null_observed[key] = _contrast_summary(
                values[fire_land], values[ems_outside_land]
            )
            offsets[key] = float(np.median(land_values))
            print(f"\n{key}; median centering offset={offsets[key]:+.5f} dB")
            _print_summary("  delineation versus all outside-fire land", summary)
            print(
                f"  outside-fire AOI-window land offset: "
                f"mean={summary['outside_mean']:+.5f} dB; "
                f"median={summary['outside_median']:+.5f} dB"
            )
            print(
                "  null-test observed contrast "
                "(inside delineation vs EMSN198_AOI outside-fire land): "
                f"{null_observed[key]['difference']:+.5f} dB "
                f"(outside n={null_observed[key]['outside_count']:,})"
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

    if not np.any(ems_outside_land):
        raise RuntimeError("No valid unburned EMSN198_AOI land is available for null comparisons.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "null_test.csv"
    keys = [
        f"{polarization}_{mode}"
        for polarization in POLARIZATIONS
        for mode in scene["changes"][polarization]
    ]
    null_results: dict[str, list[float]] = {key: [] for key in keys}
    accepted = 0
    print(
        f"\nSAME-SHAPE NULL TEST: requested={null_replicates}; "
        "at most 200 random-placement attempts per replicate. Each candidate is "
        "a cropped rasterized delineation translation, fully on valid EMSN198 "
        "land and non-overlapping with the real delineation. Scalar contrasts "
        "are written incrementally; candidate arrays are discarded after each row."
    )
    fieldnames = ["replicate", "attempts", "accepted", "column_shift_px", "row_shift_px", *keys]
    print(
        f"Incremental replicate table: {csv_path}"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for replicate, attempts, placement in _iter_random_null_placements(
            scene,
            requested=null_replicates,
            seed=seed,
            max_attempts=200,
        ):
            row: dict[str, Any] = {
                "replicate": replicate,
                "attempts": attempts,
                "accepted": placement is not None,
                "column_shift_px": "",
                "row_shift_px": "",
            }
            if placement is not None:
                accepted += 1
                column_shift, row_shift, candidate_window, candidate = placement
                row["column_shift_px"] = column_shift
                row["row_shift_px"] = row_shift
                for polarization in POLARIZATIONS:
                    for mode, values in scene["changes"][polarization].items():
                        key = f"{polarization}_{mode}"
                        difference = _null_placement_difference(
                            values,
                            candidate,
                            candidate_window,
                            land,
                            scene["ems_mask"],
                            scene["fire_mask"],
                        )
                        null_results[key].append(difference)
                        row[key] = f"{difference:.8f}"
                del candidate
            writer.writerow(row)
            output_file.flush()
            if replicate % 10 == 0:
                print(
                    f"  null placement progress: {replicate}/{null_replicates}; "
                    f"accepted={accepted}",
                    flush=True,
                )
    print(f"  placement successes: {accepted}/{null_replicates}")

    summary_lines = [
        f"Requested placements: {null_replicates}",
        f"Successful placements: {accepted}",
        "Attempt limit: 200 random offsets per replicate",
        "Null controls: remaining valid EMSN198_AOI land outside the real delineation and candidate.",
    ]
    if accepted < min(100, null_replicates):
        low_success = (
            null_replicates >= 200 and accepted < 100
        )
        if low_success:
            print(
                "  NOTE: fewer than 100 of 200 requested placements succeeded; "
                "results are reported for the smaller sample."
            )
            summary_lines.append(
                "WARNING: fewer than 100 of 200 requested placements succeeded; "
                "interpret the smaller null sample cautiously."
            )
    for key in keys:
        distribution = np.asarray(null_results[key], dtype=np.float64)
        if distribution.size == 0:
            print(f"  {key}: no successful null placements; p-value unavailable.")
            summary_lines.append(f"{key}: no successful placements; p-value unavailable.")
            continue
        observed = float(null_observed[key]["difference"])
        upper_p = _empirical_upper_tail_p(distribution, observed)
        two_sided_p = _empirical_two_sided_p(distribution, observed)
        percentile = 100.0 * float(np.mean(distribution <= observed))
        null_median = float(np.median(distribution))
        low, high = (float(value) for value in np.percentile(distribution, [2.5, 97.5]))
        conclusion = (
            "not distinguishable from background at p<0.05"
            if two_sided_p >= 0.05
            else "distinguishable from this placement null at p<0.05"
        )
        print(
            f"  {key}: observed EMS contrast={observed:+.5f} dB; "
            f"null median={null_median:+.5f}, null p2.5/p97.5={low:+.5f}/{high:+.5f}; "
            f"percentile={percentile:.1f}%; empirical p(upper)={upper_p:.4f}, "
            f"p(two-sided)={two_sided_p:.4f}; {conclusion}."
        )
        summary_lines.append(
            f"{key}: observed={observed:+.8f} dB; null median={null_median:+.8f}; "
            f"null 95% interval=[{low:+.8f}, {high:+.8f}]; "
            f"percentile={percentile:.2f}%; p_upper={upper_p:.6f}; "
            f"p_two_sided={two_sided_p:.6f}; {conclusion}."
        )
        null_results[key] = distribution.tolist()
    summary_path = OUTPUT_DIR / "null_test_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved null-test summary: {summary_path}")

    histogram_path = _save_histogram_plot(scene["changes"], land, offsets)
    print(f"\nSaved uncentered/median-centered SAR histograms: {histogram_path}")
    return {
        "scene": scene,
        "summaries": summaries,
        "null_results": null_results,
        "offsets": offsets,
        "histogram_path": histogram_path,
        "grade_masks": scene["grade_masks"],
        "land": land,
        "null_csv_path": csv_path,
        "null_summary_path": summary_path,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2024)
    arguments = parser.parse_args()
    run_diagnostics(
        null_replicates=arguments.null_replicates,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
