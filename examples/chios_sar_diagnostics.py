"""Read-only Sentinel-1 sign and background-contrast diagnostics for Chios.

Run from the repository root with:
    .venv-qgis/bin/python -m examples.chios_sar_diagnostics

This compares the mapped fire to damage-grade polygons and randomized
same-shape translations inside the EMSN198 AOI. It reports sign evidence only;
it does not select a classifier sign or compute accuracy scores.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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
        for mode in ("unfiltered", "boxcar_5x5"):
            changes[polarization][mode][~land] = np.nan

    ems_geometry, ems_srs = _open_reference_geometry("EMSN198_AOI")
    ems_geometry = _transform_geometry(ems_geometry, ems_srs, reference_srs)
    ems_mask = _rasterize_geometry(
        ems_geometry, reference_srs, output_gt, width, height
    )
    fire_mask = _rasterize_geometry(
        fire_geometry, reference_srs, output_gt, width, height
    )
    return {
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
        "ems_geometry": ems_geometry,
        "ems_mask": ems_mask,
        "fire_mask": fire_mask,
        "land": land,
        "changes": changes,
    }


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
        if not geometry.IsValid():
            geometry = geometry.MakeValid()
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


def _random_null_masks(
    scene: dict[str, Any],
    *,
    requested: int,
    seed: int,
) -> tuple[list[np.ndarray], int, int]:
    """Sample unique whole-pixel translations of the delineation inside EMS AOI."""
    if requested < 1:
        raise ValueError("requested null replicate count must be positive.")
    fire_min_x, fire_min_y, fire_max_x, fire_max_y = scene["fire_bbox"]
    ems_min_x, ems_min_y, ems_max_x, ems_max_y = scene["ems_geometry"].GetEnvelope()
    column_min = math.ceil((ems_min_x - fire_min_x) / PIXEL_SIZE)
    column_max = math.floor((ems_max_x - fire_max_x) / PIXEL_SIZE)
    row_min = math.ceil((fire_max_y - ems_max_y) / PIXEL_SIZE)
    row_max = math.floor((fire_min_y - ems_min_y) / PIXEL_SIZE)
    if column_min > column_max or row_min > row_max:
        raise ValueError("No same-shape placements fit inside the EMSN198 AOI bounds.")
    offsets = [
        (column_shift, row_shift)
        for row_shift in range(row_min, row_max + 1)
        for column_shift in range(column_min, column_max + 1)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(offsets)
    fire_mask = scene["fire_mask"]
    land = scene["land"]
    ems = scene["ems_mask"]
    accepted: list[np.ndarray] = []
    examined = 0
    for column_shift, row_shift in offsets:
        examined += 1
        candidate = _translate_mask(
            fire_mask,
            column_shift=column_shift,
            row_shift=row_shift,
        )
        if candidate.sum() != fire_mask.sum():
            continue
        if np.any(candidate & fire_mask):
            continue
        if not np.all(ems[candidate]) or not np.all(land[candidate]):
            continue
        accepted.append(candidate)
        if len(accepted) == requested:
            break
    return accepted, examined, len(offsets)


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
    fire_land = land & scene["fire_mask"]
    outside_land = land & ~scene["fire_mask"]
    print(
        "ONE-TIME SAR SIGN DIAGNOSTIC (reference comparison only; not a tuning step)"
    )
    print(
        f"AOI window={scene['grid_window']}; valid land={int(land.sum()):,}; "
        f"fire land={int(fire_land.sum()):,}; outside-fire land={int(outside_land.sum()):,}; "
        f"terrain-flagged/NoData pixels excluded from land={int((~land).sum()):,}."
    )
    offsets: dict[str, float] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    for polarization in POLARIZATIONS:
        for mode, values in scene["changes"][polarization].items():
            key = f"{polarization}_{mode}"
            land_values = values[land]
            outside_values = values[outside_land]
            summary = _contrast_summary(values[fire_land], outside_values)
            summaries[key] = summary
            offsets[key] = float(np.median(land_values))
            print(f"\n{key}; median centering offset={offsets[key]:+.5f} dB")
            _print_summary("  delineation versus outside-fire land", summary)
            print(
                f"  outside-fire global offset: mean={summary['outside_mean']:+.5f} dB; "
                f"median={summary['outside_median']:+.5f} dB"
            )
            print(
                f"  AOI-land histogram significant modes="
                f"{_significant_histogram_modes(land_values)} "
                "(256-bin histogram, 31-bin smooth, 10% peak-height cutoff)."
            )

    grade_masks, repaired = _damage_grade_masks(scene)
    print(f"\nDAMAGE-GRADE SIGN BREAKDOWN (invalid geometries repaired in memory={repaired})")
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
                print(
                    f"    {polarization} {mode}: class n={summary['inside_count']:,}, "
                    f"mean={summary['inside_mean']:+.5f}, "
                    f"median={summary['inside_median']:+.5f}, "
                    f"sd={summary['inside_std']:.5f}, "
                    f"vs outside mean difference={summary['difference']:+.5f} dB, "
                    f"pooled effect={summary['effect_size']:+.4f}"
                )

    null_masks, examined, total_offsets = _random_null_masks(
        scene,
        requested=null_replicates,
        seed=seed,
    )
    if not null_masks:
        raise RuntimeError("No valid random same-shape land placements were found.")
    print(
        f"\nSAME-SHAPE NULL TEST: accepted {len(null_masks)} unique translations "
        f"from {total_offsets:,} EMS-bounds offsets "
        f"(examined {examined:,}); constrained to EMSN198_AOI, SCL/SAR-valid land, "
        "and no delineation overlap. Null polygons may overlap each other."
    )
    null_results: dict[str, np.ndarray] = {}
    for polarization in POLARIZATIONS:
        for mode, values in scene["changes"][polarization].items():
            distribution = np.empty(len(null_masks), dtype=np.float64)
            for index, null_mask in enumerate(null_masks):
                null_inside = land & null_mask
                null_outside = land & ~null_mask
                distribution[index] = float(
                    np.mean(values[null_inside]) - np.mean(values[null_outside])
                )
            key = f"{polarization}_{mode}"
            observed = float(summaries[key]["difference"])
            upper_p = _empirical_upper_tail_p(distribution, observed)
            two_sided_p = _empirical_two_sided_p(distribution, observed)
            percentile = 100.0 * float(np.mean(distribution <= observed))
            null_results[key] = distribution
            print(
                f"  {key}: observed inside-outside={observed:+.5f} dB; "
                f"null median={np.median(distribution):+.5f}, "
                f"null p2.5/p97.5={np.percentile(distribution, [2.5, 97.5])[0]:+.5f}/"
                f"{np.percentile(distribution, [2.5, 97.5])[1]:+.5f}; "
                f"percentile={percentile:.1f}%; "
                f"empirical p(upper)={upper_p:.4f}, p(two-sided)={two_sided_p:.4f}."
            )
            if two_sided_p >= 0.05:
                print("    Conclusion: this contrast is not distinguishable from background at p<0.05.")
            else:
                print("    Conclusion: this contrast is distinguishable from this placement null at p<0.05.")

    histogram_path = _save_histogram_plot(scene["changes"], land, offsets)
    print(f"\nSaved uncentered/median-centered SAR histograms: {histogram_path}")
    return {
        "scene": scene,
        "summaries": summaries,
        "null_results": null_results,
        "offsets": offsets,
        "histogram_path": histogram_path,
        "grade_masks": grade_masks,
        "land": land,
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
