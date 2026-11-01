"""Run the frozen Checkpoint C Chios validation once.

Run from the repository root with:
    .venv-qgis/bin/python -m examples.score_checkpoint_c

The script validates the committed configuration before reading raster data,
uses AOI-window reads, and writes scored outputs only to the Checkpoint C
output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, ogr, osr

from possibility_fusion.core.indices import dnbr
from possibility_fusion.core.memberships import dnbr_possibility, sar_possibility
from possibility_fusion.core.possibility import (
    adaptive_fusion,
    conjunctive_fusion,
    decide,
    disjunctive_fusion,
)
from possibility_fusion.core.preprocessing import (
    boxcar_filter,
    center_by_median,
    db_to_linear_power,
    linear_power_to_db,
    robust_standard_deviation,
    scl_optical_reliability,
)
from possibility_fusion.core.temporal import discount_for_date_offset, temporal_reliability
from possibility_fusion.core.validation import (
    binary_metrics_from_counts,
    paired_spatial_block_bootstrap,
)
from possibility_fusion.io.raster import RasterGrid, compute_intersection_grid, read_raster_window
from examples.audit_chios import (
    AOI_BUFFER_METERS,
    DATA,
    REFERENCE_PATH,
    _find_inputs,
    _pixel_window,
    _rasterize_geometry,
    _read_boa_metadata,
    _scale_reflectance,
    _window_for_grid,
)
from examples.run_chios import (
    LS_FLAG_MASK,
    S1_POST_DATE,
    S2_POST_DATE,
    _read_float_window,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "checkpoint_c.json"
OUTPUT_DIR = ROOT / "outputs" / "chios" / "checkpoint_c"
FIRE_LAYER = "P01_FDG_FireDelineation"
GRADING_LAYER = "P01_FDG_FireGrading"
AOI_LAYER = "EMSN198_AOI"
GRADE_LABELS = {
    "High": "High severity",
    "Moderate high": "Moderate high severity",
    "Moderate low": "Moderate low severity",
    "Low": "Low severity",
}

gdal.UseExceptions()
ogr.UseExceptions()


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_committed_config() -> tuple[dict[str, Any], str, str]:
    config_bytes = CONFIG_PATH.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    committed_bytes = subprocess.run(
        ["git", "show", "HEAD:configs/checkpoint_c.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tagged_bytes = subprocess.run(
        ["git", "show", "checkpoint-c-config:configs/checkpoint_c.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    head = _git_output("rev-parse", "HEAD")
    tag_commit = _git_output("rev-parse", "checkpoint-c-config^{commit}")
    if hashlib.sha256(committed_bytes).hexdigest() != config_sha256:
        raise RuntimeError("Working config does not match the config in git HEAD.")
    if hashlib.sha256(tagged_bytes).hexdigest() != config_sha256:
        raise RuntimeError("Working config does not match tag checkpoint-c-config.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag_commit, head],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    config = json.loads(config_bytes)
    if config.get("status") != "pre-registered; not scored":
        raise RuntimeError("Checkpoint config status is not the frozen pre-scoring state.")
    return config, config_sha256, head


def _memory_layer(
    source: ogr.Layer,
    target_srs: osr.SpatialReference,
    *,
    copy_grade: bool = False,
    repair_geometry: bool = False,
) -> tuple[Any, ogr.Layer]:
    driver = ogr.GetDriverByName("Memory")
    datasource = driver.CreateDataSource("")
    layer = datasource.CreateLayer(
        source.GetName(),
        srs=target_srs,
        geom_type=ogr.wkbUnknown,
    )
    if copy_grade:
        layer.CreateField(ogr.FieldDefn("damage_grade", ogr.OFTString))
    source_srs = source.GetSpatialRef().Clone()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = target_srs.Clone()
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = (
        None
        if source_srs.IsSame(target)
        else osr.CoordinateTransformation(source_srs, target)
    )
    source.ResetReading()
    for source_feature in source:
        geometry = source_feature.GetGeometryRef()
        if geometry is None:
            raise RuntimeError(f"Null geometry in reference layer {source.GetName()}.")
        geometry = geometry.Clone()
        if repair_geometry and not geometry.IsValid():
            geometry = geometry.MakeValid()
            if geometry is None or geometry.IsEmpty():
                raise RuntimeError(
                    f"Geometry repair failed in {source.GetName()} feature "
                    f"{source_feature.GetFID()}."
                )
        if transform is not None:
            geometry.Transform(transform)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geometry)
        if copy_grade:
            feature.SetField("damage_grade", source_feature.GetField("damage_grade"))
        if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not copy feature into {source.GetName()} memory layer.")
        feature = None
        source_feature = None
    layer.ResetReading()
    return datasource, layer


def _rasterize_layer(
    layer: ogr.Layer,
    projection_wkt: str,
    geotransform: tuple[float, ...],
    width: int,
    height: int,
    *,
    attribute: str | None = None,
) -> np.ndarray:
    dataset = gdal.GetDriverByName("MEM").Create(
        "", width, height, 1, gdal.GDT_Byte
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection_wkt)
    dataset.GetRasterBand(1).Fill(0)
    options = ["ALL_TOUCHED=FALSE"]
    if attribute is not None:
        options.append(f"ATTRIBUTE={attribute}")
    error = gdal.RasterizeLayer(
        dataset,
        [1],
        layer,
        burn_values=[1],
        options=options,
    )
    if error != gdal.CE_None:
        raise RuntimeError(f"GDAL rasterization failed for layer {layer.GetName()}.")
    result = np.asarray(dataset.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
    dataset = None
    return result


def _polygonal_union(geometries: list[ogr.Geometry]) -> ogr.Geometry:
    polygons = ogr.Geometry(ogr.wkbMultiPolygon)

    def append_polygonal(geometry: ogr.Geometry) -> None:
        geometry_type = geometry.GetGeometryName().upper()
        if geometry_type == "POLYGON":
            if polygons.AddGeometry(geometry) != ogr.OGRERR_NONE:
                raise RuntimeError("Could not assemble polygon geometries for union.")
        elif geometry_type in ("MULTIPOLYGON", "GEOMETRYCOLLECTION"):
            for index in range(geometry.GetGeometryCount()):
                append_polygonal(geometry.GetGeometryRef(index))

    for geometry in geometries:
        append_polygonal(geometry)
    if polygons.GetGeometryCount() == 0:
        raise ValueError("Cannot union an empty set of polygon geometries.")
    result = polygons.UnionCascaded()
    if result is None or result.IsEmpty():
        raise RuntimeError("Could not union the reference polygon geometries.")
    return result


def _rasterize_difference(
    fire_geometries: list[ogr.Geometry],
    excluded_geometries: list[ogr.Geometry],
    spatial_reference: osr.SpatialReference,
    geotransform: tuple[float, ...],
    width: int,
    height: int,
) -> np.ndarray:
    fire_union = _polygonal_union(fire_geometries)
    excluded_union = _polygonal_union(excluded_geometries)
    difference = fire_union.Difference(excluded_union)
    if difference is None:
        raise RuntimeError("Could not subtract excluded grading geometry.")
    return _rasterize_geometry(
        difference,
        spatial_reference,
        geotransform,
        width,
        height,
    )


def _make_reference_masks(
    projection_wkt: str,
    geotransform: tuple[float, ...],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source = ogr.Open(str(REFERENCE_PATH), 0)
    if source is None:
        raise RuntimeError(f"Could not open reference GeoPackage {REFERENCE_PATH}.")
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(projection_wkt)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    fire_ds, fire_layer = _memory_layer(source.GetLayerByName(FIRE_LAYER), target_srs)
    aoi_ds, aoi_layer = _memory_layer(source.GetLayerByName(AOI_LAYER), target_srs)
    grade_ds, grade_layer = _memory_layer(
        source.GetLayerByName(GRADING_LAYER),
        target_srs,
        copy_grade=True,
        repair_geometry=True,
    )
    source = None

    fire = _rasterize_layer(
        fire_layer, projection_wkt, geotransform, width, height
    ).astype(bool)
    aoi = _rasterize_layer(
        aoi_layer, projection_wkt, geotransform, width, height
    ).astype(bool)

    fire_layer.ResetReading()
    fire_geometries = [
        feature.GetGeometryRef().Clone()
        for feature in fire_layer
    ]
    excluded_filter = '"damage_grade" = \'Unburned or Regrowth\''
    if grade_layer.SetAttributeFilter(excluded_filter) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not select Unburned or Regrowth grading polygons.")
    grade_layer.ResetReading()
    excluded_geometries = [
        feature.GetGeometryRef().Clone()
        for feature in grade_layer
    ]
    if not excluded_geometries:
        raise ValueError("No Unburned or Regrowth grading polygons were selected.")
    fire_minus_regrowth = _rasterize_difference(
        fire_geometries,
        excluded_geometries,
        target_srs,
        geotransform,
        width,
        height,
    )
    if grade_layer.SetAttributeFilter(None) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not clear the grading-layer filter.")

    grade_masks: dict[str, np.ndarray] = {}
    for short_name, source_name in GRADE_LABELS.items():
        escaped = source_name.replace("'", "''")
        if grade_layer.SetAttributeFilter(f'"damage_grade" = \'{escaped}\'') != ogr.OGRERR_NONE:
            raise RuntimeError(f"Could not select {source_name} grading polygons.")
        grade_masks[short_name] = _rasterize_layer(
            grade_layer, projection_wkt, geotransform, width, height
        ).astype(bool)
    if grade_layer.SetAttributeFilter(None) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not clear the grading-layer filter.")
    fire_layer = grade_layer = aoi_layer = None
    fire_ds = grade_ds = aoi_ds = None
    return fire, fire_minus_regrowth, aoi, grade_masks


def _read_inputs(
    s2: dict[str, Path],
    s1: dict[str, Path],
    metadata: dict[str, dict[str, tuple[int, float]]],
    grid: RasterGrid,
    grid_window: tuple[int, int, int, int],
    tile_window: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for period in ("pre", "post"):
        for band in ("B8A", "B12"):
            offset, quantification = metadata[period][band]
            digital_numbers = read_raster_window(s2[f"{period}_{band}"], tile_window)
            arrays[f"{period}_{band}"] = _scale_reflectance(
                digital_numbers,
                boa_add_offset=offset,
                boa_quantification_value=quantification,
            ).astype(np.float32)
        arrays[f"{period}_SCL"] = np.asarray(
            read_raster_window(s2[f"{period}_SCL"], tile_window),
            dtype=np.float32,
        )
        for band in ("VV", "VH"):
            source_window = _window_for_grid(
                s1[f"{period}_{band}"],
                grid.geotransform,
                grid_window,
            )
            arrays[f"{period}_{band}"] = _read_float_window(
                s1[f"{period}_{band}"], source_window
            )
        ls_path = s1.get(f"{period}_ls_map")
        if ls_path is None:
            raise FileNotFoundError(f"Missing {period} Sentinel-1 ls_map input.")
        ls_window = _window_for_grid(ls_path, grid.geotransform, grid_window)
        arrays[f"{period}_ls_map"] = _read_float_window(ls_path, ls_window)
    return arrays


def _robust_breakpoints(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values)].astype(np.float32, copy=False)
    if finite.size == 0:
        raise ValueError("Robust SAR breakpoints require finite land samples.")
    median = float(np.median(finite.astype(np.float64)))
    scale = robust_standard_deviation(finite.reshape(1, -1))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Robust SAR breakpoints require positive scaled MAD.")
    return median + scale, median + 2.0 * scale, np.inf, np.inf


def _discount(
    possibility: np.ndarray,
    sensor_alpha: np.ndarray,
    *,
    temporal_on: bool,
    source_date: str,
    source_is_earlier: bool,
    alpha_time: float,
    fire_end: str,
) -> np.ndarray:
    return discount_for_date_offset(
        possibility,
        sensor_alpha,
        alpha_time=alpha_time,
        source_is_earlier=source_is_earlier,
        is_post_fire_pair=True,
        asymmetric=True,
        asymmetry_exponent=2.0,
        combine_method="product",
        source_date=source_date,
        fire_end_date=fire_end,
        temporal_enabled=temporal_on,
    )


def _make_variant_predictions(
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    valid_land: np.ndarray,
    optical_alpha: np.ndarray,
    sar_alpha: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    primary = config["primary"]
    sar_config = primary["sar"]
    sar_window = int(sar_config["filter"]["window_pixels"][0])
    if sar_config["filter"]["window_pixels"] != [sar_window, sar_window]:
        raise ValueError("Checkpoint C requires the registered square SAR boxcar.")
    changes: dict[str, np.ndarray] = {}
    centers: dict[str, float] = {}
    for polarization in ("VV", "VH"):
        pre_db = arrays[f"pre_{polarization}"]
        post_db = arrays[f"post_{polarization}"]
        pre_filtered = linear_power_to_db(
            boxcar_filter(db_to_linear_power(pre_db), window_size=sar_window)
        )
        post_filtered = linear_power_to_db(
            boxcar_filter(db_to_linear_power(post_db), window_size=sar_window)
        )
        change = post_filtered - pre_filtered
        change[~valid_land] = np.nan
        centered, centers[polarization] = center_by_median(change, valid_land)
        changes[polarization] = centered

    positive_breakpoints = {
        polarization: _robust_breakpoints(changes[polarization][valid_land])
        for polarization in ("VV", "VH")
    }
    dnbr_values = dnbr(
        arrays["pre_B8A"],
        arrays["pre_B12"],
        arrays["post_B8A"],
        arrays["post_B12"],
    )
    dnbr_values[~valid_land] = np.nan
    optical_points = tuple(
        float(value) for value in primary["optical"]["breakpoint_values"]
    ) + (np.inf, np.inf)
    otsu_points = tuple(
        float(value) for value in config["disclosed_variants"][
            next(
                index
                for index, item in enumerate(config["disclosed_variants"])
                if item["id"] == "otsu_dnbr"
            )
        ]["optical_breakpoints"]
    ) + (np.inf, np.inf)

    temporal_config = primary["temporal_discount"]
    alpha_time = temporal_reliability(
        S2_POST_DATE,
        S1_POST_DATE,
        tau_days=float(temporal_config["tau_days"]),
    )
    fire_end = str(config["inputs"]["fire_end"])
    optical_un_discounted = dnbr_possibility(dnbr_values, optical_points)
    optical_otsu_un_discounted = dnbr_possibility(dnbr_values, otsu_points)
    optical_by_mode: dict[tuple[str, bool], np.ndarray] = {}
    for label, raw in (
        ("usgs", optical_un_discounted),
        ("otsu", optical_otsu_un_discounted),
    ):
        for temporal_on in (True, False):
            optical_by_mode[(label, temporal_on)] = _discount(
                raw,
                optical_alpha,
                temporal_on=temporal_on,
                source_date=S2_POST_DATE,
                source_is_earlier=True,
                alpha_time=alpha_time,
                fire_end=fire_end,
            )

    sar_by_mode: dict[tuple[str, bool], np.ndarray] = {}
    for polarization in ("VV", "VH"):
        raw = sar_possibility(
            changes[polarization],
            positive_breakpoints[polarization],
            burn_direction="positive",
        )
        for temporal_on in (True, False):
            sar_by_mode[(polarization, temporal_on)] = _discount(
                raw,
                sar_alpha,
                temporal_on=temporal_on,
                source_date=S1_POST_DATE,
                source_is_earlier=False,
                alpha_time=alpha_time,
                fire_end=fire_end,
            )

    predictions: dict[str, np.ndarray] = {}
    conflicts: dict[str, np.ndarray] = {}

    def add_fused(
        name: str,
        polarization: str,
        rule: str,
        temporal_on: bool,
        optical_label: str = "usgs",
    ) -> None:
        optical = optical_by_mode[(optical_label, temporal_on)]
        sar = sar_by_mode[(polarization, temporal_on)]
        if rule == "adaptive":
            fused, conflict = adaptive_fusion(optical, sar)
            conflicts[name] = conflict
        elif rule == "conjunctive":
            fused, _ = conjunctive_fusion(optical, sar)
        elif rule == "disjunctive":
            fused, _ = disjunctive_fusion(optical, sar)
        else:
            raise ValueError(f"Unknown registered fusion rule: {rule}")
        predictions[name], _ = decide(fused)

    add_fused("primary_fused", "VV", "adaptive", True)
    predictions["vh_positive"], _ = decide(sar_by_mode[("VH", True)])
    predictions["optical_only"], _ = decide(optical_by_mode[("usgs", True)])
    predictions["sar_only_vv"], _ = decide(sar_by_mode[("VV", True)])
    predictions["sar_only_vh"], _ = decide(sar_by_mode[("VH", True)])
    add_fused("fused_conjunctive", "VV", "conjunctive", True)
    add_fused("fused_disjunctive", "VV", "disjunctive", True)
    add_fused("temporal_discount_off", "VV", "adaptive", False)
    add_fused("otsu_dnbr", "VV", "adaptive", True, optical_label="otsu")

    exploratory_config = next(
        item
        for item in config["disclosed_variants"]
        if item["id"] == "vh_two_sided_exploratory"
    )
    absolute_change = np.abs(changes["VH"])
    two_sided_points = _robust_breakpoints(absolute_change[valid_land])
    two_sided_raw = sar_possibility(
        absolute_change,
        two_sided_points,
        burn_direction="positive",
    )
    two_sided_discounted = _discount(
        two_sided_raw,
        sar_alpha,
        temporal_on=True,
        source_date=S1_POST_DATE,
        source_is_earlier=False,
        alpha_time=alpha_time,
        fire_end=fire_end,
    )
    predictions[exploratory_config["id"]], _ = decide(two_sided_discounted)

    diagnostics = {
        "alpha_time": alpha_time,
        "sar_center_medians": centers,
        "sar_breakpoints": positive_breakpoints,
        "vh_two_sided_breakpoints": two_sided_points,
        "dnbr_points": optical_points[:2],
        "otsu_dnbr_points": otsu_points[:2],
    }
    return predictions, conflicts["primary_fused"], dnbr_values, diagnostics


def _metrics_row(
    variant: str,
    reference_name: str,
    reference: np.ndarray,
    classes: np.ndarray,
    common_valid: np.ndarray,
    grade_masks: dict[str, np.ndarray],
    bootstrap_intervals: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    burned_prediction = classes == 0
    tp = int(np.count_nonzero(common_valid & reference & burned_prediction))
    fp = int(np.count_nonzero(common_valid & ~reference & burned_prediction))
    fn = int(np.count_nonzero(common_valid & reference & ~burned_prediction))
    metrics = binary_metrics_from_counts(tp, fp, fn)
    decided_valid = common_valid & (classes != 2)
    decided_tp = int(np.count_nonzero(decided_valid & reference & burned_prediction))
    decided_fp = int(np.count_nonzero(decided_valid & ~reference & burned_prediction))
    decided_fn = int(np.count_nonzero(decided_valid & reference & ~burned_prediction))
    decided_metrics = binary_metrics_from_counts(
        decided_tp,
        decided_fp,
        decided_fn,
    )
    valid_count = int(np.count_nonzero(common_valid))
    undecided_count = int(np.count_nonzero(common_valid & (classes == 2)))
    row: dict[str, Any] = {
        "row_type": "variant",
        "variant": variant,
        "reference": reference_name,
        "common_pixels": valid_count,
        "burned_prevalence": float(np.count_nonzero(common_valid & reference) / valid_count),
        "burned_pixels": int(np.count_nonzero(common_valid & reference)),
        "predicted_burned_pixels": int(
            np.count_nonzero(common_valid & burned_prediction)
        ),
        "predicted_unburned_pixels": int(
            np.count_nonzero(common_valid & (classes == 1))
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "undecided_pixels": undecided_count,
        "undecided_fraction": undecided_count / valid_count,
        "iou": metrics["iou"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "decided_pixels": int(np.count_nonzero(decided_valid)),
        "decided_iou": decided_metrics["iou"],
        "decided_f1": decided_metrics["f1"],
        "decided_precision": decided_metrics["precision"],
        "decided_recall": decided_metrics["recall"],
        "iou_ci_low": bootstrap_intervals["iou"][0],
        "iou_ci_high": bootstrap_intervals["iou"][1],
        "f1_ci_low": bootstrap_intervals["f1"][0],
        "f1_ci_high": bootstrap_intervals["f1"][1],
    }
    for grade_name, grade_mask in grade_masks.items():
        grade_pixels = int(np.count_nonzero(common_valid & grade_mask))
        grade_burned = int(
            np.count_nonzero(common_valid & grade_mask & burned_prediction)
        )
        row[f"recall_{grade_name.lower().replace(' ', '_')}"] = (
            grade_burned / grade_pixels if grade_pixels else math.nan
        )
    return row


def _baseline_row(
    name: str,
    reference_name: str,
    reference: np.ndarray,
    common_valid: np.ndarray,
) -> dict[str, Any]:
    prevalence = float(np.count_nonzero(common_valid & reference) / common_valid.sum())
    if name == "predict_all_burned":
        prediction = np.ones(reference.shape, dtype=bool)
    elif name == "predict_none_burned":
        prediction = np.zeros(reference.shape, dtype=bool)
    elif name == "predict_by_prevalence":
        return {
            "row_type": "baseline",
            "variant": name,
            "reference": reference_name,
            "common_pixels": int(common_valid.sum()),
            "burned_prevalence": prevalence,
            "iou": prevalence / (2.0 - prevalence) if prevalence else 0.0,
            "f1": prevalence,
            "precision": prevalence,
            "recall": prevalence,
        }
    else:
        raise ValueError(f"Unknown baseline {name!r}.")
    tp = int(np.count_nonzero(common_valid & reference & prediction))
    fp = int(np.count_nonzero(common_valid & ~reference & prediction))
    fn = int(np.count_nonzero(common_valid & reference & ~prediction))
    return {
        "row_type": "baseline",
        "variant": name,
        "reference": reference_name,
        "common_pixels": int(common_valid.sum()),
        "burned_prevalence": prevalence,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **binary_metrics_from_counts(tp, fp, fn),
    }


def _write_raster(
    path: Path,
    values: np.ndarray,
    grid: RasterGrid,
    *,
    nodata: int | float,
    data_type: int,
) -> None:
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        grid.width,
        grid.height,
        1,
        data_type,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create raster {path}.")
    dataset.SetGeoTransform(grid.geotransform)
    dataset.SetProjection(grid.projection_wkt)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(values)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def _write_map_png(
    path: Path,
    variant: str,
    reference: np.ndarray,
    classes: np.ndarray,
    common_valid: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    truth = np.ma.masked_where(~common_valid, reference.astype(np.uint8))
    classified = np.ma.masked_where(~common_valid | (classes == 2), classes == 0)
    disagreement = np.ma.masked_where(
        ~common_valid | (classes == 2),
        (classes == 0) != reference,
    )
    axes[0].imshow(truth, cmap=ListedColormap(["#f2f2f2", "#b2182b"]), vmin=0, vmax=1)
    axes[0].set_title("FireDelineation reference")
    axes[1].imshow(
        classified,
        cmap=ListedColormap(["#2166ac", "#b2182b"]),
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"{variant}: burned / unburned")
    axes[2].imshow(
        disagreement,
        cmap=ListedColormap(["#f2f2f2", "#762a83"]),
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Disagreement (decided pixels)")
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_conflict_png(path: Path, conflict: np.ndarray, valid: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = np.ma.masked_where(~valid, conflict)
    artist = axis.imshow(image, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title("Primary VV adaptive-fusion conflict")
    axis.set_axis_off()
    figure.colorbar(artist, ax=axis, label="Conflict")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _save_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_type",
        "variant",
        "reference",
        "common_pixels",
        "burned_prevalence",
        "burned_pixels",
        "predicted_burned_pixels",
        "predicted_unburned_pixels",
        "tp",
        "fp",
        "fn",
        "undecided_pixels",
        "undecided_fraction",
        "iou",
        "f1",
        "precision",
        "recall",
        "decided_pixels",
        "decided_iou",
        "decided_f1",
        "decided_precision",
        "decided_recall",
        "iou_ci_low",
        "iou_ci_high",
        "f1_ci_low",
        "f1_ci_high",
        "recall_high",
        "recall_moderate_high",
        "recall_moderate_low",
        "recall_low",
        "comparison",
        "difference_iou",
        "difference_iou_ci_low",
        "difference_iou_ci_high",
        "difference_f1",
        "difference_f1_ci_low",
        "difference_f1_ci_high",
        "config_sha256",
        "git_commit",
        "seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _score(
    config: dict[str, Any],
    config_sha256: str,
    git_commit: str,
) -> str:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files; refusing to overwrite a scoring run."
        )

    s2, s1, _ = _find_inputs()
    metadata = _read_boa_metadata(s2)
    grid_inputs = {
        "s2_pre": s2["pre_B8A"],
        "s2_post": s2["post_B8A"],
        "s1_pre_vv": s1["pre_VV"],
        "s1_post_vv": s1["post_VV"],
        "s1_pre_vh": s1["pre_VH"],
        "s1_post_vh": s1["post_VH"],
    }
    full_grid = compute_intersection_grid(grid_inputs, reference_key="s2_pre")

    reference_source = ogr.Open(str(REFERENCE_PATH), 0)
    if reference_source is None:
        raise RuntimeError(f"Could not open reference GeoPackage {REFERENCE_PATH}.")
    fire_layer = reference_source.GetLayerByName(FIRE_LAYER)
    if fire_layer is None:
        raise RuntimeError(f"Reference layer {FIRE_LAYER} is missing.")
    fire_extent = fire_layer.GetExtent()
    fire_bounds = (fire_extent[0], fire_extent[2], fire_extent[1], fire_extent[3])
    reference_source = None
    grid_window = _pixel_window(
        full_grid.geotransform,
        full_grid.width,
        full_grid.height,
        fire_bounds,
        AOI_BUFFER_METERS,
    )
    gx, gy, width, height = grid_window
    grid = RasterGrid(
        width,
        height,
        (
            full_grid.geotransform[0] + gx * full_grid.geotransform[1],
            full_grid.geotransform[1],
            0.0,
            full_grid.geotransform[3] + gy * full_grid.geotransform[5],
            0.0,
            full_grid.geotransform[5],
        ),
        full_grid.projection_wkt,
    )
    tile_dataset = gdal.OpenEx(str(s2["pre_B8A"]), gdal.OF_RASTER | gdal.OF_READONLY)
    if tile_dataset is None:
        raise RuntimeError("Could not open the Sentinel-2 reference tile.")
    tile_window = _pixel_window(
        tile_dataset.GetGeoTransform(),
        tile_dataset.RasterXSize,
        tile_dataset.RasterYSize,
        fire_bounds,
        AOI_BUFFER_METERS,
    )
    tile_dataset = None

    arrays = _read_inputs(s2, s1, metadata, full_grid, grid_window, tile_window)
    scl_pre = arrays["pre_SCL"]
    scl_post = arrays["post_SCL"]
    land_valid = (
        np.isfinite(scl_pre)
        & np.isfinite(scl_post)
        & (scl_pre != 0)
        & (scl_post != 0)
        & (scl_pre != 6)
        & (scl_post != 6)
    )
    optical_alpha = np.minimum(
        scl_optical_reliability(
            scl_pre,
            unreliable_classes=tuple(
                config["primary"]["optical"]["reliability"]["alpha_zero_classes"]
            ),
        ),
        scl_optical_reliability(
            scl_post,
            unreliable_classes=tuple(
                config["primary"]["optical"]["reliability"]["alpha_zero_classes"]
            ),
        ),
    )
    sar_alpha = np.ones(land_valid.shape, dtype=np.float32)
    for period in ("pre", "post"):
        ls_map = arrays[f"{period}_ls_map"]
        finite_ls = np.isfinite(ls_map)
        sar_alpha[~finite_ls] = np.nan
        safe_map = np.where(finite_ls, ls_map, 0).astype(np.uint8)
        sar_alpha[(safe_map & LS_FLAG_MASK) != 0] = 0.0
    for key in (
        "pre_B8A",
        "pre_B12",
        "post_B8A",
        "post_B12",
        "pre_VV",
        "post_VV",
        "pre_VH",
        "post_VH",
    ):
        land_valid &= np.isfinite(arrays[key])
    land_valid &= np.isfinite(optical_alpha) & np.isfinite(sar_alpha)

    predictions, primary_conflict, dnbr_values, diagnostics = _make_variant_predictions(
        config,
        arrays,
        land_valid,
        optical_alpha,
        sar_alpha,
    )
    evidence_valid = land_valid & np.isfinite(dnbr_values) & np.isfinite(primary_conflict)
    for classes in predictions.values():
        evidence_valid &= classes >= 0

    fire, fire_minus_regrowth, aoi, grade_masks = _make_reference_masks(
        grid.projection_wkt,
        grid.geotransform,
        width,
        height,
    )
    common_valid = aoi & evidence_valid
    if not np.any(common_valid):
        raise ValueError("The fixed EMSN198_AOI common-valid scoring mask is empty.")
    references = {
        "firedelineation": fire,
        "firedelineation_minus_unburned_or_regrowth": fire_minus_regrowth,
    }

    bootstrap_config = config["uncertainty"]
    comparisons = {
        "primary_fused_minus_optical_only": ("primary_fused", "optical_only"),
        "primary_fused_minus_sar_only_vv": ("primary_fused", "sar_only_vv"),
    }
    prediction_burned = {
        name: classes == 0 for name, classes in predictions.items()
    }
    rows: list[dict[str, Any]] = []
    bootstrap_by_reference: dict[str, dict[str, Any]] = {}
    for reference_name, reference in references.items():
        result = paired_spatial_block_bootstrap(
            reference,
            prediction_burned,
            common_valid,
            block_shape=tuple(bootstrap_config["block_size_pixels"]),
            resamples=int(bootstrap_config["resamples"]),
            seed=int(bootstrap_config["seed"]),
            comparisons=comparisons,
        )
        bootstrap_by_reference[reference_name] = result
        for variant, classes in predictions.items():
            rows.append(
                _metrics_row(
                    variant,
                    reference_name,
                    reference,
                    classes,
                    common_valid,
                    grade_masks,
                    result["metrics"][variant],
                )
            )
        for baseline in (
            "predict_all_burned",
            "predict_none_burned",
            "predict_by_prevalence",
        ):
            rows.append(
                _baseline_row(baseline, reference_name, reference, common_valid)
            )
        for comparison_name, (candidate, baseline) in comparisons.items():
            candidate_row = next(
                row
                for row in rows
                if row["row_type"] == "variant"
                and row["reference"] == reference_name
                and row["variant"] == candidate
            )
            baseline_row = next(
                row
                for row in rows
                if row["row_type"] == "variant"
                and row["reference"] == reference_name
                and row["variant"] == baseline
            )
            difference_cis = result["differences"][comparison_name]
            rows.append(
                {
                    "row_type": "paired_difference",
                    "variant": f"{candidate}_minus_{baseline}",
                    "reference": reference_name,
                    "common_pixels": int(common_valid.sum()),
                    "comparison": comparison_name,
                    "difference_iou": candidate_row["iou"] - baseline_row["iou"],
                    "difference_iou_ci_low": difference_cis["iou"][0],
                    "difference_iou_ci_high": difference_cis["iou"][1],
                    "difference_f1": candidate_row["f1"] - baseline_row["f1"],
                    "difference_f1_ci_low": difference_cis["f1"][0],
                    "difference_f1_ci_high": difference_cis["f1"][1],
                }
            )
    for row in rows:
        row["config_sha256"] = config_sha256
        row["git_commit"] = git_commit
        row["seed"] = int(bootstrap_config["seed"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodata_classes = np.full((height, width), 255, dtype=np.uint8)
    for variant, classes in predictions.items():
        output = nodata_classes.copy()
        output[common_valid] = classes[common_valid].astype(np.uint8)
        _write_raster(
            OUTPUT_DIR / f"class_{variant}.tif",
            output,
            grid,
            nodata=255,
            data_type=gdal.GDT_Byte,
        )
        _write_map_png(
            OUTPUT_DIR / f"class_map_vs_reference_{variant}.png",
            variant,
            fire,
            classes,
            common_valid,
        )
        _write_map_png(
            OUTPUT_DIR
            / f"class_map_vs_reference_minus_unburned_or_regrowth_{variant}.png",
            variant,
            fire_minus_regrowth,
            classes,
            common_valid,
        )
    reference_values = np.full((height, width), 255, dtype=np.uint8)
    reference_values[common_valid] = fire[common_valid].astype(np.uint8)
    _write_raster(
        OUTPUT_DIR / "reference_firedelineation.tif",
        reference_values,
        grid,
        nodata=255,
        data_type=gdal.GDT_Byte,
    )
    reference_minus_values = np.full((height, width), 255, dtype=np.uint8)
    reference_minus_values[common_valid] = fire_minus_regrowth[common_valid].astype(
        np.uint8
    )
    _write_raster(
        OUTPUT_DIR / "reference_firedelineation_minus_unburned_or_regrowth.tif",
        reference_minus_values,
        grid,
        nodata=255,
        data_type=gdal.GDT_Byte,
    )
    conflict_values = np.full((height, width), np.nan, dtype=np.float32)
    conflict_values[common_valid] = primary_conflict[common_valid]
    _write_raster(
        OUTPUT_DIR / "conflict_primary_fused.tif",
        conflict_values,
        grid,
        nodata=float("nan"),
        data_type=gdal.GDT_Float32,
    )
    _write_conflict_png(
        OUTPUT_DIR / "conflict_map.png",
        primary_conflict,
        common_valid,
    )
    results_path = OUTPUT_DIR / "checkpoint_c_results.csv"
    _save_results_csv(results_path, rows)

    lines = [
        "Checkpoint C STEP 2 — pre-registered scoring results",
        "No parameters or variants were changed after scoring began.",
        f"Config SHA-256: {config_sha256}",
        f"Git commit containing frozen config: {git_commit}",
        f"PCG64 seed: {bootstrap_config['seed']}",
        f"AOI processing window: {width} x {height} = {width * height:,} pixels",
        f"Common valid scoring pixels inside EMSN198_AOI: {int(common_valid.sum()):,}",
        f"Land-valid pixels in processing window: {int(land_valid.sum()):,}",
        f"Temporal reliability exp(-3/14): {diagnostics['alpha_time']:.8f}",
        "SAR centering medians (dB): "
        + ", ".join(
            f"{name}={value:+.6f}" for name, value in diagnostics["sar_center_medians"].items()
        ),
        "SAR positive-direction breakpoints (median + 1/2 scaled MAD): "
        + ", ".join(
            f"{name}=({points[0]:.6f}, {points[1]:.6f})"
            for name, points in diagnostics["sar_breakpoints"].items()
        ),
        "Registered variant note: `vh_positive` is typed as a SAR variant and "
        "therefore duplicates `sar_only_vh`; both registered IDs are retained.",
        "dNBR breakpoints: "
        f"USGS={diagnostics['dnbr_points']}; Otsu={diagnostics['otsu_dnbr_points']}",
        "Bootstrap: paired 500 m spatial blocks, 1,000 resamples; "
        "TP/FP/FN are summed over sampled blocks before ratios.",
        "",
        "Reference prevalence and all-pixel primary metrics",
    ]
    for reference_name in references:
        lines.append(f"\n[{reference_name}]")
        prevalence_row = next(
            row
            for row in rows
            if row["row_type"] == "variant"
            and row["reference"] == reference_name
            and row["variant"] == "primary_fused"
        )
        lines.append(
            f"Burned prevalence: {prevalence_row['burned_prevalence']:.6%} "
            f"({prevalence_row['burned_pixels']:,}/{prevalence_row['common_pixels']:,})"
        )
        lines.append(
            "Variant | burned IoU [95% block CI] | F1 [95% block CI] | "
            "precision | recall | undecided | decided IoU | grade recall H/MH/ML/L"
        )
        for variant in predictions:
            row = next(
                item
                for item in rows
                if item["row_type"] == "variant"
                and item["reference"] == reference_name
                and item["variant"] == variant
            )
            lines.append(
                f"{variant} | {row['iou']:.6f} "
                f"[{row['iou_ci_low']:.6f}, {row['iou_ci_high']:.6f}] | "
                f"{row['f1']:.6f} [{row['f1_ci_low']:.6f}, {row['f1_ci_high']:.6f}] | "
                f"{row['precision']:.6f} | {row['recall']:.6f} | "
                f"{row['undecided_fraction']:.6%} | {row['decided_iou']:.6f} | "
                f"{row['recall_high']:.6f}/{row['recall_moderate_high']:.6f}/"
                f"{row['recall_moderate_low']:.6f}/{row['recall_low']:.6f}"
            )
        lines.append("Trivial baselines (all-pixel metrics):")
        for baseline in (
            "predict_all_burned",
            "predict_none_burned",
            "predict_by_prevalence",
        ):
            row = next(
                item
                for item in rows
                if item["row_type"] == "baseline"
                and item["reference"] == reference_name
                and item["variant"] == baseline
            )
            lines.append(
                f"{baseline}: IoU={row['iou']:.6f}; F1={row['f1']:.6f}; "
                f"precision={row['precision']:.6f}; recall={row['recall']:.6f}"
            )
        lines.append("Paired fused-minus-baseline comparisons (IoU and F1, 95% block CI):")
        for comparison_name in comparisons:
            row = next(
                item
                for item in rows
                if item["row_type"] == "paired_difference"
                and item["reference"] == reference_name
                and item["comparison"] == comparison_name
            )
            lines.append(
                f"{comparison_name}: IoU={row['difference_iou']:+.6f} "
                f"[{row['difference_iou_ci_low']:+.6f}, "
                f"{row['difference_iou_ci_high']:+.6f}]; "
                f"F1={row['difference_f1']:+.6f} "
                f"[{row['difference_f1_ci_low']:+.6f}, "
                f"{row['difference_f1_ci_high']:+.6f}]"
            )
    lines.extend(
        [
            "",
            "Interpretation limits: the EMS reference is optical-derived; these are "
            "validation results against that reference, not independent ground truth.",
            "SYNTHETIC cloud ablation and conflict association analysis are reserved "
            "for STEP 3 and were not run here.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    summary_path = OUTPUT_DIR / "checkpoint_c_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    config, config_sha256, git_commit = _load_committed_config()
    summary = _score(config, config_sha256, git_commit)
    print(summary)
    print(f"Results CSV: {OUTPUT_DIR / 'checkpoint_c_results.csv'}")
    print(f"Rasters and PNGs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
