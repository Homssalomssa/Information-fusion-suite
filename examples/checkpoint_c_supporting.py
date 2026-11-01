# SPDX-License-Identifier: GPL-2.0-or-later

"""Checkpoint C STEP 3 supporting and explicitly post-hoc analyses.

Run each item once from the repository root, under the configured memory cap:
    .venv-qgis/bin/python -m examples.checkpoint_c_supporting prepare
    .venv-qgis/bin/python -m examples.checkpoint_c_supporting cloud
    .venv-qgis/bin/python -m examples.checkpoint_c_supporting conflict
    .venv-qgis/bin/python -m examples.checkpoint_c_supporting diagnostics
    .venv-qgis/bin/python -m examples.checkpoint_c_supporting vh-fusion
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, ogr

from possibility_fusion.core.indices import dnbr
from possibility_fusion.core.memberships import dnbr_possibility, sar_possibility
from possibility_fusion.core.possibility import adaptive_fusion, decide
from possibility_fusion.core.preprocessing import (
    boxcar_filter,
    center_by_median,
    db_to_linear_power,
    linear_power_to_db,
    scl_optical_reliability,
)
from possibility_fusion.core.temporal import discount_for_date_offset, temporal_reliability
from possibility_fusion.core.validation import (
    binary_metrics_from_counts,
    paired_spatial_block_bootstrap,
)
from examples.audit_chios import (
    AOI_BUFFER_METERS,
    REFERENCE_PATH,
    _find_inputs,
    _pixel_window,
    _read_boa_metadata,
)
from examples.score_checkpoint_c import (
    CONFIG_PATH,
    FIRE_LAYER,
    GRADE_LABELS,
    GRADING_LAYER,
    _make_reference_masks,
    _read_inputs,
    _robust_breakpoints,
    _write_map_png,
    _write_raster,
)
from examples.run_chios import (
    LS_FLAG_MASK,
    S1_POST_DATE,
    S2_POST_DATE,
)
from possibility_fusion.io.raster import RasterGrid, compute_intersection_grid, read_raster_window


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "chios" / "checkpoint_c"
CACHE_PATH = OUTPUT_DIR / "checkpoint_c_support_cache.npz"
COMMON_GRID_BLOCK = (25, 25)
BLOCK_SEED = 20261031
VARIANT_NAMES = (
    "primary_fused",
    "optical_only",
    "sar_only_vv",
    "fused_conjunctive",
    "temporal_discount_off",
)

gdal.UseExceptions()
ogr.UseExceptions()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_config() -> tuple[dict[str, Any], str, str]:
    config_bytes = CONFIG_PATH.read_bytes()
    digest = hashlib.sha256(config_bytes).hexdigest()
    for spec, label in (
        (["git", "show", "HEAD:configs/checkpoint_c.json"], "HEAD"),
        (
            ["git", "show", "checkpoint-c-config:configs/checkpoint_c.json"],
            "checkpoint-c-config",
        ),
    ):
        tracked = subprocess.run(
            spec, cwd=ROOT, check=True, capture_output=True
        ).stdout
        if hashlib.sha256(tracked).hexdigest() != digest:
            raise RuntimeError(f"Frozen config SHA-256 does not match {label}.")
    head = _git_output("rev-parse", "HEAD")
    tag = _git_output("rev-parse", "checkpoint-c-config^{commit}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag, head],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return json.loads(config_bytes), digest, head


def _load_saved_grid() -> tuple[RasterGrid, tuple[int, int, int, int]]:
    reference_path = OUTPUT_DIR / "reference_firedelineation.tif"
    dataset = gdal.OpenEx(str(reference_path), gdal.OF_RASTER | gdal.OF_READONLY)
    if dataset is None:
        raise FileNotFoundError(f"Missing primary reference raster {reference_path}.")
    gt = dataset.GetGeoTransform()
    width, height = dataset.RasterXSize, dataset.RasterYSize
    projection = dataset.GetProjectionRef()
    dataset = None
    grid = RasterGrid(width, height, gt, projection)

    s2, s1, _ = _find_inputs()
    full_grid = compute_intersection_grid(
        {
            "s2_pre": s2["pre_B8A"],
            "s2_post": s2["post_B8A"],
            "s1_pre": s1["pre_VV"],
            "s1_post": s1["post_VV"],
            "s1_pre_vh": s1["pre_VH"],
            "s1_post_vh": s1["post_VH"],
        },
        reference_key="s2_pre",
    )
    origin_x = (gt[0] - full_grid.geotransform[0]) / full_grid.geotransform[1]
    origin_y = (gt[3] - full_grid.geotransform[3]) / full_grid.geotransform[5]
    gx, gy = int(round(origin_x)), int(round(origin_y))
    if not (
        math.isclose(origin_x, gx, abs_tol=1e-8)
        and math.isclose(origin_y, gy, abs_tol=1e-8)
    ):
        raise ValueError("Saved Checkpoint C grid is not aligned to the input grid.")
    return grid, (gx, gy, width, height)


def _load_saved_raster(path: Path) -> np.ndarray:
    dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
    if dataset is None:
        raise FileNotFoundError(f"Missing saved Checkpoint C raster {path}.")
    values = np.asarray(dataset.GetRasterBand(1).ReadAsArray())
    dataset = None
    return values


def _prepare_cache() -> None:
    config, config_sha256, git_commit = _verify_config()
    if CACHE_PATH.exists():
        raise FileExistsError(f"Supporting evidence cache already exists: {CACHE_PATH}")
    grid, grid_window = _load_saved_grid()
    s2, s1, _ = _find_inputs()
    metadata = _read_boa_metadata(s2)
    full_grid = compute_intersection_grid(
        {
            "s2_pre": s2["pre_B8A"],
            "s2_post": s2["post_B8A"],
            "s1_pre_vv": s1["pre_VV"],
            "s1_post_vv": s1["post_VV"],
            "s1_pre_vh": s1["pre_VH"],
            "s1_post_vh": s1["post_VH"],
        },
        reference_key="s2_pre",
    )
    gx, gy, width, height = grid_window
    source = ogr.Open(str(REFERENCE_PATH), 0)
    if source is None:
        raise RuntimeError(f"Could not open reference {REFERENCE_PATH}.")
    fire_layer = source.GetLayerByName(FIRE_LAYER)
    if fire_layer is None:
        raise RuntimeError(f"Missing reference layer {FIRE_LAYER}.")
    extent = fire_layer.GetExtent()
    fire_bounds = (extent[0], extent[2], extent[1], extent[3])
    source = None
    tile = gdal.OpenEx(str(s2["pre_B8A"]), gdal.OF_RASTER | gdal.OF_READONLY)
    if tile is None:
        raise RuntimeError("Could not open Sentinel-2 reference tile.")
    tile_window = _pixel_window(
        tile.GetGeoTransform(),
        tile.RasterXSize,
        tile.RasterYSize,
        fire_bounds,
        AOI_BUFFER_METERS,
    )
    tile = None
    arrays = _read_inputs(
        s2,
        s1,
        metadata,
        full_grid,
        grid_window,
        tile_window,
    )
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
        finite = np.isfinite(ls_map)
        sar_alpha[~finite] = np.nan
        safe = np.where(finite, ls_map, 0).astype(np.uint8)
        sar_alpha[(safe & LS_FLAG_MASK) != 0] = 0.0
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

    dnbr_values = dnbr(
        arrays["pre_B8A"],
        arrays["pre_B12"],
        arrays["post_B8A"],
        arrays["post_B12"],
    )
    land_valid &= np.isfinite(dnbr_values)
    dnbr_values[~land_valid] = np.nan
    optical_alpha[~land_valid] = np.nan
    sar_alpha[~land_valid] = np.nan
    optical_points = tuple(
        float(point) for point in config["primary"]["optical"]["breakpoint_values"]
    ) + (np.inf, np.inf)
    alpha_time = temporal_reliability(
        S2_POST_DATE,
        S1_POST_DATE,
        tau_days=float(config["primary"]["temporal_discount"]["tau_days"]),
    )
    optical_pi_raw = dnbr_possibility(dnbr_values, optical_points)
    optical_pi = discount_for_date_offset(
        optical_pi_raw,
        optical_alpha,
        alpha_time=alpha_time,
        source_is_earlier=True,
        is_post_fire_pair=True,
        asymmetric=True,
        asymmetry_exponent=2.0,
        combine_method="product",
        source_date=S2_POST_DATE,
        fire_end_date=str(config["inputs"]["fire_end"]),
        temporal_enabled=True,
    )
    sar_pi: dict[str, np.ndarray] = {}
    sar_centers: dict[str, float] = {}
    sar_points: dict[str, tuple[float, float, float, float]] = {}
    for polarization in ("VV", "VH"):
        pre_filtered = linear_power_to_db(
            boxcar_filter(
                db_to_linear_power(arrays[f"pre_{polarization}"]),
                window_size=5,
            )
        )
        post_filtered = linear_power_to_db(
            boxcar_filter(
                db_to_linear_power(arrays[f"post_{polarization}"]),
                window_size=5,
            )
        )
        change = post_filtered - pre_filtered
        change[~land_valid] = np.nan
        centered, sar_centers[polarization] = center_by_median(change, land_valid)
        sar_points[polarization] = _robust_breakpoints(centered[land_valid])
        raw = sar_possibility(
            centered,
            sar_points[polarization],
            burn_direction="positive",
        )
        sar_pi[polarization] = discount_for_date_offset(
            raw,
            sar_alpha,
            alpha_time=alpha_time,
            source_is_earlier=False,
            is_post_fire_pair=True,
            asymmetric=True,
            asymmetry_exponent=2.0,
            combine_method="product",
            source_date=S1_POST_DATE,
            fire_end_date=str(config["inputs"]["fire_end"]),
            temporal_enabled=True,
        )

    primary_conflict = _load_saved_raster(
        OUTPUT_DIR / "conflict_primary_fused.tif"
    ).astype(np.float32)
    primary_classes = _load_saved_raster(
        OUTPUT_DIR / "class_primary_fused.tif"
    )
    conjunctive_classes = _load_saved_raster(
        OUTPUT_DIR / "class_fused_conjunctive.tif"
    )
    temporal_off_classes = _load_saved_raster(
        OUTPUT_DIR / "class_temporal_discount_off.tif"
    )
    saved_reference = _load_saved_raster(
        OUTPUT_DIR / "reference_firedelineation.tif"
    )
    saved_reference_minus = _load_saved_raster(
        OUTPUT_DIR / "reference_firedelineation_minus_unburned_or_regrowth.tif"
    )
    if (width, height) != (grid.width, grid.height):
        raise ValueError("Reconstructed input window differs from saved score grid.")

    _, _, _, grade_masks = _make_reference_masks(
        grid.projection_wkt,
        grid.geotransform,
        grid.width,
        grid.height,
    )
    aoi = saved_reference != 255
    valid = (
        aoi
        & (saved_reference_minus != 255)
        & land_valid
        & (primary_classes != 255)
        & (conjunctive_classes != 255)
        & (temporal_off_classes != 255)
        & np.isfinite(primary_conflict)
    )
    for name in VARIANT_NAMES:
        if np.any(valid & (_load_saved_raster(OUTPUT_DIR / f"class_{name}.tif") == 255)):
            raise ValueError(f"Saved variant {name} has NoData inside common mask.")

    cache: dict[str, np.ndarray] = {
        "common_valid": valid,
        "fire": saved_reference == 1,
        "fire_minus_regrowth": saved_reference_minus == 1,
        "aoi": aoi,
        "land_valid": land_valid,
        "optical_alpha": optical_alpha.astype(np.float32),
        "sar_alpha": sar_alpha.astype(np.float32),
        "dnbr": dnbr_values.astype(np.float32),
        "optical_pi_raw": optical_pi_raw.astype(np.float32),
        "optical_pi": optical_pi.astype(np.float32),
        "sar_vv_pi": sar_pi["VV"].astype(np.float32),
        "sar_vh_pi": sar_pi["VH"].astype(np.float32),
        "primary_conflict": primary_conflict,
        "primary_classes": primary_classes.astype(np.int8),
        "conjunctive_classes": conjunctive_classes.astype(np.int8),
        "temporal_off_classes": temporal_off_classes.astype(np.int8),
        "scl_pre": scl_pre.astype(np.uint8),
        "scl_post": scl_post.astype(np.uint8),
        "config_sha256": np.asarray(config_sha256),
        "git_commit": np.asarray(git_commit),
        "seed": np.asarray(int(config["uncertainty"]["seed"]), dtype=np.int64),
        "alpha_time": np.asarray(alpha_time, dtype=np.float64),
        "grid_geotransform": np.asarray(grid.geotransform, dtype=np.float64),
        "grid_projection": np.asarray(grid.projection_wkt),
        "sar_vv_breakpoints": np.asarray(sar_points["VV"], dtype=np.float64),
        "sar_vh_breakpoints": np.asarray(sar_points["VH"], dtype=np.float64),
        "sar_vv_center": np.asarray(sar_centers["VV"], dtype=np.float64),
        "sar_vh_center": np.asarray(sar_centers["VH"], dtype=np.float64),
    }
    for grade_name, mask in grade_masks.items():
        cache[f"grade_{grade_name.lower().replace(' ', '_')}"] = mask
    temporary = CACHE_PATH.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **cache)
    temporary.replace(CACHE_PATH)
    print(
        f"Prepared supporting cache once: {CACHE_PATH}; "
        f"AOI grid={grid.width}x{grid.height}; "
        f"common valid={int(valid.sum()):,}; "
        f"VV breakpoints={sar_points['VV'][:2]}; "
        f"VH breakpoints={sar_points['VH'][:2]}."
    )


def _load_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing supporting cache {CACHE_PATH}; run the prepare item first."
        )
    loaded = np.load(CACHE_PATH, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def _variant_result_row(
    *,
    name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    region: np.ndarray,
    confidence: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    region_count = int(np.count_nonzero(region))
    tp = int(np.count_nonzero(region & truth & prediction))
    fp = int(np.count_nonzero(region & ~truth & prediction))
    fn = int(np.count_nonzero(region & truth & ~prediction))
    return {
        "variant": name,
        "pixels": region_count,
        "reference_burned": int(np.count_nonzero(region & truth)),
        "predicted_burned": int(np.count_nonzero(region & prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **binary_metrics_from_counts(tp, fp, fn),
        "iou_ci_low": confidence["iou"][0] if confidence else math.nan,
        "iou_ci_high": confidence["iou"][1] if confidence else math.nan,
        "f1_ci_low": confidence["f1"][0] if confidence else math.nan,
        "f1_ci_high": confidence["f1"][1] if confidence else math.nan,
    }


def _bootstrap_regions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    valid_region: np.ndarray,
    *,
    seed_offset: int = 0,
) -> dict[str, dict[str, tuple[float, float]]]:
    result = paired_spatial_block_bootstrap(
        truth,
        predictions,
        valid_region,
        block_shape=COMMON_GRID_BLOCK,
        resamples=1000,
        seed=BLOCK_SEED + seed_offset,
    )
    return result["metrics"]


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _select_cloud_blocks(
    fire_mask: np.ndarray,
    block_shape: tuple[int, int],
    *,
    minimum_overlap_fraction: float = 0.10,
    count: int = 3,
) -> list[tuple[int, int, int, int, int, float]]:
    height, width = fire_mask.shape
    block_height, block_width = block_shape
    candidates: list[tuple[int, int, int, int, int, float]] = []
    for row in range(0, height, block_height):
        row_end = min(row + block_height, height)
        for column in range(0, width, block_width):
            column_end = min(column + block_width, width)
            block = fire_mask[row:row_end, column:column_end]
            overlap = int(block.sum())
            fraction = overlap / block.size
            if fraction >= minimum_overlap_fraction:
                candidates.append(
                    (overlap, row, column, row_end, column_end, fraction)
                )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(candidates) < count:
        raise RuntimeError(
            f"Need {count} qualifying cloud blocks; found {len(candidates)}."
        )
    return candidates[:count]


def _cloud_analysis() -> None:
    config, config_sha256, git_commit = _verify_config()
    cache = _load_cache()
    if str(cache["config_sha256"]) != config_sha256 or str(cache["git_commit"]) != git_commit:
        raise RuntimeError("Supporting cache provenance does not match current frozen config/HEAD.")
    valid = cache["common_valid"].astype(bool)
    truth = cache["fire"].astype(bool)
    fire = cache["fire"].astype(bool)
    height, width = valid.shape
    block_height, block_width = tuple(
        config["supporting_analyses"]["synthetic_cloud_ablation"]["block_size_pixels"]
    )
    selected = _select_cloud_blocks(
        fire,
        (block_height, block_width),
        minimum_overlap_fraction=0.10,
        count=3,
    )

    selected_blocks: list[tuple[int, int, int, int]] = []
    optical_raw = cache["optical_pi_raw"].astype(np.float32, copy=False)
    sar_pi = cache["sar_vv_pi"].astype(np.float32, copy=False)
    sar_alpha = cache["sar_alpha"].astype(np.float32, copy=False)
    optical_alpha = cache["optical_alpha"].astype(np.float32, copy=False)
    alpha_time = float(cache["alpha_time"])
    geotransform = tuple(float(value) for value in cache["grid_geotransform"])
    seed = int(cache["seed"])
    lines = [
        "SYNTHETIC cloud ablation (registered locations; not observed clouds)",
        f"Config SHA-256: {config_sha256}",
        f"Git commit: {git_commit}",
        f"Registered positions selected by descending fire overlap, ties row then col; grid origin row=0, col=0; size={block_height}x{block_width}.",
        "Cloud treatment: alpha_optical=0 only inside the selected block; VV SAR and all other settings unchanged.",
        "Per-block inside CIs are not reported: each synthetic patch is a single 25x25 spatial unit, not enough independent 500 m blocks.",
        "",
    ]
    rows: list[dict[str, Any]] = []
    pooled_predictions = {
        "optical_only_SYNTHETIC": np.zeros(valid.shape, dtype=bool),
        "sar_only_vv": np.zeros(valid.shape, dtype=bool),
        "primary_adaptive_fused_SYNTHETIC": np.zeros(valid.shape, dtype=bool),
    }
    for block_index, (
        overlap,
        row,
        column,
        row_end,
        column_end,
        fraction,
    ) in enumerate(selected, 1):
        block_mask = np.zeros(valid.shape, dtype=bool)
        block_mask[row:row_end, column:column_end] = True
        selected_blocks.append((row, column, row_end, column_end))
        alpha_cloud = optical_alpha.copy()
        alpha_cloud[block_mask] = 0.0
        alpha_cloud[~valid] = np.nan
        optical_pi_cloud = discount_for_date_offset(
            optical_raw,
            alpha_cloud,
            alpha_time=alpha_time,
            source_is_earlier=True,
            is_post_fire_pair=True,
            asymmetric=True,
            asymmetry_exponent=2.0,
            combine_method="product",
            source_date=S2_POST_DATE,
            fire_end_date=str(config["inputs"]["fire_end"]),
            temporal_enabled=True,
        )
        optical_classes, _ = decide(optical_pi_cloud)
        fused_pi, _ = adaptive_fusion(optical_pi_cloud, sar_pi)
        fused_classes, _ = decide(fused_pi)
        sar_classes, _ = decide(sar_pi)
        prediction_classes = {
            "optical_only_SYNTHETIC": optical_classes,
            "sar_only_vv": sar_classes,
            "primary_adaptive_fused_SYNTHETIC": fused_classes,
        }
        burned_predictions = {
            name: classes == 0 for name, classes in prediction_classes.items()
        }
        for name, prediction in burned_predictions.items():
            pooled_predictions[name][block_mask] = prediction[block_mask]
        for region_name, region in (
            ("inside", valid & block_mask),
            ("outside", valid & ~block_mask),
        ):
            confidence_by_name: dict[str, dict[str, tuple[float, float]]] = {}
            if region_name == "outside":
                confidence_by_name = _bootstrap_regions(
                    truth,
                    burned_predictions,
                    region,
                    seed_offset=block_index,
                )
            for name, prediction in burned_predictions.items():
                confidence = (
                    confidence_by_name[name] if confidence_by_name else None
                )
                row_value = _variant_result_row(
                    name=name,
                    prediction=prediction,
                    truth=truth,
                    region=region,
                    confidence=confidence,
                )
                row_value.update(
                    {
                        "scenario": f"cloud_block_{block_index}",
                        "region": region_name,
                        "block_row_start": row,
                        "block_col_start": column,
                        "block_row_end": row_end,
                        "block_col_end": column_end,
                        "fire_overlap_pixels": overlap,
                        "fire_overlap_fraction": fraction,
                        "bootstrap_ci_note": (
                            "paired 500m-grid block bootstrap, 1000 resamples"
                            if confidence
                            else "not estimable within one synthetic 500m block"
                        ),
                        "config_sha256": config_sha256,
                        "git_commit": git_commit,
                        "seed": seed,
                    }
                )
                rows.append(row_value)
                lines.append(
                    f"block {block_index} {region_name} {name}: "
                    f"n={row_value['pixels']:,}; reference_burned="
                    f"{row_value['reference_burned']:,}; predicted_burned="
                    f"{row_value['predicted_burned']:,}; TP={row_value['tp']:,}; "
                    f"FP={row_value['fp']:,}; FN={row_value['fn']:,}; "
                    f"IoU={row_value['iou']:.6f}; F1={row_value['f1']:.6f}"
                    + (
                        f"; IoU CI=[{row_value['iou_ci_low']:.6f}, "
                        f"{row_value['iou_ci_high']:.6f}]; F1 CI="
                        f"[{row_value['f1_ci_low']:.6f}, {row_value['f1_ci_high']:.6f}]"
                        if confidence
                        else "; CI=N/A"
                    )
                )

    pooled_inside = np.zeros(valid.shape, dtype=bool)
    for row, column, row_end, column_end in selected_blocks:
        pooled_inside[row:row_end, column:column_end] = True
    pooled_region = valid & pooled_inside
    pooled_ci = _bootstrap_regions(
        truth,
        pooled_predictions,
        pooled_region,
        seed_offset=50,
    )
    for name, prediction in pooled_predictions.items():
        row_value = _variant_result_row(
            name=name,
            prediction=prediction,
            truth=truth,
            region=pooled_region,
            confidence=pooled_ci[name],
        )
        row_value.update(
            {
                "scenario": "pooled_three_cloud_blocks",
                "region": "inside",
                "bootstrap_ci_note": "paired 500m-grid block bootstrap, 1000 resamples",
                "config_sha256": config_sha256,
                "git_commit": git_commit,
                "seed": seed,
            }
        )
        rows.append(row_value)
        lines.append(
            f"pooled three blocks {name}: n={row_value['pixels']:,}; "
            f"reference_burned={row_value['reference_burned']:,}; "
            f"predicted_burned={row_value['predicted_burned']:,}; "
            f"IoU={row_value['iou']:.6f} "
            f"[{row_value['iou_ci_low']:.6f}, {row_value['iou_ci_high']:.6f}]; "
            f"F1={row_value['f1']:.6f} "
            f"[{row_value['f1_ci_low']:.6f}, {row_value['f1_ci_high']:.6f}]"
        )

    _save_csv(OUTPUT_DIR / "synthetic_cloud_ablation.csv", rows)
    (OUTPUT_DIR / "synthetic_cloud_ablation.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 2:
        return math.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def _conflict_analysis() -> None:
    config, config_sha256, git_commit = _verify_config()
    cache = _load_cache()
    valid = cache["common_valid"].astype(bool)
    conflict = cache["primary_conflict"].astype(np.float32)
    scl = cache["scl_post"].astype(np.uint8)
    truth = cache["fire"].astype(bool)
    geotransform = tuple(float(value) for value in cache["grid_geotransform"])
    pixel_size = abs(geotransform[1])

    from scipy.ndimage import distance_transform_edt

    water = scl == 6
    if not np.any(water):
        raise ValueError("No SCL class 6 pixels available for a coast-distance proxy.")
    distance_m = distance_transform_edt(~water) * pixel_size
    distance_groups = (
        ("0-100m", 0.0, 100.0),
        ("100-500m", 100.0, 500.0),
        ("500-1000m", 500.0, 1000.0),
        (">1000m", 1000.0, np.inf),
    )

    rows: list[dict[str, Any]] = []
    lines = [
        "Conflict associations (descriptive; not causal)",
        f"Config SHA-256: {config_sha256}",
        f"Git commit: {git_commit}",
        "Distance is to nearest valid SCL class 6 water pixel, a raster water-proximity proxy rather than a surveyed coastline.",
        "No DEM/elevation raster is available in data/chios2024; slope association is unavailable.",
        "",
    ]

    def add_group(group_type: str, group: str, mask: np.ndarray) -> None:
        selected = valid & mask & np.isfinite(conflict)
        values = conflict[selected]
        row: dict[str, Any] = {
            "group_type": group_type,
            "group": group,
            "pixels": int(values.size),
            "mean_conflict": float(np.mean(values)) if values.size else math.nan,
            "median_conflict": float(np.median(values)) if values.size else math.nan,
            "p90_conflict": float(np.percentile(values, 90)) if values.size else math.nan,
            "fraction_conflict_ge_0_5": (
                float(np.mean(values >= 0.5)) if values.size else math.nan
            ),
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "seed": int(cache["seed"]),
        }
        rows.append(row)
        lines.append(
            f"{group_type}={group}: n={row['pixels']:,}; mean="
            f"{row['mean_conflict']:.6f}; median={row['median_conflict']:.6f}; "
            f"p90={row['p90_conflict']:.6f}; fraction>=0.5="
            f"{row['fraction_conflict_ge_0_5']:.6%}"
        )

    for group, lower, upper in distance_groups:
        add_group(
            "distance_to_SCL6_water",
            group,
            (distance_m >= lower) & (distance_m < upper),
        )
    lines.append(
        "Spearman(conflict, distance to SCL6 water)="
        f"{_correlation(conflict[valid], distance_m[valid]):.6f}"
    )
    for class_value in sorted(int(value) for value in np.unique(scl[valid])):
        add_group(
            "post_SCL_class",
            str(class_value),
            scl == class_value,
        )
    grade_keys = (
        "high",
        "moderate_high",
        "moderate_low",
        "low",
    )
    for grade_key in grade_keys:
        grade_mask = cache[f"grade_{grade_key}"].astype(bool)
        add_group("reference_damage_grade", grade_key.replace("_", " "), grade_mask)
    _save_csv(OUTPUT_DIR / "conflict_associations.csv", rows)
    (OUTPUT_DIR / "conflict_associations.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def _diagnostics() -> None:
    config, config_sha256, git_commit = _verify_config()
    cache = _load_cache()
    valid = cache["common_valid"].astype(bool)
    truth = cache["fire"].astype(bool)
    primary = cache["primary_classes"].astype(np.int8)
    conjunctive = cache["conjunctive_classes"].astype(np.int8)
    temporal_off = cache["temporal_off_classes"].astype(np.int8)
    optical = _load_saved_raster(OUTPUT_DIR / "class_optical_only.tif").astype(
        np.int8
    )
    grade_names = (
        ("high", "High"),
        ("moderate_high", "Moderate high"),
        ("moderate_low", "Moderate low"),
        ("low", "Low"),
    )
    lines = [
        "POST-HOC diagnostics (descriptive; not registered primary outcomes)",
        f"Config SHA-256: {config_sha256}",
        f"Git commit: {git_commit}",
        "",
    ]
    difference = valid & (primary != conjunctive)
    lines.append(
        f"(a) Adaptive vs conjunctive decisions differ on {int(difference.sum()):,} "
        f"of {int(valid.sum()):,} common pixels ({difference.sum()/valid.sum():.6%})."
    )
    if not np.any(difference):
        lines.append(
            "The metrics match because every common pixel has the same decided "
            "class or undecided status; IoU/F1 and all other label metrics are "
            "functions of those class decisions, not the underlying possibility "
            "magnitudes."
        )
    else:
        lines.append(
            "The decisions are not identical; any equal aggregate metrics arise "
            "from equal confusion counts, not pixelwise equivalence."
        )

    missed = valid & truth & (optical == 0) & (primary != 0)
    lines.append(
        "\n(b) Among reference-burned pixels detected by optical-only but missed "
        "by primary fused:"
    )
    rows: list[dict[str, Any]] = []
    for grade_key, grade_label in grade_names:
        group = missed & cache[f"grade_{grade_key}"].astype(bool)
        n = int(group.sum())
        undecided = int(np.count_nonzero(group & (primary == 2)))
        unburned = int(np.count_nonzero(group & (primary == 1)))
        other = int(n - undecided - unburned)
        lines.append(
            f"{grade_label}: n={n:,}; undecided={undecided:,} "
            f"({undecided/n:.6%} of misses)" if n else f"{grade_label}: n=0"
        )
        lines.append(
            f"  called unburned={unburned:,} ({unburned/n:.6%} of misses); "
            f"other={other:,}" if n else "  called unburned=0"
        )
        rows.append(
            {
                "diagnostic": "optical_detected_primary_missed",
                "damage_grade": grade_label,
                "pixels": n,
                "primary_undecided": undecided,
                "undecided_share": undecided / n if n else math.nan,
                "primary_called_unburned": unburned,
                "unburned_share": unburned / n if n else math.nan,
                "other": other,
                "config_sha256": config_sha256,
                "git_commit": git_commit,
                "seed": int(cache["seed"]),
            }
        )

    lines.append(
        f"\n(c) POST-HOC temporal_discount_off minus primary, against "
        f"FireDelineation: common pixels={int(valid.sum()):,}; "
        f"point IoU difference="
    )
    paired = paired_spatial_block_bootstrap(
        truth,
        {"temporal_discount_off": temporal_off == 0, "primary_fused": primary == 0},
        valid,
        block_shape=tuple(config["uncertainty"]["block_size_pixels"]),
        resamples=int(config["uncertainty"]["resamples"]),
        seed=int(config["uncertainty"]["seed"]),
        comparisons={
            "temporal_discount_off_minus_primary": (
                "temporal_discount_off",
                "primary_fused",
            )
        },
    )
    difference_cis = paired["differences"]["temporal_discount_off_minus_primary"]
    prim_tp = int(np.count_nonzero(valid & truth & (primary == 0)))
    prim_fp = int(np.count_nonzero(valid & ~truth & (primary == 0)))
    prim_fn = int(np.count_nonzero(valid & truth & (primary != 0)))
    off_tp = int(np.count_nonzero(valid & truth & (temporal_off == 0)))
    off_fp = int(np.count_nonzero(valid & ~truth & (temporal_off == 0)))
    off_fn = int(np.count_nonzero(valid & truth & (temporal_off != 0)))
    prim_metrics = binary_metrics_from_counts(prim_tp, prim_fp, prim_fn)
    off_metrics = binary_metrics_from_counts(off_tp, off_fp, off_fn)
    point_iou_diff = off_metrics["iou"] - prim_metrics["iou"]
    point_f1_diff = off_metrics["f1"] - prim_metrics["f1"]
    lines[-1] += f"{point_iou_diff:+.6f}"
    lines.append(
        f"IoU difference 95% paired spatial-block CI="
        f"[{difference_cis['iou'][0]:+.6f}, {difference_cis['iou'][1]:+.6f}]; "
        f"F1 difference={point_f1_diff:+.6f}, 95% CI="
        f"[{difference_cis['f1'][0]:+.6f}, {difference_cis['f1'][1]:+.6f}]."
    )
    rows.extend(
        [
            {
                "diagnostic": "temporal_discount_off_minus_primary",
                "reference": "firedelineation",
                "metric": "IoU",
                "difference": point_iou_diff,
                "ci_low": difference_cis["iou"][0],
                "ci_high": difference_cis["iou"][1],
                "config_sha256": config_sha256,
                "git_commit": git_commit,
                "seed": int(cache["seed"]),
            },
            {
                "diagnostic": "temporal_discount_off_minus_primary",
                "reference": "firedelineation",
                "metric": "F1",
                "difference": point_f1_diff,
                "ci_low": difference_cis["f1"][0],
                "ci_high": difference_cis["f1"][1],
                "config_sha256": config_sha256,
                "git_commit": git_commit,
                "seed": int(cache["seed"]),
            },
        ]
    )
    _save_csv(OUTPUT_DIR / "posthoc_diagnostics.csv", rows)
    (OUTPUT_DIR / "posthoc_diagnostics.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def _vh_fusion() -> None:
    config, config_sha256, git_commit = _verify_config()
    cache = _load_cache()
    valid = cache["common_valid"].astype(bool)
    fire = cache["fire"].astype(bool)
    fire_minus_regrowth = cache["fire_minus_regrowth"].astype(bool)
    optical = cache["optical_pi"].astype(np.float32, copy=False)
    sar_vh = cache["sar_vh_pi"].astype(np.float32, copy=False)
    fused, conflict = adaptive_fusion(optical, sar_vh)
    classes, _ = decide(fused)
    rows: list[dict[str, Any]] = []
    lines = [
        "POST-HOC supplementary run: optical + VH-positive SAR adaptive fusion",
        "This supplementary result does not replace or alter the pre-registered primary.",
        f"Config SHA-256: {config_sha256}",
        f"Git commit: {git_commit}",
        f"PCG64 seed: {int(cache['seed'])}",
        f"VH center median: {float(cache['sar_vh_center']):+.6f} dB",
        f"VH robust breakpoints: {tuple(cache['sar_vh_breakpoints'][:2])}",
        "",
    ]
    predictions = {"fused_vh_positive_POSTHOC": classes == 0}
    for reference_name, reference in (
        ("firedelineation", fire),
        ("firedelineation_minus_unburned_or_regrowth", fire_minus_regrowth),
    ):
        result = paired_spatial_block_bootstrap(
            reference,
            predictions,
            valid,
            block_shape=tuple(config["uncertainty"]["block_size_pixels"]),
            resamples=int(config["uncertainty"]["resamples"]),
            seed=int(config["uncertainty"]["seed"]),
        )
        for name, prediction in predictions.items():
            metric_cis = result["metrics"][name]
            row = _variant_result_row(
                name=name,
                prediction=prediction,
                truth=reference,
                region=valid,
                confidence=metric_cis,
            )
            row.update(
                {
                    "reference": reference_name,
                    "label": "POST-HOC supplementary; never replaces primary",
                    "config_sha256": config_sha256,
                    "git_commit": git_commit,
                    "seed": int(cache["seed"]),
                }
            )
            rows.append(row)
            lines.append(
                f"{reference_name}: IoU={row['iou']:.6f} "
                f"[{row['iou_ci_low']:.6f}, {row['iou_ci_high']:.6f}]; "
                f"F1={row['f1']:.6f} "
                f"[{row['f1_ci_low']:.6f}, {row['f1_ci_high']:.6f}]; "
                f"precision={row['precision']:.6f}; recall={row['recall']:.6f}; "
                f"predicted burned={row['predicted_burned']:,}; "
                f"reference burned={row['reference_burned']:,}"
            )
    _write_raster(
        OUTPUT_DIR / "class_fused_vh_positive_POSTHOC.tif",
        np.where(valid, classes, 255).astype(np.uint8),
        RasterGrid(
            int(classes.shape[1]),
            int(classes.shape[0]),
            tuple(float(value) for value in cache["grid_geotransform"]),
            str(cache["grid_projection"]),
        ),
        nodata=255,
        data_type=gdal.GDT_Byte,
    )
    _write_map_png(
        OUTPUT_DIR / "class_map_vs_reference_fused_vh_positive_POSTHOC.png",
        "POST-HOC fused VH positive",
        fire,
        classes,
        valid,
    )
    _write_map_png(
        OUTPUT_DIR
        / "class_map_vs_reference_minus_unburned_or_regrowth_fused_vh_positive_POSTHOC.png",
        "POST-HOC fused VH positive",
        fire_minus_regrowth,
        classes,
        valid,
    )
    _save_csv(OUTPUT_DIR / "posthoc_vh_fusion.csv", rows)
    (OUTPUT_DIR / "posthoc_vh_fusion.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "item",
        choices=("prepare", "cloud", "conflict", "diagnostics", "vh-fusion"),
    )
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if arguments.item == "prepare":
        _prepare_cache()
    elif arguments.item == "cloud":
        _cloud_analysis()
    elif arguments.item == "conflict":
        _conflict_analysis()
    elif arguments.item == "diagnostics":
        _diagnostics()
    else:
        _vh_fusion()


if __name__ == "__main__":
    main()
