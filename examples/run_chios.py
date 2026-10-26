"""Run the AOI-window-only Chios possibility-fusion pipeline.

Run from the repository root with:
    .venv-qgis/bin/python -m examples.run_chios

The delineation is read only after breakpoints have been selected and is used
solely for the permitted SAR-change sign diagnostic, never to tune parameters.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, ogr, osr

from dst_fusion.core.indices import dnbr
from dst_fusion.core.memberships import (
    DNBR_DEFAULT_BREAKPOINTS,
    dnbr_possibility,
    sar_possibility,
    suggest_otsu_breakpoints,
)
from dst_fusion.core.possibility import (
    adaptive_fusion,
    conjunctive_fusion,
    decide,
    disjunctive_fusion,
)
from dst_fusion.core.preprocessing import (
    boxcar_filter,
    center_by_median,
    db_to_linear_power,
    linear_power_to_db,
    robust_standard_deviation,
    scl_optical_reliability,
)
from dst_fusion.core.temporal import (
    check_temporal_plausibility,
    discount_for_date_offset,
    temporal_reliability,
)
from dst_fusion.io.raster import (
    RasterGrid,
    compute_intersection_grid,
    inspect_raster,
    read_raster_window,
)
from examples.audit_chios import (
    AOI_BUFFER_METERS,
    _find_inputs,
    _get_reference_geometry,
    _pixel_window,
    _rasterize_geometry,
    _read_boa_metadata,
    _read_window,
    _scale_reflectance,
    _window_for_grid,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "chios"
S2_PRE_DATE = "2024-06-28"
S2_POST_DATE = "2024-07-08"
S1_PRE_DATE = "2024-06-29"
S1_POST_DATE = "2024-07-11"
FIRE_START = "2024-07-01"
FIRE_END = "2024-07-03"
CLASS_LABELS = ("Burned", "Unburned", "Undecided")
LS_FLAG_MASK = 4 | 16
SAR_NODATA_SENTINEL = -3.4028235e38
gdal.UseExceptions()
ogr.UseExceptions()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fire-start", default=FIRE_START)
    parser.add_argument("--fire-end", default=FIRE_END)
    parser.add_argument("--tau-days", type=float, default=14.0)
    parser.add_argument("--asymmetry-exponent", type=float, default=2.0)
    parser.add_argument("--asymmetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--temporal-discount",
        choices=("both", "on", "off"),
        default="both",
        help="Report both temporal-discount states by default.",
    )
    parser.add_argument(
        "--speckle",
        choices=("both", "boxcar", "none"),
        default="both",
        help="Run the 5x5 linear-power boxcar and unfiltered cases by default.",
    )
    parser.add_argument("--speckle-window", type=int, default=5)
    parser.add_argument("--alpha-opt-cloud", type=float, default=0.0)
    parser.add_argument("--alpha-sar", type=float, default=1.0)
    parser.add_argument(
        "--terrain-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discount SAR where ls_map layover/shadow bits are set.",
    )
    parser.add_argument("--burn-direction", choices=("negative", "positive"), default="negative")
    parser.add_argument(
        "--dnbr-breakpoints",
        choices=("otsu", "usgs"),
        default="usgs",
        help="Use USGS defaults by default; retain AOI Otsu as a labelled variant.",
    )
    parser.add_argument(
        "--sar-breakpoints",
        choices=("robust", "otsu"),
        default="robust",
        help="Use median + 1/2 robust standard deviations by default.",
    )
    parser.add_argument(
        "--center-sar",
        action="store_true",
        help="Subtract the AOI-valid-land median of each SAR change before memberships.",
    )
    parser.add_argument("--otsu-bins", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    if not np.isfinite(arguments.tau_days) or arguments.tau_days <= 0:
        parser.error("--tau-days must be finite and greater than zero.")
    if (
        not np.isfinite(arguments.asymmetry_exponent)
        or arguments.asymmetry_exponent < 1
    ):
        parser.error("--asymmetry-exponent must be finite and at least one.")
    if arguments.speckle_window < 1 or arguments.speckle_window % 2 == 0:
        parser.error("--speckle-window must be a positive odd integer.")
    for name in ("alpha_opt_cloud", "alpha_sar"):
        value = getattr(arguments, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    if arguments.otsu_bins < 2:
        parser.error("--otsu-bins must be at least 2.")
    return arguments


def _read_float_window(path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    """Read one AOI window as float32 and replace its declared NoData with NaN."""
    info = inspect_raster(path)
    values = np.asarray(read_raster_window(path, window), dtype=np.float32)
    invalid = ~np.isfinite(values)
    if info.nodata is not None:
        if math.isnan(info.nodata):
            invalid |= np.isnan(values)
        else:
            invalid |= values == np.float32(info.nodata)
    invalid |= values == np.float32(SAR_NODATA_SENTINEL)
    values[invalid] = np.nan
    return values


def _read_aoi_inputs(
    s2: dict[str, Path],
    s1: dict[str, Path],
    metadata: dict[str, dict[str, tuple[int, float]]],
    grid_gt: tuple[float, ...],
    grid_window: tuple[int, int, int, int],
    tile_window: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for period in ("pre", "post"):
        for band in ("B8A", "B12"):
            offset, quantification = metadata[period][band]
            arrays[f"{period}_{band}"] = _scale_reflectance(
                _read_window(s2[f"{period}_{band}"], tile_window),
                boa_add_offset=offset,
                boa_quantification_value=quantification,
            ).astype(np.float32)
        arrays[f"{period}_SCL"] = _read_window(
            s2[f"{period}_SCL"], tile_window
        ).astype(np.float32)
        sar_window = _window_for_grid(s1[f"{period}_VH"], grid_gt, grid_window)
        arrays[f"{period}_VH"] = _read_float_window(
            s1[f"{period}_VH"], sar_window
        )
        ls_path = s1.get(f"{period}_ls_map")
        if ls_path is None:
            raise FileNotFoundError(
                f"Expected HyP3 {period} layover/shadow _ls_map.tif for SAR reliability."
            )
        ls_window = _window_for_grid(ls_path, grid_gt, grid_window)
        arrays[f"{period}_ls_map"] = _read_float_window(ls_path, ls_window)
    return arrays


def _build_validity_and_reliability(
    arrays: dict[str, np.ndarray],
    *,
    alpha_opt_cloud: float,
    alpha_sar: float,
    terrain_mask: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common valid land, optical alpha, and SAR alpha for this AOI."""
    scl_pre = arrays["pre_SCL"]
    scl_post = arrays["post_SCL"]
    optical_pre = np.minimum(
        scl_optical_reliability(scl_pre, unreliable_classes=(3, 8, 9, 10)),
        scl_optical_reliability(scl_post, unreliable_classes=(3, 8, 9, 10)),
    )
    optical_pre[optical_pre == 0] = alpha_opt_cloud

    sar_alpha = np.full(scl_pre.shape, alpha_sar, dtype=np.float32)
    for period in ("pre", "post"):
        ls_map = arrays[f"{period}_ls_map"]
        sar_alpha[~np.isfinite(ls_map)] = np.nan
        if terrain_mask:
            safe_ls_map = np.where(np.isfinite(ls_map), ls_map, 0).astype(np.uint8)
            flagged = (safe_ls_map & LS_FLAG_MASK) != 0
            sar_alpha[flagged] = 0.0

    valid = np.ones(scl_pre.shape, dtype=bool)
    for key in ("pre_B8A", "pre_B12", "post_B8A", "post_B12", "pre_VH", "post_VH"):
        valid &= np.isfinite(arrays[key])
    valid &= np.isfinite(scl_pre) & np.isfinite(scl_post)
    valid &= (scl_pre != 0) & (scl_post != 0)
    valid &= (scl_pre != 6) & (scl_post != 6)
    valid &= np.isfinite(optical_pre) & np.isfinite(sar_alpha)
    return valid, optical_pre, sar_alpha


def _breakpoint_text(points: tuple[float, float, float, float]) -> str:
    return "(" + ", ".join("inf" if math.isinf(value) else f"{value:.6g}" for value in points) + ")"


def _get_breakpoints(
    dnbr_values: np.ndarray,
    sar_change: np.ndarray,
    *,
    dnbr_method: str,
    sar_method: str,
    burn_direction: str,
    bins: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    dnbr_points = (
        suggest_otsu_breakpoints(dnbr_values, bins=bins)
        if dnbr_method == "otsu"
        else DNBR_DEFAULT_BREAKPOINTS
    )
    oriented_change = -sar_change if burn_direction == "negative" else sar_change
    if sar_method == "otsu":
        sar_points = suggest_otsu_breakpoints(oriented_change, bins=bins)
    else:
        sample = np.asarray(oriented_change, dtype=np.float32)
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            raise ValueError("Robust SAR breakpoints require finite land samples.")
        median = float(np.median(finite.astype(np.float64)))
        scale = robust_standard_deviation(finite.reshape(1, -1))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Robust SAR breakpoints require positive robust spread.")
        sar_points = (median + scale, median + 2.0 * scale, np.inf, np.inf)
    return dnbr_points, sar_points


def _save_histograms(
    dnbr_values: np.ndarray,
    sar_by_mode: dict[str, np.ndarray],
    dnbr_points: tuple[float, float, float, float],
    sar_points: dict[str, tuple[float, float, float, float]],
    *,
    dnbr_method: str,
    sar_method: str,
    burn_direction: str,
    output_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(sar_by_mode) + 1,
        figsize=(7 * (len(sar_by_mode) + 1), 5),
        squeeze=False,
        constrained_layout=True,
    )
    dnbr_axis = axes[0, 0]
    dnbr_axis.hist(dnbr_values[np.isfinite(dnbr_values)], bins=256, color="#d97932")
    dnbr_axis.axvline(dnbr_points[0], color="black", linestyle="--", label="ramp start")
    dnbr_axis.axvline(dnbr_points[1], color="red", linestyle="-", label=f"{dnbr_method} upper threshold")
    dnbr_axis.set_title(f"dNBR histogram ({dnbr_method})")
    dnbr_axis.set_xlabel("dNBR")
    dnbr_axis.set_ylabel("AOI pixel count")
    dnbr_axis.legend()
    for axis, (mode, values) in zip(axes[0, 1:], sar_by_mode.items()):
        axis.hist(values[np.isfinite(values)], bins=256, color="#497eaa")
        points = sar_points[mode]
        axis.axvline(points[0], color="black", linestyle="--", label="ramp start")
        axis.axvline(points[1], color="red", linestyle="-", label=f"{sar_method} upper threshold")
        axis.set_title(
            f"SAR oriented change ({mode}; {sar_method}; burn_direction={burn_direction})"
        )
        axis.set_xlabel("Oriented SAR change (dB)")
        axis.set_ylabel("AOI pixel count")
        axis.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "chios_aoi_histograms.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _apply_temporal_discount(
    possibility: np.ndarray,
    sensor_alpha: np.ndarray,
    *,
    alpha_time: float,
    source_is_earlier: bool,
    source_date: str,
    fire_end: str,
    temporal_enabled: bool,
    exponent: float,
    asymmetric: bool,
) -> np.ndarray:
    return discount_for_date_offset(
        possibility,
        sensor_alpha,
        alpha_time=alpha_time,
        source_is_earlier=source_is_earlier,
        is_post_fire_pair=True,
        asymmetric=asymmetric,
        asymmetry_exponent=exponent,
        combine_method="product",
        source_date=source_date,
        fire_end_date=fire_end,
        temporal_enabled=temporal_enabled,
    )


def _write_raster(
    path: Path,
    values: np.ndarray,
    grid: RasterGrid,
    *,
    description: str,
    nodata: float,
    gdal_type: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        grid.width,
        grid.height,
        1,
        gdal_type,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create output raster: {path}")
    dataset.SetGeoTransform(grid.geotransform)
    dataset.SetProjection(grid.projection_wkt)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.SetDescription(description)
    band.WriteArray(values)
    band.FlushCache()
    dataset.FlushCache()
    band = dataset = None


def _write_possibilities(
    path: Path,
    values: np.ndarray,
    grid: RasterGrid,
    *,
    descriptions: tuple[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        grid.width,
        grid.height,
        2,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create possibility raster: {path}")
    dataset.SetGeoTransform(grid.geotransform)
    dataset.SetProjection(grid.projection_wkt)
    for index, description in enumerate(descriptions, start=1):
        band = dataset.GetRasterBand(index)
        band.SetNoDataValue(float("nan"))
        band.SetDescription(description)
        band.WriteArray(values[index - 1].astype(np.float32))
        band.FlushCache()
    dataset.FlushCache()
    dataset = None


def _save_preview(
    outputs: dict[str, np.ndarray],
    output_path: Path,
    grid: RasterGrid,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    class_cmap = ListedColormap(("#b51f2a", "#d9e4d7", "#f0bd4a"))
    class_norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), class_cmap.N)
    titles = (
        ("optical", "Optical only"),
        ("sar", "SAR only"),
        ("adaptive", "Fused: Dubois-Prade adaptive"),
        ("conjunctive", "Fused: conjunctive"),
        ("disjunctive", "Fused: disjunctive"),
        ("necessity", "Adaptive necessity"),
        ("conflict", "Fusion conflict"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(12, 18), constrained_layout=True)
    extent = (
        grid.geotransform[0],
        grid.geotransform[0] + grid.width * grid.geotransform[1],
        grid.geotransform[3] + grid.height * grid.geotransform[5],
        grid.geotransform[3],
    )
    for axis, (key, title) in zip(axes.flat, titles):
        values = outputs[key]
        if key in {"optical", "sar", "adaptive", "conjunctive", "disjunctive"}:
            image = axis.imshow(
                np.ma.masked_less(values, 0),
                extent=extent,
                cmap=class_cmap,
                norm=class_norm,
                interpolation="nearest",
            )
            colorbar = figure.colorbar(image, ax=axis, ticks=(0, 1, 2), shrink=0.8)
            colorbar.ax.set_yticklabels(CLASS_LABELS)
        else:
            image = axis.imshow(
                values,
                extent=extent,
                cmap="magma" if key == "conflict" else "viridis",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            figure.colorbar(image, ax=axis, shrink=0.8)
        axis.set_title(title)
        axis.set_xlabel("Easting (EPSG:32635, m)")
        axis.set_ylabel("Northing (EPSG:32635, m)")
        axis.set_aspect("equal")
    axes.flat[-1].axis("off")
    figure.suptitle(title, fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _class_counts(classes: np.ndarray, valid: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.count_nonzero(valid & (classes == code))) for code in (0, 1, 2))


def _print_variant_summary(
    label: str,
    decisions: dict[str, np.ndarray],
    conflict: np.ndarray,
    valid: np.ndarray,
) -> None:
    print(f"\nCLASS OUTPUT COUNTS ({label}); no reference-based scoring:")
    for name, result in decisions.items():
        burned, unburned, undecided = _class_counts(result, valid)
        print(
            f"  {name}: burned={burned:,}; unburned={unburned:,}; "
            f"undecided={undecided:,}; undecided fraction={undecided / max(1, int(valid.sum())):.3%}"
        )
    finite_conflict = conflict[valid & np.isfinite(conflict)]
    mean_conflict = float(np.mean(finite_conflict)) if finite_conflict.size else math.nan
    print(f"  mean conflict={mean_conflict:.5f} over {finite_conflict.size:,} valid pixels")


def main() -> None:
    args = _arguments()
    if args.dnbr_breakpoints == "otsu":
        dnbr_method = "Otsu variant (AOI-window histogram only)"
    else:
        dnbr_method = "USGS dNBR defaults (0.10, 0.27)"

    s2, s1, _ = _find_inputs()
    metadata = _read_boa_metadata(s2)
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
    if not fire_srs.IsSame(reference_srs):
        fire_geometry.Transform(osr.CoordinateTransformation(fire_srs, reference_srs))
    envelope = fire_geometry.GetEnvelope()
    fire_bbox = (envelope[0], envelope[2], envelope[1], envelope[3])
    grid_window = _pixel_window(
        grid.geotransform,
        grid.width,
        grid.height,
        fire_bbox,
        AOI_BUFFER_METERS,
    )

    tile_ds = gdal.OpenEx(str(s2["pre_B8A"]), gdal.OF_RASTER | gdal.OF_READONLY)
    if tile_ds is None:
        raise RuntimeError("Could not open the Sentinel-2 B8A reference grid.")
    tile_gt = tile_ds.GetGeoTransform()
    tile_width, tile_height = tile_ds.RasterXSize, tile_ds.RasterYSize
    tile_ds = None
    tile_window = _pixel_window(
        tile_gt,
        tile_width,
        tile_height,
        fire_bbox,
        AOI_BUFFER_METERS,
    )
    gx, gy, width, height = grid_window
    output_grid = RasterGrid(
        width,
        height,
        (
            grid.geotransform[0] + gx * grid.geotransform[1],
            grid.geotransform[1],
            0.0,
            grid.geotransform[3] + gy * grid.geotransform[5],
            0.0,
            grid.geotransform[5],
        ),
        grid.projection_wkt,
    )
    print(
        "AOI-only read windows: "
        f"intersection grid={grid_window}; S2 tile={tile_window}; "
        f"pixels={width * height:,}; full-scene arrays were not read."
    )
    arrays = _read_aoi_inputs(
        s2,
        s1,
        metadata,
        grid.geotransform,
        grid_window,
        tile_window,
    )
    valid_land, optical_alpha, sar_alpha = _build_validity_and_reliability(
        arrays,
        alpha_opt_cloud=args.alpha_opt_cloud,
        alpha_sar=args.alpha_sar,
        terrain_mask=args.terrain_mask,
    )

    dnbr_values = dnbr(
        arrays["pre_B8A"],
        arrays["pre_B12"],
        arrays["post_B8A"],
        arrays["post_B12"],
    )
    raw_sar_change = arrays["post_VH"] - arrays["pre_VH"]
    valid_land &= np.isfinite(dnbr_values) & np.isfinite(raw_sar_change)
    dnbr_values[~valid_land] = np.nan
    raw_sar_change[~valid_land] = np.nan

    speckle_modes = ("boxcar", "none") if args.speckle == "both" else (args.speckle,)
    sar_change_by_mode: dict[str, np.ndarray] = {}
    sar_breakpoints: dict[str, tuple[float, float, float, float]] = {}
    sar_center_offsets: dict[str, float] = {}
    dnbr_points: tuple[float, float, float, float] | None = None
    for mode in speckle_modes:
        if mode == "boxcar":
            pre_power = db_to_linear_power(arrays["pre_VH"])
            post_power = db_to_linear_power(arrays["post_VH"])
            pre_filtered = linear_power_to_db(
                boxcar_filter(pre_power, window_size=args.speckle_window)
            )
            post_filtered = linear_power_to_db(
                boxcar_filter(post_power, window_size=args.speckle_window)
            )
            change = post_filtered - pre_filtered
        else:
            change = raw_sar_change.copy()
        change[~valid_land] = np.nan
        if args.center_sar:
            change, sar_center_offsets[mode] = center_by_median(change, valid_land)
        points_dn, points_sar = _get_breakpoints(
            dnbr_values,
            change[valid_land],
            dnbr_method=args.dnbr_breakpoints,
            sar_method=args.sar_breakpoints,
            burn_direction=args.burn_direction,
            bins=args.otsu_bins,
        )
        if dnbr_points is None:
            dnbr_points = points_dn
        sar_change_by_mode[mode] = change
        sar_breakpoints[mode] = points_sar
    assert dnbr_points is not None

    # Breakpoints are fixed from AOI histograms/USGS defaults before reading
    # the delineation geometry for the explicitly allowed sign diagnostic.
    fire_geometry, fire_srs = _get_reference_geometry()[0]
    if not fire_srs.IsSame(reference_srs):
        fire_geometry.Transform(osr.CoordinateTransformation(fire_srs, reference_srs))
    window_gt = output_grid.geotransform
    delineation_mask = _rasterize_geometry(
        fire_geometry,
        reference_srs,
        window_gt,
        width,
        height,
    )
    diagnostic_valid = valid_land & np.isfinite(raw_sar_change)
    inside = diagnostic_valid & delineation_mask
    outside = diagnostic_valid & ~delineation_mask
    inside_mean = float(np.mean(raw_sar_change[inside])) if np.any(inside) else math.nan
    outside_mean = float(np.mean(raw_sar_change[outside])) if np.any(outside) else math.nan
    observed_sign = (
        "negative, consistent with configured burn_direction"
        if inside_mean < outside_mean and args.burn_direction == "negative"
        else "positive, consistent with configured burn_direction"
        if inside_mean > outside_mean and args.burn_direction == "positive"
        else "opposite to configured burn_direction"
        if inside_mean != outside_mean
        else "no mean sign difference"
    )

    print("\nPARAMETERS SELECTED WITHOUT REFERENCE-LABEL TUNING")
    print(f"  dNBR breakpoints [{dnbr_method}]: {_breakpoint_text(dnbr_points)}")
    print(
        "  SAR breakpoint method: "
        + (
            "AOI-land median + 1/2 scaled MAD (label-free robust scale)"
            if args.sar_breakpoints == "robust"
            else "AOI histogram Otsu (not recommended for unimodal SAR)"
        )
    )
    if args.center_sar:
        for mode, offset in sar_center_offsets.items():
            print(f"  {mode} AOI-valid-land SAR median removed before membership: {offset:+.5f} dB")
    else:
        print("  SAR centering: off")
    for mode, points in sar_breakpoints.items():
        print(
            f"  VH SAR breakpoints [{mode} {args.sar_breakpoints}, "
            f"burn_direction={args.burn_direction}]: {_breakpoint_text(points)}"
        )
    print(
        "  One-time permitted burn-direction diagnostic (not a tuning step): "
        f"raw VH post-pre mean inside delineation={inside_mean:.5f} dB "
        f"(n={int(inside.sum()):,}); outside={outside_mean:.5f} dB "
        f"(n={int(outside.sum()):,}); {observed_sign}."
    )

    dates = {
        "s2_pre_date": S2_PRE_DATE,
        "s2_post_date": S2_POST_DATE,
        "s1_pre_date": S1_PRE_DATE,
        "s1_post_date": S1_POST_DATE,
    }
    warnings = check_temporal_plausibility(
        **dates,
        fire_start=args.fire_start,
        fire_end=args.fire_end,
    )
    print("\nTEMPORAL PLAUSIBILITY WARNINGS")
    if warnings:
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("  none")
    alpha_time = temporal_reliability(S2_POST_DATE, S1_POST_DATE, tau_days=args.tau_days)
    earlier_source = "optical" if S2_POST_DATE < S1_POST_DATE else "SAR"
    print(
        f"  Post-fire offset: {abs((np.datetime64(S2_POST_DATE) - np.datetime64(S1_POST_DATE)).astype(int))} days; "
        f"{earlier_source} is earlier; alpha_time=exp(-dt/tau)="
        f"exp(-3/{args.tau_days:g})={alpha_time:.6f}; "
        f"uniform floor with sensor alpha=1 is {1-alpha_time:.6f} when enabled, "
        "0 when disabled."
    )
    pre_gap = abs(
        (np.datetime64(S2_PRE_DATE) - np.datetime64(S1_PRE_DATE)).astype(int)
    )
    print(
        f"  Pre-fire date gap={pre_gap} days (warning threshold is the core default); "
        "pre-fire gap is reported, not used to tune evidence."
    )

    temporal_modes = (
        (True, False)
        if args.temporal_discount == "both"
        else (args.temporal_discount == "on",)
    )
    modes_to_run: list[tuple[str, str, float, bool]] = []
    for mode in speckle_modes:
        for temporal_on in temporal_modes:
            temporal_label = "on" if temporal_on else "off"
            modes_to_run.append((mode, temporal_label, 1.0, False))
            if args.asymmetry:
                modes_to_run.append(
                    (mode, temporal_label, args.asymmetry_exponent, True)
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hist_path = _save_histograms(
        dnbr_values[valid_land],
        {mode: sar_change_by_mode[mode][valid_land] * (-1 if args.burn_direction == "negative" else 1)
         for mode in speckle_modes},
        dnbr_points,
        sar_breakpoints,
        dnbr_method=args.dnbr_breakpoints,
        sar_method=args.sar_breakpoints,
        burn_direction=args.burn_direction,
        output_dir=args.output_dir,
    )
    print(f"\nSaved AOI-only evidence histograms: {hist_path}")

    canonical = ("boxcar" if "boxcar" in speckle_modes else speckle_modes[0],
                 "on" if True in temporal_modes else "off",
                 args.asymmetry_exponent if args.asymmetry else 1.0,
                 args.asymmetry)
    canonical_outputs: dict[str, np.ndarray] | None = None
    canonical_possibilities: dict[str, np.ndarray] | None = None
    canonical_decisions: dict[str, np.ndarray] | None = None
    for mode, temporal_label, exponent, asymmetric in modes_to_run:
        temporal_on = temporal_label == "on"
        sar_pi = sar_possibility(
            sar_change_by_mode[mode],
            sar_breakpoints[mode],
            burn_direction=args.burn_direction,
        )
        optical_pi = dnbr_possibility(dnbr_values, dnbr_points)
        optical_discounted = _apply_temporal_discount(
            optical_pi,
            optical_alpha,
            alpha_time=alpha_time,
            source_is_earlier=S2_POST_DATE < S1_POST_DATE,
            source_date=S2_POST_DATE,
            fire_end=args.fire_end,
            temporal_enabled=temporal_on,
            exponent=exponent,
            asymmetric=asymmetric,
        )
        sar_discounted = _apply_temporal_discount(
            sar_pi,
            sar_alpha,
            alpha_time=alpha_time,
            source_is_earlier=S1_POST_DATE < S2_POST_DATE,
            source_date=S1_POST_DATE,
            fire_end=args.fire_end,
            temporal_enabled=temporal_on,
            exponent=exponent,
            asymmetric=asymmetric,
        )
        optical_discounted[:, ~valid_land] = np.nan
        sar_discounted[:, ~valid_land] = np.nan
        adaptive, conflict = adaptive_fusion(optical_discounted, sar_discounted)
        conjunctive, _ = conjunctive_fusion(optical_discounted, sar_discounted)
        disjunctive, _ = disjunctive_fusion(optical_discounted, sar_discounted)
        optical_class, _ = decide(optical_discounted)
        sar_class, _ = decide(sar_discounted)
        adaptive_class, necessity = decide(adaptive)
        conjunctive_class, _ = decide(conjunctive)
        disjunctive_class, _ = decide(disjunctive)
        for classes in (
            optical_class,
            sar_class,
            adaptive_class,
            conjunctive_class,
            disjunctive_class,
        ):
            classes[~valid_land] = -1
        conflict[~valid_land] = np.nan
        necessity[~valid_land] = np.nan
        centered_label = "_centered" if args.center_sar else ""
        run_label = f"{mode}{centered_label}_temporal_{temporal_label}_k{exponent:g}"
        decisions = {
            "optical": optical_class,
            "sar": sar_class,
            "adaptive": adaptive_class,
            "conjunctive": conjunctive_class,
            "disjunctive": disjunctive_class,
        }
        _print_variant_summary(run_label, decisions, conflict, valid_land)
        output_arrays = {
            **decisions,
            "necessity": necessity,
            "conflict": conflict,
        }
        output_paths = {
            "adaptive": (f"fused_class_{run_label}.tif", adaptive_class, "Fused class: 0 Burned, 1 Unburned, 2 Undecided"),
            "conjunctive": (f"conjunctive_class_{run_label}.tif", conjunctive_class, "Conjunctive fused class: 0 Burned, 1 Unburned, 2 Undecided"),
            "disjunctive": (f"disjunctive_class_{run_label}.tif", disjunctive_class, "Disjunctive fused class: 0 Burned, 1 Unburned, 2 Undecided"),
            "optical": (f"optical_only_class_{run_label}.tif", optical_class, "Optical-only class: 0 Burned, 1 Unburned, 2 Undecided"),
            "sar": (f"sar_only_class_{run_label}.tif", sar_class, "SAR-only class: 0 Burned, 1 Unburned, 2 Undecided"),
        }
        for filename, classes, description in output_paths.values():
            _write_raster(
                args.output_dir / filename,
                classes.astype(np.int16),
                output_grid,
                description=description,
                nodata=-1,
                gdal_type=gdal.GDT_Int16,
            )
        _write_raster(
            args.output_dir / f"necessity_{run_label}.tif",
            necessity.astype(np.float32),
            output_grid,
            description="Necessity: 1 minus runner-up possibility",
            nodata=float("nan"),
            gdal_type=gdal.GDT_Float32,
        )
        _write_raster(
            args.output_dir / f"conflict_{run_label}.tif",
            conflict.astype(np.float32),
            output_grid,
            description="Conflict: 1 minus conjunctive height h",
            nodata=float("nan"),
            gdal_type=gdal.GDT_Float32,
        )
        _write_possibilities(
            args.output_dir / f"possibility_adaptive_{run_label}.tif",
            adaptive,
            output_grid,
            descriptions=("Adaptive possibility: Burned", "Adaptive possibility: Unburned"),
        )
        _write_possibilities(
            args.output_dir / f"possibility_optical_{run_label}.tif",
            optical_discounted,
            output_grid,
            descriptions=("Optical possibility: Burned", "Optical possibility: Unburned"),
        )
        _write_possibilities(
            args.output_dir / f"possibility_sar_{run_label}.tif",
            sar_discounted,
            output_grid,
            descriptions=("SAR possibility: Burned", "SAR possibility: Unburned"),
        )

        if (mode, temporal_label, exponent, asymmetric) == canonical:
            canonical_outputs = output_arrays
            canonical_possibilities = {
                "adaptive": adaptive,
                "optical": optical_discounted,
                "sar": sar_discounted,
            }
            canonical_decisions = decisions

    if canonical_outputs is None or canonical_possibilities is None or canonical_decisions is None:
        raise RuntimeError("Could not identify the requested canonical output configuration.")
    for key, filename, description in (
        ("adaptive", "fused_class.tif", "Adaptive fused class: 0 Burned, 1 Unburned, 2 Undecided"),
        ("conjunctive", "conjunctive_class.tif", "Conjunctive fused class: 0 Burned, 1 Unburned, 2 Undecided"),
        ("disjunctive", "disjunctive_class.tif", "Disjunctive fused class: 0 Burned, 1 Unburned, 2 Undecided"),
        ("optical", "optical_only_class.tif", "Optical-only class: 0 Burned, 1 Unburned, 2 Undecided"),
        ("sar", "sar_only_class.tif", "SAR-only class: 0 Burned, 1 Unburned, 2 Undecided"),
    ):
        _write_raster(
            args.output_dir / filename,
            canonical_decisions[key].astype(np.int16),
            output_grid,
            description=description,
            nodata=-1,
            gdal_type=gdal.GDT_Int16,
        )
    for key in ("necessity", "conflict"):
        _write_raster(
            args.output_dir / f"{key}.tif",
            canonical_outputs[key].astype(np.float32),
            output_grid,
            description=(
                "Necessity: 1 minus runner-up possibility"
                if key == "necessity"
                else "Conflict: 1 minus conjunctive height h"
            ),
            nodata=float("nan"),
            gdal_type=gdal.GDT_Float32,
        )
    for key, array in canonical_possibilities.items():
        _write_possibilities(
            args.output_dir / f"possibility_{key}.tif",
            array,
            output_grid,
            descriptions=(
                f"{key.capitalize()} possibility: Burned",
                f"{key.capitalize()} possibility: Unburned",
            ),
        )
    preview_path = args.output_dir / "chios_fusion_preview.png"
    _save_preview(
        canonical_outputs,
        preview_path,
        output_grid,
        title=(
            f"burn_direction={args.burn_direction}; "
            f"{'median-centered' if args.center_sar else 'uncentered'}; "
            f"{'filtered' if 'boxcar' in speckle_modes else 'unfiltered'}; "
            f"temporal={'on' if True in temporal_modes else 'off'}; "
            f"dNBR={dnbr_method}"
        ),
    )
    print(f"\nSaved canonical map preview: {preview_path}")
    print(
        "Canonical output configuration: "
        f"speckle={canonical[0]}, temporal={canonical[1]}, "
        f"k={canonical[2]:g}, asymmetric={canonical[3]}"
    )
    print(f"GeoTIFF and PNG outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
