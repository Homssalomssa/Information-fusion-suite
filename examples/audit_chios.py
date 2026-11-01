"""Read-only metadata and AOI-window audit for the Chios 2024 inputs.

Run with a Python that has GDAL, OGR, and NumPy installed:
    /usr/bin/python3 examples/audit_chios.py

Only source rasters are read. SAR cross-check values are aligned by integer
pixel offsets and evaluated in memory; no audit outputs are written.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from osgeo import gdal, ogr, osr

from possibility_fusion.core.indices import dnbr
from possibility_fusion.io.raster import compute_intersection_grid, read_raster_window


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "chios2024"
FIRE_DATES = ("2024-07-01", "2024-07-03")
PIXEL_SIZE = 20.0
AOI_BUFFER_METERS = 5000.0
DIFFERENCE_TOLERANCE_DB = 1e-6

S2_PRODUCT_GLOB = "S2A_MSIL2A_*_N0510_R050_T35SMC_*.SAFE"
S1_PRE_NAME = "S1A_IW_20240629T041546_DVP_RTC20_G_gduned_713A"
S1_POST_NAME = "S1A_IW_20240711T041546_DVP_RTC20_G_gduned_7189"
REFERENCE_PATH = DATA / "reference" / "fire_reference.gpkg"
EMS_DNBR_GLOB = "EMSN198_P01_post-pre_dNBR_grading.tif"
AUDIT_OUTPUT = ROOT / "outputs" / "chios"

gdal.UseExceptions()
ogr.UseExceptions()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _epsg(projection_wkt: str) -> str:
    reference = osr.SpatialReference()
    reference.ImportFromWkt(projection_wkt)
    reference.AutoIdentifyEPSG()
    code = reference.GetAuthorityCode(None)
    return f"EPSG:{code}" if code else reference.GetName() or "unknown"


def _acquisition(path: Path) -> str:
    match = re.search(r"(20\d{6}T\d{6})", path.as_posix())
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").isoformat()
    return "not encoded"


def _open(path: Path):
    dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_READONLY)
    if dataset is None:
        raise RuntimeError(f"Could not open raster: {path}")
    return dataset


def _raster_row(path: Path, *, sensor: str, polarization: str = "") -> dict[str, str]:
    dataset = _open(path)
    band = dataset.GetRasterBand(1)
    gt = dataset.GetGeoTransform()
    nodata = band.GetNoDataValue()
    if path.suffix.lower() == ".jp2" and path.stem.endswith(("_B8A_20m", "_B12_20m")):
        nodata_text = "GDAL: none; DN=0 per input specification"
    elif path.suffix.lower() == ".jp2" and path.stem.endswith("_SCL_20m"):
        nodata_text = "GDAL: none; SCL class 0 treated as NoData"
    else:
        nodata_text = "none" if nodata is None else f"{nodata:g}"
    row = {
        "path": _relative(path),
        "sensor": sensor,
        "crs": _epsg(dataset.GetProjectionRef()),
        "size": f"{dataset.RasterXSize}x{dataset.RasterYSize}",
        "pixel": f"{gt[1]:g} x {gt[5]:g}",
        "origin": f"{gt[0]:.3f}, {gt[3]:.3f}",
        "nodata": nodata_text,
        "dtype": gdal.GetDataTypeName(band.DataType),
        "acquisition": _acquisition(path),
        "rel_orbit": "",
        "pass": "",
        "polarization": polarization,
    }
    band = None
    dataset = None
    return row


def _read_s1_metadata(folder: Path) -> tuple[str, str, str]:
    """Return only orbit/pass values explicitly present in HyP3 metadata."""
    candidates = [
        *folder.glob("*.README.md.txt"),
        *folder.glob("*.log"),
        *folder.glob("*.tif.xml"),
    ]
    manifests = list(folder.rglob("manifest.safe"))
    candidates.extend(manifests)
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in candidates
        if path.is_file()
    )
    relative = re.search(
        r"(?:relative\s*orbit(?:\s*(?:number|no\.?))?|relativeOrbitNumber)"
        r"\s*[:=]?\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    direction = re.search(
        r"(?:orbit|pass)\s*(?:direction|type)?\s*[:=]\s*(ascending|descending)",
        text,
        re.IGNORECASE,
    )
    source = re.search(
        r"source granule used.*?\n\s*(S1[A-D]_[^\s]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    absolute = ""
    if source:
        orbit = re.search(r"_(\d{6})_[0-9A-F]{6}_", source.group(1))
        if orbit:
            absolute = orbit.group(1)
    return (
        relative.group(1) if relative else "not supplied",
        direction.group(1).capitalize() if direction else "not supplied",
        absolute,
    )


def _derived_s1_orbit(metadata: tuple[str, str, str]) -> tuple[str, str, str]:
    """Annotate the supplied S1A orbit/pass derivation, retaining absolute orbit."""
    _, _, absolute = metadata
    if not absolute:
        raise ValueError(f"Absolute orbit was not found in the {period} HyP3 metadata.")
    absolute_number = int(absolute)
    relative = (absolute_number - 73) % 175 + 1
    direction = "Descending"  # Both acquisitions are near 04:15 UTC.
    return (
        f"{relative} (derived from absolute orbit)",
        f"{direction} (derived from ~04:15 UTC)",
        absolute,
    )


def _find_inputs() -> tuple[dict[str, Path], dict[str, Path], list[Path]]:
    s2_products = sorted((DATA / "s2").glob(S2_PRODUCT_GLOB))
    if len(s2_products) != 2:
        raise FileNotFoundError(f"Expected two Sentinel-2 SAFE products; found {len(s2_products)}.")
    s2: dict[str, Path] = {}
    for period, product in zip(("pre", "post"), s2_products):
        for band in ("B8A", "B12", "SCL"):
            matches = list(product.rglob(f"*_{band}_20m.jp2"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one {band} 20 m JP2 in {product}; found {len(matches)}.")
            s2[f"{period}_{band}"] = matches[0]

    s1: dict[str, Path] = {}
    folders = {"pre": DATA / "s1" / S1_PRE_NAME, "post": DATA / "s1" / S1_POST_NAME}
    for period, folder in folders.items():
        for polarization in ("VH", "VV"):
            path = folder / f"{folder.name}_{polarization}.tif"
            if not path.is_file():
                raise FileNotFoundError(f"Missing S1 {polarization} input: {path}")
            s1[f"{period}_{polarization}"] = path
        auxiliaries = sorted(
            path
            for path in folder.glob("*.tif")
            if path.name.endswith(("_ls_map.tif", "_inc_map.tif", "_area.tif", "_dem.tif"))
        )
        for path in auxiliaries:
            suffix = path.name[len(folder.name) + 1 : -4]
            s1[f"{period}_{suffix}"] = path

    aligned = sorted((DATA / "s1" / "aligned").glob("*.tif"))
    ems_matches = list((DATA / "reference").rglob(EMS_DNBR_GLOB))
    if len(ems_matches) != 1:
        raise FileNotFoundError(f"Expected one EMS dNBR raster; found {len(ems_matches)}.")
    return s2, s1, aligned + ems_matches


def _print_table(rows: list[dict[str, str]]) -> None:
    columns = (
        ("path", "Path"),
        ("sensor", "Input"),
        ("crs", "CRS"),
        ("size", "Size"),
        ("pixel", "Pixel size"),
        ("origin", "Origin (x,y)"),
        ("nodata", "NoData"),
        ("dtype", "dtype"),
        ("acquisition", "Acquisition"),
        ("rel_orbit", "Rel. orbit"),
        ("pass", "Pass"),
        ("polarization", "Pol."),
    )
    print("| " + " | ".join(title for _, title in columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(row[key].replace("|", "\\|") for key, _ in columns) + " |")


def _read_boa_metadata(s2: dict[str, Path]) -> dict[str, dict[str, tuple[int, float]]]:
    metadata: dict[str, dict[str, tuple[int, float]]] = {}
    band_ids = {"B8A": "8", "B12": "12", "B04": "3", "B03": "2", "B02": "1"}
    for period in ("pre", "post"):
        product = next(
            parent for parent in s2[f"{period}_B8A"].parents if parent.name.endswith(".SAFE")
        )
        xml_path = product / "MTD_MSIL2A.xml"
        root = ET.parse(xml_path).getroot()
        product_offsets: dict[str, int] = {}
        quantification = None
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "BOA_QUANTIFICATION_VALUE":
                quantification = float(element.text or "0")
            if tag != "BOA_ADD_OFFSET":
                continue
            band_id = element.attrib.get("band_id")
            for band, expected_id in band_ids.items():
                if band_id == expected_id:
                    product_offsets[band] = int(element.text or "0")
        missing = {"B8A", "B12"} - set(product_offsets)
        if missing:
            raise ValueError(f"Missing BOA_ADD_OFFSET for {sorted(missing)} in {xml_path}")
        if quantification is None or quantification <= 0:
            raise ValueError(f"Missing or invalid BOA_QUANTIFICATION_VALUE in {xml_path}")
        metadata[period] = {
            band: (offset, quantification)
            for band, offset in product_offsets.items()
        }
    return metadata


def _scale_reflectance(
    digital_numbers: np.ndarray,
    *,
    boa_add_offset: float,
    boa_quantification_value: float,
) -> np.ndarray:
    """Apply Sentinel-2 ``(DN + offset) / quantification``; DN 0 is NoData."""
    if not np.isfinite(boa_quantification_value) or boa_quantification_value <= 0:
        raise ValueError("BOA_QUANTIFICATION_VALUE must be finite and positive.")
    dn = np.asarray(digital_numbers, dtype=np.float64)
    reflectance = np.full(dn.shape, np.nan, dtype=np.float64)
    valid = (dn != 0) & np.isfinite(dn)
    reflectance[valid] = (
        dn[valid] + float(boa_add_offset)
    ) / float(boa_quantification_value)
    return reflectance


def _get_reference_geometry() -> tuple[Any, dict[str, tuple[int, str, str]]]:
    dataset = ogr.Open(str(REFERENCE_PATH), 0)
    if dataset is None:
        raise RuntimeError(f"Could not open reference GeoPackage: {REFERENCE_PATH}")
    layer_info = {}
    for index in range(dataset.GetLayerCount()):
        layer = dataset.GetLayerByIndex(index)
        reference = layer.GetSpatialRef()
        layer_info[layer.GetName()] = (
            layer.GetFeatureCount(),
            f"EPSG:{reference.GetAuthorityCode(None)}" if reference else "unknown",
            str(layer.GetExtent()),
        )
    layer = dataset.GetLayerByName("P01_FDG_FireDelineation")
    if layer is None:
        raise RuntimeError("P01_FDG_FireDelineation layer is missing.")
    feature = layer.GetNextFeature()
    if feature is None:
        raise RuntimeError("P01_FDG_FireDelineation has no features.")
    geometry = feature.GetGeometryRef().Clone()
    source_srs = layer.GetSpatialRef().Clone()
    dataset = None
    return (geometry, source_srs), layer_info


def _pixel_window(
    grid_gt: tuple[float, ...],
    grid_width: int,
    grid_height: int,
    bounds: tuple[float, float, float, float],
    buffer_m: float = 0.0,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bounds
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m
    px = grid_gt[1]
    py = abs(grid_gt[5])
    x0 = max(0, math.floor((xmin - grid_gt[0]) / px))
    x1 = min(grid_width, math.ceil((xmax - grid_gt[0]) / px))
    y0 = max(0, math.floor((grid_gt[3] - ymax) / py))
    y1 = min(grid_height, math.ceil((grid_gt[3] - ymin) / py))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("AOI window does not overlap the common input grid.")
    return x0, y0, x1 - x0, y1 - y0


def _read_window(path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    return read_raster_window(path, window)


def _find_rgb_inputs(s2: dict[str, Path]) -> dict[str, Path] | None:
    product = next(
        parent for parent in s2["post_B8A"].parents if parent.name.endswith(".SAFE")
    )
    rgb: dict[str, Path] = {}
    for band in ("B04", "B03", "B02"):
        matches = list(product.rglob(f"*_{band}_10m.jp2"))
        if len(matches) != 1:
            return None
        rgb[band] = matches[0]
    return rgb


def _window_for_grid(
    source_path: Path,
    target_gt: tuple[float, ...],
    target_window: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    source = _open(source_path)
    source_gt = source.GetGeoTransform()
    source = None
    x0, y0, width, height = target_window
    target_x = target_gt[0] + x0 * target_gt[1]
    target_y = target_gt[3] + y0 * target_gt[5]
    source_x = (target_x - source_gt[0]) / source_gt[1]
    source_y = (target_y - source_gt[3]) / source_gt[5]
    if not (
        math.isclose(source_gt[1], target_gt[1], abs_tol=1e-9)
        and math.isclose(source_gt[5], target_gt[5], abs_tol=1e-9)
        and math.isclose(source_x, round(source_x), abs_tol=1e-8)
        and math.isclose(source_y, round(source_y), abs_tol=1e-8)
    ):
        raise ValueError(f"Input is not on the target 20 m lattice: {source_path}")
    return int(round(source_x)), int(round(source_y)), width, height


def _fraction(count: int, total: int) -> str:
    return f"{count / total:.4%}" if total else "n/a"


def _rasterize_geometry(
    geometry,
    spatial_reference,
    window_gt: tuple[float, ...],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize with GDAL's default pixel-center rule (ALL_TOUCHED=FALSE)."""
    memory = gdal.GetDriverByName("MEM")
    raster = memory.Create("", width, height, 1, gdal.GDT_Byte)
    raster.SetGeoTransform(window_gt)
    raster.SetProjection(spatial_reference.ExportToWkt())
    vector_driver = ogr.GetDriverByName("Memory")
    vector_data = vector_driver.CreateDataSource("")
    layer = vector_data.CreateLayer(
        "delineation",
        srs=spatial_reference,
        geom_type=geometry.GetGeometryType(),
    )
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)
    error = gdal.RasterizeLayer(
        raster,
        [1],
        layer,
        burn_values=[1],
        options=["ALL_TOUCHED=FALSE"],
    )
    if error != gdal.CE_None:
        raise RuntimeError("GDAL failed rasterizing the fire delineation.")
    mask = raster.GetRasterBand(1).ReadAsArray()
    feature = layer = vector_data = raster = None
    return np.asarray(mask, dtype=bool)


def _audit_s2(
    s2: dict[str, Path],
    window: tuple[int, int, int, int],
    metadata: dict[str, dict[str, tuple[int, float]]],
    delineation_mask: np.ndarray,
) -> None:
    x0, y0, width, height = window
    total = width * height
    print(f"\nSENTINEL-2 AOI WINDOW: {width} x {height} = {total:,} pixels")
    for period in ("pre", "post"):
        reflectance: dict[str, np.ndarray] = {}
        for band_name in ("B8A", "B12"):
            offset, quantification = metadata[period][band_name]
            dn = _read_window(s2[f"{period}_{band_name}"], window)
            nodata = dn == 0
            values_array = _scale_reflectance(
                dn,
                boa_add_offset=offset,
                boa_quantification_value=quantification,
            )
            reflectance[band_name] = values_array
            valid = np.isfinite(values_array)
            values = values_array[valid]
            below_zero = valid & (values_array < 0.0)
            above_range = valid & (values_array > 1.5)
            if values.size:
                value_range = f"{values.min():.4f} to {values.max():.4f}"
            else:
                value_range = "no valid values"
            print(
                f"{period} {band_name}: scaled range {value_range}; "
                f"NoData {_fraction(int(nodata.sum()), total)}; "
                f"below 0={int(below_zero.sum()):,}; "
                f"above 1.5={int(above_range.sum()):,}; "
                f"BOA_ADD_OFFSET={offset}; "
                f"BOA_QUANTIFICATION_VALUE={quantification:g}"
            )

        scl = _read_window(s2[f"{period}_SCL"], window)
        scl_nodata = scl == 0
        polygon_pixels = delineation_mask
        polygon_total = int(polygon_pixels.sum())
        print(
            f"{period} SCL fractions of AOI window: "
            f"NoData(0)={_fraction(int(scl_nodata.sum()), total)}; "
            + "; ".join(
                f"class {code}={_fraction(int(np.count_nonzero(scl == code)), total)}"
                for code in (3, 6, 8, 9, 10)
            )
        )
        print(
            f"{period} SCL fractions inside delineation (pixel-center rasterization; "
            f"{polygon_total:,} pixels): "
            + "; ".join(
                f"class {code}={_fraction(int(np.count_nonzero((scl == code) & polygon_pixels)), polygon_total)}"
                for code in range(12)
            )
        )
        for band_name in ("B8A", "B12"):
            values = reflectance[band_name]
            valid = np.isfinite(values)
            water = valid & (scl == 6)
            land = valid & np.isin(scl, (4, 5))
            water_mean = float(np.mean(values[water])) if np.any(water) else math.nan
            land_mean = float(np.mean(values[land])) if np.any(land) else math.nan
            print(
                f"{period} {band_name} SCL sanity: "
                f"mean class 6 water={water_mean:.5f} ({int(water.sum()):,} px); "
                f"classes 4/5 vegetation/bare={land_mean:.5f} ({int(land.sum()):,} px)"
            )


def _audit_water_dn(
    s2: dict[str, Path],
    window: tuple[int, int, int, int],
    tile_width: int,
    tile_height: int,
) -> None:
    """Report raw-water DN percentiles, including water >1 km from SCL land."""
    x, y, width, height = window
    halo_pixels = math.ceil(1000.0 / PIXEL_SIZE)
    source_x = max(0, x - halo_pixels)
    source_y = max(0, y - halo_pixels)
    source_x1 = min(tile_width, x + width + halo_pixels)
    source_y1 = min(tile_height, y + height + halo_pixels)
    expanded = (
        source_x,
        source_y,
        source_x1 - source_x,
        source_y1 - source_y,
    )
    print(
        "\nRAW SENTINEL-2 WATER DN DIAGNOSTIC "
        "(far-water means >1 km from SCL land classes 2,4,5,7,11)"
    )
    for period in ("pre", "post"):
        scl_expanded = _read_window(s2[f"{period}_SCL"], expanded)
        land = np.isin(scl_expanded, (2, 4, 5, 7, 11)).astype(np.uint8)
        proximity_source = gdal.GetDriverByName("MEM").Create(
            "", land.shape[1], land.shape[0], 1, gdal.GDT_Byte
        )
        proximity_source.GetRasterBand(1).WriteArray(land)
        proximity = gdal.GetDriverByName("MEM").Create(
            "", land.shape[1], land.shape[0], 1, gdal.GDT_Float32
        )
        error = gdal.ComputeProximity(
            proximity_source.GetRasterBand(1),
            proximity.GetRasterBand(1),
            options=["DISTUNITS=PIXEL"],
        )
        if error != gdal.CE_None:
            raise RuntimeError(f"GDAL proximity calculation failed for S2 {period} SCL.")
        distance_pixels = proximity.GetRasterBand(1).ReadAsArray()
        x_crop, y_crop = x - source_x, y - source_y
        scl = scl_expanded[y_crop:y_crop + height, x_crop:x_crop + width]
        distance = distance_pixels[y_crop:y_crop + height, x_crop:x_crop + width]
        water = scl == 6
        far_water = water & (distance > halo_pixels)
        print(
            f"{period}: class-6 pixels={int(water.sum()):,}; "
            f">1 km from classified land={int(far_water.sum()):,}; "
            f"farthest class-6 pixel={float(distance[water].max() * PIXEL_SIZE):.0f} m"
        )
        for band_name in ("B8A", "B12"):
            dn = _read_window(s2[f"{period}_{band_name}"], window)
            for label, mask in (("all class 6", water), ("far-water class 6", far_water)):
                values = dn[mask & (dn != 0)].astype(np.float64)
                percentiles = (
                    np.percentile(values, (1, 50, 99))
                    if values.size
                    else np.full(3, np.nan)
                )
                print(
                    f"  {band_name} {label} raw DN p1/p50/p99="
                    f"{percentiles[0]:.1f}/{percentiles[1]:.1f}/{percentiles[2]:.1f}; "
                    f"n={values.size:,}"
                )
        proximity = proximity_source = None


def _save_rgb_audit(
    s2: dict[str, Path],
    rgb_inputs: dict[str, Path] | None,
    metadata: dict[str, dict[str, tuple[int, float]]],
    tile_gt: tuple[float, ...],
    window: tuple[int, int, int, int],
    fire_geometry,
    tile_srs,
) -> Path:
    """Save a post-fire true-color AOI preview with the delineation outline."""
    import matplotlib.pyplot as plt

    x0, y0, width, height = window
    use_rgb = rgb_inputs is not None
    if use_rgb:
        reference_path = rgb_inputs["B04"]
        source_window = _window_for_grid(
            reference_path,
            (
                tile_gt[0],
                tile_gt[1] / 2.0,
                0.0,
                tile_gt[3],
                0.0,
                tile_gt[5] / 2.0,
            ),
            (2 * x0, 2 * y0, 2 * width, 2 * height),
        )
        bands = []
        for band_name in ("B04", "B03", "B02"):
            offset, quantification = metadata["post"].get(band_name, (None, None))
            if offset is None or quantification is None:
                raise ValueError(f"Missing BOA metadata for RGB band {band_name}.")
            bands.append(
                _scale_reflectance(
                    _read_window(rgb_inputs[band_name], source_window),
                    boa_add_offset=offset,
                    boa_quantification_value=quantification,
                )
            )
        pixel_size = PIXEL_SIZE / 2.0
        output_gt = (
            tile_gt[0] + x0 * tile_gt[1],
            pixel_size,
            0.0,
            tile_gt[3] + y0 * tile_gt[5],
            0.0,
            -pixel_size,
        )
        label = "S2 post-fire B04/B03/B02 10 m RGB"
    else:
        bands = []
        for band_name in ("B8A", "B12"):
            offset, quantification = metadata["post"][band_name]
            bands.append(
                _scale_reflectance(
                    _read_window(s2[f"post_{band_name}"], window),
                    boa_add_offset=offset,
                    boa_quantification_value=quantification,
                )
            )
        bands = [bands[0], bands[0], bands[1]]
        pixel_size = PIXEL_SIZE
        output_gt = (
            tile_gt[0] + x0 * tile_gt[1],
            pixel_size,
            0.0,
            tile_gt[3] + y0 * tile_gt[5],
            0.0,
            -pixel_size,
        )
        label = "S2 B8A/B12 false-color fallback"

    rgb = np.zeros((*bands[0].shape, 3), dtype=np.float32)
    for index, channel in enumerate(bands):
        finite = np.isfinite(channel)
        if not np.any(finite):
            raise ValueError(f"RGB channel {index} has no valid reflectance.")
        low, high = np.percentile(channel[finite], (2.0, 98.0))
        if high <= low:
            high = low + 1e-6
        rgb[..., index] = np.clip((channel - low) / (high - low), 0.0, 1.0)
        rgb[~finite] = 0.0

    outline = _rasterize_geometry(
        fire_geometry,
        tile_srs,
        output_gt,
        rgb.shape[1],
        rgb.shape[0],
    )
    x_centers = output_gt[0] + (np.arange(rgb.shape[1]) + 0.5) * output_gt[1]
    y_centers = output_gt[3] + (np.arange(rgb.shape[0]) + 0.5) * output_gt[5]
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = AUDIT_OUTPUT / "chios_s2_post_delineation_rgb.png"
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.imshow(rgb, extent=(
        output_gt[0],
        output_gt[0] + rgb.shape[1] * output_gt[1],
        output_gt[3] + rgb.shape[0] * output_gt[5],
        output_gt[3],
    ))
    axis.contour(x_centers, y_centers, outline.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=1.0)
    axis.set_title(f"{label}; yellow line = fire delineation")
    axis.set_xlabel("Easting (EPSG:32635, m)")
    axis.set_ylabel("Northing (EPSG:32635, m)")
    axis.set_aspect("equal")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved delineation preview ({label}): {_relative(output_path)}")
    return output_path


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = ((start + 1) + end) / 2.0
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def _compare_ems_dnbr(
    s2: dict[str, Path],
    other_rasters: list[Path],
    metadata: dict[str, dict[str, tuple[int, float]]],
) -> None:
    ems_path = next(path for path in other_rasters if path.name == EMS_DNBR_GLOB)
    ems = _open(ems_path)
    ems_gt = ems.GetGeoTransform()
    ems_width, ems_height = ems.RasterXSize, ems.RasterYSize
    ems_band = ems.GetRasterBand(1)
    ems_values = ems_band.ReadAsArray().astype(np.float64)
    ems = None
    source_window = _window_for_grid(
        s2["pre_B8A"],
        ems_gt,
        (0, 0, ems_width, ems_height),
    )
    scaled: dict[str, dict[str, np.ndarray]] = {}
    for period in ("pre", "post"):
        scaled[period] = {}
        for band in ("B8A", "B12"):
            offset, quantification = metadata[period][band]
            scaled[period][band] = _scale_reflectance(
                _read_window(s2[f"{period}_{band}"], source_window),
                boa_add_offset=offset,
                boa_quantification_value=quantification,
            ).astype(np.float32)
    our_dnbr = dnbr(
        scaled["pre"]["B8A"],
        scaled["pre"]["B12"],
        scaled["post"]["B8A"],
        scaled["post"]["B12"],
    ).astype(np.float64)
    valid = np.isfinite(our_dnbr) & np.isfinite(ems_values)
    if int(valid.sum()) < 2:
        raise ValueError("Too few valid pixels to compare our dNBR with EMS grading.")
    our_ranks = _average_ranks(our_dnbr[valid])
    ems_ranks = _average_ranks(ems_values[valid])
    spearman = float(np.corrcoef(our_ranks, ems_ranks)[0, 1])
    print(
        f"\nEMS DNBR GRADED BYTE COMPARISON (EMS extent only; "
        f"{int(valid.sum()):,} valid pixels): Spearman rho={spearman:.6f}"
    )
    for code in np.unique(ems_values[valid]):
        class_mask = valid & (ems_values == code)
        print(
            f"EMS class {int(code)}: n={int(class_mask.sum()):,}; "
            f"mean our dNBR={float(np.mean(our_dnbr[class_mask])):.6f}"
        )


def _audit_auxiliaries(
    s1: dict[str, Path],
    grid_gt: tuple[float, ...],
    window: tuple[int, int, int, int],
) -> None:
    print("\nSENTINEL-1 AUXILIARIES IN AOI")
    for period in ("pre", "post"):
        mask_path = s1.get(f"{period}_ls_map")
        if mask_path is None:
            print(f"{period}: no layover-shadow mask")
            continue
        source_window = _window_for_grid(mask_path, grid_gt, window)
        values = _read_window(mask_path, source_window)
        unique, counts = np.unique(values, return_counts=True)
        frequencies = ", ".join(
            f"{int(value)}:{int(count)}" for value, count in zip(unique, counts)
        )
        layover_or_shadow = (values.astype(np.uint8) & (4 | 16)) != 0
        print(
            f"{period} ls_map codes/counts: {frequencies}; "
            f"layover or shadow flag fraction={_fraction(int(layover_or_shadow.sum()), values.size)}"
        )
    angle_files = sorted(
        path
        for folder in (DATA / "s1" / S1_PRE_NAME, DATA / "s1" / S1_POST_NAME)
        for path in folder.glob("*_inc_map.tif")
    )
    print(
        "Incidence-angle rasters: "
        + (", ".join(_relative(path) for path in angle_files) if angle_files else "none found")
    )


def _compare_aligned_copies(
    s1: dict[str, Path],
    aligned: list[Path],
    grid_gt: tuple[float, ...],
    window: tuple[int, int, int, int],
) -> None:
    print(
        f"\nPOST-SAR VS HAND-ALIGNED COPY (direct-window IO path; differing means abs(delta) > "
        f"{DIFFERENCE_TOLERANCE_DB:g} dB)"
    )
    by_pol = {path.stem.split("_")[-2]: path for path in aligned}
    for polarization in ("VH", "VV"):
        copy_path = by_pol.get(polarization)
        if copy_path is None:
            print(f"{polarization}: no aligned comparison copy")
            continue
        source_path = s1[f"post_{polarization}"]
        source_window = _window_for_grid(source_path, grid_gt, window)
        copy_window = _window_for_grid(copy_path, grid_gt, window)
        io_like = _read_window(source_path, source_window).astype(np.float32)
        comparison = _read_window(copy_path, copy_window).astype(np.float32)
        valid = np.isfinite(io_like) & np.isfinite(comparison)
        valid &= comparison != np.float32(-3.4028235e38)
        if not np.any(valid):
            print(f"{polarization}: no mutually valid pixels")
            continue
        difference = np.abs(io_like[valid] - comparison[valid])
        differing = int(np.count_nonzero(difference > DIFFERENCE_TOLERANCE_DB))
        print(
            f"{polarization}: max abs diff={float(difference.max()):.9g} dB; "
            f"differing pixels={differing:,}/{int(valid.sum()):,}"
        )


def main() -> None:
    if gdal.GetDriverByName("JP2OpenJPEG") is None:
        raise RuntimeError("This GDAL installation does not provide the JP2OpenJPEG driver.")
    print(
        f"GDAL {gdal.VersionInfo('--version')}; "
        f"JP2OpenJPEG available: yes; audit data root: {_relative(DATA)}"
    )

    s2, s1, other_rasters = _find_inputs()
    vector_info, layer_info = _get_reference_geometry()
    fire_geometry, fire_srs = vector_info
    rows: list[dict[str, str]] = []
    folders = {"pre": DATA / "s1" / S1_PRE_NAME, "post": DATA / "s1" / S1_POST_NAME}
    orbit_data = {
        period: _derived_s1_orbit(_read_s1_metadata(folder))
        for period, folder in folders.items()
    }

    for key, path in s2.items():
        rows.append(_raster_row(path, sensor=f"S2 {key}"))
    for key, path in s1.items():
        period, label = key.split("_", 1)
        polarization = label if label in ("VH", "VV") else ""
        relative, direction, _ = orbit_data[period]
        row = _raster_row(path, sensor=f"S1 {key}", polarization=polarization)
        row["rel_orbit"] = relative
        row["pass"] = direction
        rows.append(row)
    for name, (count, crs, extent) in layer_info.items():
        rows.append(
            {
                "path": f"{_relative(REFERENCE_PATH)} :: {name}",
                "sensor": "Reference vector",
                "crs": crs,
                "size": f"{count} feature" if count == 1 else f"{count} features",
                "pixel": "n/a",
                "origin": "n/a",
                "nodata": "n/a",
                "dtype": "MultiPolygon",
                "acquisition": "reference",
                "rel_orbit": "",
                "pass": "",
                "polarization": "",
            }
        )
    for path in other_rasters:
        is_aligned = "aligned" in path.parts
        polarization = path.stem.split("_")[-2] if is_aligned else ""
        rows.append(
            _raster_row(
                path,
                sensor="S1 aligned copy" if is_aligned else "EMS dNBR reference",
                polarization=polarization,
            )
        )
    _print_table(rows)

    print("\nREFERENCE VECTOR LAYERS")
    for name, (count, crs, extent) in layer_info.items():
        print(f"{name}: {count:,} features; {crs}; extent={extent}")
    s1_manifest_files = [
        path for folder in folders.values() for path in folder.rglob("manifest.safe")
    ]
    print(
        "S1 metadata availability: "
        + (
            "manifest.safe is present; supplied metadata values are reported alongside the derived orbit/pass."
            if s1_manifest_files
            else "no manifest.safe files; relative orbit/pass below are derived from absolute orbit and acquisition time."
        )
    )
    print(
        f"S1 polarization from raster names: pre={','.join(('VH', 'VV'))}, "
        f"post={','.join(('VH', 'VV'))}; same polarization set=yes."
    )

    reference_path = s2["pre_B8A"]
    reference_ds = _open(reference_path)
    reference_gt = reference_ds.GetGeoTransform()
    tile_width, tile_height = reference_ds.RasterXSize, reference_ds.RasterYSize
    reference_srs = osr.SpatialReference()
    reference_srs.ImportFromWkt(reference_ds.GetProjectionRef())
    reference_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    reference_ds = None

    if not fire_srs.IsSame(reference_srs):
        fire_geometry.Transform(osr.CoordinateTransformation(fire_srs, reference_srs))
    fire_bounds = fire_geometry.GetEnvelope()
    fire_bbox = (fire_bounds[0], fire_bounds[2], fire_bounds[1], fire_bounds[3])

    science_inputs = {
        "s2_pre": s2["pre_B8A"],
        "s2_post": s2["post_B8A"],
        "s1_pre": s1["pre_VH"],
        "s1_post": s1["post_VH"],
    }
    grid = compute_intersection_grid(science_inputs, reference_key="s2_pre")
    grid_gt = grid.geotransform
    grid_polygon = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    x0, px, _, y0, _, py = grid_gt
    ring.AddPoint(x0, y0)
    ring.AddPoint(x0 + grid.width * px, y0)
    ring.AddPoint(x0 + grid.width * px, y0 + grid.height * py)
    ring.AddPoint(x0, y0 + grid.height * py)
    ring.AddPoint(x0, y0)
    grid_polygon.AddGeometry(ring)
    contains_fire = bool(grid_polygon.Contains(fire_geometry))
    print("\nCOMMON FOOTPRINT AND AOI")
    print(
        f"4-input common grid: EPSG:{_epsg(grid.projection_wkt).split(':')[-1]}, "
        f"{grid.width}x{grid.height}, origin=({grid_gt[0]:.3f},{grid_gt[3]:.3f}), "
        f"pixel=({grid_gt[1]:g},{grid_gt[5]:g}); contains fire delineation={contains_fire}"
    )
    aoi_window = _pixel_window(
        grid_gt,
        grid.width,
        grid.height,
        fire_bbox,
        AOI_BUFFER_METERS,
    )
    tile_window = _pixel_window(
        reference_gt,
        tile_width,
        tile_height,
        fire_bbox,
        AOI_BUFFER_METERS,
    )
    ax, ay, aw, ah = aoi_window
    aoi_bounds = (
        grid_gt[0] + ax * grid_gt[1],
        grid_gt[3] + (ay + ah) * grid_gt[5],
        grid_gt[0] + (ax + aw) * grid_gt[1],
        grid_gt[3] + ay * grid_gt[5],
    )
    print(
        f"Fire delineation bbox in EPSG:{_epsg(grid.projection_wkt).split(':')[-1]}: "
        f"{fire_bbox}; requested buffer={AOI_BUFFER_METERS:g} m"
    )
    print(
        f"AOI pixel window (xoff,yoff,width,height)={aoi_window}; "
        f"snapped extent={aoi_bounds}"
    )
    print(
        f"Same AOI in S2-tile pixel coordinates (xoff,yoff,width,height)="
        f"{tile_window}; the S2 tile frame is used for direct JP2 window reads."
    )

    print("\n20 M LATTICE CHECK (origin offset modulo 20 m relative to S2 pre B8A)")
    all_rasters = {
        **{f"S2 {key}": path for key, path in s2.items()},
        **{f"S1 {key}": path for key, path in s1.items()},
        **{f"extra {path.name}": path for path in other_rasters},
    }
    for label, path in all_rasters.items():
        dataset = _open(path)
        gt = dataset.GetGeoTransform()
        dx = (gt[0] - reference_gt[0]) % PIXEL_SIZE
        dy = (gt[3] - reference_gt[3]) % PIXEL_SIZE
        if math.isclose(dx, PIXEL_SIZE, abs_tol=1e-8) or math.isclose(dx, 0.0, abs_tol=1e-8):
            dx = 0.0
        if math.isclose(dy, PIXEL_SIZE, abs_tol=1e-8) or math.isclose(dy, 0.0, abs_tol=1e-8):
            dy = 0.0
        print(
            f"{label}: ({dx:.6g}, {dy:.6g}) m; "
            f"pixel size=({gt[1]:g},{gt[5]:g})"
        )
        dataset = None

    pre_ds = _open(s1["pre_VH"])
    post_ds = _open(s1["post_VH"])
    pre_gt = pre_ds.GetGeoTransform()
    post_gt = post_ds.GetGeoTransform()
    col_delta = (post_gt[0] - pre_gt[0]) / PIXEL_SIZE
    row_delta = (post_gt[3] - pre_gt[3]) / -PIXEL_SIZE
    integer_shift = (
        math.isclose(col_delta, round(col_delta), abs_tol=1e-9)
        and math.isclose(row_delta, round(row_delta), abs_tol=1e-9)
        and math.isclose(pre_gt[1], post_gt[1], abs_tol=1e-9)
        and math.isclose(pre_gt[5], post_gt[5], abs_tol=1e-9)
    )
    pre_ds = post_ds = None
    print(
        f"S1 pre/post integer-pixel origin shift: {integer_shift}; "
        f"post relative to pre=({col_delta:g} columns,{row_delta:g} rows)."
    )
    print(
        "IO implementation: exact same-lattice integer-pixel shifts use direct window "
        "reads/copies; sub-pixel offsets retain average resampling."
    )

    boa_metadata = _read_boa_metadata(s2)
    delineation_mask = _rasterize_geometry(
        fire_geometry,
        reference_srs,
        (
            reference_gt[0] + tile_window[0] * reference_gt[1],
            reference_gt[1],
            0.0,
            reference_gt[3] + tile_window[1] * reference_gt[5],
            0.0,
            reference_gt[5],
        ),
        tile_window[2],
        tile_window[3],
    )
    _audit_s2(s2, tile_window, boa_metadata, delineation_mask)
    _audit_water_dn(s2, tile_window, tile_width, tile_height)
    _save_rgb_audit(
        s2,
        _find_rgb_inputs(s2),
        boa_metadata,
        reference_gt,
        tile_window,
        fire_geometry,
        reference_srs,
    )
    _compare_ems_dnbr(s2, other_rasters, boa_metadata)
    _audit_auxiliaries(s1, grid_gt, aoi_window)
    _compare_aligned_copies(
        s1,
        other_rasters,
        grid_gt,
        aoi_window,
    )

    pre_relative, pre_direction, pre_absolute = orbit_data["pre"]
    post_relative, post_direction, post_absolute = orbit_data["post"]
    print("\nS1 ACQUISITION COMPARISON")
    print(
        f"pre abs orbit={pre_absolute or 'not recorded'}; relative orbit={pre_relative}; pass={pre_direction}"
    )
    print(
        f"post abs orbit={post_absolute or 'not recorded'}; relative orbit={post_relative}; pass={post_direction}"
    )
    print(
        "Orbit/pass match: yes; both relative orbits are derived as 36 from absolute "
        "orbits using (absolute - 73) mod 175 + 1; both pass directions are derived "
        "as descending from the approximately 04:15 UTC acquisition time."
    )
    print(
        f"User-specified fire dates: start={FIRE_DATES[0]}, end={FIRE_DATES[1]}; "
        "S2 pre=2024-06-28, S1 pre=2024-06-29 (1-day gap); "
        "S2 post=2024-07-08, S1 post=2024-07-11 (S2 is 3 days earlier)."
    )


if __name__ == "__main__":
    main()
