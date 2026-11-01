"""Block-wise GDAL raster inspection, intersection, alignment, and SAR conversion."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RasterGrid:
    """North-up target grid in the reference raster's coordinate system."""

    width: int
    height: int
    geotransform: tuple[float, float, float, float, float, float]
    projection_wkt: str


@dataclass(frozen=True)
class RasterInfo:
    """Metadata needed to align a single-band raster."""

    path: Path
    width: int
    height: int
    geotransform: tuple[float, float, float, float, float, float]
    projection_wkt: str
    nodata: float | None


@dataclass(frozen=True)
class RasterBlock:
    """A bounded raster window and its 2D float32 values."""

    window: tuple[int, int, int, int]
    data: NDArray[np.float32]

    @property
    def x_offset(self) -> int:
        return self.window[0]

    @property
    def y_offset(self) -> int:
        return self.window[1]


@dataclass(frozen=True)
class AlignedBlock:
    """Matching aligned input blocks with a shared NoData mask."""

    window: tuple[int, int, int, int]
    arrays: Mapping[str, NDArray[np.float32]]
    read_window: tuple[int, int, int, int] | None = None

    @property
    def x_offset(self) -> int:
        return self.window[0]

    @property
    def y_offset(self) -> int:
        return self.window[1]

    @property
    def core_slices(self) -> tuple[slice, slice]:
        """Return the block window inside arrays that may include a halo."""
        read_window = self.read_window or self.window
        x_offset, y_offset, width, height = self.window
        read_x, read_y, _, _ = read_window
        x_start = x_offset - read_x
        y_start = y_offset - read_y
        return (
            slice(y_start, y_start + height),
            slice(x_start, x_start + width),
        )


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned output paths and their common target grid."""

    grid: RasterGrid
    outputs: Mapping[str, Path]


def _gdal_modules():
    """Load GDAL lazily so the package remains importable without osgeo."""
    try:
        from osgeo import gdal, osr
    except ImportError as error:
        raise ImportError(
            "Raster IO requires GDAL's Python bindings (osgeo.gdal and osgeo.osr)."
        ) from error
    gdal.UseExceptions()
    return gdal, osr


def _open_raster(path: str | Path):
    gdal, _ = _gdal_modules()
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open raster: {path}")
    if dataset.RasterCount != 1:
        raise ValueError(f"Raster must have exactly one band: {path}")
    return dataset


def inspect_raster(path: str | Path) -> RasterInfo:
    """Read dimensions, CRS, geotransform, and declared NoData for one raster."""
    dataset = _open_raster(path)
    projection = dataset.GetProjection()
    if not projection:
        raise ValueError(f"Raster has no coordinate reference system: {path}")
    geotransform = tuple(float(value) for value in dataset.GetGeoTransform())
    band = dataset.GetRasterBand(1)
    info = RasterInfo(
        path=Path(path),
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
        geotransform=geotransform,
        projection_wkt=projection,
        nodata=band.GetNoDataValue(),
    )
    dataset = None
    return info


def read_raster_window(
    path: str | Path,
    window: tuple[int, int, int, int],
) -> NDArray:
    """Read a native-type ``(x_offset, y_offset, width, height)`` pixel window."""
    x_offset, y_offset, width, height = window
    if min(x_offset, y_offset) < 0 or min(width, height) < 1:
        raise ValueError("Raster window offsets must be non-negative and sizes positive.")
    dataset = _open_raster(path)
    if x_offset + width > dataset.RasterXSize or y_offset + height > dataset.RasterYSize:
        raise ValueError(f"Raster window is outside the source dimensions: {path}")
    data = dataset.GetRasterBand(1).ReadAsArray(x_offset, y_offset, width, height)
    dataset = None
    if data is None:
        raise RuntimeError(f"GDAL failed reading raster window from {path}.")
    return np.asarray(data)


def _validate_north_up(info: RasterInfo) -> None:
    gt = info.geotransform
    if not math.isclose(gt[2], 0.0, abs_tol=1e-12) or not math.isclose(
        gt[4], 0.0, abs_tol=1e-12
    ):
        raise ValueError(f"Rotated rasters are not supported: {info.path}")
    if gt[1] <= 0.0 or gt[5] >= 0.0:
        raise ValueError(f"Raster must be north-up with positive x and negative y pixel sizes: {info.path}")


def _raster_bounds(info: RasterInfo) -> tuple[float, float, float, float]:
    _validate_north_up(info)
    gt = info.geotransform
    return (
        gt[0],
        gt[3] + info.height * gt[5],
        gt[0] + info.width * gt[1],
        gt[3],
    )


def compute_intersection_grid(
    input_paths: Mapping[str, str | Path],
    *,
    reference_key: str,
    expected_resolution: float = 20.0,
    resolution_tolerance: float = 0.1,
) -> RasterGrid:
    """Compute the intersection footprint on the reference raster's exact grid.

    Every input bounds is transformed into the reference CRS. The intersection
    is snapped inward to whole reference pixels, preserving the Sentinel-2
    origin, pixel size, and 20 m reference alignment.
    """
    if not input_paths:
        raise ValueError("input_paths must contain at least one raster.")
    if reference_key not in input_paths:
        raise ValueError(f"reference_key {reference_key!r} is not present in input_paths.")
    if expected_resolution <= 0.0 or resolution_tolerance < 0.0:
        raise ValueError("expected_resolution must be positive and tolerance non-negative.")

    _, osr = _gdal_modules()
    reference = inspect_raster(input_paths[reference_key])
    _validate_north_up(reference)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(reference.projection_wkt)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not target_srs.IsProjected():
        raise ValueError("The 20 m reference raster must use a projected CRS.")
    linear_units_to_meters = target_srs.GetLinearUnits()
    if not math.isclose(linear_units_to_meters, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The reference projected CRS must use meters.")

    pixel_width = reference.geotransform[1]
    pixel_height = abs(reference.geotransform[5])
    if not (
        math.isclose(
            pixel_width * linear_units_to_meters,
            expected_resolution,
            abs_tol=resolution_tolerance,
        )
        and math.isclose(
            pixel_height * linear_units_to_meters,
            expected_resolution,
            abs_tol=resolution_tolerance,
        )
    ):
        raise ValueError(
            f"Reference grid must be {expected_resolution:g} m square pixels; "
            f"got {pixel_width:g} x {pixel_height:g}."
        )

    xmin = ymin = -np.inf
    xmax = ymax = np.inf

    for key, path in input_paths.items():
        info = inspect_raster(path)
        source_srs = osr.SpatialReference()
        source_srs.ImportFromWkt(info.projection_wkt)
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_srs, target_srs)
        source_bounds = _raster_bounds(info)
        try:
            bounds = transform.TransformBounds(*source_bounds, 21)
        except Exception as error:
            raise ValueError(f"Could not transform bounds for input {key!r}: {path}") from error
        xmin = max(xmin, bounds[0])
        ymin = max(ymin, bounds[1])
        xmax = min(xmax, bounds[2])
        ymax = min(ymax, bounds[3])

    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Input rasters have no overlapping footprint.")

    ref_gt = reference.geotransform
    col_start = math.ceil((xmin - ref_gt[0]) / pixel_width - 1e-10)
    col_end = math.floor((xmax - ref_gt[0]) / pixel_width + 1e-10)
    row_start = math.ceil((ref_gt[3] - ymax) / pixel_height - 1e-10)
    row_end = math.floor((ref_gt[3] - ymin) / pixel_height + 1e-10)
    width = col_end - col_start
    height = row_end - row_start
    if width <= 0 or height <= 0:
        raise ValueError("Intersection does not contain a complete reference-grid pixel.")

    geotransform = (
        ref_gt[0] + col_start * pixel_width,
        ref_gt[1],
        0.0,
        ref_gt[3] + row_start * ref_gt[5],
        0.0,
        ref_gt[5],
    )
    return RasterGrid(width, height, geotransform, reference.projection_wkt)


def compute_grid_for_extent(
    reference_path: str | Path,
    extent_bounds: tuple[float, float, float, float],
    extent_projection_wkt: str,
    *,
    expected_resolution: float = 20.0,
    resolution_tolerance: float = 0.1,
) -> RasterGrid:
    """Snap a CRS-aware extent inward to the reference raster's pixel lattice."""
    if len(extent_bounds) != 4 or any(
        not math.isfinite(float(value)) for value in extent_bounds
    ):
        raise ValueError("extent_bounds must contain four finite values.")
    xmin, ymin, xmax, ymax = map(float, extent_bounds)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("extent_bounds must have positive width and height.")
    if expected_resolution <= 0.0 or resolution_tolerance < 0.0:
        raise ValueError("expected_resolution must be positive and tolerance non-negative.")

    _, osr = _gdal_modules()
    reference = inspect_raster(reference_path)
    _validate_north_up(reference)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(reference.projection_wkt)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not target_srs.IsProjected():
        raise ValueError("The reference raster must use a projected CRS.")
    units_to_meters = target_srs.GetLinearUnits()
    if not math.isclose(units_to_meters, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The reference projected CRS must use meters.")

    pixel_width = reference.geotransform[1]
    pixel_height = abs(reference.geotransform[5])
    if not (
        math.isclose(pixel_width, expected_resolution, abs_tol=resolution_tolerance)
        and math.isclose(pixel_height, expected_resolution, abs_tol=resolution_tolerance)
    ):
        raise ValueError(
            f"Reference grid must be {expected_resolution:g} m square pixels; "
            f"got {pixel_width:g} x {pixel_height:g}."
        )

    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(extent_projection_wkt)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    try:
        bounds = transform.TransformBounds(xmin, ymin, xmax, ymax, 21)
    except Exception as error:
        raise ValueError("Could not transform AOI extent to the reference CRS.") from error

    ref_xmin, ref_ymin, ref_xmax, ref_ymax = _raster_bounds(reference)
    xmin = max(bounds[0], ref_xmin)
    ymin = max(bounds[1], ref_ymin)
    xmax = min(bounds[2], ref_xmax)
    ymax = min(bounds[3], ref_ymax)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("AOI extent does not overlap the reference raster.")

    ref_gt = reference.geotransform
    col_start = math.ceil((xmin - ref_gt[0]) / pixel_width - 1e-10)
    col_end = math.floor((xmax - ref_gt[0]) / pixel_width + 1e-10)
    row_start = math.ceil((ref_gt[3] - ymax) / pixel_height - 1e-10)
    row_end = math.floor((ref_gt[3] - ymin) / pixel_height + 1e-10)
    width = col_end - col_start
    height = row_end - row_start
    if width <= 0 or height <= 0:
        raise ValueError("AOI extent does not contain a complete reference-grid pixel.")
    geotransform = (
        ref_gt[0] + col_start * pixel_width,
        ref_gt[1],
        0.0,
        ref_gt[3] + row_start * ref_gt[5],
        0.0,
        ref_gt[5],
    )
    return RasterGrid(width, height, geotransform, reference.projection_wkt)


def _window_bounds(
    grid: RasterGrid,
    x_offset: int,
    y_offset: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    gt = grid.geotransform
    xmin = gt[0] + x_offset * gt[1]
    xmax = xmin + width * gt[1]
    ymax = gt[3] + y_offset * gt[5]
    ymin = ymax + height * gt[5]
    return xmin, ymin, xmax, ymax


def _normalize_nodata(
    values: NDArray[np.float32],
    nodata: float | None,
) -> NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32)
    invalid = ~np.isfinite(result)
    if nodata is not None:
        if math.isnan(nodata):
            invalid |= np.isnan(result)
        else:
            invalid |= result == nodata
    invalid |= result == np.float32(-3.4028235e38)
    if np.any(invalid):
        result = result.copy()
        result[invalid] = np.nan
    return result


def iter_raster_blocks(
    path: str | Path,
    *,
    block_size: int = 512,
) -> Iterator[RasterBlock]:
    """Read a single-band raster in bounded 2D float32 windows."""
    if block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    dataset = _open_raster(path)
    band = dataset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    try:
        for y_offset in range(0, dataset.RasterYSize, block_size):
            height = min(block_size, dataset.RasterYSize - y_offset)
            for x_offset in range(0, dataset.RasterXSize, block_size):
                width = min(block_size, dataset.RasterXSize - x_offset)
                data = band.ReadAsArray(x_offset, y_offset, width, height)
                if data is None:
                    raise RuntimeError(f"GDAL failed reading raster block from {path}.")
                yield RasterBlock(
                    (x_offset, y_offset, width, height),
                    _normalize_nodata(data, nodata),
                )
    finally:
        band = None
        dataset = None


def iter_aligned_blocks(
    input_paths: Mapping[str, str | Path],
    *,
    block_size: int = 512,
    halo: int = 0,
) -> Iterator[AlignedBlock]:
    """Yield same-grid blocks, propagating any input NoData across all arrays."""
    if not input_paths:
        raise ValueError("input_paths must contain at least one raster.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    if isinstance(halo, bool) or not isinstance(halo, int) or halo < 0:
        raise ValueError("halo must be a non-negative integer.")
    infos = {key: inspect_raster(path) for key, path in input_paths.items()}
    first = next(iter(infos.values()))
    _, osr = _gdal_modules()
    first_srs = osr.SpatialReference()
    first_srs.ImportFromWkt(first.projection_wkt)
    first_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    for key, info in infos.items():
        if (info.width, info.height) != (first.width, first.height):
            raise ValueError("Aligned raster inputs must have matching dimensions.")
        if not np.allclose(info.geotransform, first.geotransform, rtol=0.0, atol=1e-9):
            raise ValueError(f"Aligned raster input {key!r} has a different geotransform.")
        other_srs = osr.SpatialReference()
        other_srs.ImportFromWkt(info.projection_wkt)
        other_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not bool(first_srs.IsSame(other_srs)):
            raise ValueError(f"Aligned raster input {key!r} has a different CRS.")
    datasets = {key: _open_raster(path) for key, path in input_paths.items()}
    bands = {key: dataset.GetRasterBand(1) for key, dataset in datasets.items()}
    try:
        for y_offset in range(0, first.height, block_size):
            height = min(block_size, first.height - y_offset)
            for x_offset in range(0, first.width, block_size):
                width = min(block_size, first.width - x_offset)
                read_x = max(0, x_offset - halo)
                read_y = max(0, y_offset - halo)
                read_right = min(first.width, x_offset + width + halo)
                read_bottom = min(first.height, y_offset + height + halo)
                read_width = read_right - read_x
                read_height = read_bottom - read_y
                arrays: dict[str, NDArray[np.float32]] = {}
                for key, band in bands.items():
                    data = band.ReadAsArray(read_x, read_y, read_width, read_height)
                    if data is None:
                        raise RuntimeError(f"GDAL failed reading block for input {key!r}.")
                    arrays[key] = _normalize_nodata(data, band.GetNoDataValue())
                valid = np.ones((read_height, read_width), dtype=bool)
                for array in arrays.values():
                    valid &= np.isfinite(array)
                for array in arrays.values():
                    array[~valid] = np.nan
                yield AlignedBlock(
                    (x_offset, y_offset, width, height),
                    arrays,
                    (read_x, read_y, read_width, read_height),
                )
    finally:
        bands.clear()
        datasets.clear()


def _create_float_output(path: str | Path, grid: RasterGrid):
    gdal, _ = _gdal_modules()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(output_path),
        grid.width,
        grid.height,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create output raster: {output_path}")
    dataset.SetGeoTransform(grid.geotransform)
    dataset.SetProjection(grid.projection_wkt)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(float("nan"))
    return dataset, band


def create_raster_output(
    path: str | Path,
    grid: RasterGrid,
    *,
    data_type: int,
    nodata: float,
):
    """Create a tiled GeoTIFF band on ``grid`` with explicit NoData metadata."""
    gdal, _ = _gdal_modules()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(output_path),
        grid.width,
        grid.height,
        1,
        data_type,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create output raster: {output_path}")
    dataset.SetGeoTransform(grid.geotransform)
    dataset.SetProjection(grid.projection_wkt)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    return dataset, band


def write_raster_block(
    band,
    values: NDArray,
    x_offset: int,
    y_offset: int,
) -> None:
    """Write one 2D array to a GDAL raster band at the given pixel offset."""
    result = np.asarray(values)
    if result.ndim != 2:
        raise ValueError("Raster output blocks must be two-dimensional.")
    gdal, _ = _gdal_modules()
    if band.WriteArray(result, x_offset, y_offset) != gdal.CE_None:
        raise RuntimeError(f"GDAL failed writing raster block at ({x_offset}, {y_offset}).")


def _integer_window_offset(info: RasterInfo, grid: RasterGrid) -> tuple[int, int] | None:
    """Return source pixel offsets when the target is an exact same-lattice subset."""
    gt = info.geotransform
    target_gt = grid.geotransform
    if not (
        math.isclose(gt[1], target_gt[1], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(gt[5], target_gt[5], rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(gt[2], 0.0, abs_tol=1e-12)
        and math.isclose(gt[4], 0.0, abs_tol=1e-12)
    ):
        return None
    _, osr = _gdal_modules()
    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(info.projection_wkt)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(grid.projection_wkt)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not bool(source_srs.IsSame(target_srs)):
        return None
    source_x = (target_gt[0] - gt[0]) / gt[1]
    source_y = (target_gt[3] - gt[3]) / gt[5]
    if not (
        math.isclose(source_x, round(source_x), rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(source_y, round(source_y), rel_tol=0.0, abs_tol=1e-8)
    ):
        return None
    x_offset, y_offset = int(round(source_x)), int(round(source_y))
    if (
        x_offset < 0
        or y_offset < 0
        or x_offset + grid.width > info.width
        or y_offset + grid.height > info.height
    ):
        return None
    return x_offset, y_offset


def _copy_integer_aligned(
    source_path: str | Path,
    output_path: str | Path,
    grid: RasterGrid,
    *,
    block_size: int,
) -> Path | None:
    """Copy exact source windows when input and target grids differ by whole pixels."""
    info = inspect_raster(source_path)
    offset = _integer_window_offset(info, grid)
    if offset is None:
        return None
    gdal, _ = _gdal_modules()
    destination, output_band = _create_float_output(output_path, grid)
    source = _open_raster(source_path)
    source_band = source.GetRasterBand(1)
    try:
        for y_offset in range(0, grid.height, block_size):
            height = min(block_size, grid.height - y_offset)
            for x_offset in range(0, grid.width, block_size):
                width = min(block_size, grid.width - x_offset)
                data = source_band.ReadAsArray(
                    offset[0] + x_offset,
                    offset[1] + y_offset,
                    width,
                    height,
                )
                if data is None:
                    raise RuntimeError(f"GDAL failed reading aligned window from {source_path}.")
                block = _normalize_nodata(data, info.nodata)
                if output_band.WriteArray(block, x_offset, y_offset) != gdal.CE_None:
                    raise RuntimeError(
                        f"GDAL failed writing aligned block at ({x_offset}, {y_offset})."
                    )
        output_band.FlushCache()
        destination.FlushCache()
    finally:
        source_band = None
        source = None
        output_band = None
        destination = None
    return Path(output_path)


def _write_output_block(
    band,
    values: NDArray[np.float32],
    x_offset: int,
    y_offset: int,
) -> None:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("Raster output blocks must be two-dimensional.")
    gdal, _ = _gdal_modules()
    if band.WriteArray(result, x_offset, y_offset) != gdal.CE_None:
        raise RuntimeError(f"GDAL failed writing raster block at ({x_offset}, {y_offset}).")


def _warp_average_block(
    source_dataset,
    source_nodata: float | None,
    grid: RasterGrid,
    x_offset: int,
    y_offset: int,
    width: int,
    height: int,
    *,
    resample_alg: int,
):
    gdal, _ = _gdal_modules()
    destination = gdal.Warp(
        "",
        source_dataset,
        format="MEM",
        outputBounds=_window_bounds(grid, x_offset, y_offset, width, height),
        width=width,
        height=height,
        dstSRS=grid.projection_wkt,
        resampleAlg=resample_alg,
        srcNodata=float("nan") if source_nodata is None else source_nodata,
        dstNodata=float("nan"),
        outputType=gdal.GDT_Float32,
        errorThreshold=0.0,
        multithread=False,
        warpOptions=["INIT_DEST=NO_DATA"],
    )
    if destination is None:
        raise RuntimeError("GDAL Warp failed while aligning a raster block.")
    array = destination.GetRasterBand(1).ReadAsArray()
    destination = None
    if array is None:
        raise RuntimeError("GDAL failed reading a warped raster block.")
    return np.asarray(array, dtype=np.float32)


def _convert_db_to_power(source_path: str | Path, destination_path: Path) -> Path:
    """Create a tiled linear-power temporary raster using bounded source reads."""
    gdal, _ = _gdal_modules()
    info = inspect_raster(source_path)
    output = gdal.GetDriverByName("GTiff").Create(
        str(destination_path),
        info.width,
        info.height,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if output is None:
        raise RuntimeError(f"Could not create temporary SAR power raster: {destination_path}")
    output.SetGeoTransform(info.geotransform)
    output.SetProjection(info.projection_wkt)
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(float("nan"))

    for block in iter_raster_blocks(source_path, block_size=512):
        x_offset, y_offset, _, _ = block.window
        valid = np.isfinite(block.data)
        with np.errstate(over="ignore", invalid="ignore"):
            power64 = np.power(10.0, block.data.astype(np.float64) / 10.0)
        if np.any(valid & (~np.isfinite(power64) | (power64 > np.finfo(np.float32).max))):
            output = None
            raise ValueError("SAR dB value exceeds float32 linear-power range.")
        power = np.full(block.data.shape, np.nan, dtype=np.float32)
        power[valid] = power64[valid].astype(np.float32)
        _write_output_block(output_band, power, x_offset, y_offset)

    output.FlushCache()
    output = None
    return destination_path


def _resample_to_grid(
    source_path: str | Path,
    output_path: str | Path,
    grid: RasterGrid,
    *,
    block_size: int,
    decibels: bool,
    resample_alg: int,
) -> Path:
    if not decibels:
        copied = _copy_integer_aligned(
            source_path,
            output_path,
            grid,
            block_size=block_size,
        )
        if copied is not None:
            return copied
    info = inspect_raster(source_path)
    dataset, band = _create_float_output(output_path, grid)
    source = _open_raster(source_path)
    source_nodata = info.nodata
    try:
        for y_offset in range(0, grid.height, block_size):
            height = min(block_size, grid.height - y_offset)
            for x_offset in range(0, grid.width, block_size):
                width = min(block_size, grid.width - x_offset)
                aligned = _warp_average_block(
                    source,
                    source_nodata,
                    grid,
                    x_offset,
                    y_offset,
                    width,
                    height,
                    resample_alg=resample_alg,
                )
                if decibels:
                    valid = np.isfinite(aligned) & (aligned > 0.0)
                    converted = np.full(aligned.shape, np.nan, dtype=np.float32)
                    converted[valid] = (
                        10.0 * np.log10(aligned[valid].astype(np.float64))
                    ).astype(np.float32)
                    aligned = converted
                _write_output_block(band, aligned, x_offset, y_offset)
        band.FlushCache()
        dataset.FlushCache()
    finally:
        source = None
        band = None
        dataset = None
    return Path(output_path)


def resample_sar_db_to_grid(
    source_path: str | Path,
    output_path: str | Path,
    grid: RasterGrid,
    *,
    block_size: int = 512,
) -> Path:
    """Resample SAR dB by averaging linear power, then converting back to dB.

    The source is converted to linear power in bounded blocks. GDAL average
    resampling is applied to power values, and each output tile is converted to
    dB before writing. Nearest-neighbor resampling is never used.
    """
    if block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    gdal, _ = _gdal_modules()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copied = _copy_integer_aligned(
        source_path,
        output_path,
        grid,
        block_size=block_size,
    )
    if copied is not None:
        return copied
    with tempfile.TemporaryDirectory(prefix="sar-power-", dir=output_path.parent) as tmp:
        power_path = Path(tmp) / "sar_power.tif"
        _convert_db_to_power(source_path, power_path)
        return _resample_to_grid(
            power_path,
            output_path,
            grid,
            block_size=block_size,
            decibels=True,
            resample_alg=gdal.GRA_Average,
        )


def align_rasters_to_grid(
    input_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    grid: RasterGrid,
    *,
    sar_db_keys: tuple[str, ...] = (),
    categorical_keys: tuple[str, ...] = (),
    block_size: int = 512,
) -> AlignmentResult:
    """Align rasters to a caller-supplied grid with type-appropriate resampling."""
    if not input_paths:
        raise ValueError("input_paths must contain at least one raster.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    if set(sar_db_keys) - set(input_paths):
        raise ValueError("Every SAR key must be present in input_paths.")
    if set(categorical_keys) - set(input_paths):
        raise ValueError("Every categorical key must be present in input_paths.")
    if set(sar_db_keys) & set(categorical_keys):
        raise ValueError("SAR and categorical raster keys must be disjoint.")
    gdal, _ = _gdal_modules()
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for key, source_path in input_paths.items():
        output_path = destination_dir / f"{key}.tif"
        if key in sar_db_keys:
            outputs[key] = resample_sar_db_to_grid(
                source_path,
                output_path,
                grid,
                block_size=block_size,
            )
        else:
            outputs[key] = _resample_to_grid(
                source_path,
                output_path,
                grid,
                block_size=block_size,
                decibels=False,
                resample_alg=(
                    gdal.GRA_NearestNeighbour
                    if key in categorical_keys
                    else gdal.GRA_Average
                ),
            )
    return AlignmentResult(grid, outputs)


def align_rasters(
    input_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    reference_key: str,
    sar_db_keys: tuple[str, ...] = (),
    categorical_keys: tuple[str, ...] = (),
    block_size: int = 512,
) -> AlignmentResult:
    """Align all single-band rasters to the S2 reference-grid intersection.

    Every output is Float32 with NaN NoData. Continuous inputs use average
    resampling; keys listed in ``sar_db_keys`` use linear-power averaging
    before conversion back to dB.
    """
    grid = compute_intersection_grid(input_paths, reference_key=reference_key)
    return align_rasters_to_grid(
        input_paths,
        output_dir,
        grid,
        sar_db_keys=sar_db_keys,
        categorical_keys=categorical_keys,
        block_size=block_size,
    )
