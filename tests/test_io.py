"""Generated GeoTIFF integration tests for raster alignment and block IO."""

import math

import numpy as np
import pytest

gdal = pytest.importorskip("osgeo.gdal")
osr = pytest.importorskip("osgeo.osr")
gdal.UseExceptions()

from possibility_fusion.io.raster import (  # noqa: E402
    align_rasters,
    align_rasters_to_grid,
    compute_intersection_grid,
    compute_grid_for_extent,
    iter_aligned_blocks,
    iter_raster_blocks,
    read_raster_window,
    resample_sar_db_to_grid,
)


def _spatial_reference(epsg: int):
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(epsg)
    reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return reference


def _create_raster(
    path,
    data: np.ndarray,
    *,
    geotransform: tuple[float, float, float, float, float, float],
    epsg: int = 32631,
    nodata: float | None = np.nan,
):
    height, width = data.shape
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        width,
        height,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(_spatial_reference(epsg).ExportToWkt())
    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(data.astype(np.float32))
    dataset.FlushCache()
    dataset = None


def test_intersection_uses_s2_reference_grid_across_crs_and_resolutions(tmp_path):
    reference_path = tmp_path / "s2_reference.tif"
    other_path = tmp_path / "other_crs.tif"
    _create_raster(
        reference_path,
        np.ones((10, 10), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    target_srs = _spatial_reference(32631)
    other_srs = _spatial_reference(4326)
    transform = osr.CoordinateTransformation(target_srs, other_srs)
    corners = [
        transform.TransformPoint(500040.0, 840.0),
        transform.TransformPoint(500160.0, 960.0),
    ]
    xmin = min(point[0] for point in corners)
    xmax = max(point[0] for point in corners)
    ymin = min(point[1] for point in corners)
    ymax = max(point[1] for point in corners)
    pixel_size = 0.00002
    width = math.ceil((xmax - xmin) / pixel_size)
    height = math.ceil((ymax - ymin) / pixel_size)
    _create_raster(
        other_path,
        np.ones((height, width), dtype=np.float32),
        geotransform=(xmin, pixel_size, 0.0, ymax, 0.0, -pixel_size),
        epsg=4326,
    )

    grid = compute_intersection_grid(
        {"s2": reference_path, "other": other_path},
        reference_key="s2",
    )

    assert grid.width < 10
    assert grid.height < 10
    assert grid.geotransform[1] == pytest.approx(20.0)
    assert grid.geotransform[5] == pytest.approx(-20.0)
    assert grid.projection_wkt


def test_sar_downsampling_averages_linear_power_not_db(tmp_path):
    reference_path = tmp_path / "s2.tif"
    sar_path = tmp_path / "sar_db.tif"
    output_path = tmp_path / "sar_aligned.tif"
    _create_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )
    sar = np.array(
        [
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    _create_raster(
        sar_path,
        sar,
        geotransform=(500000.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
    )

    grid = compute_intersection_grid(
        {"s2": reference_path, "sar": sar_path},
        reference_key="s2",
    )
    resample_sar_db_to_grid(sar_path, output_path, grid, block_size=1)

    aligned = gdal.Open(str(output_path)).ReadAsArray()
    expected = 10.0 * np.log10(np.mean(np.power(10.0, sar[:2, :2] / 10.0)))
    assert aligned[0, 0] == pytest.approx(expected, abs=1e-4)
    assert aligned[0, 0] != pytest.approx(float(np.mean(sar[:2, :2])))


def test_align_rasters_writes_common_grid_and_resamples_sar(tmp_path):
    reference_path = tmp_path / "s2.tif"
    sar_path = tmp_path / "sar.tif"
    output_dir = tmp_path / "aligned"
    transform = (500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0)
    _create_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        geotransform=transform,
    )
    _create_raster(
        sar_path,
        np.full((4, 4), 10.0, dtype=np.float32),
        geotransform=(500000.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
    )

    result = align_rasters(
        {"s2": reference_path, "sar": sar_path},
        output_dir,
        reference_key="s2",
        sar_db_keys=("sar",),
        block_size=1,
    )

    assert result.grid.width == result.grid.height == 2
    assert set(result.outputs) == {"s2", "sar"}
    s2 = gdal.Open(str(result.outputs["s2"]))
    sar = gdal.Open(str(result.outputs["sar"]))
    assert s2.GetGeoTransform() == pytest.approx(result.grid.geotransform)
    assert sar.GetGeoTransform() == pytest.approx(result.grid.geotransform)
    np.testing.assert_allclose(sar.ReadAsArray(), 10.0)


def test_align_rasters_uses_exact_window_for_integer_pixel_shift(tmp_path):
    reference_path = tmp_path / "s2.tif"
    shifted_path = tmp_path / "shifted.tif"
    reference = np.arange(12, dtype=np.float32).reshape(3, 4)
    shifted = np.array(
        [
            [100.0, 201.0, 202.0, 203.0],
            [100.0, 211.0, 212.0, 213.0],
            [100.0, 221.0, 222.0, 223.0],
        ],
        dtype=np.float32,
    )
    _create_raster(
        reference_path,
        reference,
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )
    _create_raster(
        shifted_path,
        shifted,
        geotransform=(499980.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    result = align_rasters(
        {"s2": reference_path, "shifted": shifted_path},
        tmp_path / "aligned",
        reference_key="s2",
        sar_db_keys=("shifted",),
        block_size=2,
    )

    aligned = gdal.Open(str(result.outputs["shifted"])).ReadAsArray()
    expected = shifted[:, 1:]
    np.testing.assert_array_equal(aligned, expected)
    np.testing.assert_array_equal(aligned.view(np.uint32), expected.view(np.uint32))


def test_align_rasters_averages_subpixel_shift(tmp_path):
    reference_path = tmp_path / "s2.tif"
    shifted_path = tmp_path / "shifted.tif"
    _create_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )
    source = np.tile(np.array([0.0, 10.0, 20.0], dtype=np.float32), (2, 1))
    _create_raster(
        shifted_path,
        source,
        geotransform=(499990.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    result = align_rasters(
        {"s2": reference_path, "shifted": shifted_path},
        tmp_path / "aligned",
        reference_key="s2",
    )

    aligned = gdal.Open(str(result.outputs["shifted"])).ReadAsArray()
    assert aligned.shape == (2, 2)
    np.testing.assert_allclose(aligned, [[5.0, 15.0], [5.0, 15.0]])


def test_aligned_block_reader_propagates_any_layer_nodata_to_all_layers(tmp_path):
    first_path = tmp_path / "first.tif"
    second_path = tmp_path / "second.tif"
    transform = (500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0)
    _create_raster(first_path, np.ones((2, 2), dtype=np.float32), geotransform=transform)
    second = np.full((2, 2), 2.0, dtype=np.float32)
    second[0, 1] = np.nan
    _create_raster(second_path, second, geotransform=transform)

    blocks = list(
        iter_aligned_blocks(
            {"first": first_path, "second": second_path},
            block_size=1,
        )
    )

    nodata_block = next(block for block in blocks if block.x_offset == 1 and block.y_offset == 0)
    assert np.isnan(nodata_block.arrays["first"][0, 0])
    assert np.isnan(nodata_block.arrays["second"][0, 0])


def test_raster_blocks_are_bounded_and_cover_each_pixel_once(tmp_path):
    path = tmp_path / "values.tif"
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    _create_raster(
        path,
        values,
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    blocks = list(iter_raster_blocks(path, block_size=3))

    reconstructed = np.full(values.shape, np.nan, dtype=np.float32)
    for block in blocks:
        x_offset, y_offset, x_size, y_size = block.window
        assert x_size <= 3 and y_size <= 3
        reconstructed[y_offset:y_offset + y_size, x_offset:x_offset + x_size] = block.data
    np.testing.assert_array_equal(reconstructed, values)


def test_native_window_reader_uses_source_tile_pixel_coordinates(tmp_path):
    path = tmp_path / "s2_tile.tif"
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    _create_raster(
        path,
        values,
        geotransform=(400000.0, 20.0, 0.0, 4300000.0, 0.0, -20.0),
    )

    actual = read_raster_window(path, (1, 4, 3, 2))

    np.testing.assert_array_equal(actual, values[4:6, 1:4])


def test_extent_grid_snaps_to_reference_pixel_lattice(tmp_path):
    reference = tmp_path / "reference.tif"
    _create_raster(
        reference,
        np.ones((5, 6), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    grid = compute_grid_for_extent(
        reference,
        (500019.0, 920.0, 500081.0, 981.0),
        _spatial_reference(32631).ExportToWkt(),
    )

    assert grid.width == 3
    assert grid.height == 3
    assert grid.geotransform == (500020.0, 20.0, 0.0, 980.0, 0.0, -20.0)


def test_aligned_blocks_include_filter_halo_but_keep_core_window(tmp_path):
    first_path = tmp_path / "first_halo.tif"
    second_path = tmp_path / "second_halo.tif"
    transform = (500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0)
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    _create_raster(first_path, values, geotransform=transform)
    _create_raster(second_path, values + 1, geotransform=transform)

    first_block = next(
        iter_aligned_blocks(
            {"first": first_path, "second": second_path},
            block_size=2,
            halo=1,
        )
    )

    assert first_block.window == (0, 0, 2, 2)
    assert first_block.read_window == (0, 0, 3, 3)
    assert first_block.core_slices == (slice(0, 2), slice(0, 2))
    assert first_block.arrays["first"].shape == (3, 3)


def test_raster_reader_normalizes_common_sar_nodata_sentinel(tmp_path):
    path = tmp_path / "sar_sentinel.tif"
    values = np.array(
        [[-10.0, -3.4028235e38], [-9.0, -8.0]],
        dtype=np.float32,
    )
    _create_raster(
        path,
        values,
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
        nodata=None,
    )

    block = next(iter_raster_blocks(path, block_size=2))

    assert block.data[0, 0] == -10.0
    assert np.isnan(block.data[0, 1])


def test_align_rasters_to_explicit_grid_uses_nearest_for_categorical(tmp_path):
    reference = tmp_path / "categorical_reference.tif"
    source = tmp_path / "categorical_source.tif"
    _create_raster(
        reference,
        np.ones((2, 2), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )
    _create_raster(
        source,
        np.array([[4.0, 8.0, 10.0], [5.0, 9.0, 11.0]], dtype=np.float32),
        geotransform=(499990.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )
    grid = compute_grid_for_extent(
        reference,
        (500000.0, 960.0, 500040.0, 1000.0),
        _spatial_reference(32631).ExportToWkt(),
    )

    result = align_rasters_to_grid(
        {"reference": reference, "categorical": source},
        tmp_path / "aligned_categorical",
        grid,
        categorical_keys=("categorical",),
    )

    aligned = gdal.Open(str(result.outputs["categorical"])).ReadAsArray()
    assert set(np.unique(aligned)) <= {4.0, 5.0, 8.0, 9.0, 10.0, 11.0}
