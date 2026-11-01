"""Headless end-to-end tests for the QGIS Processing algorithm."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

qgis_core = pytest.importorskip("qgis.core")
from osgeo import gdal, osr
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProject,
    QgsRectangle,
    QgsReferencedRectangle,
)

from dst_fusion.algorithms.fuse_burned_area import FuseBurnedAreaAlgorithm
from dst_fusion.provider import DstFusionProvider
from dst_fusion.io.raster import inspect_raster, read_raster_window



@pytest.fixture(scope="module")
def qgis_application():
    application = QgsApplication.instance()
    if application is None:
        QgsApplication.setPrefixPath("/usr", True)
        application = QgsApplication([], False)
        application.initQgis()
    yield application


@pytest.fixture(scope="module")
def processing_provider(qgis_application):
    registry = QgsApplication.processingRegistry()
    provider = DstFusionProvider()
    assert registry.addProvider(provider)
    assert registry.algorithmById("dst_fusion:fuse_burned_area") is not None
    yield provider
    registry.removeProvider(provider)


def _create_raster(
    path: Path,
    values: np.ndarray,
    *,
    geotransform: tuple[float, float, float, float, float, float],
    dtype: int = gdal.GDT_Float32,
    nodata: float | None = np.nan,
) -> Path:
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), values.shape[1], values.shape[0], 1, dtype)
    if dataset is None:
        raise RuntimeError(f"Could not create test raster {path}.")
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(32635)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    dataset.SetGeoTransform(geotransform)
    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(values)
    dataset = None
    return path


def _extent(xmin: float, ymin: float, xmax: float, ymax: float):
    crs = QgsCoordinateReferenceSystem("EPSG:32635")
    return QgsReferencedRectangle(QgsRectangle(xmin, ymin, xmax, ymax), crs)


def _run_algorithm(parameters: dict[str, object]) -> dict[str, object]:
    algorithm = FuseBurnedAreaAlgorithm()
    algorithm.initAlgorithm()
    context = QgsProcessingContext()
    context.setProject(QgsProject.instance())
    feedback = QgsProcessingFeedback()
    results, success = algorithm.run(parameters, context, feedback)
    assert success, feedback.textLog()
    return results

def _assert_undecided_mask_matches_classes(class_path, mask_path) -> None:
    """The undecided mask must agree with the class raster, pixel for pixel."""
    grid = inspect_raster(Path(class_path))
    window = (0, 0, grid.width, grid.height)
    classes = read_raster_window(str(class_path), window)
    mask = read_raster_window(str(mask_path), window)
    assert int(np.count_nonzero(mask == 1)) == int(np.count_nonzero(classes == 2))
    assert int(np.count_nonzero(mask == 0)) == int(
        np.count_nonzero((classes == 0) | (classes == 1))
    )
    assert int(np.count_nonzero(mask == 255)) == int(np.count_nonzero(classes == -1))
    assert np.array_equal(mask == 1, classes == 2)

def _base_parameters(
    *,
    b8a_pre: Path,
    b12_pre: Path,
    b8a_post: Path,
    b12_post: Path,
    sar_pre: Path,
    sar_post: Path,
    extent,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "B8A_PRE": str(b8a_pre),
        "B12_PRE": str(b12_pre),
        "B8A_POST": str(b8a_post),
        "B12_POST": str(b12_post),
        "S2_INPUT_MODE": 0,
        "BOA_ADD_OFFSET": -1000.0,
        "BOA_QUANT": 10000.0,
        "SCL_PRE": None,
        "SCL_POST": None,
        "LAND_MASK": None,
        "SAR_PRE": str(sar_pre),
        "SAR_POST": str(sar_post),
        "S2_PRE_DATE": "2024-06-01",
        "S2_POST_DATE": "2024-06-08",
        "SAR_PRE_DATE": "2024-06-02",
        "SAR_POST_DATE": "2024-06-09",
        "FIRE_START": None,
        "FIRE_END": None,
        "TAU_DAYS": 14.0,
        "BURN_DIRECTION": 2,
        "SAR_FILTER_SIZE": 5,
        "CENTER_SAR": True,
        "SAR_RELIABILITY": 1.0,
        "LS_MAP": [],
        "DNBR_LOW": 0.10,
        "DNBR_HIGH": 0.27,
        "SAR_MAD_K1": 1.0,
        "SAR_MAD_K2": 2.0,
        "FUSION_RULE": 0,
        "TEMPORAL_MODE": 2,
        "AOI_EXTENT": extent,
        "NECESSITY_THRESHOLD": 0.0,
        "OUTPUT_CLASS": str(output_dir / "class.tif"),
        "OUTPUT_NECESSITY": str(output_dir / "necessity.tif"),
        "OUTPUT_CONFLICT": str(output_dir / "conflict.tif"),
        "OUTPUT_PI_BURNED": None,
        "OUTPUT_PI_UNBURNED": None,
        "OUTPUT_UNDECIDED_MASK": str(output_dir / "undecided.tif"),
    }


def test_synthetic_cloud_scene_recovers_burned_pixels(processing_provider, tmp_path):
    rows, columns = np.mgrid[:100, :100]
    burned = (rows >= 25) & (rows < 75) & (columns >= 25) & (columns < 75)
    cloud = burned & (rows >= 35) & (rows < 65) & (columns >= 35) & (columns < 65)
    rng = np.random.default_rng(42)
    sar_pre_values = (-10.0 + rng.normal(0.0, 0.1, burned.shape)).astype(np.float32)
    sar_post_values = (
        sar_pre_values
        + rng.normal(0.0, 0.1, burned.shape)
        + burned.astype(np.float32) * 3.0
    ).astype(np.float32)
    scl_pre = np.full(burned.shape, 4, dtype=np.uint8)
    scl_post = np.full(burned.shape, 4, dtype=np.uint8)
    scl_pre[cloud] = 8
    scl_post[cloud] = 8
    b8a_pre = np.full(burned.shape, 0.6, dtype=np.float32)
    b8a_pre[0, 0] = np.nan
    geotransform = (500000.0, 20.0, 0.0, 100000.0, 0.0, -20.0)

    inputs = {
        "b8a_pre": _create_raster(
            tmp_path / "b8a_pre.tif",
            b8a_pre,
            geotransform=geotransform,
        ),
        "b12_pre": _create_raster(
            tmp_path / "b12_pre.tif",
            np.full(burned.shape, 0.2, dtype=np.float32),
            geotransform=geotransform,
        ),
        "b8a_post": _create_raster(
            tmp_path / "b8a_post.tif",
            np.where(burned, 0.2, 0.6).astype(np.float32),
            geotransform=geotransform,
        ),
        "b12_post": _create_raster(
            tmp_path / "b12_post.tif",
            np.where(burned, 0.4, 0.2).astype(np.float32),
            geotransform=geotransform,
        ),
        "sar_pre": _create_raster(
            tmp_path / "sar_pre.tif",
            sar_pre_values,
            geotransform=geotransform,
        ),
        "sar_post": _create_raster(
            tmp_path / "sar_post.tif",
            sar_post_values,
            geotransform=geotransform,
        ),
    }
    inputs["scl_pre"] = _create_raster(
        tmp_path / "scl_pre.tif",
        scl_pre,
        geotransform=geotransform,
        dtype=gdal.GDT_Byte,
        nodata=None,
    )
    inputs["scl_post"] = _create_raster(
        tmp_path / "scl_post.tif",
        scl_post,
        geotransform=geotransform,
        dtype=gdal.GDT_Byte,
        nodata=None,
    )
    parameters = _base_parameters(
        b8a_pre=inputs["b8a_pre"],
        b12_pre=inputs["b12_pre"],
        b8a_post=inputs["b8a_post"],
        b12_post=inputs["b12_post"],
        sar_pre=inputs["sar_pre"],
        sar_post=inputs["sar_post"],
        extent=_extent(500000.0, 98000.0, 502000.0, 100000.0),
        output_dir=tmp_path,
    )
    parameters["SCL_PRE"] = str(inputs["scl_pre"])
    parameters["SCL_POST"] = str(inputs["scl_post"])

    _run_algorithm(parameters)

    classes = read_raster_window(parameters["OUTPUT_CLASS"], (0, 0, 100, 100))
    assert int(np.count_nonzero(classes[cloud] == 0)) >= int(cloud.sum() * 0.95)
    assert classes[0, 0] == -1
    undecided_mask = read_raster_window(
        parameters["OUTPUT_UNDECIDED_MASK"],
        (0, 0, 100, 100),
    )
    assert undecided_mask[0, 0] == 255

    _assert_undecided_mask_matches_classes(
        parameters["OUTPUT_CLASS"],
        parameters["OUTPUT_UNDECIDED_MASK"],
    )
    for path in (
        Path(parameters["OUTPUT_CLASS"]),
        Path(parameters["OUTPUT_NECESSITY"]),
        Path(parameters["OUTPUT_CONFLICT"]),
        Path(parameters["OUTPUT_UNDECIDED_MASK"]),
    ):
        assert path.is_file()
        assert Path(f"{path}.qml").is_file()
    dataset = gdal.Open(str(parameters["OUTPUT_CLASS"]))
    metadata = dataset.GetMetadata()
    dataset = None
    assert metadata["BURN_DIRECTION"] == "positive"
    assert metadata["TEMPORAL_MODE"] == "never"


def test_chios_frozen_primary_matches_scoring_mask_counts(processing_provider, tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    chios_data = repository_root / "data" / "chios2024"
    if not chios_data.is_dir():
        pytest.skip("Chios source data is not installed.")

    from examples.audit_chios import REFERENCE_PATH, _find_inputs
    from examples.score_checkpoint_c import _make_reference_masks
    if not REFERENCE_PATH.is_file():
        pytest.skip("Chios reference data is not installed.")

    try:
        s2_inputs, s1_inputs, _ = _find_inputs()
    except FileNotFoundError as error:
        pytest.skip(f"Chios source data is incomplete: {error}")
    required = (
        "pre_B8A",
        "pre_B12",
        "post_B8A",
        "post_B12",
        "pre_SCL",
        "post_SCL",
    )
    missing = [key for key in required if key not in s2_inputs]
    missing.extend(
        key
        for key in ("pre_VV", "post_VV", "pre_ls_map", "post_ls_map")
        if key not in s1_inputs
    )
    if missing:
        pytest.skip(f"Chios source inputs are incomplete: {', '.join(missing)}")

    parameters = _base_parameters(
        b8a_pre=s2_inputs["pre_B8A"],
        b12_pre=s2_inputs["pre_B12"],
        b8a_post=s2_inputs["post_B8A"],
        b12_post=s2_inputs["post_B12"],
        sar_pre=s1_inputs["pre_VV"],
        sar_post=s1_inputs["post_VV"],
        extent=_extent(403780.0, 4245000.0, 417820.0, 4260140.0),
        output_dir=tmp_path,
    )
    parameters.update(
        {
            "S2_INPUT_MODE": 1,
            "SCL_PRE": str(s2_inputs["pre_SCL"]),
            "SCL_POST": str(s2_inputs["post_SCL"]),
            "LS_MAP": [str(s1_inputs["pre_ls_map"]), str(s1_inputs["post_ls_map"])],
            "S2_PRE_DATE": "2024-06-28",
            "S2_POST_DATE": "2024-07-08",
            "SAR_PRE_DATE": "2024-06-29",
            "SAR_POST_DATE": "2024-07-11",
            "FIRE_START": "2024-07-01",
            "FIRE_END": "2024-07-03",
            "TAU_DAYS": 14.0,
            "BURN_DIRECTION": 2,
            "SAR_FILTER_SIZE": 5,
            "CENTER_SAR": True,
            "SAR_RELIABILITY": 1.0,
            "DNBR_LOW": 0.10,
            "DNBR_HIGH": 0.27,
            "SAR_MAD_K1": 1.0,
            "SAR_MAD_K2": 2.0,
            "FUSION_RULE": 0,
            "TEMPORAL_MODE": 1,
        }
    )

    _run_algorithm(parameters)

    _assert_undecided_mask_matches_classes(
        parameters["OUTPUT_CLASS"],
        parameters["OUTPUT_UNDECIDED_MASK"],
    )

    class_path = Path(parameters["OUTPUT_CLASS"])
    grid = inspect_raster(class_path)
    classes = read_raster_window(
        class_path,
        (0, 0, grid.width, grid.height),
    )
    _, _, aoi, _ = _make_reference_masks(
        grid.projection_wkt,
        grid.geotransform,
        grid.width,
        grid.height,
    )
    scoring_mask = aoi & (classes >= 0)
    necessity = read_raster_window(
        parameters["OUTPUT_NECESSITY"],
        (0, 0, grid.width, grid.height),
    )
    conflict = read_raster_window(
        parameters["OUTPUT_CONFLICT"],
        (0, 0, grid.width, grid.height),
    )
    scoring_mask &= np.isfinite(necessity) & np.isfinite(conflict)
    assert int(scoring_mask.sum()) == 111191
    counts = {
        code: int(np.count_nonzero(scoring_mask & (classes == code)))
        for code in (0, 1, 2)
    }
    assert counts == {0: 13961, 1: 85382, 2: 11848}

    full_window_counts = {
        code: int(np.count_nonzero(classes == code))
        for code in (0, 1, 2)
    }
    print(
        "Full Chios AOI-window class totals (separate from scoring mask): "
        f"{full_window_counts}"
    )
def test_algorithm_metadata_methods_work(processing_provider):
    algorithm = FuseBurnedAreaAlgorithm()
    assert algorithm.name()
    assert algorithm.displayName()
    assert algorithm.group()
    assert algorithm.groupId()
    assert isinstance(algorithm.shortHelpString(), str)
    assert isinstance(algorithm.createInstance(), FuseBurnedAreaAlgorithm)


def test_acquisition_date_detection_and_manual_override(processing_provider):
    algorithm = FuseBurnedAreaAlgorithm()
    algorithm.initAlgorithm()
    context = QgsProcessingContext()
    feedback = QgsProcessingFeedback()
    source_path = "/data/T35SMC_20240628T090601_B8A_20m.jp2"

    detected = algorithm._resolve_acquisition_date(
        {"S2_PRE_DATE": None},
        "S2_PRE_DATE",
        context,
        feedback,
        "B8A_PRE",
        source_path,
    )
    overridden = algorithm._resolve_acquisition_date(
        {"S2_PRE_DATE": "2024-07-01"},
        "S2_PRE_DATE",
        context,
        feedback,
        "B8A_PRE",
        source_path,
    )

    assert detected == date(2024, 6, 28)
    assert "Detected S2_PRE_DATE=2024-06-28 from layer B8A_PRE." in feedback.textLog()
    assert overridden == date(2024, 7, 1)


def test_acquisition_date_failure_names_layer_and_parameter(processing_provider):
    algorithm = FuseBurnedAreaAlgorithm()
    algorithm.initAlgorithm()

    with pytest.raises(QgsProcessingException, match="B8A_PRE.*Enter S2_PRE_DATE manually"):
        algorithm._resolve_acquisition_date(
            {"S2_PRE_DATE": None},
            "S2_PRE_DATE",
            QgsProcessingContext(),
            QgsProcessingFeedback(),
            "B8A_PRE",
            "/data/image_without_acquisition_date.tif",
        )


def test_sar_polarization_mismatch_is_rejected(processing_provider):
    with pytest.raises(QgsProcessingException, match="different polarizations"):
        FuseBurnedAreaAlgorithm._validate_sar_polarizations(
            {
                "sar_pre": "/data/S1_pre_VV.tif",
                "sar_post": "/data/S1_post_VH.tif",
            }
        )