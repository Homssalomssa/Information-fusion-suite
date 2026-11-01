"""QGIS Processing wrapper for Sentinel burned-area possibility fusion."""

from __future__ import annotations

import math
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsColorRampShader,
    QgsPalettedRasterRenderer,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDateTime,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsRasterLayer,
    QgsSingleBandPseudoColorRenderer,
    QgsRasterShader,
)
from qgis.PyQt.QtCore import QDateTime, QCoreApplication
from qgis.PyQt.QtGui import QColor

from ..core.acquisitions import detect_acquisition_date, detect_sar_polarization
from ..core.indices import dnbr
from ..core.memberships import dnbr_possibility, sar_possibility
from ..core.possibility import (
    adaptive_fusion,
    conjunctive_fusion,
    decide,
    disjunctive_fusion,
    discount,
)
from ..core.preprocessing import (
    boxcar_filter,
    db_to_linear_power,
    linear_power_to_db,
    robust_standard_deviation,
    scl_optical_reliability,
)
from ..core.temporal import (
    check_temporal_plausibility,
    combine_reliability,
    temporal_reliability,
)
from ..io.raster import (
    RasterGrid,
    align_rasters_to_grid,
    compute_grid_for_extent,
    create_raster_output,
    inspect_raster,
    iter_aligned_blocks,
    iter_raster_blocks,
    write_raster_block,
)
from ..io.sentinel import infer_s2_input_mode, resolve_boa_parameters

EXACT_STATS_PIXEL_LIMIT = 2_000_000
HISTOGRAM_BINS = 65_536
BLOCK_SIZE = 256
SAR_FLAG_MASK = 4 | 16
SAR_NODATA_MIN = -60.0
SAR_NODATA_MAX = 20.0


class FuseBurnedAreaAlgorithm(QgsProcessingAlgorithm):
    """Fuse Sentinel-2 dNBR and Sentinel-1 change into burned-area outputs."""

    B8A_PRE = "B8A_PRE"
    B12_PRE = "B12_PRE"
    B8A_POST = "B8A_POST"
    B12_POST = "B12_POST"
    S2_INPUT_MODE = "S2_INPUT_MODE"
    BOA_ADD_OFFSET = "BOA_ADD_OFFSET"
    BOA_QUANT = "BOA_QUANT"
    SCL_PRE = "SCL_PRE"
    SCL_POST = "SCL_POST"
    LAND_MASK = "LAND_MASK"
    SAR_PRE = "SAR_PRE"
    SAR_POST = "SAR_POST"
    LS_MAP = "LS_MAP"
    S2_PRE_DATE = "S2_PRE_DATE"
    S2_POST_DATE = "S2_POST_DATE"
    SAR_PRE_DATE = "SAR_PRE_DATE"
    SAR_POST_DATE = "SAR_POST_DATE"
    FIRE_START = "FIRE_START"
    FIRE_END = "FIRE_END"
    TAU_DAYS = "TAU_DAYS"
    BURN_DIRECTION = "BURN_DIRECTION"
    SAR_FILTER_SIZE = "SAR_FILTER_SIZE"
    CENTER_SAR = "CENTER_SAR"
    SAR_RELIABILITY = "SAR_RELIABILITY"
    DNBR_LOW = "DNBR_LOW"
    DNBR_HIGH = "DNBR_HIGH"
    SAR_MAD_K1 = "SAR_MAD_K1"
    SAR_MAD_K2 = "SAR_MAD_K2"
    FUSION_RULE = "FUSION_RULE"
    TEMPORAL_MODE = "TEMPORAL_MODE"
    AOI_EXTENT = "AOI_EXTENT"
    NECESSITY_THRESHOLD = "NECESSITY_THRESHOLD"
    OUTPUT_CLASS = "OUTPUT_CLASS"
    OUTPUT_NECESSITY = "OUTPUT_NECESSITY"
    OUTPUT_CONFLICT = "OUTPUT_CONFLICT"
    OUTPUT_PI_BURNED = "OUTPUT_PI_BURNED"
    OUTPUT_PI_UNBURNED = "OUTPUT_PI_UNBURNED"
    OUTPUT_UNDECIDED_MASK = "OUTPUT_UNDECIDED_MASK"

    def name(self) -> str:
        return "fuse_burned_area"

    def displayName(self) -> str:
        return self.tr("Fuse burned-area evidence")
    
    def tr(self, string: str) -> str:
        return QCoreApplication.translate("Processing", string)

    def group(self) -> str:
        return self.tr("Burned-area mapping")

    def groupId(self) -> str:
        return "burned_area"

    def shortHelpString(self) -> str:
        return self.tr(
            "Combines Sentinel-2 dNBR and same-polarization Sentinel-1 dB change "
            "using possibility fusion. SAR sign is scene-dependent and must be "
            "selected explicitly. The tau, SAR filter, and MAD multiplier defaults "
            "are starting values, not validated beyond one event."
        )

    def createInstance(self):
        return FuseBurnedAreaAlgorithm()

    def initAlgorithm(self, config: Mapping[str, Any] | None = None) -> None:
        for name, description in (
            (self.B8A_PRE, "Sentinel-2 pre-fire B8A, 20 m"),
            (self.B12_PRE, "Sentinel-2 pre-fire B12, 20 m"),
            (self.B8A_POST, "Sentinel-2 post-fire B8A, 20 m"),
            (self.B12_POST, "Sentinel-2 post-fire B12, 20 m"),
        ):
            self.addParameter(QgsProcessingParameterRasterLayer(name, description))

        self.addParameter(
            QgsProcessingParameterEnum(
                self.S2_INPUT_MODE,
                "S2 input units",
                options=["Auto", "Reflectance", "DN with BOA offset"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BOA_ADD_OFFSET,
                "Optional BOA additive offset override (DN mode)",
                type=QgsProcessingParameterNumber.Double,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BOA_QUANT,
                "Optional BOA quantification override (DN mode)",
                type=QgsProcessingParameterNumber.Double,
                minValue=0.000001,
                optional=True,
            )
        )
        for name, description in (
            (self.SCL_PRE, "Optional Sentinel-2 pre-fire SCL"),
            (self.SCL_POST, "Optional Sentinel-2 post-fire SCL"),
        ):
            self.addParameter(
                QgsProcessingParameterRasterLayer(name, description, optional=True)
            )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.LAND_MASK,
                "Optional land mask (finite nonzero cells are land)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.SAR_PRE,
                "Pre-fire Sentinel-1 backscatter in dB",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.SAR_POST,
                "Post-fire Sentinel-1 backscatter in dB; same polarization as pre-fire",
            )
        )
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.LS_MAP,
                "Optional SAR layover/shadow mask(s); one shared mask or pre/post pair",
                layerType=Qgis.ProcessingSourceType.Raster,
                optional=True,
            )
        )
        for name, description in (
            (self.S2_PRE_DATE, "Sentinel-2 pre-fire acquisition date"),
            (self.S2_POST_DATE, "Sentinel-2 post-fire acquisition date"),
            (self.SAR_PRE_DATE, "Sentinel-1 pre-fire acquisition date"),
            (self.SAR_POST_DATE, "Sentinel-1 post-fire acquisition date"),
        ):
            self.addParameter(
                QgsProcessingParameterDateTime(
                    name,
                    description,
                    type=Qgis.ProcessingDateTimeParameterDataType.Date,
                    optional=True,
                )
            )
        for name, description in (
            (self.FIRE_START, "Optional fire start date; supply with fire end"),
            (self.FIRE_END, "Optional fire end date; supply with fire start"),
        ):
            self.addParameter(
                QgsProcessingParameterDateTime(
                    name,
                    description,
                    type=Qgis.ProcessingDateTimeParameterDataType.Date,
                    optional=True,
                )
            )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TAU_DAYS,
                "Uniform temporal discount tau in days (default not validated beyond one event)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=14.0,
                minValue=0.000001,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.BURN_DIRECTION,
                "SAR burn direction (scene-dependent; choose explicitly)",
                options=["-- choose explicitly --", "Negative change", "Positive change"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAR_FILTER_SIZE,
                "SAR boxcar filter size in pixels (0 off; default not validated beyond one event)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CENTER_SAR,
                "Center SAR change by valid AOI median",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAR_RELIABILITY,
                "SAR reliability scalar",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DNBR_LOW,
                "Lower dNBR breakpoint (USGS default 0.10)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.10,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DNBR_HIGH,
                "Upper dNBR breakpoint (USGS default 0.27)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.27,
            )
        )
        for name, default in ((self.SAR_MAD_K1, 1.0), (self.SAR_MAD_K2, 2.0)):
            self.addParameter(
                QgsProcessingParameterNumber(
                    name,
                    f"{name} scaled-MAD multiplier (default not validated beyond one event)",
                    type=QgsProcessingParameterNumber.Double,
                    defaultValue=default,
                    minValue=0.0,
                )
            )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FUSION_RULE,
                "Possibility fusion rule",
                options=["Adaptive", "Conjunctive", "Disjunctive"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.TEMPORAL_MODE,
                "Uniform temporal discount mode",
                options=["Auto", "Always", "Never"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.AOI_EXTENT,
                "AOI processing extent (statistics are extent-dependent)",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NECESSITY_THRESHOLD,
                "Decision necessity threshold",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        for name, description, optional in (
            (self.OUTPUT_CLASS, "Class raster", False),
            (self.OUTPUT_NECESSITY, "Necessity raster", False),
            (self.OUTPUT_CONFLICT, "Conflict raster", False),
            (self.OUTPUT_PI_BURNED, "Optional Burned possibility raster", True),
            (self.OUTPUT_PI_UNBURNED, "Optional Unburned possibility raster", True),
            (self.OUTPUT_UNDECIDED_MASK, "Optional undecided mask raster", True),
        ):
            self.addParameter(
                QgsProcessingParameterRasterDestination(
                    name,
                    description,
                    optional=optional,
                )
            )

    def processAlgorithm(
        self,
        parameters: Mapping[str, Any],
        context,
        feedback,
    ) -> dict[str, Any]:
        paths = self._input_paths(parameters, context)
        input_paths = dict(paths)
        self._validate_sar_polarizations(paths)
        input_layers = {
            name: self.parameterAsRasterLayer(parameters, name, context)
            for name in (
                self.B8A_PRE,
                self.B12_PRE,
                self.B8A_POST,
                self.B12_POST,
                self.SAR_PRE,
                self.SAR_POST,
            )
        }
        reference_layer = input_layers[self.B8A_PRE]
        extent = self.parameterAsExtent(
            parameters,
            self.AOI_EXTENT,
            context,
            reference_layer.crs(),
        )
        grid = compute_grid_for_extent(
            paths["b8a_pre"],
            (extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()),
            reference_layer.crs().toWkt(),
        )

        selected_input_mode = self.parameterAsEnum(parameters, self.S2_INPUT_MODE, context)
        if selected_input_mode not in (0, 1, 2):
            raise QgsProcessingException("Unknown S2_INPUT_MODE selection.")
        offset_override = self._optional_number_parameter(
            parameters,
            self.BOA_ADD_OFFSET,
            context,
        )
        quantification_override = self._optional_number_parameter(
            parameters,
            self.BOA_QUANT,
            context,
        )
        if selected_input_mode == 0:
            try:
                inference = infer_s2_input_mode(
                    {key: input_paths[key] for key in ("b8a_pre", "b12_pre", "b8a_post", "b12_post")}
                )
            except ValueError as error:
                raise QgsProcessingException(str(error)) from error
            input_mode = 1 if inference.mode == "dn" else 0
            feedback.pushInfo(
                f"Automatically selected S2 mode {inference.mode}: {inference.reason}."
            )
        else:
            input_mode = 0 if selected_input_mode == 1 else 1
            selected_name = "reflectance" if input_mode == 0 else "DN with BOA offset"
            feedback.pushInfo(f"S2 mode explicitly selected: {selected_name}.")

        boa_offset = offset_override
        boa_quant = quantification_override
        if input_mode == 1:
            try:
                boa_parameters = resolve_boa_parameters(
                    {key: input_paths[key] for key in ("b8a_pre", "b12_pre", "b8a_post", "b12_post")},
                    offset_override=offset_override,
                    quantification_override=quantification_override,
                )
            except ValueError as error:
                raise QgsProcessingException(str(error)) from error
            for warning in boa_parameters.warnings:
                feedback.pushWarning(warning)
            boa_offset = boa_parameters.offset
            boa_quant = boa_parameters.quantification
            feedback.pushInfo(
                f"S2 DN calibration: BOA_ADD_OFFSET={boa_offset:g} "
                f"(source: {boa_parameters.offset_source}); BOA_QUANTIFICATION_VALUE="
                f"{boa_quant:g} (source: {boa_parameters.quantification_source})."
            )
            for key in ("b8a_pre", "b12_pre", "b8a_post", "b12_post"):
                paths[key] = self._scale_dn_raster(
                    paths[key],
                    key,
                    boa_offset,
                    boa_quant,
                    context,
                )

        burn_direction_index = self.parameterAsEnum(
            parameters,
            self.BURN_DIRECTION,
            context,
        )
        if burn_direction_index == 0:
            raise QgsProcessingException(
                "Choose a SAR burn direction explicitly; the sign is scene-dependent."
            )
        burn_direction = "negative" if burn_direction_index == 1 else "positive"

        filter_size = self.parameterAsInt(parameters, self.SAR_FILTER_SIZE, context)
        if filter_size < 0 or (filter_size > 0 and filter_size % 2 == 0):
            raise QgsProcessingException(
                "SAR_FILTER_SIZE must be 0 or a positive odd integer."
            )
        dnbr_low = self.parameterAsDouble(parameters, self.DNBR_LOW, context)
        dnbr_high = self.parameterAsDouble(parameters, self.DNBR_HIGH, context)
        k1 = self.parameterAsDouble(parameters, self.SAR_MAD_K1, context)
        k2 = self.parameterAsDouble(parameters, self.SAR_MAD_K2, context)
        if not math.isfinite(dnbr_low) or not math.isfinite(dnbr_high) or dnbr_low >= dnbr_high:
            raise QgsProcessingException("DNBR_LOW must be finite and less than DNBR_HIGH.")
        if not math.isfinite(k1) or not math.isfinite(k2) or k1 < 0.0 or k1 >= k2:
            raise QgsProcessingException(
                "SAR_MAD_K1 must be non-negative and less than SAR_MAD_K2."
            )

        scl_supplied = "scl_pre" in paths
        land_mask_supplied = "land_mask" in paths
        if not scl_supplied and not land_mask_supplied:
            feedback.pushWarning("No SCL or LAND_MASK supplied; water is not masked.")
        tau_days = self.parameterAsDouble(parameters, self.TAU_DAYS, context)
        if not math.isfinite(tau_days) or tau_days <= 0.0:
            raise QgsProcessingException("TAU_DAYS must be finite and greater than zero.")
        sar_reliability = self.parameterAsDouble(
            parameters,
            self.SAR_RELIABILITY,
            context,
        )
        necessity_threshold = self.parameterAsDouble(
            parameters,
            self.NECESSITY_THRESHOLD,
            context,
        )
        if not math.isfinite(necessity_threshold) or not 0.0 <= necessity_threshold <= 1.0:
            raise QgsProcessingException("NECESSITY_THRESHOLD must be in [0, 1].")
        if not math.isfinite(sar_reliability) or not 0.0 <= sar_reliability <= 1.0:
            raise QgsProcessingException("SAR_RELIABILITY must be in [0, 1].")

        date_sources = {
            self.S2_PRE_DATE: ("B8A_PRE", "b8a_pre"),
            self.S2_POST_DATE: ("B8A_POST", "b8a_post"),
            self.SAR_PRE_DATE: ("SAR_PRE", "sar_pre"),
            self.SAR_POST_DATE: ("SAR_POST", "sar_post"),
        }
        dates = {
            name: self._resolve_acquisition_date(
                parameters,
                name,
                context,
                feedback,
                layer_name,
                input_paths[path_key],
            )
            for name, (layer_name, path_key) in date_sources.items()
        }
        fire_start = self._optional_date_parameter(parameters, self.FIRE_START, context)
        fire_end = self._optional_date_parameter(parameters, self.FIRE_END, context)
        if (fire_start is None) != (fire_end is None):
            raise QgsProcessingException("FIRE_START and FIRE_END must be supplied together.")
        for message in check_temporal_plausibility(
            dates[self.S2_PRE_DATE],
            dates[self.S2_POST_DATE],
            dates[self.SAR_PRE_DATE],
            dates[self.SAR_POST_DATE],
            fire_start=fire_start,
            fire_end=fire_end,
        ):
            feedback.pushWarning(message)
        temporal_mode = self.parameterAsEnum(parameters, self.TEMPORAL_MODE, context)
        if temporal_mode not in (0, 1, 2):
            raise QgsProcessingException("Unknown TEMPORAL_MODE selection.")
        if temporal_mode == 1:
            temporal_enabled = True
        elif temporal_mode == 2:
            temporal_enabled = False
        else:
            temporal_enabled = fire_end is None or any(
                acquisition < fire_end
                for acquisition in (dates[self.S2_POST_DATE], dates[self.SAR_POST_DATE])
            )
        alpha_time = temporal_reliability(
            dates[self.S2_POST_DATE],
            dates[self.SAR_POST_DATE],
            tau_days=tau_days,
        )
        alpha_time = alpha_time if temporal_enabled else 1.0

        fusion_rule_index = self.parameterAsEnum(parameters, self.FUSION_RULE, context)
        fusion_functions = (adaptive_fusion, conjunctive_fusion, disjunctive_fusion)
        if not 0 <= fusion_rule_index < len(fusion_functions):
            raise QgsProcessingException("Unknown FUSION_RULE selection.")
        fusion_function = fusion_functions[fusion_rule_index]

        block_size = BLOCK_SIZE
        filter_radius = max(0, filter_size // 2)
        source_paths = dict(paths)
        categorical_keys = tuple(
            key
            for key in source_paths
            if key.startswith("scl_") or key == "land_mask" or key.startswith("ls_map_")
        )
        sar_keys = ("sar_pre", "sar_post")
        aligned_directory = Path(tempfile.mkdtemp(prefix="dst-fusion-aligned-"))
        try:
            alignment = align_rasters_to_grid(
                source_paths,
                aligned_directory,
                grid,
                sar_db_keys=sar_keys,
                categorical_keys=categorical_keys,
                block_size=block_size,
            )
            aligned_paths = alignment.outputs
            with tempfile.TemporaryDirectory(prefix="dst-fusion-stats-") as stats_dir:
                change_path = Path(stats_dir) / "sar_change.tif"
                change_dataset, change_band = create_raster_output(
                    change_path,
                    grid,
                    data_type=gdal.GDT_Float32,
                    nodata=float("nan"),
                )
                pre_histogram = np.zeros(16_384, dtype=np.int64)
                post_histogram = np.zeros(16_384, dtype=np.int64)
                sar_range_underflow = [0, 0]
                sar_range_overflow = [0, 0]
                maximum_reflectance = -math.inf
                for block, sar_pre, sar_post in self._evidence_blocks(
                    aligned_paths,
                    block_size=block_size,
                    halo=filter_radius,
                    filter_size=filter_size,
                ):
                    window = block.window
                    x_offset, y_offset, _, _ = window
                    arrays = block.arrays
                    valid = self._valid_land(arrays)
                    if input_mode == 0:
                        for key in ("b8a_pre", "b12_pre", "b8a_post", "b12_post"):
                            band_values = arrays[key]
                            finite_band = band_values[np.isfinite(band_values)]
                            if finite_band.size:
                                maximum_reflectance = max(
                                    maximum_reflectance,
                                    float(np.max(finite_band)),
                                )
                    pre_values = arrays["sar_pre"]
                    post_values = arrays["sar_post"]
                    pre_histogram, sar_range_underflow[0], sar_range_overflow[0] = (
                        self._update_dbrange_histogram(
                            pre_histogram,
                            pre_values[valid],
                            sar_range_underflow[0],
                            sar_range_overflow[0],
                        )
                    )
                    post_histogram, sar_range_underflow[1], sar_range_overflow[1] = (
                        self._update_dbrange_histogram(
                            post_histogram,
                            post_values[valid],
                            sar_range_underflow[1],
                            sar_range_overflow[1],
                        )
                    )
                    valid &= np.isfinite(sar_pre) & np.isfinite(sar_post)
                    change = np.full(valid.shape, np.nan, dtype=np.float32)
                    change[valid] = sar_post[valid] - sar_pre[valid]
                    write_raster_block(change_band, change, x_offset, y_offset)
                change_band.FlushCache()
                change_dataset.FlushCache()
                change_band = None
                change_dataset = None
                if input_mode == 0 and maximum_reflectance > 1.5:
                    feedback.pushWarning(
                        "S2_INPUT_MODE is reflectance, but a finite band value exceeds 1.5; "
                        "the inputs may be unscaled."
                    )
                self._warn_if_sar_not_db(
                    feedback,
                    pre_histogram,
                    post_histogram,
                    sar_range_underflow,
                    sar_range_overflow,
                )

                center_offset, sar_breakpoints, stats_mode = self._sar_statistics(
                    change_path,
                    center= self.parameterAsBool(parameters, self.CENTER_SAR, context),
                    burn_direction=burn_direction,
                    k1=k1,
                    k2=k2,
                    feedback=feedback,
                )
                if feedback.isCanceled():
                    raise QgsProcessingException("Processing was canceled.")

                outputs = self._create_outputs(
                    parameters,
                    context,
                    grid,
                    input_mode=input_mode,
                    boa_offset=boa_offset,
                    boa_quant=boa_quant,
                    burn_direction=burn_direction,
                    filter_size=filter_size,
                    center_offset=center_offset,
                    sar_breakpoints=sar_breakpoints,
                    stats_mode=stats_mode,
                    tau_days=tau_days,
                    temporal_mode=("auto", "always", "never")[temporal_mode],
                    temporal_enabled=temporal_enabled,
                    alpha_time=alpha_time,
                    dnbr_low=dnbr_low,
                    dnbr_high=dnbr_high,
                    k1=k1,
                    k2=k2,
                    sar_reliability=sar_reliability,
                    necessity_threshold=necessity_threshold,
                    fusion_rule=("adaptive", "conjunctive", "disjunctive")[
                        fusion_rule_index
                    ],
                    dates=dates,
                    fire_start=fire_start,
                    fire_end=fire_end,
                    paths=source_paths,
                    input_paths=input_paths,
                    sar_change_center=center_offset,
                )
                try:
                    total_blocks = math.ceil(grid.width / block_size) * math.ceil(
                        grid.height / block_size
                    )
                    for block_index, (block, sar_pre, sar_post) in enumerate(
                        self._evidence_blocks(
                            aligned_paths,
                            block_size=block_size,
                            halo=filter_radius,
                            filter_size=filter_size,
                        ),
                        start=1,
                    ):
                        if feedback.isCanceled():
                            raise QgsProcessingException("Processing was canceled.")
                        classes, necessity, conflict, pi_burned, pi_unburned = (
                            self._classify_block(
                                block.arrays,
                                sar_pre,
                                sar_post,
                                center_offset=center_offset,
                                burn_direction=burn_direction,
                                sar_breakpoints=sar_breakpoints,
                                dnbr_low=dnbr_low,
                                dnbr_high=dnbr_high,
                                alpha_time=alpha_time,
                                sar_reliability=sar_reliability,
                                necessity_threshold=necessity_threshold,
                                fusion_function=fusion_function,
                            )
                        )
                        x_offset, y_offset, _, _ = block.window
                        undecided_mask = np.full(
                            classes.shape,
                            255,
                            dtype=np.uint8,
                        )
                        valid_classes = classes >= 0
                        undecided_mask[valid_classes] = (
                            classes[valid_classes] == 2
                        ).astype(np.uint8)
                        for key, values in (
                            ("class", classes),
                            ("necessity", necessity),
                            ("conflict", conflict),
                            ("pi_burned", pi_burned),
                            ("pi_unburned", pi_unburned),
                            ("undecided", undecided_mask),
                        ):
                            writer = outputs.get(key)
                            if writer is not None:
                                write_raster_block(
                                    writer[1],
                                    values,
                                    x_offset,
                                    y_offset,
                                )
                        feedback.setProgress(20.0 + 80.0 * block_index / total_blocks)
                finally:
                    self._close_outputs(outputs)
                undecided_path = outputs["paths"].get(self.OUTPUT_UNDECIDED_MASK)
                class_path = outputs["paths"].get(self.OUTPUT_CLASS)
                if undecided_path and class_path:
                    self._rebuild_undecided_mask(class_path, undecided_path, feedback)
                for parameter, output_path in outputs["paths"].items():
                    self._write_qml(
                        output_path,
                        style_type=(
                            "class"
                            if parameter == self.OUTPUT_CLASS
                            else "mask"
                            if parameter == self.OUTPUT_UNDECIDED_MASK
                            else "continuous"
                        ),
                    )
                return {
                    parameter: str(path)
                    for parameter, path in outputs["paths"].items()
                }
        finally:
            self._remove_directory(aligned_directory)

    def _input_paths(self, parameters, context) -> dict[str, str]:
        keys = (
            (self.B8A_PRE, "b8a_pre"),
            (self.B12_PRE, "b12_pre"),
            (self.B8A_POST, "b8a_post"),
            (self.B12_POST, "b12_post"),
            (self.SAR_PRE, "sar_pre"),
            (self.SAR_POST, "sar_post"),
        )
        paths: dict[str, str] = {}
        for parameter, key in keys:
            paths[key] = self._raster_path(parameters, parameter, context)
        scl_pre = self._optional_raster_path(parameters, self.SCL_PRE, context)
        scl_post = self._optional_raster_path(parameters, self.SCL_POST, context)
        if (scl_pre is None) != (scl_post is None):
            raise QgsProcessingException("SCL_PRE and SCL_POST must be supplied together.")
        if scl_pre is not None and scl_post is not None:
            paths["scl_pre"] = scl_pre
            paths["scl_post"] = scl_post
        land_mask = self._optional_raster_path(parameters, self.LAND_MASK, context)
        if land_mask is not None:
            paths["land_mask"] = land_mask
        ls_layers = self.parameterAsLayerList(parameters, self.LS_MAP, context)
        if len(ls_layers) not in (0, 1, 2):
            raise QgsProcessingException("LS_MAP accepts one shared mask or a pre/post pair.")
        for index, layer in enumerate(ls_layers):
            if not isinstance(layer, QgsRasterLayer) or not layer.isValid():
                raise QgsProcessingException("Every LS_MAP input must be a valid raster layer.")
            path = self._path_from_layer(layer)
            paths[f"ls_map_{index}"] = path
        if len(ls_layers) == 1:
            paths["ls_map_1"] = paths["ls_map_0"]
        return paths

    def _raster_path(self, parameters, name: str, context) -> str:
        layer = self.parameterAsRasterLayer(parameters, name, context)
        if layer is None or not layer.isValid():
            raise QgsProcessingException(f"{name} must be a valid raster layer.")
        path = self._path_from_layer(layer)
        info = inspect_raster(path)
        if info.width < 1 or info.height < 1:
            raise QgsProcessingException(f"{name} has no raster cells.")
        return path

    def _optional_raster_path(self, parameters, name: str, context) -> str | None:
        layer = self.parameterAsRasterLayer(parameters, name, context)
        if layer is None:
            return None
        if not layer.isValid():
            raise QgsProcessingException(f"{name} is not a valid raster layer.")
        return self._path_from_layer(layer)

    @staticmethod
    def _path_from_layer(layer: QgsRasterLayer) -> str:
        source = layer.source().split("|", 1)[0]
        path = Path(source)
        if not path.is_file():
            raise QgsProcessingException(f"Raster source is not a readable file: {source}")
        return str(path)

    def _resolve_acquisition_date(
        self,
        parameters,
        name: str,
        context,
        feedback,
        layer_name: str,
        source_path: str,
    ) -> date:
        manual_date = self._optional_date_parameter(parameters, name, context)
        if manual_date is not None:
            return manual_date
        detected_date = detect_acquisition_date(source_path)
        if detected_date is None:
            raise QgsProcessingException(
                f"{name} could not be detected from layer {layer_name}. "
                f"Enter {name} manually."
            )
        feedback.pushInfo(
            f"Detected {name}={detected_date.isoformat()} from layer {layer_name}."
        )
        return detected_date

    @staticmethod
    def _validate_sar_polarizations(paths: Mapping[str, str]) -> None:
        pre_polarization = detect_sar_polarization(paths["sar_pre"])
        post_polarization = detect_sar_polarization(paths["sar_post"])
        if (
            pre_polarization is not None
            and post_polarization is not None
            and pre_polarization != post_polarization
        ):
            raise QgsProcessingException(
                "SAR_PRE and SAR_POST filenames identify different polarizations "
                f"({pre_polarization} and {post_polarization}). Use matching "
                "polarizations or rename/replace the layers."
            )

    def _optional_date_parameter(self, parameters, name: str, context) -> date | None:
        value = self.parameterAsDateTime(parameters, name, context)
        if not isinstance(value, QDateTime) or not value.isValid():
            return None
        qdate = value.date()
        return date(qdate.year(), qdate.month(), qdate.day())

    def _optional_number_parameter(
        self,
        parameters: Mapping[str, Any],
        name: str,
        context,
    ) -> float | None:
        raw_value = parameters.get(name)
        if raw_value is None or raw_value == "":
            return None
        value = self.parameterAsDouble(parameters, name, context)
        if not math.isfinite(value):
            raise QgsProcessingException(f"{name} must be finite.")
        if name == self.BOA_QUANT and value <= 0.0:
            raise QgsProcessingException("BOA_QUANT must be greater than zero.")
        return value

    def _scale_dn_raster(
        self,
        source_path: str,
        key: str,
        offset: float,
        quantification: float,
        context,
    ) -> str:
        info = inspect_raster(source_path)
        grid = RasterGrid(
            info.width,
            info.height,
            info.geotransform,
            info.projection_wkt,
        )
        directory = Path(context.temporaryFolder())
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{key}_reflectance.tif"
        dataset, band = create_raster_output(
            output_path,
            grid,
            data_type=gdal.GDT_Float32,
            nodata=float("nan"),
        )
        try:
            for block in iter_raster_blocks(source_path, block_size=BLOCK_SIZE):
                x_offset, y_offset, _, _ = block.window
                valid = np.isfinite(block.data) & (block.data != 0.0)
                scaled = np.full(block.data.shape, np.nan, dtype=np.float32)
                scaled[valid] = (
                    (block.data[valid].astype(np.float64) + offset) / quantification
                ).astype(np.float32)
                write_raster_block(band, scaled, x_offset, y_offset)
            band.FlushCache()
            dataset.FlushCache()
        finally:
            band = None
            dataset = None
        return str(output_path)

    def _evidence_blocks(
        self,
        aligned_paths: Mapping[str, Path],
        *,
        block_size: int,
        halo: int,
        filter_size: int,
    ) -> Iterator[tuple[Any, np.ndarray, np.ndarray]]:
        core_blocks = iter_aligned_blocks(aligned_paths, block_size=block_size)
        pre_blocks = iter_aligned_blocks(
            {"sar_pre": aligned_paths["sar_pre"]},
            block_size=block_size,
            halo=halo,
        )
        post_blocks = iter_aligned_blocks(
            {"sar_post": aligned_paths["sar_post"]},
            block_size=block_size,
            halo=halo,
        )
        for core, pre, post in zip(core_blocks, pre_blocks, post_blocks, strict=True):
            if core.window != pre.window or core.window != post.window:
                raise RuntimeError("SAR and evidence blocks are not synchronized.")
            pre_values = pre.arrays["sar_pre"]
            post_values = post.arrays["sar_post"]
            if filter_size > 0:
                pre_values = linear_power_to_db(
                    boxcar_filter(
                        db_to_linear_power(pre_values),
                        window_size=filter_size,
                    )
                )
                post_values = linear_power_to_db(
                    boxcar_filter(
                        db_to_linear_power(post_values),
                        window_size=filter_size,
                    )
                )
            pre_values = pre_values[pre.core_slices]
            post_values = post_values[post.core_slices]
            yield core, pre_values, post_values

    def _valid_land(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        shape = next(iter(arrays.values())).shape
        valid = np.ones(shape, dtype=bool)
        for values in arrays.values():
            valid &= np.isfinite(values)
        if "scl_pre" in arrays:
            for name in ("scl_pre", "scl_post"):
                scl = arrays[name]
                valid &= (scl != 0) & (scl != 6)
        if "land_mask" in arrays:
            valid &= arrays["land_mask"] != 0
        for key in ("ls_map_0", "ls_map_1"):
            if key in arrays:
                mask = arrays[key]
                finite = np.isfinite(mask)
                if np.any(finite & ((mask < 0) | (mask != np.floor(mask)))):
                    raise QgsProcessingException(f"{key} must contain non-negative integer flags.")
        return valid

    @staticmethod
    def _update_dbrange_histogram(
        histogram: np.ndarray,
        values: np.ndarray,
        underflow: int,
        overflow: int,
    ) -> tuple[np.ndarray, int, int]:
        finite = values[np.isfinite(values)]
        underflow += int(np.count_nonzero(finite < -100.0))
        overflow += int(np.count_nonzero(finite > 50.0))
        in_range = finite[(finite >= -100.0) & (finite <= 50.0)]
        if in_range.size:
            histogram += np.histogram(in_range, bins=histogram.size, range=(-100.0, 50.0))[0]
        return histogram, underflow, overflow

    def _warn_if_sar_not_db(
        self,
        feedback,
        pre_histogram: np.ndarray,
        post_histogram: np.ndarray,
        underflow: list[int],
        overflow: list[int],
    ) -> None:
        for label, histogram, lower, upper in (
            ("SAR_PRE", pre_histogram, underflow[0], overflow[0]),
            ("SAR_POST", post_histogram, underflow[1], overflow[1]),
        ):
            total = int(histogram.sum()) + lower + upper
            if total == 0:
                raise QgsProcessingException(f"{label} has no finite samples.")
            p01 = self._histogram_quantile(histogram, lower, upper, 0.01)
            p99 = self._histogram_quantile(histogram, lower, upper, 0.99)
            if p01 < SAR_NODATA_MIN or p99 > SAR_NODATA_MAX:
                feedback.pushWarning(
                    f"{label} values have 1st/99th percentiles {p01:.2f}/{p99:.2f}; "
                    "they are outside the broad -60 to +20 dB sanity range."
                )

    @staticmethod
    def _histogram_quantile(
        histogram: np.ndarray,
        underflow: int,
        overflow: int,
        quantile: float,
    ) -> float:
        total = int(histogram.sum()) + underflow + overflow
        target = quantile * max(0, total - 1)
        if target < underflow:
            return -math.inf
        target -= underflow
        if target >= int(histogram.sum()):
            return math.inf
        index = int(np.searchsorted(np.cumsum(histogram), target + 1, side="left"))
        return -100.0 + (index + 0.5) * (150.0 / histogram.size)

    def _sar_statistics(
        self,
        change_path: Path,
        *,
        center: bool,
        burn_direction: str,
        k1: float,
        k2: float,
        feedback,
    ) -> tuple[float, tuple[float, float, float, float], str]:
        pixel_count = 0
        minimum = math.inf
        maximum = -math.inf
        samples: list[np.ndarray] = []
        for block in iter_raster_blocks(change_path, block_size=BLOCK_SIZE):
            values = block.data[np.isfinite(block.data)]
            pixel_count += int(values.size)
            if values.size:
                minimum = min(minimum, float(np.min(values)))
                maximum = max(maximum, float(np.max(values)))
                if pixel_count <= EXACT_STATS_PIXEL_LIMIT:
                    samples.append(values.copy())
                else:
                    samples.clear()
        if pixel_count == 0:
            raise QgsProcessingException("No valid land samples remain for SAR statistics.")

        if pixel_count <= EXACT_STATS_PIXEL_LIMIT:
            raw_values = np.concatenate(samples).astype(np.float32, copy=False)
            center_offset = (
                float(np.median(raw_values.astype(np.float64))) if center else 0.0
            )
            centered = raw_values - np.float32(center_offset)
            oriented = centered if burn_direction == "positive" else -centered
            median = float(np.median(oriented.astype(np.float64)))
            scale = robust_standard_deviation(oriented.reshape(1, -1))
            stats_mode = "exact"
        else:
            feedback.pushWarning(
                f"AOI has {pixel_count:,} valid pixels, above the {EXACT_STATS_PIXEL_LIMIT:,} "
                "exact-statistics limit; deterministic histogram estimates are used."
            )
            if center:
                center_offset = self._histogram_median(
                    change_path,
                    transform=lambda values: values,
                    minimum=minimum,
                    maximum=maximum,
                )
            else:
                center_offset = 0.0
            centered_min = float(np.float32(minimum) - np.float32(center_offset))
            centered_max = float(np.float32(maximum) - np.float32(center_offset))
            if burn_direction == "negative":
                oriented_min, oriented_max = -centered_max, -centered_min
            else:
                oriented_min, oriented_max = centered_min, centered_max
            oriented_min = float(np.nextafter(oriented_min, -math.inf))
            oriented_max = float(np.nextafter(oriented_max, math.inf))
            median = self._histogram_median(
                change_path,
                transform=lambda values: (
                    -(values - np.float32(center_offset))
                    if burn_direction == "negative"
                    else values - np.float32(center_offset)
                ),
                minimum=oriented_min,
                maximum=oriented_max,
            )
            deviation_max = max(abs(oriented_min - median), abs(oriented_max - median))
            mad = self._histogram_median(
                change_path,
                transform=lambda values: np.abs(
                    (
                        -(values - np.float32(center_offset))
                        if burn_direction == "negative"
                        else values - np.float32(center_offset)
                    )
                    - median
                ),
                minimum=0.0,
                maximum=deviation_max,
            )
            scale = float(1.4826 * mad)
            stats_mode = "histogram_approximate"
        if not math.isfinite(scale) or scale <= 0.0:
            raise QgsProcessingException("Robust SAR scale must be finite and positive.")
        if not math.isfinite(median):
            raise QgsProcessingException("Robust SAR median must be finite.")
        breakpoints = (
            median + k1 * scale,
            median + k2 * scale,
            math.inf,
            math.inf,
        )
        return center_offset, breakpoints, stats_mode

    @staticmethod
    def _histogram_median(
        path: Path,
        *,
        transform,
        minimum: float,
        maximum: float,
    ) -> float:
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise QgsProcessingException("Cannot estimate statistics from an empty raster.")
        if maximum <= minimum:
            return float(minimum)
        histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
        for block in iter_raster_blocks(path, block_size=BLOCK_SIZE):
            values = block.data[np.isfinite(block.data)]
            if values.size:
                sample = np.asarray(transform(values), dtype=np.float64)
                histogram += np.histogram(
                    sample,
                    bins=HISTOGRAM_BINS,
                    range=(minimum, maximum),
                )[0]
        total = int(histogram.sum())
        if total == 0:
            raise QgsProcessingException("No finite values remain for histogram statistics.")
        index = int(np.searchsorted(np.cumsum(histogram), (total + 1) // 2, side="left"))
        index = min(index, HISTOGRAM_BINS - 1)
        return float(
            minimum + (index + 0.5) * ((maximum - minimum) / HISTOGRAM_BINS)
        )

    def _classify_block(
        self,
        arrays: Mapping[str, np.ndarray],
        sar_pre: np.ndarray,
        sar_post: np.ndarray,
        *,
        center_offset: float,
        burn_direction: str,
        sar_breakpoints: tuple[float, float, float, float],
        dnbr_low: float,
        dnbr_high: float,
        alpha_time: float,
        sar_reliability: float,
        necessity_threshold: float,
        fusion_function,
    ) -> tuple[np.ndarray, ...]:
        valid = self._valid_land(arrays)
        valid &= np.isfinite(sar_pre) & np.isfinite(sar_post)
        change = sar_post - sar_pre
        change[~valid] = np.nan
        if center_offset:
            change[valid] -= np.float32(center_offset)
        dnbr_values = dnbr(
            arrays["b8a_pre"],
            arrays["b12_pre"],
            arrays["b8a_post"],
            arrays["b12_post"],
            unscaled_threshold=np.finfo(np.float32).max,
        )
        dnbr_values[~valid] = np.nan
        optical = dnbr_possibility(
            dnbr_values,
            (dnbr_low, dnbr_high, math.inf, math.inf),
        )
        if "scl_pre" in arrays:
            alpha_pre = scl_optical_reliability(arrays["scl_pre"])
            alpha_post = scl_optical_reliability(arrays["scl_post"])
            alpha_optical = np.minimum(alpha_pre, alpha_post)
        else:
            alpha_optical = np.ones(valid.shape, dtype=np.float32)
        alpha_optical[~valid] = np.nan
        alpha_sar = np.full(valid.shape, sar_reliability, dtype=np.float32)
        for key in ("ls_map_0", "ls_map_1"):
            if key in arrays:
                flags = np.zeros(valid.shape, dtype=np.int64)
                finite_flags = np.isfinite(arrays[key])
                flags[finite_flags] = arrays[key][finite_flags].astype(np.int64)
                alpha_sar[(flags & SAR_FLAG_MASK) != 0] = 0.0
        alpha_sar[~valid] = np.nan
        alpha_optical = combine_reliability(alpha_optical, alpha_time)
        alpha_sar = combine_reliability(alpha_sar, alpha_time)
        optical = discount(optical, alpha_optical)
        oriented_change = change if burn_direction == "positive" else -change
        sar = sar_possibility(
            oriented_change,
            sar_breakpoints,
            burn_direction="positive",
        )
        sar = discount(sar, alpha_sar)
        fused, conflict = fusion_function(optical, sar)
        classes, necessity = decide(fused, necessity_threshold)
        invalid = ~valid
        classes[invalid] = -1
        necessity[invalid] = np.nan
        conflict[invalid] = np.nan
        return classes, necessity, conflict, fused[0], fused[1]

    def _create_outputs(
        self,
        parameters,
        context,
        grid: RasterGrid,
        **metadata_values,
    ) -> dict[str, Any]:
        output_specs = {
            "class": (self.OUTPUT_CLASS, gdal.GDT_Int16, -1.0),
            "necessity": (self.OUTPUT_NECESSITY, gdal.GDT_Float32, float("nan")),
            "conflict": (self.OUTPUT_CONFLICT, gdal.GDT_Float32, float("nan")),
            "pi_burned": (self.OUTPUT_PI_BURNED, gdal.GDT_Float32, float("nan")),
            "pi_unburned": (self.OUTPUT_PI_UNBURNED, gdal.GDT_Float32, float("nan")),
            "undecided": (self.OUTPUT_UNDECIDED_MASK, gdal.GDT_Byte, 255.0),
        }
        output_paths: dict[str, str] = {}
        for key, (parameter, _, _) in output_specs.items():
            path = self.parameterAsOutputLayer(parameters, parameter, context)
            if path:
                output_paths[parameter] = path
        writers: dict[str, Any] = {"paths": output_paths}
        metadata = self._metadata(metadata_values)
        for key, (parameter, data_type, nodata) in output_specs.items():
            path = output_paths.get(parameter)
            if not path:
                continue
            dataset, band = create_raster_output(
                path,
                grid,
                data_type=data_type,
                nodata=nodata,
            )
            dataset.SetMetadata(metadata)
            writers[key] = (dataset, band)
        return writers

    @staticmethod
    def _metadata(values: Mapping[str, Any]) -> dict[str, str]:
        metadata = {"ALGORITHM": "dst_fusion:fuse_burned_area"}
        for key, value in values.items():
            if key in ("paths", "input_paths", "dates"):
                for subkey, subvalue in value.items():
                    metadata[f"{key}_{subkey}".upper()] = str(subvalue)
            elif value is not None:
                metadata[key.upper()] = str(value)
        return metadata

    @staticmethod
    def _close_outputs(outputs: dict[str, Any]) -> None:
        for key, writer in outputs.items():
            if key == "paths":
                continue
            dataset, band = writer
            band.FlushCache()
            dataset.FlushCache()
            outputs[key] = None

    @staticmethod
    def _rebuild_undecided_mask(class_path, mask_path, feedback) -> None:
        """Derive the undecided mask from the finished class raster."""
        src = gdal.Open(class_path)
        dst = gdal.Open(mask_path, gdal.GA_Update)
        if src is None or dst is None:
            raise QgsProcessingException("Could not reopen outputs to build the undecided mask.")
        sband, dband = src.GetRasterBand(1), dst.GetRasterBand(1)
        width, height = src.RasterXSize, src.RasterYSize
        step = 256
        for y in range(0, height, step):
            rows = min(step, height - y)
            classes = sband.ReadAsArray(0, y, width, rows)
            mask = np.full(classes.shape, 255, dtype=np.uint8)
            valid = classes >= 0
            mask[valid] = (classes[valid] == 2).astype(np.uint8)
            dband.WriteArray(mask, 0, y)
        dband.FlushCache()
        try:
            dband.ComputeStatistics(False)
        except RuntimeError as exc:
            feedback.pushWarning(f"Could not compute undecided-mask statistics: {exc}")
        src = None
        dst = None

    @staticmethod
    def _write_qml(path: str, *, style_type: str) -> None:
        layer = QgsRasterLayer(path, Path(path).stem)
        if not layer.isValid():
            raise QgsProcessingException(f"Could not load output layer for styling: {path}")
        if style_type in ("class", "mask"):
            if style_type == "class":
                palette = [
                    QgsPalettedRasterRenderer.Class(0, QColor(210, 45, 45), "Burned"),
                    QgsPalettedRasterRenderer.Class(1, QColor(75, 145, 75), "Unburned"),
                    QgsPalettedRasterRenderer.Class(2, QColor(150, 150, 150), "Undecided"),
                ]
            else:
                palette = [
                    QgsPalettedRasterRenderer.Class(0, QColor(0, 0, 0, 0), "Decided"),
                    QgsPalettedRasterRenderer.Class(1, QColor(150, 150, 150), "Undecided"),
                ]
            layer.setRenderer(
                QgsPalettedRasterRenderer(layer.dataProvider(), 1, palette)
            )
        else:
            shader = QgsRasterShader()
            ramp = QgsColorRampShader()
            ramp.setColorRampType(QgsColorRampShader.Interpolated)
            ramp.setColorRampItemList(
                [
                    QgsColorRampShader.ColorRampItem(
                        0.0,
                        QColor(255, 255, 230),
                        "0",
                    ),
                    QgsColorRampShader.ColorRampItem(
                        0.5,
                        QColor(255, 150, 50),
                        "0.5",
                    ),
                    QgsColorRampShader.ColorRampItem(
                        1.0,
                        QColor(150, 0, 0),
                        "1",
                    ),
                ]
            )
            shader.setRasterShaderFunction(ramp)
            layer.setRenderer(
                QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
            )
        style_path = f"{path}.qml"
        result, message = layer.saveNamedStyle(style_path)
        if not result:
            raise QgsProcessingException(f"Could not save QML style for {path}: {message}")

    @staticmethod
    def _remove_directory(directory: Path) -> None:
        import shutil

        shutil.rmtree(directory, ignore_errors=False)
