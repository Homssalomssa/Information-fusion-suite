# SPDX-License-Identifier: GPL-2.0-or-later

"""Sentinel-2 input mode and BOA metadata discovery."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .raster import _open_raster, iter_raster_blocks

_BASELINE_PATTERN = re.compile(r"(?:^|[^A-Z0-9])N(\d{4})(?!\d)", re.IGNORECASE)
_BAND_IDS = {"b8a_pre": "8", "b8a_post": "8", "b12_pre": "12", "b12_post": "12"}


@dataclass(frozen=True)
class S2InputInference:
    mode: str
    reason: str


@dataclass(frozen=True)
class BoaParameters:
    offset: float
    quantification: float
    offset_source: str
    quantification_source: str
    warnings: tuple[str, ...]


def infer_s2_input_mode(input_paths: Mapping[str, str | Path]) -> S2InputInference:
    """Classify four S2 inputs as integer DN or low-range float reflectance."""
    if set(input_paths) != set(_BAND_IDS):
        raise ValueError("S2 mode detection requires all four B8A/B12 pre/post inputs.")

    classifications: dict[str, tuple[str, str]] = {}
    for key, path in input_paths.items():
        dataset = _open_raster(path)
        band = dataset.GetRasterBand(1)
        from osgeo import gdal

        dtype_name = gdal.GetDataTypeName(band.DataType)
        dataset = None
        if dtype_name == "Byte" or dtype_name.startswith(("Int", "UInt")):
            classifications[key] = ("dn", dtype_name)
            continue
        if not dtype_name.startswith("Float"):
            raise ValueError(
                f"{key} has unsupported raster type {dtype_name}; "
                "choose S2_INPUT_MODE explicitly."
            )

        maximum = -np.inf
        for block in iter_raster_blocks(path):
            finite = block.data[np.isfinite(block.data)]
            if finite.size:
                maximum = max(maximum, float(np.max(finite)))
        if not np.isfinite(maximum):
            raise ValueError(
                f"{key} has no finite samples for automatic S2 mode detection; "
                "choose S2_INPUT_MODE explicitly."
            )
        if maximum > 1.5:
            raise ValueError(
                f"{key} is floating-point with maximum {maximum:g} (> 1.5); "
                "choose S2_INPUT_MODE explicitly."
            )
        classifications[key] = ("reflectance", f"{dtype_name}, max={maximum:g}")

    modes = {classification[0] for classification in classifications.values()}
    if len(modes) != 1:
        detail = ", ".join(
            f"{key}={mode} ({reason})"
            for key, (mode, reason) in sorted(classifications.items())
        )
        raise ValueError(
            f"S2 inputs do not share one unambiguous mode ({detail}); "
            "choose S2_INPUT_MODE explicitly."
        )
    mode = modes.pop()
    detail = "; ".join(
        f"{key}: {reason}" for key, (_, reason) in sorted(classifications.items())
    )
    return S2InputInference(mode, detail)


def resolve_boa_parameters(
    input_paths: Mapping[str, str | Path],
    *,
    offset_override: float | None = None,
    quantification_override: float | None = None,
) -> BoaParameters:
    """Read BOA calibration from SAFE XML, or infer offset from baseline names."""
    if set(input_paths) != set(_BAND_IDS):
        raise ValueError("BOA metadata discovery requires all four B8A/B12 pre/post inputs.")

    offsets: list[tuple[float, str]] = []
    quantifications: list[tuple[float, str]] = []
    warnings: list[str] = []
    for key, source in input_paths.items():
        source_path = Path(source)
        xml_path = _find_boa_xml(source_path)
        if xml_path is not None:
            offset, quantification = _read_boa_xml(xml_path, _BAND_IDS[key], key)
            offsets.append((offset, str(xml_path)))
            quantifications.append((quantification, str(xml_path)))
            continue

        baseline = _find_processing_baseline(source_path)
        if baseline is None:
            if offset_override is None:
                raise ValueError(
                    f"No MTD_MSIL2A.xml or processing baseline was found for {key} "
                    f"({source_path}); provide explicit BOA offset/calibration values."
                )
            offsets.append((offset_override, "manual BOA_ADD_OFFSET parameter"))
        else:
            offset = -1000.0 if baseline >= 400 else 0.0
            offsets.append((offset, f"processing baseline N{baseline:04d}"))
            warnings.append(
                f"No MTD_MSIL2A.xml found for {key}; inferred BOA_ADD_OFFSET={offset:g} "
                f"from processing baseline N{baseline:04d}."
            )

    if offset_override is not None:
        offset, offset_source = offset_override, "manual BOA_ADD_OFFSET parameter"
    else:
        distinct_offsets = {value for value, _ in offsets}
        if len(distinct_offsets) != 1:
            raise ValueError(
                "S2 inputs have different BOA additive offsets; "
                "provide an explicit BOA_ADD_OFFSET."
            )
        offset = offsets[0][0]
        offset_sources = sorted({source for _, source in offsets})
        offset_source = ", ".join(offset_sources)

    if quantification_override is not None:
        quantification = quantification_override
        quantification_source = "manual BOA_QUANT parameter"
    elif quantifications:
        distinct_quantifications = {value for value, _ in quantifications}
        if len(distinct_quantifications) != 1:
            raise ValueError(
                "S2 inputs have different BOA quantification values; "
                "provide an explicit BOA_QUANT."
            )
        quantification = quantifications[0][0]
        quantification_source = ", ".join(
            sorted({source for _, source in quantifications})
        )
    else:
        quantification = 10000.0
        quantification_source = "standard BOA_QUANTIFICATION_VALUE fallback (10000)"
        warnings.append(
            "No MTD_MSIL2A.xml was found for the S2 inputs; using "
            "BOA_QUANTIFICATION_VALUE=10000. Set BOA_QUANT to override."
        )

    if not np.isfinite(offset):
        raise ValueError("BOA_ADD_OFFSET must be finite.")
    if not np.isfinite(quantification) or quantification <= 0.0:
        raise ValueError("BOA_QUANTIFICATION_VALUE must be finite and positive.")
    return BoaParameters(
        offset,
        quantification,
        offset_source,
        quantification_source,
        tuple(dict.fromkeys(warnings)),
    )


def _find_boa_xml(source_path: Path) -> Path | None:
    for parent in source_path.parents:
        candidate = parent / "MTD_MSIL2A.xml"
        if candidate.is_file():
            return candidate
    return None


def _find_processing_baseline(source_path: Path) -> int | None:
    for component in (source_path.name, *(parent.name for parent in source_path.parents)):
        match = _BASELINE_PATTERN.search(component)
        if match:
            return int(match.group(1))
    return None


def _read_boa_xml(xml_path: Path, band_id: str, input_name: str) -> tuple[float, float]:
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"Could not read BOA metadata from {xml_path}: {error}") from error

    offset_values: list[float] = []
    quantification_values: list[float] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "BOA_QUANTIFICATION_VALUE":
            quantification_values.append(float(element.text or "nan"))
        elif tag == "BOA_ADD_OFFSET" and element.attrib.get("band_id") == band_id:
            offset_values.append(float(element.text or "nan"))
    if len(offset_values) != 1 or len(quantification_values) != 1:
        raise ValueError(
            f"Missing or ambiguous BOA_ADD_OFFSET for band {band_id} or "
            f"BOA_QUANTIFICATION_VALUE in {xml_path} ({input_name})."
        )
    return offset_values[0], quantification_values[0]
