"""Pure helpers for extracting acquisition metadata from raster paths."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

_ACQUISITION_PATTERN = re.compile(r"(?<!\d)(\d{8})T\d{6}(?!\d)")
_SAR_POLARIZATION_PATTERN = re.compile(r"(?:^|[_-])(VV|VH)(?=[_.-]|$)", re.IGNORECASE)


def detect_acquisition_date(source_path: str | Path) -> date | None:
    """Return a Sentinel acquisition date embedded in a filename or SAFE name."""
    path = Path(source_path)
    candidates = [path.name]
    candidates.extend(
        parent.name
        for parent in path.parents
        if parent.name.upper().endswith(".SAFE")
    )
    for candidate in candidates:
        match = _ACQUISITION_PATTERN.search(candidate)
        if match:
            try:
                return date(
                    int(match.group(1)[:4]),
                    int(match.group(1)[4:6]),
                    int(match.group(1)[6:8]),
                )
            except ValueError:
                continue
    return None


def detect_sar_polarization(source_path: str | Path) -> str | None:
    """Return VV or VH when the raster filename identifies a polarization."""
    match = _SAR_POLARIZATION_PATTERN.search(Path(source_path).name)
    return match.group(1).upper() if match else None
