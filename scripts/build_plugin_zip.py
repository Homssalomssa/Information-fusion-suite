# SPDX-License-Identifier: GPL-2.0-or-later
"""Build a runtime-only QGIS plugin archive."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIRECTORY = "possibility_fusion"
PLUGIN_ROOT = REPOSITORY_ROOT / PLUGIN_DIRECTORY
RUNTIME_FILES = {"metadata.txt", "icon.png", "LICENSE"}
RUNTIME_PACKAGES = {"algorithms", "core", "io"}


def _plugin_files() -> list[Path]:
    files = []
    for path in PLUGIN_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PLUGIN_ROOT)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if relative.parts[0] in RUNTIME_PACKAGES and path.suffix == ".py":
            files.append(path)
        elif len(relative.parts) == 1 and (
            path.name in RUNTIME_FILES or path.suffix == ".py"
        ):
            files.append(path)
    if not files:
        raise RuntimeError(f"No runtime plugin files found under {PLUGIN_ROOT}.")
    return sorted(files, key=lambda item: item.relative_to(PLUGIN_ROOT).as_posix())


def build_zip(output_path: Path | None = None) -> Path:
    metadata = configparser.ConfigParser()
    with (PLUGIN_ROOT / "metadata.txt").open(encoding="utf-8") as source:
        metadata.read_file(source)
    version = metadata["general"]["version"]
    if output_path is None:
        output_path = REPOSITORY_ROOT / "dist" / f"{PLUGIN_DIRECTORY}-{version}.zip"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in _plugin_files():
            relative = path.relative_to(PLUGIN_ROOT)
            archive.write(path, (Path(PLUGIN_DIRECTORY) / relative).as_posix())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output zip path (default: dist/possibility_fusion-<version>.zip).",
    )
    arguments = parser.parse_args()
    output = build_zip(arguments.output)
    print(output)
    with zipfile.ZipFile(output) as archive:
        for name in archive.namelist():
            print(name)


if __name__ == "__main__":
    main()
