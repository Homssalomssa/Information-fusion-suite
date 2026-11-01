# Contributing

## Development setup

Use the system QGIS and GDAL Python bindings through a virtual environment with
system site packages. From the repository root, the current environment recipe
is:

```sh
/usr/bin/python3 -m venv --system-site-packages --without-pip .venv-qgis
python3 -m pip --python .venv-qgis/bin/python install pip
.venv-qgis/bin/python -m pip install -r requirements-dev.txt
```

If pip is unavailable, bootstrap it with a Python/pip installation available
on your machine. Keep QGIS, Qt, and GDAL bindings from one compatible system
installation.

## Tests

Run the full suite headlessly, with Qt offscreen and the project memory limit:

```sh
ulimit -v 8000000
QT_QPA_PLATFORM=offscreen .venv-qgis/bin/python -m pytest -p no:cacheprovider -q
```

Run targeted tests by appending their path, for example
`tests/test_processing_algorithm.py`. Add or update tests before changing
behavior. Preserve the frozen Chios test parameters, expected scoring-mask
counts, synthetic regression, and explicit tolerances unless the user approves
a separate, documented change.

## Architecture

- `possibility_fusion/core/`: pure NumPy/Python numerical logic, independent
  from GIS runtimes.
- `possibility_fusion/io/`: raster inspection, alignment, calibration, and
  block-based GDAL IO.
- `possibility_fusion/algorithms/`: QGIS Processing parameter definitions,
  feedback, orchestration, and output styling.
- `possibility_fusion/plugin.py` and `provider.py`: plugin lifecycle and
  Processing provider registration.

**No QGIS or GDAL imports in `core/`.** Keep GIS-specific concerns in `io/` or
the algorithm wrapper, and test core functions without requiring QGIS.

## Adding a Processing algorithm

1. Define numerical behavior and pure helpers in `core/` first; add focused
   unit tests.
2. Add raster-specific work under `io/` when needed, with bounded reads and
   explicit NoData handling.
3. Implement a `QgsProcessingAlgorithm` wrapper in `algorithms/`, register it
   in `PossibilityFusionProvider.loadAlgorithms()`, and expose errors and
   warnings through Processing exceptions/feedback.
4. Add a headless Processing integration test for parameter wiring, outputs,
   NoData, and representative synthetic behavior. Avoid tuning frozen
   event-specific parameters to the validation reference.
5. Update help, user docs, and packaging inclusion as appropriate. Run the
   targeted test first, then the full suite and plugin ZIP check.

## Reporting validation results

For every real-scene validation, report the scene/event, sensor products and
acquisition dates, reference source and known reference bias, AOI/grid and
valid-pixel rule, selected SAR sign and who/what informed it, parameters
changed or tuned, and all registered primary/secondary comparisons.
Pre-register the primary outcome, variants, thresholds, mask, and analysis
before inspecting reference-comparison metrics. Record exploratory changes
and results separately; do not replace a frozen primary result with a tuned
variant. Include test commands/counts and distinguish validation against an
optical-derived reference from independent ground truth.
