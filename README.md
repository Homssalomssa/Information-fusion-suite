# Possibility Fusion

**EXPERIMENTAL** QGIS Processing plugin for combining Sentinel-2 optical
change and Sentinel-1 radar change into a burned-area map.

The plugin turns pre- and post-fire observations into two-class possibility
evidence and combines them. It is not a probability estimator and does not
choose the direction of radar change for you.

## Requirements

- QGIS 3.40 or newer.
- The QGIS Processing framework enabled.
- Sentinel-2 L2A B8A and B12 rasters at 20 m, and Sentinel-1 calibrated
  backscatter rasters in dB. QGIS's bundled GDAL and NumPy are used.

## Install from a ZIP

1. Build or download `possibility_fusion-0.1.0.zip`.
2. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Select the ZIP, install it, and enable **Possibility Fusion** if prompted.
4. Open the **Processing Toolbox** and search for **Fuse burned-area evidence**
   (algorithm ID `possibility_fusion:fuse_burned_area`).

## Inputs

Required:

- Sentinel-2 B8A and B12, each for pre-fire and post-fire dates.
- Sentinel-1 pre-fire and post-fire backscatter in dB. Use the same
  polarization for both rasters.
- Acquisition dates for both observations from both sensors. Leave a date
  empty for detection from Sentinel filenames, or from the enclosing S2
  `.SAFE` directory name. Enter a date manually when the name cannot be
  recognized.

Optional:

- A matching pair of Sentinel-2 Scene Classification Layer rasters.
- A binary land mask and/or SAR layover/shadow flag raster(s).
- Fire start/end dates (supply them as a pair).

## Outputs

- **Class raster:** `0` Burned, `1` Unburned, `2` Undecided, `-1` NoData.
- **Necessity raster:** 0–1 support for the selected class; higher values
  indicate stronger necessity.
- **Conflict raster:** 0–1 disagreement between the input evidence; 0 means no
  conflict and 1 means maximal conflict.
- Optional undecided mask: `1` undecided, `0` decided, `255` NoData.
- Optional Burned and Unburned possibility rasters.

## Parameter guide

- **SAR burn direction:** Choose `negative` or `positive` based on independent
  scene-specific evidence about whether burned surfaces show a radar decrease
  or increase. The correct sign varies by scene and polarization, so the
  plugin deliberately requires a choice.
- **S2 input units:** `auto` treats integer rasters as DN and floating-point
  rasters with maximum at most 1.5 as reflectance. For DN, calibration is read
  from `MTD_MSIL2A.xml`, or inferred from the processing baseline with a
  warning. Use `Reflectance` or `DN with BOA offset` explicitly if automatic
  classification is ambiguous. BOA offset and quantification values can be
  supplied as overrides.
- **Temporal mode:** `auto` applies temporal discounting when the fire end is
  unknown or a post-fire acquisition predates it; `always` applies it
  regardless of dates; `never` skips it. `TAU_DAYS` controls the discount
  strength.
- **AOI extent:** This controls the analysis window, including where SAR
  centering and robust SAR thresholds are calculated. Select a representative
  scene extent.
- **Optical thresholds, SAR filter, and MAD multipliers:** These change how
  evidence is formed. The defaults are experimental starting points, not
  generally validated settings.
- **SCL / land / layover-shadow masks:** These affect validity or sensor
  reliability. Without an SCL or land mask, water is not masked.

## Worked synthetic example

Consider four aligned 100×100-pixel, 20 m Sentinel-2 rasters and two matching
Sentinel-1 dB rasters. In a synthetic square from rows and columns 25 through
74, suppose pre-fire reflectance is B8A `0.6`, B12 `0.2`, while post-fire
reflectance is B8A `0.2`, B12 `0.4`. Set pre-fire SAR to `-10 dB` everywhere
and post-fire SAR to `-7 dB` in the square. These example observations show
increased radar backscatter in the square, so select `positive` burn direction
for this described case; do not assume that sign for another scene.

Load the four optical rasters and the two SAR rasters in QGIS, set the AOI to
the 100×100 grid, choose reflectance mode, supply or detect all four dates,
and choose the positive direction. Run **Fuse burned-area evidence** and
inspect the class, necessity, and conflict rasters together. The example
values are illustrative; actual class boundaries depend on the selected
fusion and threshold settings. No sample raster files are included.

## Limitations

- **EXPERIMENTAL:** validation currently covers one event.
- The available reference delineation is optical-derived, which can favor
  optical evidence in comparisons.
- SAR burn direction is scene-dependent and must be chosen by the user.
- Defaults have not been tested on other events.
- Results depend on input quality, masking, AOI, acquisitions, and chosen
  parameters; inspect outputs and validate against independent references
  before operational use.
