# QGIS Raster Processor Plugin

A QGIS plugin for interpolation, raster resampling, and raster smoothing with a stronger focus on robust defaults, low-memory processing, and clear metric inputs.

## Highlights
- Interpolation resolution is specified as a grid cell size in meters (X x Y), with an option to link both axes to the same value
- Resampling uses a target raster resolution in meters instead of a scale factor
- Smoothing uses a radius in meters instead of ambiguous pixel factors
- Large IDW and nearest-neighbour interpolation jobs are processed tile-wise to avoid `unable to allocate xx GB` errors
- XYZ files can be assigned an explicit CRS
- Output size, CRS, and safety limits are shown before processing
- Duplicate XY points are merged automatically by averaging Z values
- GeoTIFF output is tiled, compressed, and written as BigTIFF when needed
- Large IDW/Nearest interpolation and smoothing jobs can optionally use multiple worker processes to speed up tile processing

## Recommended workflow
1. Use projected data in a CRS with meter units.
2. Pick `IDW` for the most robust interpolation workflow on large datasets.
3. Increase the target resolution value if the summary shows an extremely large output raster.
4. Use `Linear`, `Cubic`, and `RBF` only for smaller jobs. The plugin limits them intentionally for stability.

## Interpolation
- Input can be a QGIS point layer or an XYZ/CSV file
- `IDW` and `Nearest` are designed for large jobs
- `Linear`, `Cubic`, and `RBF` remain available for smaller, more detailed surfaces
- Extrapolation outside the convex hull can be enabled explicitly

## Resampling
- Set the target resolution directly in meters
- Uses GDAL warp with tiled compressed GeoTIFF output
- Supports `Nearest`, `Bilinear`, `Cubic`, `Average`, and `Lanczos`

## Smoothing
- Uses a smoothing radius in meters
- Supports Gaussian and uniform mean filtering
- Processes raster tiles with overlap so large rasters do not need to fit fully into RAM
- Preserves NoData regions with weight-aware filtering

## Performance
- The "Performance" section can process tiles in parallel across multiple worker processes for `IDW`/`Nearest` interpolation and for smoothing.
- Disabled by default. Enable it and pick a worker count for large tile jobs where per-core speedup outweighs the process startup cost.
- If worker processes cannot be started (e.g. no standalone Python interpreter can be found next to the QGIS installation), the plugin logs a warning and automatically continues with a single process.

## Notes
- The plugin expects projected CRS with meters. Layers in degrees should be reprojected first.
- Very large outputs can still take a long time and produce large files even though RAM usage is controlled.
