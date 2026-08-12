# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A QGIS 3 plugin ("Raster Processor") providing memory-safe point interpolation, raster resampling, and raster smoothing, all with metric (meters) inputs. It is a single-window Qt dialog driven entirely by `main.py`; there is no separate build step, package manager, or test suite — the plugin is deployed by copying/symlinking this folder into the QGIS profile's `python/plugins` directory and reloading QGIS.

## Repository layout

- `__init__.py` — QGIS plugin entry point. `classFactory(iface)` imports and returns `RasterProcessorPlugin` from `main.py`. Keep this file minimal; QGIS reads `metadata.txt` + this file to discover the plugin.
- `main.py` — the entire plugin UI: `RasterProcessorPlugin` (QGIS plugin lifecycle: `initGui`/`unload`/`run`) and `RasterProcessorDialog` (the `QDialog` with all UI + orchestration logic).
- `parallel_workers.py` — pure-compute tile functions (IDW/Nearest interpolation, raster smoothing) used both in-process and as `ProcessPoolExecutor` worker tasks. Deliberately has **no** `qgis.PyQt`/`qgis.core`/`qgis.gui` imports — see "Parallel tile processing" below for why that matters.
- `metadata.txt` — QGIS plugin manifest (name, version, author, `qgisMinimumVersion`, etc.). Bump `version=` here when making a release-worthy change.
- `README.md` — user-facing feature description and recommended workflow; keep in sync with UI/behavior changes.
- `icon.ico` — toolbar/menu icon referenced by `metadata.txt`.
- `symbology-style.db` / `symbology-style.db-journal` — QGIS-generated local style database, not referenced anywhere in code; unrelated to plugin logic.

## Running / testing

There are no automated tests, linters, or build scripts in this repo. To exercise changes:
1. Copy or symlink this directory into the QGIS profile plugins folder (e.g. `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\raster_processor`).
2. In QGIS: Plugin Reloader (or restart QGIS), then launch via the "Raster Processor" toolbar icon/menu entry.
3. Watch the in-dialog log panel and the QGIS Log Messages panel (tag `RasterProcessor`) for errors — `self.log()` writes to both.

Since there's no headless test harness, validate logic changes by reasoning through the code and, where feasible, by running the relevant pure-numpy/GDAL routines outside QGIS (e.g. in a plain Python/conda env with `gdal`, `numpy`, `scipy` installed) rather than assuming QGIS Qt widgets are mockable.

## Architecture

`RasterProcessorDialog` has three mutually-exclusive processing modes selected by radio buttons, each with its own settings `QGroupBox` shown/hidden via `toggle_mode()`:

1. **Interpolate** — turns a QGIS point layer or an XYZ/CSV file into a raster.
2. **Resample** — changes a raster's pixel resolution via `gdal.Warp`.
3. **Smooth** — applies Gaussian or uniform (moving-average) filtering to a raster.

All three converge on `process()` → `_process_interpolation` / `_process_resample` / `_process_smooth`, each of which writes a GeoTIFF via GDAL and calls `_add_output_to_project()` to load the result back into QGIS.

### Interpolation pipeline (`_process_interpolation`)
- Points are loaded from either a vector layer (`_load_points_from_vector`) or a text file (`_read_xyz_file`, whitespace/comma/semicolon-delimited), always requiring a **projected CRS in meters** (`_validate_metric_crs`).
- Duplicate XY points are merged by averaging Z (`_aggregate_duplicate_points`).
- Z values can be converted from TWT/OWT travel-time units to meters via velocity (`_vertical_conversion_factor`, `_convert_vertical_values`) — this is domain logic for seismic/geophysics-style depth conversion.
- Input structure detection (`_detect_regular_grid`): if points form a complete-enough regular XY grid (`REGULAR_GRID_COMPLETENESS = 0.95`), the plugin takes a fast path (`_process_regular_grid_input`) that rasterizes directly and uses `gdal.Warp`/`CreateCopy` for resampling, skipping interpolation entirely. This can be forced or disabled via the "Input structure" combo (`auto` / `scattered` / `regular`).
- For scattered points: `IDW`/`Nearest` (`_interpolate_idw_or_nearest`) are processed **tile-by-tile** (`DEFAULT_TILE_SIZE = 512`) via `parallel_workers.idw_or_nearest_tile` (a `cKDTree` query per tile) so large outputs don't blow up RAM. `Linear`/`Cubic`/`RBF` (`_interpolate_advanced_method`) are computed in one shot via `scipy.interpolate.griddata`/`RBFInterpolator` and are therefore hard-capped by `ADVANCED_METHOD_LIMITS` (max pixels/points per method) — don't remove these caps without replacing them with tiling, since that's the whole reason they exist. Only the tile-based `IDW`/`Nearest` path supports parallel workers; the single-shot advanced methods don't.
- Extrapolation outside the input convex hull is masked out by default (`_build_hull_equations` + `_points_inside_hull`) unless the user opts in.
- Hard/soft safety limits: `HARD_OUTPUT_PIXEL_LIMIT` (raises) and `LARGE_OUTPUT_PIXEL_WARNING` (logs only), enforced by `_validate_output_pixels`.
- Output grid cell size is independently configurable per axis (`resolution_x_spin`/`resolution_y_spin`, optionally kept in sync via the "Link X/Y" checkbox → `_sync_linked_resolution`). `grid_spec` dicts and `_estimate_output_grid(...)` carry `resolution_x`/`resolution_y` separately throughout the interpolation/regular-grid code paths — don't collapse them back into a single value.

### Resample / Smooth
- Resample is a thin wrapper around `gdal.Warp` with resolution + resampling algorithm.
- Smooth reads raster tiles with a halo (padding sized to the filter radius: 3σ for Gaussian, radius-in-pixels for uniform) so tile boundaries don't produce filtering artifacts, applies `scipy.ndimage.gaussian_filter`/`uniform_filter` via `parallel_workers.smooth_tile` with NoData-aware weighting (filter the data and a validity mask separately, then divide), and writes only the inner (non-halo) region back per tile. Processed per band, one `tile_windows` pass at a time.

### Parallel tile processing
- Both tile loops (`_interpolate_idw_or_nearest`, `_process_smooth`) build a plain-data task list, then dispatch it through either `_run_*_tiles_sequential` (in-process, calls `parallel_workers.*_tile` directly) or `_run_*_tiles_parallel` (`ProcessPoolExecutor`), chosen by `_effective_worker_count(total_tiles)`. Whether parallel or sequential, the actual math always runs through the same `parallel_workers` functions — don't reintroduce a separate inline implementation in `main.py`.
- `_effective_worker_count` returns `1` (i.e. sequential) unless the "Process large tile jobs in parallel" checkbox is on, more than one tile exists, and — on Windows — a standalone Python interpreter can be located (`_resolve_worker_python_executable`, checked via `multiprocessing.set_executable`). This matters because `sys.executable` inside an embedded QGIS process is the QGIS binary itself, not a plain interpreter multiprocessing's spawn bootstrap can launch.
- `parallel_workers.py` has no Qt/QGIS imports specifically so that a spawned worker process only needs to `import` that module (not `main.py`), avoiding the Qt/QGIS native-DLL loading problems a bare worker process would otherwise hit.
- Every parallel call site is wrapped in `try`/`except` and falls back to the sequential runner (logging a `Qgis.Warning`) on any failure (e.g. `BrokenProcessPool`). Parallelism is always best-effort — never let a parallel-path failure surface as a processing error the user has to work around.
- IDW/Nearest workers get the full point cloud once via `ProcessPoolExecutor(initializer=parallel_workers.init_idw_worker, initargs=(x, y, z))`, building one `cKDTree` per worker (not per tile). Smoothing workers instead get a `(source_path, band_number)` pair via `init_smooth_worker` and open their own read-only GDAL dataset, since a GDAL `Dataset`/`Band` handle can't be shared across processes.

### UI/state conventions
- Every settings widget's change signal is wired to `update_summary()`, which re-renders a live "Summary" panel (`_interpolation_summary`/`_resample_summary`/`_smooth_summary`) showing CRS, estimated output size, and safety-limit warnings *before* the user clicks "Process". When adding a new setting, wire its signal the same way so the summary stays accurate.
- The dialog is resizable (`setMinimumSize` only, no fixed size) and its settings groups + summary live inside a `QScrollArea` (`setup_ui`), while the progress bar, Process/Close buttons, and log stay pinned outside the scroll area so they're always visible. Add new settings widgets to the scrollable `layout`, not the outer one.
- All distances/resolutions in the UI are in **meters**; the plugin actively rejects non-metric CRSes (`_validate_metric_crs`) rather than silently reprojecting.
- GDAL exceptions are enabled globally (`gdal.UseExceptions()`); output GeoTIFFs always use `_gdal_creation_options()` (`COMPRESS=LZW, TILED=YES, BIGTIFF=IF_SAFER, PREDICTOR=3`).
- `DEFAULT_NODATA = -9999.0` is the fallback NoData value used when a source raster has none.
