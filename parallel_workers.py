"""Pure-compute tile workers used by multiprocessing.Pool / ProcessPoolExecutor.

This module intentionally avoids importing qgis.PyQt / qgis.core / qgis.gui.
Worker processes spawned on Windows re-import whichever module a task function
lives in, and the Qt/QGIS bindings are not safely importable outside the main
QGIS process. Keeping this module limited to numpy/scipy/gdal lets tile jobs
run in plain worker processes.
"""

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

gdal.UseExceptions()

_idw_tree = None
_idw_values = None

_smooth_dataset = None
_smooth_band = None


def _points_inside_hull(coords, hull_equations):
    if hull_equations is None:
        return np.ones(coords.shape[0], dtype=bool)
    lhs = np.matmul(coords, hull_equations[:, :2].T) + hull_equations[:, 2]
    return np.all(lhs <= 1e-9, axis=1)


def init_idw_worker(x, y, z):
    global _idw_tree, _idw_values
    _idw_tree = cKDTree(np.column_stack((x, y)))
    _idw_values = np.asarray(z, dtype=np.float64)


def idw_or_nearest_tile(task):
    (
        xoff, yoff, xsize, ysize,
        xmin, ymax, resolution_x, resolution_y,
        method, power, neighbours, hull_equations, nodata,
    ) = task

    x_coords = xmin + (np.arange(xsize) + xoff + 0.5) * resolution_x
    y_coords = ymax - (np.arange(ysize) + yoff + 0.5) * resolution_y
    tile_x, tile_y = np.meshgrid(x_coords, y_coords)
    query_coords = np.column_stack((tile_x.ravel(), tile_y.ravel()))

    if method == "nearest":
        _, indices = _idw_tree.query(query_coords, k=1)
        values = _idw_values[np.asarray(indices, dtype=np.int64)]
    else:
        distances, indices = _idw_tree.query(query_coords, k=neighbours)
        if np.ndim(distances) == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        exact_matches = distances[:, 0] <= 1e-12
        safe_distances = np.maximum(distances, 1e-12)
        weights = 1.0 / np.power(safe_distances, power)
        values = np.sum(weights * _idw_values[indices], axis=1) / np.sum(weights, axis=1)
        if np.any(exact_matches):
            values[exact_matches] = _idw_values[indices[exact_matches, 0]]

    if hull_equations is not None:
        inside_mask = _points_inside_hull(query_coords, hull_equations)
        values[~inside_mask] = nodata

    tile_array = values.reshape((ysize, xsize)).astype(np.float32)
    return xoff, yoff, tile_array


def init_smooth_worker(source_path, band_number):
    global _smooth_dataset, _smooth_band
    _smooth_dataset = gdal.Open(source_path, gdal.GA_ReadOnly)
    _smooth_band = _smooth_dataset.GetRasterBand(band_number)


def smooth_tile(task):
    (
        xoff, yoff, xsize, ysize, halo_x, halo_y,
        raster_x_size, raster_y_size,
        method_key, sigma_x, sigma_y, size_x, size_y,
        source_nodata, target_nodata,
    ) = task

    read_xoff = max(0, xoff - halo_x)
    read_yoff = max(0, yoff - halo_y)
    read_xend = min(raster_x_size, xoff + xsize + halo_x)
    read_yend = min(raster_y_size, yoff + ysize + halo_y)
    read_xsize = read_xend - read_xoff
    read_ysize = read_yend - read_yoff

    array = _smooth_band.ReadAsArray(read_xoff, read_yoff, read_xsize, read_ysize)
    array = np.asarray(array, dtype=np.float64)

    valid_mask = np.isfinite(array)
    if source_nodata is not None:
        valid_mask &= ~np.isclose(array, source_nodata)

    value_array = np.where(valid_mask, array, 0.0)
    weight_array = valid_mask.astype(np.float64)

    if method_key == "gaussian":
        filtered_values = gaussian_filter(value_array, sigma=(sigma_y, sigma_x), mode="nearest")
        filtered_weights = gaussian_filter(weight_array, sigma=(sigma_y, sigma_x), mode="nearest")
    else:
        filtered_values = uniform_filter(value_array, size=(size_y, size_x), mode="nearest")
        filtered_weights = uniform_filter(weight_array, size=(size_y, size_x), mode="nearest")

    smoothed = np.full_like(filtered_values, target_nodata, dtype=np.float64)
    nonzero_weights = filtered_weights > 1e-12
    smoothed[nonzero_weights] = filtered_values[nonzero_weights] / filtered_weights[nonzero_weights]

    inner_xoff = xoff - read_xoff
    inner_yoff = yoff - read_yoff
    inner = smoothed[
        inner_yoff : inner_yoff + ysize,
        inner_xoff : inner_xoff + xsize,
    ].astype(np.float32)
    return xoff, yoff, inner
