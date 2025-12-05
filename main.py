"""
QGIS Plugin: Raster Processor
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtWidgets import QAction, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QProgressBar, QFileDialog, QGroupBox, QRadioButton, QCheckBox
from qgis.PyQt.QtGui import QIcon
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterRasterLayer, QgsProcessingParameterNumber,
                       QgsProcessingParameterRasterDestination, QgsProcessingParameterEnum,
                       QgsProcessingParameterField, QgsRasterLayer, QgsVectorLayer,
                       QgsProject, QgsMessageLog, Qgis, QgsRasterBlock, QgsCoordinateReferenceSystem)
from qgis import processing
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter, uniform_filter
from osgeo import gdal, osr
import multiprocessing as mp
from functools import partial
import os

class RasterProcessorPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        

    def initGui(self):
        self.action = QAction("Advanced Raster Processor", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Raster Processor", self.action)
        

    def unload(self):
        self.iface.removePluginMenu("&Raster Processor", self.action)
        self.iface.removeToolBarIcon(self.action)
        

    def run(self):
        dlg = RasterProcessorDialog(self.iface)
        dlg.exec_()


class RasterProcessorDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("Advanced Raster Processor")
        self.setMinimumWidth(600)
        self.setup_ui()
        

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Mode selection
        mode_group = QGroupBox("Processing")
        mode_layout = QVBoxLayout()
        self.interpolate_radio = QRadioButton("Interpolate point cloud")
        self.resample_radio = QRadioButton("Resample raster")
        self.smooth_radio = QRadioButton("Smooth raster")
        self.interpolate_radio.setChecked(True)
        mode_layout.addWidget(self.interpolate_radio)
        mode_layout.addWidget(self.resample_radio)
        mode_layout.addWidget(self.smooth_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Interpolation options
        self.interp_group = QGroupBox("Interpolation settings")
        interp_layout = QVBoxLayout()

        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Point layer:"))
        self.vector_combo = QComboBox()
        vector_layout.addWidget(self.vector_combo)
        interp_layout.addLayout(vector_layout)

        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Z-value field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)
        interp_layout.addLayout(field_layout)

        self.populate_vector_layers()
        self.vector_combo.currentIndexChanged.connect(self.update_fields)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["IDW (fast), Cubic, Linear, Nearest, RBF (precise)"])
        method_layout.addWidget(self.method_combo)
        interp_layout.addLayout(method_layout)
        
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution (m):"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.1, 1000)
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setDecimals(2)
        res_layout.addWidget(self.resolution_spin)
        interp_layout.addLayout(res_layout)
        
        self.interp_group.setLayout(interp_layout)
        layout.addWidget(self.interp_group)
        
        # Resample options
        self.resample_group = QGroupBox("Resample settings")
        resample_layout = QVBoxLayout()
        
        raster_layout = QHBoxLayout()
        raster_layout.addWidget(QLabel("Raster:"))
        self.raster_combo = QComboBox()
        self.populate_raster_layers()
        raster_layout.addWidget(self.raster_combo)
        resample_layout.addLayout(raster_layout)
        
        factor_layout = QHBoxLayout()
        factor_layout.addWidget(QLabel("Resample factor:"))
        self.resample_factor_spin = QDoubleSpinBox()
        self.resample_factor_spin.setRange(0.1, 10)
        self.resample_factor_spin.setValue(2.0)
        self.resample_factor_spin.setDecimals(2)
        factor_layout.addWidget(self.resample_factor_spin)
        resample_layout.addLayout(factor_layout)
        
        resample_method_layout = QHBoxLayout()
        resample_method_layout.addWidget(QLabel("Method:"))
        self.resample_method_combo = QComboBox()
        self.resample_method_combo.addItems(["Bilinear", "Cubic", "Nearest", "Average", "Lanczos"])
        resample_method_layout.addWidget(self.resample_method_combo)
        resample_layout.addLayout(resample_method_layout)
        
        self.resample_group.setLayout(resample_layout)
        self.resample_group.setVisible(False)
        layout.addWidget(self.resample_group)
        
        # Smooth options
        self.smooth_group = QGroupBox("Smooth settings")
        smooth_layout = QVBoxLayout()
        
        smooth_raster_layout = QHBoxLayout()
        smooth_raster_layout.addWidget(QLabel("Raster:"))
        self.smooth_raster_combo = QComboBox()
        self.populate_raster_layers_smooth()
        smooth_raster_layout.addWidget(self.smooth_raster_combo)
        smooth_layout.addLayout(smooth_raster_layout)
        
        smooth_factor_layout = QHBoxLayout()
        smooth_factor_layout.addWidget(QLabel("Smooth factor:"))
        self.smooth_factor_spin = QDoubleSpinBox()
        self.smooth_factor_spin.setRange(0.1, 20)
        self.smooth_factor_spin.setValue(1.0)
        self.smooth_factor_spin.setDecimals(2)
        smooth_factor_layout.addWidget(self.smooth_factor_spin)
        smooth_layout.addLayout(smooth_factor_layout)
        
        smooth_method_layout = QHBoxLayout()
        smooth_method_layout.addWidget(QLabel("Method:"))
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItems(["Gaussian", "Uniform (Mean)"])
        smooth_method_layout.addWidget(self.smooth_method_combo)
        smooth_layout.addLayout(smooth_method_layout)
        
        self.smooth_group.setLayout(smooth_layout)
        self.smooth_group.setVisible(False)
        layout.addWidget(self.smooth_group)
        
        # Output
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.output_path = QLabel("Not set")
        output_layout.addWidget(self.output_path)
        self.output_btn = QPushButton("...")
        self.output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)
        
        # Parallel processing
        parallel_layout = QHBoxLayout()
        self.parallel_check = QCheckBox("Parallel processing")
        self.parallel_check.setChecked(True)
        parallel_layout.addWidget(self.parallel_check)
        parallel_layout.addWidget(QLabel("Cores:"))
        self.cores_spin = QSpinBox()
        self.cores_spin.setRange(1, mp.cpu_count())
        self.cores_spin.setValue(max(1, mp.cpu_count() - 1))
        parallel_layout.addWidget(self.cores_spin)
        layout.addLayout(parallel_layout)
        
        # Progress
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Process")
        self.run_btn.clicked.connect(self.process)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect radio buttons
        self.interpolate_radio.toggled.connect(self.toggle_mode)
        self.resample_radio.toggled.connect(self.toggle_mode)
        self.smooth_radio.toggled.connect(self.toggle_mode)
        

    def toggle_mode(self):
        self.interp_group.setVisible(self.interpolate_radio.isChecked())
        self.resample_group.setVisible(self.resample_radio.isChecked())
        self.smooth_group.setVisible(self.smooth_radio.isChecked())
        

    def populate_vector_layers(self):
        self.vector_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                self.vector_combo.addItem(layer.name(), layer)
        self.update_fields()


    def populate_raster_layers(self):
        self.raster_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.raster_combo.addItem(layer.name(), layer)


    def populate_raster_layers_smooth(self):
        self.smooth_raster_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsRasterLayer):
                self.smooth_raster_combo.addItem(layer.name(), layer)


    def update_fields(self):
        self.field_combo.clear()
        layer = self.vector_combo.currentData()
        if layer:
            for field in layer.fields():
                if field.type() in [QVariant.Int, QVariant.Double]:
                    self.field_combo.addItem(field.name())


    def select_output(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Output raster", "", "GeoTIFF (*.tif)")
        if filename:
            self.output_path.setText(filename)


    def process(self):
        output = self.output_path.text()
        if output == "Not set":
            self.iface.messageBar().pushMessage("Error", "Please select an output file", Qgis.Critical)
            return
            
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        
        try:
            if self.interpolate_radio.isChecked():
                self.process_interpolation(output)
            elif self.resample_radio.isChecked():
                self.process_resample(output)
            else:
                self.process_smooth(output)
                
            self.iface.messageBar().pushMessage("Success", "Processing completed", Qgis.Success)
            layer = QgsRasterLayer(output, os.path.basename(output))
            QgsProject.instance().addMapLayer(layer)
        except Exception as e:
            self.iface.messageBar().pushMessage("Error", str(e), Qgis.Critical)
        finally:
            self.run_btn.setEnabled(True)
            self.progress.setValue(100)


    def process_interpolation(self, output):
        layer = self.vector_combo.currentData()
        field = self.field_combo.currentText()
        resolution = self.resolution_spin.value()
        method_idx = self.method_combo.currentIndex()
        
        # Extract points - chunked for large datasets
        self.progress.setValue(10)
        chunk_size = 100000
        points_list = []
        values_list = []
        
        total_features = layer.featureCount()
        features = layer.getFeatures()
        
        for i, feat in enumerate(features):
            if i % chunk_size == 0 and i > 0:
                self.iface.messageBar().pushMessage("Info", 
                    f"Loading points: {i}/{total_features}", Qgis.Info, 2)
            
            geom = feat.geometry()
            if geom.isMultipart():
                pts = geom.asMultiPoint()
            else:
                pts = [geom.asPoint()]
            for pt in pts:
                points_list.append([pt.x(), pt.y()])
                values_list.append(feat[field])
        
        points = np.array(points_list, dtype=np.float32)
        values = np.array(values_list, dtype=np.float32)
        del points_list, values_list
        
        # Calculate grid dimensions
        self.progress.setValue(30)
        extent = layer.extent()
        x_min, x_max = extent.xMinimum(), extent.xMaximum()
        y_min, y_max = extent.yMinimum(), extent.yMaximum()
        
        x_res = int((x_max - x_min) / resolution)
        y_res = int((y_max - y_min) / resolution)
        
        # Tile-based processing for large rasters
        tile_size = 2048
        n_tiles_x = (x_res + tile_size - 1) // tile_size
        n_tiles_y = (y_res + tile_size - 1) // tile_size
        
        self.iface.messageBar().pushMessage("Info", 
            f"Processing {n_tiles_x * n_tiles_y} tiles ({x_res}x{y_res} pixels)", Qgis.Info, 3)
        
        # Prepare raster file
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output, x_res, y_res, 1, gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 
                                        'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
        
        geotransform = (x_min, resolution, 0, y_max, 0, -resolution)
        out_ds.SetGeoTransform(geotransform)
        
        srs = osr.SpatialReference()
        srs.ImportFromWkt(layer.crs().toWkt())
        out_ds.SetProjection(srs.ExportToWkt())
        
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(-9999)
        
        # Tile-wise interpolation
        methods = ['linear', 'cubic', 'linear', 'nearest', 'rbf']
        method = methods[method_idx]
        
        total_tiles = n_tiles_x * n_tiles_y
        processed_tiles = 0
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Compute tile boundaries
                x_start = tx * tile_size
                y_start = ty * tile_size
                x_end = min((tx + 1) * tile_size, x_res)
                y_end = min((ty + 1) * tile_size, y_res)
                
                tile_x_size = x_end - x_start
                tile_y_size = y_end - y_start
                
                # Tile coordinates
                xi = np.linspace(x_min + x_start * resolution, 
                                 x_min + x_end * resolution, tile_x_size, dtype=np.float32)
                yi = np.linspace(y_max - y_start * resolution, 
                                 y_max - y_end * resolution, tile_y_size, dtype=np.float32)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                
                # Only relevant points for this tile (with buffer)
                buffer = resolution * 10
                tile_x_min = xi.min() - buffer
                tile_x_max = xi.max() + buffer
                tile_y_min = yi.min() - buffer
                tile_y_max = yi.max() + buffer
                
                mask = ((points[:, 0] >= tile_x_min) & (points[:, 0] <= tile_x_max) &
                        (points[:, 1] >= tile_y_min) & (points[:, 1] <= tile_y_max))
                
                tile_points = points[mask]
                tile_values = values[mask]
                
                if len(tile_points) == 0:
                    zi_tile = np.full((tile_y_size, tile_x_size), -9999, dtype=np.float32)
                else:
                    # Interpolation for the tile
                    if method_idx == 0:  # IDW - optimized
                        zi_tile = self.idw_interpolation_tile(tile_points, tile_values, 
                                                              xi_grid, yi_grid, power=2)
                    elif method == 'rbf':
                        if len(tile_points) > 10000:
                            # Subsample for very large point sets
                            sample_idx = np.random.choice(len(tile_points), 10000, replace=False)
                            tile_points_sub = tile_points[sample_idx]
                            tile_values_sub = tile_values[sample_idx]
                        else:
                            tile_points_sub = tile_points
                            tile_values_sub = tile_values
                        
                        rbf = RBFInterpolator(tile_points_sub, tile_values_sub, 
                                              kernel='thin_plate_spline', smoothing=0.1)
                        grid_points = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
                        zi_tile = rbf(grid_points).reshape(xi_grid.shape).astype(np.float32)
                    else:
                        zi_tile = griddata(tile_points, tile_values, (xi_grid, yi_grid), 
                                           method=method, fill_value=-9999).astype(np.float32)
                
                # Write tile
                out_band.WriteArray(zi_tile, x_start, y_start)
                
                processed_tiles += 1
                progress = 50 + int(30 * processed_tiles / total_tiles)
                self.progress.setValue(progress)
                
                if processed_tiles % 10 == 0:
                    self.iface.messageBar().pushMessage("Info", 
                        f"Tile {processed_tiles}/{total_tiles}", Qgis.Info, 1)
                
                del zi_tile, xi_grid, yi_grid, tile_points, tile_values
        
        out_band.FlushCache()
        out_ds = None
        self.progress.setValue(80)
        

    def idw_interpolation_tile(self, points, values, xi, yi, power=2):
        """IDW for a single tile - memory-efficient"""
        grid_shape = xi.shape
        xi_flat = xi.ravel()
        yi_flat = yi.ravel()
        zi = np.full(len(xi_flat), -9999, dtype=np.float32)
        
        # Batch processing in smaller chunks
        batch_size = 10000
        n_batches = (len(xi_flat) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(xi_flat))
            
            xi_batch = xi_flat[start_idx:end_idx]
            yi_batch = yi_flat[start_idx:end_idx]
            
            # Vectorized distance computation
            xi_reshaped = xi_batch[:, np.newaxis]
            yi_reshaped = yi_batch[:, np.newaxis]
            
            distances = np.sqrt((points[:, 0] - xi_reshaped)**2 + 
                                (points[:, 1] - yi_reshaped)**2)
            
            # Use only nearest N points for performance
            k = min(12, len(points))
            nearest_idx = np.argpartition(distances, k, axis=1)[:, :k]
            
            for i, idx_array in enumerate(nearest_idx):
                dists = distances[i, idx_array]
                vals = values[idx_array]
                
                if np.min(dists) < 1e-10:
                    zi[start_idx + i] = vals[np.argmin(dists)]
                else:
                    weights = 1.0 / (dists**power)
                    zi[start_idx + i] = np.sum(weights * vals) / np.sum(weights)
        
        return zi.reshape(grid_shape)
        

    def process_resample(self, output):
        layer = self.raster_combo.currentData()
        factor = self.resample_factor_spin.value()
        method_idx = self.resample_method_combo.currentIndex()
        
        methods_gdal = [gdal.GRA_Bilinear, gdal.GRA_Cubic, gdal.GRA_NearestNeighbour, 
                        gdal.GRA_Average, gdal.GRA_Lanczos]
        method = methods_gdal[method_idx]
        
        self.progress.setValue(30)
        
        # GDAL Warp with optimized options for large files
        src_ds = gdal.Open(layer.source())
        width = int(src_ds.RasterXSize * factor)
        height = int(src_ds.RasterYSize * factor)
        
        self.iface.messageBar().pushMessage("Info", 
            f"Resampling to {width}x{height} pixels", Qgis.Info, 3)
        
        self.progress.setValue(60)
        
        warp_options = gdal.WarpOptions(
            format='GTiff',
            width=width,
            height=height,
            resampleAlg=method,
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 
                             'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
            warpMemoryLimit=512,  # MB - limits RAM usage
            multithread=True,
            callback=gdal.TermProgress_nocb
        )
        
        gdal.Warp(output, src_ds, options=warp_options)
        src_ds = None
        
        self.progress.setValue(80)
        
        
    def process_smooth(self, output):
        layer = self.smooth_raster_combo.currentData()
        factor = self.smooth_factor_spin.value()
        method = self.smooth_method_combo.currentIndex()
        
        self.progress.setValue(30)
        
        ds = gdal.Open(layer.source())
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        
        x_size = ds.RasterXSize
        y_size = ds.RasterYSize
        
        # Prepare output
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output, x_size, y_size, 1, gdal.GDT_Float32, 
                               options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES',
                                        'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)
        
        # Tile-based processing
        tile_size = 1024
        kernel_size = int(2 * factor + 1)
        overlap = kernel_size * 2
        
        n_tiles_x = (x_size + tile_size - 1) // tile_size
        n_tiles_y = (y_size + tile_size - 1) // tile_size
        total_tiles = n_tiles_x * n_tiles_y
        processed = 0
        
        self.iface.messageBar().pushMessage("Info", 
            f"Processing {total_tiles} tiles for smoothing", Qgis.Info, 3)
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Load tile with overlap
                x_start = max(0, tx * tile_size - overlap)
                y_start = max(0, ty * tile_size - overlap)
                x_end = min(x_size, (tx + 1) * tile_size + overlap)
                y_end = min(y_size, (ty + 1) * tile_size + overlap)
                
                tile_data = band.ReadAsArray(x_start, y_start, 
                                             x_end - x_start, y_end - y_start)
                
                if tile_data is None:
                    continue
                
                # Mask NoData
                if nodata is not None:
                    mask = (tile_data == nodata)
                else:
                    mask = None
                
                tile_data_float = tile_data.astype(np.float32)
                
                if method == 0:  # Gaussian
                    smoothed = gaussian_filter(tile_data_float, sigma=factor)
                else:  # Uniform
                    smoothed = uniform_filter(tile_data_float, size=kernel_size)
                
                if mask is not None:
                    smoothed[mask] = nodata
                
                # Write only actual tile area (without overlap)
                write_x_start = tx * tile_size
                write_y_start = ty * tile_size
                write_x_size = min(tile_size, x_size - write_x_start)
                write_y_size = min(tile_size, y_size - write_y_start)
                
                offset_x = write_x_start - x_start
                offset_y = write_y_start - y_start
                
                output_tile = smoothed[offset_y:offset_y + write_y_size, 
                                      offset_x:offset_x + write_x_size]
                
                out_band.WriteArray(output_tile, write_x_start, write_y_start)
                
                processed += 1
                progress = 50 + int(30 * processed / total_tiles)
                self.progress.setValue(progress)
                
                if processed % 10 == 0:
                    self.iface.messageBar().pushMessage("Info", 
                        f"Smoothing tile {processed}/{total_tiles}", Qgis.Info, 1)
                
                del tile_data, smoothed, output_tile
        
        out_band.FlushCache()
        out_ds = None
        ds = None
        self.progress.setValue(80)
        

    def save_raster(self, data, filename, x_min, y_max, resolution, crs):
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data.shape
        
        out_ds = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
        
        geotransform = (x_min, resolution, 0, y_max, 0, -resolution)
        out_ds.SetGeoTransform(geotransform)
        
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs.toWkt())
        out_ds.SetProjection(srs.ExportToWkt())
        
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.SetNoDataValue(-9999)
        out_band.FlushCache()
        out_ds = None


def classFactory(iface):
    return RasterProcessorPlugin(iface)
