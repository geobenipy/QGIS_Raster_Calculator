"""
QGIS Plugin: Raster Processor
Full version with logging, XYZ file support, raster resampling,
smoothing, and optional extrapolation control.
Requirements:
- QGIS
- NumPy
- SciPy
- GDAL
"""

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import (
    QAction, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QDoubleSpinBox, QProgressBar, QFileDialog, QGroupBox, QRadioButton, QCheckBox,
    QTextEdit, QLineEdit, QMessageBox
)
from qgis.core import QgsVectorLayer, QgsRasterLayer, QgsProject, QgsMessageLog, Qgis
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree, Delaunay
from osgeo import gdal, osr
import numpy as np
import os
import re
import csv

PLUGIN_TAG = "RasterProcessor"
NODATA_VALUE = -9999.0


class RasterProcessorPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None

    def initGui(self):
        self.action = QAction("Raster Processor", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Raster Processor", self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removePluginMenu("&Raster Processor", self.action)
            self.iface.removeToolBarIcon(self.action)

    def run(self):
        dlg = RasterProcessorDialog(self.iface)
        dlg.exec_()


class RasterProcessorDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("Raster Processor")
        self.setMinimumWidth(820)
        self.setMinimumHeight(700)
        self.xyz_file_path = ""
        self.raster_file_path = ""
        self.smooth_raster_path = ""
        self.setup_ui()
        self.refresh_layers()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout()
        self.interpolate_radio = QRadioButton("Interpolate XYZ / Point Cloud")
        self.resample_radio = QRadioButton("Resample Raster")
        self.smooth_radio = QRadioButton("Smooth Raster")
        self.interpolate_radio.setChecked(True)
        mode_layout.addWidget(self.interpolate_radio)
        mode_layout.addWidget(self.resample_radio)
        mode_layout.addWidget(self.smooth_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Interpolation settings
        self.interp_group = QGroupBox("Interpolation Settings")
        interp_layout = QVBoxLayout()
        self.extrap_check = QCheckBox("Allow extrapolation")
        self.extrap_check.setChecked(False)
        interp_layout.addWidget(self.extrap_check)

        # Vector layer selection
        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Point Layer:"))
        self.vector_combo = QComboBox()
        vector_layout.addWidget(self.vector_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_layers)
        vector_layout.addWidget(self.refresh_btn)
        interp_layout.addLayout(vector_layout)

        # XYZ file selection
        xyz_layout = QHBoxLayout()
        xyz_layout.addWidget(QLabel("XYZ file (.txt/.csv/.xyz):"))
        self.xyz_path_edit = QLineEdit()
        self.xyz_path_edit.setReadOnly(True)
        xyz_layout.addWidget(self.xyz_path_edit)
        self.xyz_btn = QPushButton("Browse")
        self.xyz_btn.clicked.connect(self.select_xyz_file)
        xyz_layout.addWidget(self.xyz_btn)
        interp_layout.addLayout(xyz_layout)

        # Z-value field
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Z-value field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)
        interp_layout.addLayout(field_layout)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["IDW", "Linear", "Cubic", "Nearest", "RBF"])
        method_layout.addWidget(self.method_combo)
        interp_layout.addLayout(method_layout)

        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution (map units per pixel):"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.0001, 1e6)
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setDecimals(4)
        res_layout.addWidget(self.resolution_spin)
        interp_layout.addLayout(res_layout)

        self.interp_group.setLayout(interp_layout)
        layout.addWidget(self.interp_group)

        # Resample group
        self.resample_group = QGroupBox("Resample Settings")
        resample_layout = QVBoxLayout()
        raster_layout = QHBoxLayout()
        raster_layout.addWidget(QLabel("Raster Layer:"))
        self.raster_combo = QComboBox()
        raster_layout.addWidget(self.raster_combo)
        self.raster_refresh_btn = QPushButton("Refresh")
        self.raster_refresh_btn.clicked.connect(self.refresh_layers)
        raster_layout.addWidget(self.raster_refresh_btn)
        resample_layout.addLayout(raster_layout)

        raster_file_layout = QHBoxLayout()
        raster_file_layout.addWidget(QLabel("Or raster file:"))
        self.raster_path_edit = QLineEdit()
        self.raster_path_edit.setReadOnly(True)
        raster_file_layout.addWidget(self.raster_path_edit)
        self.raster_btn = QPushButton("Browse")
        self.raster_btn.clicked.connect(self.select_raster_file)
        raster_file_layout.addWidget(self.raster_btn)
        resample_layout.addLayout(raster_file_layout)

        factor_layout = QHBoxLayout()
        factor_layout.addWidget(QLabel("Scale factor:"))
        self.resample_factor_spin = QDoubleSpinBox()
        self.resample_factor_spin.setRange(0.1, 10.0)
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

        # Smooth group
        self.smooth_group = QGroupBox("Smooth Settings")
        smooth_layout = QVBoxLayout()
        smooth_raster_layout = QHBoxLayout()
        smooth_raster_layout.addWidget(QLabel("Raster Layer:"))
        self.smooth_raster_combo = QComboBox()
        smooth_raster_layout.addWidget(self.smooth_raster_combo)
        self.smooth_raster_refresh_btn = QPushButton("Refresh")
        self.smooth_raster_refresh_btn.clicked.connect(self.refresh_layers)
        smooth_raster_layout.addWidget(self.smooth_raster_refresh_btn)
        smooth_layout.addLayout(smooth_raster_layout)

        smooth_file_layout = QHBoxLayout()
        smooth_file_layout.addWidget(QLabel("Or raster file:"))
        self.smooth_raster_path_edit = QLineEdit()
        self.smooth_raster_path_edit.setReadOnly(True)
        smooth_file_layout.addWidget(self.smooth_raster_path_edit)
        self.smooth_raster_btn = QPushButton("Browse")
        self.smooth_raster_btn.clicked.connect(self.select_smooth_raster_file)
        smooth_file_layout.addWidget(self.smooth_raster_btn)
        smooth_layout.addLayout(smooth_file_layout)

        smooth_factor_layout = QHBoxLayout()
        smooth_factor_layout.addWidget(QLabel("Smooth factor:"))
        self.smooth_factor_spin = QDoubleSpinBox()
        self.smooth_factor_spin.setRange(0.1, 100.0)
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

        # Output selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.output_path = QLabel("Not set")
        output_layout.addWidget(self.output_path)
        self.output_btn = QPushButton("...")
        self.output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)

        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
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

        # Log window
        layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

        # Connect radio toggles
        self.interpolate_radio.toggled.connect(self.toggle_mode)
        self.resample_radio.toggled.connect(self.toggle_mode)
        self.smooth_radio.toggled.connect(self.toggle_mode)
        self.vector_combo.currentIndexChanged.connect(self.update_fields)

    def log(self, message, level=Qgis.Info):
        QgsMessageLog.logMessage(str(message), PLUGIN_TAG, level)
        self.log_text.append(str(message))
        self.iface.messageBar().pushMessage(PLUGIN_TAG, str(message), level, 3)

    def toggle_mode(self):
        self.interp_group.setVisible(self.interpolate_radio.isChecked())
        self.resample_group.setVisible(self.resample_radio.isChecked())
        self.smooth_group.setVisible(self.smooth_radio.isChecked())

    # -----------------------------
    # Layer / file helpers
    # -----------------------------
    def refresh_layers(self):
        self.populate_vector_layers()
        self.populate_raster_layers()
        self.populate_smooth_layers()
        self.log("Layer lists refreshed.")

    def populate_vector_layers(self):
        self.vector_combo.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsVectorLayer):
                self.vector_combo.addItem(layer.name(), layer)
        self.update_fields()

    def populate_raster_layers(self):
        self.raster_combo.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                self.raster_combo.addItem(layer.name(), layer)

    def populate_smooth_layers(self):
        self.smooth_raster_combo.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                self.smooth_raster_combo.addItem(layer.name(), layer)

    def update_fields(self):
        self.field_combo.clear()
        layer = self.vector_combo.currentData()
        if not layer:
            return
        for field in layer.fields():
            if field.type() in (QVariant.Int, QVariant.Double, QVariant.LongLong, QVariant.UInt, QVariant.ULongLong):
                self.field_combo.addItem(field.name())
        if self.field_combo.count() == 0:
            for field in layer.fields():
                self.field_combo.addItem(field.name())

    def select_output(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Output raster", "", "GeoTIFF (*.tif)")
        if filename:
            if not filename.lower().endswith(".tif"):
                filename += ".tif"
            self.output_path.setText(filename)

    def select_xyz_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select XYZ file", "", "XYZ/Text/CSV (*.txt *.csv *.xyz);;All files (*)")
        if filename:
            self.xyz_file_path = filename
            self.xyz_path_edit.setText(filename)
            self.log(f"Selected XYZ file: {filename}")

    def select_raster_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select raster file", "", "Raster files (*.tif *.tiff *.img *.asc *.grd);;All files (*)")
        if filename:
            self.raster_file_path = filename
            self.raster_path_edit.setText(filename)
            self.log(f"Selected raster file: {filename}")

    def select_smooth_raster_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select raster file", "", "Raster files (*.tif *.tiff *.img *.asc *.grd);;All files (*)")
        if filename:
            self.smooth_raster_path = filename
            self.smooth_raster_path_edit.setText(filename)
            self.log(f"Selected raster for smoothing: {filename}")

    # -----------------------------
    # Main process
    # -----------------------------
    def process(self):
        try:
            output = self.output_path.text()
            if not output or output == "Not set":
                raise ValueError("Please select an output file.")

            self.run_btn.setEnabled(False)
            self.progress.setValue(0)

            if self.interpolate_radio.isChecked():
                self.process_interpolation(output)
            elif self.resample_radio.isChecked():
                self.process_resample(output)
            else:
                self.process_smooth(output)

            self.progress.setValue(100)
            self.log(f"Processing completed: {output}")

            # Load output into QGIS
            layer = QgsRasterLayer(output, os.path.basename(output))
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.log("Output added to project.")
            else:
                self.log("Output created but could not be loaded.", Qgis.Warning)

        except Exception as e:
            self.log(f"Error: {e}", Qgis.Critical)
            QMessageBox.critical(self, "Raster Processor", str(e))
        finally:
            self.run_btn.setEnabled(True)

    # -----------------------------
    # Interpolation
    # -----------------------------
    def process_interpolation(self, output_path):
        """Interpolates XYZ or vector points to raster with optional extrapolation."""
        # --- Load points ---
        points = []
        values = []

        layer = self.vector_combo.currentData()
        z_field = self.field_combo.currentText()
        if layer:
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom.isEmpty():
                    continue
                if geom.isMultipart():
                    pts = geom.asMultiPoint()
                else:
                    pts = [geom.asPoint()]
                for pt in pts:
                    val = feat[z_field]
                    if val is None or val == NODATA_VALUE:
                        continue
                    points.append([pt.x(), pt.y()])
                    values.append(val)
        elif self.xyz_file_path:
            with open(self.xyz_file_path, "r") as f:
                for line in f:
                    toks = re.split(r"[\s,;]+", line.strip())
                    if len(toks) < 3:
                        continue
                    try:
                        x, y, z = map(float, toks[:3])
                        if z != NODATA_VALUE:
                            points.append([x, y])
                            values.append(z)
                    except:
                        continue
        else:
            raise ValueError("No input points selected.")

        points = np.array(points)
        values = np.array(values, dtype=np.float32)
        if len(points) < 3:
            raise ValueError("Not enough points for interpolation.")

        # --- Create grid ---
        resolution = float(self.resolution_spin.value())
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        xi = np.arange(xmin, xmax + resolution, resolution)
        yi = np.arange(ymin, ymax + resolution, resolution)
        grid_x, grid_y = np.meshgrid(xi, yi)

        self.log(f"Grid size: {grid_x.shape[1]} x {grid_x.shape[0]} pixels")

        # --- Interpolation ---
        method = self.method_combo.currentText().lower()
        extrapolate = self.extrap_check.isChecked()
        grid_z = np.full_like(grid_x, NODATA_VALUE, dtype=np.float32)

        if method == "idw":
            k = min(8, len(points))
            tree = cKDTree(points)
            xi_flat = np.c_[grid_x.ravel(), grid_y.ravel()]
            dists, idx = tree.query(xi_flat, k=k)
            dists[dists == 0] = 1e-12
            weights = 1.0 / (dists ** 2)
            vals = np.take(values, idx)
            zi_flat = np.sum(weights * vals, axis=1) / np.sum(weights, axis=1)
            if not extrapolate:
                hull = Delaunay(points)
                mask = hull.find_simplex(xi_flat) < 0
                zi_flat[mask] = NODATA_VALUE
            grid_z = zi_flat.reshape(grid_x.shape)

        elif method in ["linear", "cubic", "nearest"]:
            zi = griddata(points, values, (grid_x, grid_y), method=method, fill_value=NODATA_VALUE)
            grid_z = zi.astype(np.float32)
            if not extrapolate:
                hull = Delaunay(points)
                mask = hull.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()]) < 0
                grid_z.ravel()[mask] = NODATA_VALUE

        elif method == "rbf":
            rbf = RBFInterpolator(points, values, neighbors=15, smoothing=0.0)
            zi_flat = rbf(np.c_[grid_x.ravel(), grid_y.ravel()])
            grid_z = zi_flat.reshape(grid_x.shape).astype(np.float32)
            if not extrapolate:
                hull = Delaunay(points)
                mask = hull.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()]) < 0
                grid_z.ravel()[mask] = NODATA_VALUE

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # --- Save raster ---
        self.save_raster(output_path, grid_z, xmin, ymax, resolution)

    # -----------------------------
    # Resample
    # -----------------------------
    def process_resample(self, output_path):
        """Resamples raster by scale factor."""
        layer = self.raster_combo.currentData()
        raster_file = self.raster_file_path if self.raster_file_path else None
        if layer:
            src = gdal.Open(layer.source())
        elif raster_file:
            src = gdal.Open(raster_file)
        else:
            raise ValueError("No raster selected for resampling.")

        factor = float(self.resample_factor_spin.value())
        method = self.resample_method_combo.currentText().lower()

        # Map method to GDAL
        method_map = {"nearest": gdal.GRA_NearestNeighbour,
                      "bilinear": gdal.GRA_Bilinear,
                      "cubic": gdal.GRA_Cubic,
                      "average": gdal.GRA_Average,
                      "lanczos": gdal.GRA_Lanczos}
        resample_alg = method_map.get(method, gdal.GRA_Bilinear)

        src_band = src.GetRasterBand(1)
        xsize = int(src.RasterXSize * factor)
        ysize = int(src.RasterYSize * factor)

        geotransform = list(src.GetGeoTransform())
        geotransform[1] /= factor
        geotransform[5] /= factor

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(output_path, xsize, ysize, 1, gdal.GDT_Float32)
        dst.SetGeoTransform(geotransform)
        dst.SetProjection(src.GetProjection())

        gdal.ReprojectImage(src, dst, src.GetProjection(), src.GetProjection(), resample_alg)
        dst = None
        src = None
        self.log(f"Resampled raster saved: {output_path}")

    # -----------------------------
    # Smoothing
    # -----------------------------
    def process_smooth(self, output_path):
        """Smooth raster using Gaussian or Uniform filter."""
        layer = self.smooth_raster_combo.currentData()
        raster_file = self.smooth_raster_path if self.smooth_raster_path else None
        if layer:
            src = gdal.Open(layer.source())
        elif raster_file:
            src = gdal.Open(raster_file)
        else:
            raise ValueError("No raster selected for smoothing.")

        method = self.smooth_method_combo.currentText()
        factor = float(self.smooth_factor_spin.value())
        band = src.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float32)

        if method == "Gaussian":
            smooth_arr = gaussian_filter(arr, sigma=factor)
        else:
            smooth_arr = uniform_filter(arr, size=int(factor))

        geotransform = src.GetGeoTransform()
        proj = src.GetProjection()
        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(output_path, src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Float32)
        dst.SetGeoTransform(geotransform)
        dst.SetProjection(proj)
        dst.GetRasterBand(1).WriteArray(smooth_arr)
        dst.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
        dst = None
        src = None
        self.log(f"Smoothed raster saved: {output_path}")

    # -----------------------------
    # Utility
    # -----------------------------
    def save_raster(self, path, array, xmin, ymax, resolution):
        nrows, ncols = array.shape
        geotransform = (xmin, resolution, 0, ymax, 0, -resolution)
        driver = gdal.GetDriverByName("GTiff")
        out_raster = driver.Create(path, ncols, nrows, 1, gdal.GDT_Float32)
        out_raster.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        out_raster.SetProjection(srs.ExportToWkt())
        outband = out_raster.GetRasterBand(1)
        outband.WriteArray(array)
        outband.SetNoDataValue(NODATA_VALUE)
        outband.FlushCache()
        out_raster = None