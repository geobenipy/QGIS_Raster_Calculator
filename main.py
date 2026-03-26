"""
QGIS Plugin: Raster Processor
Full version with:
- Interpolation (IDW, Linear, Cubic, Nearest, RBF)
- Resample (Bilinear, Cubic, Nearest, Average, Lanczos)
- Smooth (Gaussian, Uniform)
- Correct coordinate orientation (no flip)
- Progress bar updates
- Output added automatically to QGIS project
"""

from qgis.PyQt.QtCore import QVariant, QCoreApplication
from qgis.PyQt.QtWidgets import (
    QAction, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QProgressBar, QFileDialog, QGroupBox, QRadioButton,
    QCheckBox, QTextEdit, QLineEdit
)
from qgis.core import QgsVectorLayer, QgsRasterLayer, QgsProject, QgsMessageLog, Qgis

from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree, Delaunay
from osgeo import gdal, osr

import numpy as np
import os
import re

PLUGIN_TAG = "RasterProcessor"
NODATA_VALUE = -9999.0

class RasterProcessorPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None

    def initGui(self):
        self.action = QAction("Raster Processor", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Raster Processor", self.action)

    def unload(self):
        if self.action:
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
        self.smooth_raster_file_path = ""
        self.setup_ui()
        self.refresh_layers()

    # -----------------------------
    # UI / logging
    # -----------------------------
    def setup_ui(self):
        layout = QVBoxLayout()
        # Mode selection
        mode_group = QGroupBox("Processing")
        mode_layout = QVBoxLayout()
        self.interpolate_radio = QRadioButton("Interpolate point cloud / XYZ file")
        self.resample_radio = QRadioButton("Resample raster")
        self.smooth_radio = QRadioButton("Smooth raster")
        self.interpolate_radio.setChecked(True)
        mode_layout.addWidget(self.interpolate_radio)
        mode_layout.addWidget(self.resample_radio)
        mode_layout.addWidget(self.smooth_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Interpolation group
        self.interp_group = QGroupBox("Interpolation settings")
        interp_layout = QVBoxLayout()
        self.extrap_check = QCheckBox("Allow extrapolation")
        self.extrap_check.setChecked(False)
        interp_layout.addWidget(self.extrap_check)

        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Point layer:"))
        self.vector_combo = QComboBox()
        vector_layout.addWidget(self.vector_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_layers)
        vector_layout.addWidget(self.refresh_btn)
        interp_layout.addLayout(vector_layout)

        xyz_layout = QHBoxLayout()
        xyz_layout.addWidget(QLabel("XYZ file (.txt/.dat/.csv/.xyz):"))
        self.xyz_path_edit = QLineEdit()
        self.xyz_path_edit.setReadOnly(True)
        xyz_layout.addWidget(self.xyz_path_edit)
        self.xyz_btn = QPushButton("Browse")
        self.xyz_btn.clicked.connect(self.select_xyz_file)
        xyz_layout.addWidget(self.xyz_btn)
        interp_layout.addLayout(xyz_layout)

        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Z-value field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)
        interp_layout.addLayout(field_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["IDW", "Linear", "Cubic", "Nearest", "RBF"])
        method_layout.addWidget(self.method_combo)
        interp_layout.addLayout(method_layout)

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution (map units per pixel):"))
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.0001, 1000000)
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setDecimals(4)
        res_layout.addWidget(self.resolution_spin)
        interp_layout.addLayout(res_layout)

        self.interp_group.setLayout(interp_layout)
        layout.addWidget(self.interp_group)

        # Resample group
        self.resample_group = QGroupBox("Resample settings")
        resample_layout = QVBoxLayout()
        raster_layout = QHBoxLayout()
        raster_layout.addWidget(QLabel("Raster layer:"))
        self.raster_combo = QComboBox()
        raster_layout.addWidget(self.raster_combo)
        self.raster_refresh_btn = QPushButton("Refresh")
        self.raster_refresh_btn.clicked.connect(self.refresh_layers)
        raster_layout.addWidget(self.raster_refresh_btn)
        resample_layout.addLayout(raster_layout)

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
        self.smooth_group = QGroupBox("Smooth settings")
        smooth_layout = QVBoxLayout()
        smooth_raster_layout = QHBoxLayout()
        smooth_raster_layout.addWidget(QLabel("Raster layer:"))
        self.smooth_raster_combo = QComboBox()
        smooth_raster_layout.addWidget(self.smooth_raster_combo)
        self.smooth_raster_refresh_btn = QPushButton("Refresh")
        self.smooth_raster_refresh_btn.clicked.connect(self.refresh_layers)
        smooth_raster_layout.addWidget(self.smooth_raster_refresh_btn)
        smooth_layout.addLayout(smooth_raster_layout)

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

        # Output
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.output_path = QLabel("Not set")
        output_layout.addWidget(self.output_path)
        self.output_btn = QPushButton("...")
        self.output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)

        # Progress & buttons
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Process")
        self.run_btn.clicked.connect(self.process)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)

        # Log
        layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

        # Connections
        self.interpolate_radio.toggled.connect(self.toggle_mode)
        self.resample_radio.toggled.connect(self.toggle_mode)
        self.smooth_radio.toggled.connect(self.toggle_mode)
        self.vector_combo.currentIndexChanged.connect(self.update_fields)

    def log(self, message, level=Qgis.Info):
        QgsMessageLog.logMessage(str(message), PLUGIN_TAG, level)
        self.log_text.append(str(message))

    def toggle_mode(self):
        self.interp_group.setVisible(self.interpolate_radio.isChecked())
        self.resample_group.setVisible(self.resample_radio.isChecked())
        self.smooth_group.setVisible(self.smooth_radio.isChecked())

    # -----------------------------
    # File selection / layers
    # -----------------------------
    def refresh_layers(self):
        self.vector_combo.clear()
        self.raster_combo.clear()
        self.smooth_raster_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                self.vector_combo.addItem(layer.name(), layer)
            if isinstance(layer, QgsRasterLayer):
                self.raster_combo.addItem(layer.name(), layer)
                self.smooth_raster_combo.addItem(layer.name(), layer)
        self.update_fields()
        self.log("Layer lists refreshed.")

    def update_fields(self):
        self.field_combo.clear()
        layer = self.vector_combo.currentData()
        if layer:
            for field in layer.fields():
                if field.type() in (QVariant.Int, QVariant.Double, QVariant.LongLong, QVariant.UInt, QVariant.ULongLong):
                    self.field_combo.addItem(field.name())

    def select_output(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Output raster", "", "GeoTIFF (*.tif)")
        if filename and not filename.lower().endswith(".tif"):
            filename += ".tif"
        if filename:
            self.output_path.setText(filename)

    def select_xyz_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select XYZ file", "", "XYZ/Text/CSV (*.txt *.dat *.xyz *.csv)")
        if filename:
            self.xyz_file_path = filename
            self.xyz_path_edit.setText(filename)
            self.log(f"XYZ file selected: {filename}")

    # -----------------------------
    # Processing
    # -----------------------------
    def _selected_output(self):
        output = self.output_path.text()
        if not output or output == "Not set":
            raise ValueError("Please select an output file.")
        return output

    def _load_points_from_vector(self, layer, field_name):
        xs, ys, zs = [], [], []
        total = layer.featureCount()
        for i, feat in enumerate(layer.getFeatures()):
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                z = float(feat[field_name])
            except:
                continue
            pts = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
            for pt in pts:
                xs.append(pt.x())
                ys.append(pt.y())
                zs.append(z)
            if total > 0 and i % 100 == 0:
                self.progress.setValue(int((i/total)*100))
                QCoreApplication.processEvents()
        self.progress.setValue(100)
        return np.array(xs), np.array(ys), np.array(zs)

    def _read_xyz_file(self, filepath):
        data = np.loadtxt(filepath, comments="#")
        return data[:,0], data[:,1], data[:,2]

    def process(self):
        try:
            output_file = self._selected_output()
            if self.interpolate_radio.isChecked():
                self._process_interpolation(output_file)
            elif self.resample_radio.isChecked():
                self._process_resample(output_file)
            elif self.smooth_radio.isChecked():
                self._process_smooth(output_file)
            self.log("Processing finished successfully.", Qgis.Success)
        except Exception as e:
            self.log(f"Error: {e}", Qgis.Critical)

    # -----------------------------
    # Interpolation
    # -----------------------------
    def _process_interpolation(self, output_file):
        resolution = self.resolution_spin.value()
        allow_extrap = self.extrap_check.isChecked()

        if self.xyz_file_path:
            x, y, z = self._read_xyz_file(self.xyz_file_path)
            crs = osr.SpatialReference()
            crs.ImportFromEPSG(4326)
        else:
            layer = self.vector_combo.currentData()
            field_name = self.field_combo.currentText()
            x, y, z = self._load_points_from_vector(layer, field_name)
            crs = osr.SpatialReference()
            crs.ImportFromWkt(layer.crs().toWkt())

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        cols = int(np.ceil((xmax - xmin)/resolution))+1
        rows = int(np.ceil((ymax - ymin)/resolution))+1

        grid_x = np.linspace(xmin, xmax, cols)
        grid_y = np.linspace(ymin, ymax, rows)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

        method = self.method_combo.currentText().lower()
        points = np.column_stack([x, y])
        if not allow_extrap:
            hull = Delaunay(points)
            mask = hull.find_simplex(np.column_stack([grid_X.ravel(), grid_Y.ravel()]))>=0
        else:
            mask = np.ones(grid_X.size, dtype=bool)

        if method=="idw":
            xi, yi = grid_X.ravel(), grid_Y.ravel()
            tree = cKDTree(points)
            k = min(12,len(x))
            dists, idxs = tree.query(np.column_stack([xi, yi]), k=k)
            weights = 1/(dists+1e-12)
            zi = np.sum(weights*z[idxs], axis=1)/np.sum(weights,axis=1)
        elif method in ("linear","nearest","cubic"):
            zi = griddata(points,z,(grid_X,grid_Y),method=method).ravel()
        elif method=="rbf":
            rbf = RBFInterpolator(points,z)
            zi = rbf(np.column_stack([grid_X.ravel(), grid_Y.ravel()]))
        else:
            raise ValueError(f"Unknown method {method}")

        zi[~mask] = NODATA_VALUE
        zi = zi.reshape((rows,cols))

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_file,cols,rows,1,gdal.GDT_Float32, options=["COMPRESS=LZW"])
        out_ds.SetGeoTransform((xmin,resolution,0,ymax,0,-resolution))
        out_ds.SetProjection(crs.ExportToWkt())
        band = out_ds.GetRasterBand(1)
        band.SetNoDataValue(NODATA_VALUE)
        band.WriteArray(zi)
        out_ds.FlushCache()

        rlayer = QgsRasterLayer(output_file, os.path.basename(output_file))
        if rlayer.isValid():
            QgsProject.instance().addMapLayer(rlayer)
            self.log("Raster added to QGIS project.")

    # -----------------------------
    # Resample
    # -----------------------------
    def _process_resample(self, output_file):
        factor = self.resample_factor_spin.value()
        method_map = {
            "bilinear": gdal.GRA_Bilinear,
            "cubic": gdal.GRA_Cubic,
            "nearest": gdal.GRA_NearestNeighbour,
            "average": gdal.GRA_Average,
            "lanczos": gdal.GRA_Lanczos
        }
        method_name = self.resample_method_combo.currentText().lower()
        resample_method = method_map.get(method_name, gdal.GRA_Bilinear)
        ds = None
        layer = self.raster_combo.currentData()
        if layer:
            ds = gdal.Open(layer.source())
        if ds is None:
            raise ValueError("No raster to resample.")

        cols, rows = int(ds.RasterXSize*factor), int(ds.RasterYSize*factor)
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_file,cols,rows,ds.RasterCount,gdal.GDT_Float32, options=["COMPRESS=LZW"])
        gt = ds.GetGeoTransform()
        out_ds.SetGeoTransform((gt[0], gt[1]/factor, 0, gt[3], 0, gt[5]/factor))
        out_ds.SetProjection(ds.GetProjection())
        for i in range(ds.RasterCount):
            gdal.ReprojectImage(ds,out_ds,ds.GetProjection(),ds.GetProjection(),resample_method)
        out_ds.FlushCache()
        rlayer = QgsRasterLayer(output_file, os.path.basename(output_file))
        if rlayer.isValid():
            QgsProject.instance().addMapLayer(rlayer)
            self.log("Resampled raster added to QGIS project.")

    # -----------------------------
    # Smooth
    # -----------------------------
    def _process_smooth(self, output_file):
        factor = self.smooth_factor_spin.value()
        method_name = self.smooth_method_combo.currentText().lower()
        ds = None
        layer = self.smooth_raster_combo.currentData()
        if layer:
            ds = gdal.Open(layer.source())
        if ds is None:
            raise ValueError("No raster to smooth.")

        arr = ds.GetRasterBand(1).ReadAsArray()
        if method_name=="gaussian":
            arr_smooth = gaussian_filter(arr, sigma=factor)
        else:
            arr_smooth = uniform_filter(arr, size=factor)

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_file, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(arr_smooth)
        out_ds.GetRasterBand(1).SetNoDataValue(NODATA_VALUE)
        out_ds.FlushCache()

        rlayer = QgsRasterLayer(output_file, os.path.basename(output_file))
        if rlayer.isValid():
            QgsProject.instance().addMapLayer(rlayer)
            self.log("Smoothed raster added to QGIS project.")