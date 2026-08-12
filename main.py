"""
QGIS Plugin: Raster Processor
Memory-safe raster interpolation, resampling, and smoothing for QGIS.
"""

from concurrent.futures import ProcessPoolExecutor
from math import ceil
import multiprocessing
import os
import re
import sys

import numpy as np
from osgeo import gdal
from qgis.PyQt.QtCore import QCoreApplication, Qt, QVariant
from qgis.PyQt.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    Qgis,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsUnitTypes,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsProjectionSelectionWidget
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import ConvexHull, QhullError

from . import parallel_workers

gdal.UseExceptions()

PLUGIN_TAG = "RasterProcessor"
DEFAULT_NODATA = -9999.0
DEFAULT_TILE_SIZE = 512
LARGE_OUTPUT_PIXEL_WARNING = 80_000_000
HARD_OUTPUT_PIXEL_LIMIT = 500_000_000
REGULAR_GRID_COMPLETENESS = 0.95

ADVANCED_METHOD_LIMITS = {
    "linear": {"max_pixels": 8_000_000, "max_points": 250_000},
    "cubic": {"max_pixels": 4_000_000, "max_points": 120_000},
    "rbf": {"max_pixels": 1_000_000, "max_points": 40_000},
}

INTERPOLATION_HINTS = {
    "idw": "Recommended default. Robust for large datasets and processed tile-wise to avoid RAM errors.",
    "nearest": "Fastest option. Best when you want exact nearest-point assignment without smoothing.",
    "linear": "Uses linear triangulation. Good for moderate datasets, but intentionally limited for stability.",
    "cubic": "Creates a smoother surface, but requires much more memory and is limited to small jobs.",
    "rbf": "Experimental smooth interpolation for small datasets only.",
}

SMOOTHING_HINTS = {
    "gaussian": "Smooth, distance-weighted filter. The radius is interpreted in meters and converted to pixels.",
    "uniform": "Moving-window mean filter. The radius is interpreted in meters and converted to pixels.",
}

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
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.setMinimumSize(650, 500)
        self.resize(900, 820)
        self.xyz_file_path = ""
        self.setup_ui()
        self.refresh_layers()
        self.toggle_mode()
        self.update_source_mode()
        self.update_method_help()
        self.update_value_conversion_hint()
        self.update_summary()

    # -----------------------------
    # UI / logging
    # -----------------------------
    def setup_ui(self):
        layout = QVBoxLayout()

        intro_label = QLabel(
            "This plugin works best with projected CRS in meters. Interpolation and smoothing are "
            "processed in tiles where possible to keep memory usage low."
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        mode_group = QGroupBox("Processing mode")
        mode_layout = QVBoxLayout()
        self.interpolate_radio = QRadioButton("Interpolate point layer / XYZ file")
        self.resample_radio = QRadioButton("Resample raster")
        self.smooth_radio = QRadioButton("Smooth raster")
        self.interpolate_radio.setChecked(True)
        mode_layout.addWidget(self.interpolate_radio)
        mode_layout.addWidget(self.resample_radio)
        mode_layout.addWidget(self.smooth_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        self.interp_group = QGroupBox("Interpolation settings")
        interp_layout = QVBoxLayout()

        self.vector_source_radio = QRadioButton("Use QGIS point layer")
        self.xyz_source_radio = QRadioButton("Use XYZ / CSV file")
        self.vector_source_radio.setChecked(True)
        interp_layout.addWidget(self.vector_source_radio)

        vector_layout = QHBoxLayout()
        vector_layout.addWidget(QLabel("Point layer:"))
        self.vector_combo = QComboBox()
        vector_layout.addWidget(self.vector_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_layers)
        vector_layout.addWidget(self.refresh_btn)
        interp_layout.addLayout(vector_layout)

        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Z value field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)
        interp_layout.addLayout(field_layout)

        interp_layout.addWidget(self.xyz_source_radio)

        xyz_layout = QHBoxLayout()
        xyz_layout.addWidget(QLabel("XYZ file:"))
        self.xyz_path_edit = QLineEdit()
        self.xyz_path_edit.setReadOnly(True)
        xyz_layout.addWidget(self.xyz_path_edit)
        self.xyz_btn = QPushButton("Browse")
        self.xyz_btn.clicked.connect(self.select_xyz_file)
        xyz_layout.addWidget(self.xyz_btn)
        self.xyz_clear_btn = QPushButton("Clear")
        self.xyz_clear_btn.clicked.connect(self.clear_xyz_file)
        xyz_layout.addWidget(self.xyz_clear_btn)
        interp_layout.addLayout(xyz_layout)

        xyz_crs_layout = QHBoxLayout()
        xyz_crs_layout.addWidget(QLabel("XYZ CRS:"))
        self.xyz_crs_selector = QgsProjectionSelectionWidget()
        xyz_crs_layout.addWidget(self.xyz_crs_selector)
        interp_layout.addLayout(xyz_crs_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["IDW", "Nearest", "Linear", "Cubic", "RBF"])
        method_layout.addWidget(self.method_combo)
        interp_layout.addLayout(method_layout)

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Grid cell size:"))
        self.resolution_x_spin = QDoubleSpinBox()
        self.resolution_x_spin.setRange(0.01, 1_000_000.0)
        self.resolution_x_spin.setValue(1.0)
        self.resolution_x_spin.setDecimals(2)
        self.resolution_x_spin.setSuffix(" m")
        res_layout.addWidget(self.resolution_x_spin)
        res_layout.addWidget(QLabel("x"))
        self.resolution_y_spin = QDoubleSpinBox()
        self.resolution_y_spin.setRange(0.01, 1_000_000.0)
        self.resolution_y_spin.setValue(1.0)
        self.resolution_y_spin.setDecimals(2)
        self.resolution_y_spin.setSuffix(" m")
        res_layout.addWidget(self.resolution_y_spin)
        self.link_resolution_check = QCheckBox("Link X/Y")
        self.link_resolution_check.setChecked(True)
        res_layout.addWidget(self.link_resolution_check)
        interp_layout.addLayout(res_layout)

        structure_layout = QHBoxLayout()
        structure_layout.addWidget(QLabel("Input structure:"))
        self.grid_structure_combo = QComboBox()
        self.grid_structure_combo.addItem("Auto-detect", "auto")
        self.grid_structure_combo.addItem("Scattered points", "scattered")
        self.grid_structure_combo.addItem("Regular XYZ grid", "regular")
        structure_layout.addWidget(self.grid_structure_combo)
        interp_layout.addLayout(structure_layout)

        regular_method_layout = QHBoxLayout()
        regular_method_layout.addWidget(QLabel("Regular-grid resampling:"))
        self.regular_grid_method_combo = QComboBox()
        self.regular_grid_method_combo.addItems(["Nearest", "Bilinear", "Cubic", "Average", "Lanczos"])
        regular_method_layout.addWidget(self.regular_grid_method_combo)
        interp_layout.addLayout(regular_method_layout)

        conversion_group = QGroupBox("Vertical value conversion")
        conversion_layout = QVBoxLayout()

        value_unit_layout = QHBoxLayout()
        value_unit_layout.addWidget(QLabel("Input value unit:"))
        self.value_unit_combo = QComboBox()
        self.value_unit_combo.addItem("Meters (no conversion)", "meters")
        self.value_unit_combo.addItem("Milliseconds TWT", "ms_twt")
        self.value_unit_combo.addItem("Seconds TWT", "s_twt")
        self.value_unit_combo.addItem("Milliseconds OWT", "ms_owt")
        self.value_unit_combo.addItem("Seconds OWT", "s_owt")
        value_unit_layout.addWidget(self.value_unit_combo)
        conversion_layout.addLayout(value_unit_layout)

        velocity_layout = QHBoxLayout()
        velocity_layout.addWidget(QLabel("Reference velocity:"))
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(100.0, 10_000.0)
        self.velocity_spin.setValue(2000.0)
        self.velocity_spin.setDecimals(1)
        self.velocity_spin.setSuffix(" m/s")
        velocity_layout.addWidget(self.velocity_spin)
        conversion_layout.addLayout(velocity_layout)

        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Vertical offset:"))
        self.vertical_offset_spin = QDoubleSpinBox()
        self.vertical_offset_spin.setRange(-100_000.0, 100_000.0)
        self.vertical_offset_spin.setValue(0.0)
        self.vertical_offset_spin.setDecimals(2)
        self.vertical_offset_spin.setSuffix(" m")
        offset_layout.addWidget(self.vertical_offset_spin)
        conversion_layout.addLayout(offset_layout)

        self.use_absolute_values_check = QCheckBox("Use absolute values before conversion")
        self.use_absolute_values_check.setChecked(False)
        conversion_layout.addWidget(self.use_absolute_values_check)

        self.value_conversion_hint_label = QLabel()
        self.value_conversion_hint_label.setWordWrap(True)
        conversion_layout.addWidget(self.value_conversion_hint_label)

        conversion_group.setLayout(conversion_layout)
        interp_layout.addWidget(conversion_group)

        advanced_group = QGroupBox("Advanced interpolation")
        advanced_layout = QVBoxLayout()
        self.extrap_check = QCheckBox("Allow extrapolation outside the convex hull")
        self.extrap_check.setChecked(False)
        advanced_layout.addWidget(self.extrap_check)

        idw_neighbors_layout = QHBoxLayout()
        idw_neighbors_layout.addWidget(QLabel("IDW neighbours:"))
        self.idw_neighbors_spin = QSpinBox()
        self.idw_neighbors_spin.setRange(1, 64)
        self.idw_neighbors_spin.setValue(12)
        idw_neighbors_layout.addWidget(self.idw_neighbors_spin)
        advanced_layout.addLayout(idw_neighbors_layout)

        idw_power_layout = QHBoxLayout()
        idw_power_layout.addWidget(QLabel("IDW power:"))
        self.idw_power_spin = QDoubleSpinBox()
        self.idw_power_spin.setRange(0.1, 10.0)
        self.idw_power_spin.setValue(2.0)
        self.idw_power_spin.setDecimals(2)
        idw_power_layout.addWidget(self.idw_power_spin)
        advanced_layout.addLayout(idw_power_layout)

        advanced_group.setLayout(advanced_layout)
        interp_layout.addWidget(advanced_group)

        self.interp_hint_label = QLabel()
        self.interp_hint_label.setWordWrap(True)
        interp_layout.addWidget(self.interp_hint_label)

        self.interp_group.setLayout(interp_layout)
        layout.addWidget(self.interp_group)

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

        target_resample_layout = QHBoxLayout()
        target_resample_layout.addWidget(QLabel("Target resolution:"))
        self.resample_resolution_spin = QDoubleSpinBox()
        self.resample_resolution_spin.setRange(0.01, 1_000_000.0)
        self.resample_resolution_spin.setValue(1.0)
        self.resample_resolution_spin.setDecimals(2)
        self.resample_resolution_spin.setSuffix(" m/pixel")
        target_resample_layout.addWidget(self.resample_resolution_spin)
        resample_layout.addLayout(target_resample_layout)

        resample_method_layout = QHBoxLayout()
        resample_method_layout.addWidget(QLabel("Method:"))
        self.resample_method_combo = QComboBox()
        self.resample_method_combo.addItems(["Nearest", "Bilinear", "Cubic", "Average", "Lanczos"])
        resample_method_layout.addWidget(self.resample_method_combo)
        resample_layout.addLayout(resample_method_layout)

        self.resample_info_label = QLabel()
        self.resample_info_label.setWordWrap(True)
        resample_layout.addWidget(self.resample_info_label)

        self.resample_group.setLayout(resample_layout)
        self.resample_group.setVisible(False)
        layout.addWidget(self.resample_group)

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

        smooth_radius_layout = QHBoxLayout()
        smooth_radius_layout.addWidget(QLabel("Smoothing radius:"))
        self.smooth_radius_spin = QDoubleSpinBox()
        self.smooth_radius_spin.setRange(0.10, 1_000_000.0)
        self.smooth_radius_spin.setValue(3.0)
        self.smooth_radius_spin.setDecimals(2)
        self.smooth_radius_spin.setSuffix(" m")
        smooth_radius_layout.addWidget(self.smooth_radius_spin)
        smooth_layout.addLayout(smooth_radius_layout)

        smooth_method_layout = QHBoxLayout()
        smooth_method_layout.addWidget(QLabel("Method:"))
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItems(["Gaussian", "Uniform"])
        smooth_method_layout.addWidget(self.smooth_method_combo)
        smooth_layout.addLayout(smooth_method_layout)

        self.smooth_info_label = QLabel()
        self.smooth_info_label.setWordWrap(True)
        smooth_layout.addWidget(self.smooth_info_label)

        self.smooth_group.setLayout(smooth_layout)
        self.smooth_group.setVisible(False)
        layout.addWidget(self.smooth_group)

        performance_group = QGroupBox("Performance")
        performance_layout = QVBoxLayout()

        self.parallel_check = QCheckBox("Process large tile jobs in parallel (multiple CPU cores)")
        self.parallel_check.setChecked(False)
        performance_layout.addWidget(self.parallel_check)

        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Worker processes:"))
        self.worker_count_spin = QSpinBox()
        self.worker_count_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.worker_count_spin.setValue(max(1, (os.cpu_count() or 2) - 1))
        self.worker_count_spin.setEnabled(False)
        worker_layout.addWidget(self.worker_count_spin)
        performance_layout.addLayout(worker_layout)

        performance_hint_label = QLabel(
            "Applies to tile-based IDW/Nearest interpolation and raster smoothing. Falls back "
            "to a single process automatically if worker processes cannot be started."
        )
        performance_hint_label.setWordWrap(True)
        performance_layout.addWidget(performance_hint_label)

        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output GeoTIFF:"))
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        output_layout.addWidget(self.output_path_edit)
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)

        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        scroll_content = QWidget()
        scroll_content.setLayout(layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(scroll_area)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        outer_layout.addWidget(self.progress)
        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Process")
        self.run_btn.clicked.connect(self.process)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.close_btn)
        outer_layout.addLayout(button_layout)

        outer_layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(50)
        self.log_text.setMaximumHeight(140)
        outer_layout.addWidget(self.log_text)

        self.setLayout(outer_layout)

        self.interpolate_radio.toggled.connect(self.toggle_mode)
        self.resample_radio.toggled.connect(self.toggle_mode)
        self.smooth_radio.toggled.connect(self.toggle_mode)

        self.vector_source_radio.toggled.connect(self.update_source_mode)
        self.xyz_source_radio.toggled.connect(self.update_source_mode)
        self.vector_combo.currentIndexChanged.connect(self.update_fields)
        self.vector_combo.currentIndexChanged.connect(self.update_summary)
        self.field_combo.currentIndexChanged.connect(self.update_summary)
        self.method_combo.currentIndexChanged.connect(self.update_method_help)
        self.method_combo.currentIndexChanged.connect(self.update_summary)
        self.resolution_x_spin.valueChanged.connect(
            lambda: self._sync_linked_resolution(self.resolution_x_spin, self.resolution_y_spin)
        )
        self.resolution_y_spin.valueChanged.connect(
            lambda: self._sync_linked_resolution(self.resolution_y_spin, self.resolution_x_spin)
        )
        self.link_resolution_check.toggled.connect(
            lambda: self._sync_linked_resolution(self.resolution_x_spin, self.resolution_y_spin)
        )
        self.resolution_x_spin.valueChanged.connect(self.update_summary)
        self.resolution_y_spin.valueChanged.connect(self.update_summary)
        self.link_resolution_check.toggled.connect(self.update_summary)
        self.grid_structure_combo.currentIndexChanged.connect(self.update_summary)
        self.regular_grid_method_combo.currentIndexChanged.connect(self.update_summary)
        self.extrap_check.toggled.connect(self.update_summary)
        self.idw_neighbors_spin.valueChanged.connect(self.update_summary)
        self.idw_power_spin.valueChanged.connect(self.update_summary)
        self.xyz_crs_selector.crsChanged.connect(self.update_summary)
        self.value_unit_combo.currentIndexChanged.connect(self.update_value_conversion_hint)
        self.velocity_spin.valueChanged.connect(self.update_value_conversion_hint)
        self.vertical_offset_spin.valueChanged.connect(self.update_value_conversion_hint)
        self.use_absolute_values_check.toggled.connect(self.update_value_conversion_hint)

        self.raster_combo.currentIndexChanged.connect(self.update_summary)
        self.resample_resolution_spin.valueChanged.connect(self.update_summary)
        self.resample_method_combo.currentIndexChanged.connect(self.update_summary)

        self.smooth_raster_combo.currentIndexChanged.connect(self.update_summary)
        self.smooth_radius_spin.valueChanged.connect(self.update_summary)
        self.smooth_method_combo.currentIndexChanged.connect(self.update_summary)

        self.parallel_check.toggled.connect(self.worker_count_spin.setEnabled)
        self.parallel_check.toggled.connect(self.update_summary)
        self.worker_count_spin.valueChanged.connect(self.update_summary)

    def log(self, message, level=Qgis.Info):
        message = str(message)
        QgsMessageLog.logMessage(message, PLUGIN_TAG, level)
        self.log_text.append(message)

    def _set_progress(self, value):
        self.progress.setValue(max(0, min(100, int(value))))
        QCoreApplication.processEvents()

    def toggle_mode(self):
        self.interp_group.setVisible(self.interpolate_radio.isChecked())
        self.resample_group.setVisible(self.resample_radio.isChecked())
        self.smooth_group.setVisible(self.smooth_radio.isChecked())
        self.update_summary()

    def update_source_mode(self):
        use_vector = self.vector_source_radio.isChecked()
        self.vector_combo.setEnabled(use_vector)
        self.field_combo.setEnabled(use_vector)
        self.refresh_btn.setEnabled(use_vector)
        self.xyz_path_edit.setEnabled(not use_vector)
        self.xyz_btn.setEnabled(not use_vector)
        self.xyz_clear_btn.setEnabled(not use_vector)
        self.xyz_crs_selector.setEnabled(not use_vector)
        self.update_summary()

    def _sync_linked_resolution(self, source_spin, target_spin):
        if not self.link_resolution_check.isChecked():
            return
        if abs(target_spin.value() - source_spin.value()) < 1e-9:
            return
        target_spin.blockSignals(True)
        target_spin.setValue(source_spin.value())
        target_spin.blockSignals(False)

    def update_method_help(self):
        method_key = self.method_combo.currentText().lower()
        hint = INTERPOLATION_HINTS.get(method_key, "")
        if method_key == "idw":
            hint += (
                f" Current settings: {self.idw_neighbors_spin.value()} neighbours, "
                f"power {self.idw_power_spin.value():.2f}."
            )
        self.interp_hint_label.setText(hint)
        self.update_summary()

    def update_value_conversion_hint(self):
        unit = self.value_unit_combo.currentData()
        uses_velocity = unit != "meters"
        self.velocity_spin.setEnabled(uses_velocity)

        if unit == "meters":
            hint = "Values are used directly as meters."
        else:
            factor = self._vertical_conversion_factor()
            hint = (
                f"Conversion factor: {factor:.4f} m per input unit. "
                "TWT values are converted with depth = time * velocity / 2."
            )
        if self.use_absolute_values_check.isChecked():
            hint += " Absolute values are used before scaling."
        if abs(self.vertical_offset_spin.value()) > 1e-9:
            hint += f" Offset after conversion: {self.vertical_offset_spin.value():.2f} m."

        self.value_conversion_hint_label.setText(hint)
        self.update_summary()

    # -----------------------------
    # File selection / layers
    # -----------------------------
    def refresh_layers(self):
        current_vector = self.vector_combo.currentText()
        current_raster = self.raster_combo.currentText()
        current_smooth = self.smooth_raster_combo.currentText()

        self.vector_combo.clear()
        self.raster_combo.clear()
        self.smooth_raster_combo.clear()
        layers = QgsProject.instance().mapLayers().values()
        for layer in layers:
            if isinstance(layer, QgsVectorLayer):
                if QgsWkbTypes.geometryType(layer.wkbType()) == QgsWkbTypes.PointGeometry:
                    self.vector_combo.addItem(layer.name(), layer)
            if isinstance(layer, QgsRasterLayer):
                self.raster_combo.addItem(layer.name(), layer)
                self.smooth_raster_combo.addItem(layer.name(), layer)

        self._restore_combo_text(self.vector_combo, current_vector)
        self._restore_combo_text(self.raster_combo, current_raster)
        self._restore_combo_text(self.smooth_raster_combo, current_smooth)

        if QgsProject.instance().crs().isValid() and not self.xyz_crs_selector.crs().isValid():
            self.xyz_crs_selector.setCrs(QgsProject.instance().crs())

        self.update_fields()
        self.update_summary()
        self.log("Layer lists refreshed.")

    def _restore_combo_text(self, combo, text):
        if not text:
            return
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def update_fields(self):
        self.field_combo.clear()
        layer = self.vector_combo.currentData()
        if layer:
            for field in layer.fields():
                if field.type() in (
                    QVariant.Int,
                    QVariant.Double,
                    QVariant.LongLong,
                    QVariant.UInt,
                    QVariant.ULongLong,
                ):
                    self.field_combo.addItem(field.name())

    def select_output(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Output raster", "", "GeoTIFF (*.tif)")
        if filename and not filename.lower().endswith(".tif"):
            filename += ".tif"
        if filename:
            self.output_path_edit.setText(filename)
            self.update_summary()

    def select_xyz_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select XYZ file",
            "",
            "XYZ/Text/CSV (*.txt *.dat *.xyz *.csv)",
        )
        if filename:
            self.xyz_file_path = filename
            self.xyz_path_edit.setText(filename)
            self.log(f"XYZ file selected: {filename}")
            self.update_summary()

    def clear_xyz_file(self):
        self.xyz_file_path = ""
        self.xyz_path_edit.clear()
        self.update_summary()

    # -----------------------------
    # Processing
    # -----------------------------
    def _selected_output(self):
        output = self.output_path_edit.text().strip()
        if not output:
            raise ValueError("Please select an output GeoTIFF.")
        return output

    def _selected_interpolation_crs(self):
        if self.vector_source_radio.isChecked():
            layer = self.vector_combo.currentData()
            return layer.crs() if layer else None
        return self.xyz_crs_selector.crs()

    def _crs_label(self, crs):
        if not crs or not crs.isValid():
            return "CRS not set"
        unit_name = QgsUnitTypes.toString(crs.mapUnits())
        authid = crs.authid() or "custom CRS"
        return f"{authid} ({unit_name})"

    def _validate_metric_crs(self, crs, context):
        if not crs or not crs.isValid():
            raise ValueError(f"{context} has no valid CRS.")
        if crs.mapUnits() != QgsUnitTypes.DistanceMeters:
            unit_name = QgsUnitTypes.toString(crs.mapUnits())
            raise ValueError(
                f"{context} uses '{unit_name}'. Please use a projected CRS in meters."
            )

    def _format_bytes(self, num_bytes):
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _vertical_conversion_factor(self):
        unit = self.value_unit_combo.currentData()
        velocity = self.velocity_spin.value()
        if unit == "meters":
            return 1.0
        if unit == "ms_twt":
            return velocity * 0.0005
        if unit == "s_twt":
            return velocity * 0.5
        if unit == "ms_owt":
            return velocity * 0.001
        if unit == "s_owt":
            return velocity
        return 1.0

    def _value_conversion_summary(self):
        unit = self.value_unit_combo.currentData()
        if unit == "meters":
            summary = "Vertical values: meters"
        else:
            summary = (
                f"Vertical conversion: {self.value_unit_combo.currentText()} "
                f"with {self.velocity_spin.value():.1f} m/s"
            )
        if self.use_absolute_values_check.isChecked():
            summary += ", absolute values"
        if abs(self.vertical_offset_spin.value()) > 1e-9:
            summary += f", offset {self.vertical_offset_spin.value():.2f} m"
        return summary

    def _convert_vertical_values(self, values):
        array = np.asarray(values, dtype=np.float64)
        if self.use_absolute_values_check.isChecked():
            array = np.abs(array)
        factor = self._vertical_conversion_factor()
        offset = self.vertical_offset_spin.value()
        return (array * factor) + offset

    def _normalized_source(self, source):
        return source.split("|", 1)[0] if source else source

    def _raster_source_path(self, layer):
        if not layer:
            raise ValueError("No raster layer selected.")
        return self._normalized_source(layer.dataProvider().dataSourceUri() or layer.source())

    def _open_raster_dataset(self, layer):
        source = self._raster_source_path(layer)
        dataset = gdal.Open(source, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Could not open raster source: {source}")
        return dataset

    def _dataset_resolution(self, dataset):
        geotransform = dataset.GetGeoTransform()
        return abs(geotransform[1]), abs(geotransform[5])

    def _dataset_bounds(self, dataset):
        geotransform = dataset.GetGeoTransform()
        xmin = geotransform[0]
        ymax = geotransform[3]
        xmax = xmin + dataset.RasterXSize * geotransform[1]
        ymin = ymax + dataset.RasterYSize * geotransform[5]
        return min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)

    def _estimate_output_grid(self, xmin, ymin, xmax, ymax, resolution_x, resolution_y):
        width = max(0.0, xmax - xmin)
        height = max(0.0, ymax - ymin)
        cols = max(1, int(ceil(width / resolution_x)))
        rows = max(1, int(ceil(height / resolution_y)))
        pixels = rows * cols
        return rows, cols, pixels

    def _validate_output_pixels(self, rows, cols, context):
        pixels = rows * cols
        if pixels > HARD_OUTPUT_PIXEL_LIMIT:
            raise ValueError(
                f"{context} would create {pixels:,} pixels. Please increase the target resolution."
            )
        if pixels > LARGE_OUTPUT_PIXEL_WARNING:
            self.log(
                f"Large output detected ({pixels:,} pixels). Processing remains tile-based, "
                "but runtime and disk usage will still be high.",
                Qgis.Warning,
            )

    def _gdal_creation_options(self):
        return [
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
            "PREDICTOR=3",
        ]

    def _tile_windows(self, rows, cols, tile_size=DEFAULT_TILE_SIZE):
        for yoff in range(0, rows, tile_size):
            ysize = min(tile_size, rows - yoff)
            for xoff in range(0, cols, tile_size):
                xsize = min(tile_size, cols - xoff)
                yield xoff, yoff, xsize, ysize

    def _selected_grid_mode(self):
        return self.grid_structure_combo.currentData()

    def _detect_regular_grid(self, x, y, min_completeness=REGULAR_GRID_COMPLETENESS):
        rounded_x = np.round(np.asarray(x, dtype=np.float64), 6)
        rounded_y = np.round(np.asarray(y, dtype=np.float64), 6)
        unique_x = np.unique(rounded_x)
        unique_y = np.unique(rounded_y)

        if unique_x.size < 2 or unique_y.size < 2:
            return None

        x_steps = np.unique(np.round(np.diff(unique_x), 6))
        y_steps = np.unique(np.round(np.diff(unique_y), 6))
        if x_steps.size != 1 or y_steps.size != 1:
            return None
        if x_steps[0] <= 0 or y_steps[0] <= 0:
            return None

        total_cells = int(unique_x.size * unique_y.size)
        completeness = float(len(rounded_x)) / float(total_cells)
        if completeness < min_completeness:
            return None

        return {
            "rounded_x": rounded_x,
            "rounded_y": rounded_y,
            "unique_x": unique_x,
            "unique_y": unique_y,
            "x_res": float(x_steps[0]),
            "y_res": float(y_steps[0]),
            "total_cells": total_cells,
            "completeness": completeness,
        }

    def _points_inside_hull(self, coords, hull_equations):
        if hull_equations is None:
            return np.ones(coords.shape[0], dtype=bool)
        lhs = np.matmul(coords, hull_equations[:, :2].T) + hull_equations[:, 2]
        return np.all(lhs <= 1e-9, axis=1)

    def _build_hull_equations(self, points):
        if points.shape[0] < 3:
            return None
        try:
            hull = ConvexHull(points)
            return hull.equations
        except QhullError:
            self.log(
                "Could not derive a convex hull from the input points. Extrapolation mask disabled.",
                Qgis.Warning,
            )
            return None

    def _aggregate_duplicate_points(self, x, y, z):
        coords = np.column_stack((x, y))
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        if unique_coords.shape[0] == coords.shape[0]:
            return x, y, z

        z_sum = np.bincount(inverse, weights=z)
        z_count = np.bincount(inverse)
        z_mean = z_sum / np.maximum(z_count, 1)
        removed = coords.shape[0] - unique_coords.shape[0]
        self.log(
            f"Merged {removed:,} duplicate XY points by averaging their Z values.",
            Qgis.Warning,
        )
        return unique_coords[:, 0], unique_coords[:, 1], z_mean

    def _load_points_from_vector(self, layer, field_name):
        if not layer:
            raise ValueError("Please select a point layer.")
        if not field_name:
            raise ValueError("Please select a numeric Z value field.")

        xs, ys, zs = [], [], []
        total = max(int(layer.featureCount()), 1)
        for index, feature in enumerate(layer.getFeatures(), start=1):
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            try:
                z_value = float(feature[field_name])
            except (TypeError, ValueError):
                continue

            points = geometry.asMultiPoint() if geometry.isMultipart() else [geometry.asPoint()]
            for point in points:
                xs.append(point.x())
                ys.append(point.y())
                zs.append(z_value)

            if index % 500 == 0:
                self._set_progress((index / total) * 20)

        if not xs:
            raise ValueError("No valid point/Z combinations were found in the selected layer.")

        return (
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(zs, dtype=np.float64),
        )

    def _read_xyz_file(self, filepath):
        xs, ys, zs = [], [], []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [part for part in re.split(r"[,\s;]+", line) if part]
                if len(parts) < 3:
                    continue
                try:
                    x_value = float(parts[0])
                    y_value = float(parts[1])
                    z_value = float(parts[2])
                except ValueError:
                    if line_number == 1:
                        continue
                    raise ValueError(
                        f"XYZ file contains an invalid numeric value in line {line_number}."
                    )
                xs.append(x_value)
                ys.append(y_value)
                zs.append(z_value)

        if not xs:
            raise ValueError("No valid XYZ rows were found in the selected file.")

        return (
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(zs, dtype=np.float64),
        )

    def _collect_interpolation_points(self):
        if self.vector_source_radio.isChecked():
            layer = self.vector_combo.currentData()
            if layer is None:
                raise ValueError("Please select a point layer.")
            self._validate_metric_crs(layer.crs(), "Point layer")
            field_name = self.field_combo.currentText()
            x, y, z = self._load_points_from_vector(layer, field_name)
            crs = layer.crs()
        else:
            if not self.xyz_file_path:
                raise ValueError("Please select an XYZ / CSV file.")
            crs = self.xyz_crs_selector.crs()
            self._validate_metric_crs(crs, "XYZ file CRS")
            x, y, z = self._read_xyz_file(self.xyz_file_path)

        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.all(finite_mask):
            dropped = int((~finite_mask).sum())
            self.log(f"Dropped {dropped:,} invalid point rows.", Qgis.Warning)
            x, y, z = x[finite_mask], y[finite_mask], z[finite_mask]

        if x.size == 0:
            raise ValueError("No valid input points remain after validation.")

        x, y, z = self._aggregate_duplicate_points(x, y, z)
        return x, y, z, crs

    def _grid_spec_from_points(self, x, y, resolution_x, resolution_y):
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        rows, cols, pixels = self._estimate_output_grid(xmin, ymin, xmax, ymax, resolution_x, resolution_y)
        self._validate_output_pixels(rows, cols, "Interpolation")
        return {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "rows": rows,
            "cols": cols,
            "pixels": pixels,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
        }

    def _create_output_dataset(self, output_file, cols, rows, band_count, projection_wkt):
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            output_file,
            cols,
            rows,
            band_count,
            gdal.GDT_Float32,
            options=self._gdal_creation_options(),
        )
        if dataset is None:
            raise RuntimeError(f"Could not create output raster: {output_file}")
        dataset.SetProjection(projection_wkt)
        return dataset

    def _add_output_to_project(self, output_file):
        raster_layer = QgsRasterLayer(output_file, os.path.basename(output_file))
        if raster_layer.isValid():
            QgsProject.instance().addMapLayer(raster_layer)
            self.log("Output raster added to the QGIS project.")
        else:
            self.log("Output raster was written but could not be loaded into QGIS.", Qgis.Warning)

    def _build_regular_grid_dataset(self, x, y, z, crs, grid_info):
        cols = grid_info["unique_x"].size
        rows = grid_info["unique_y"].size
        array = np.full((rows, cols), DEFAULT_NODATA, dtype=np.float32)

        x_index = np.searchsorted(grid_info["unique_x"], grid_info["rounded_x"])
        y_index = np.searchsorted(grid_info["unique_y"], grid_info["rounded_y"])
        row_index = rows - 1 - y_index
        array[row_index, x_index] = np.asarray(z, dtype=np.float32)

        dataset = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float32)
        dataset.SetProjection(crs.toWkt())
        dataset.SetGeoTransform(
            (
                grid_info["unique_x"][0] - (grid_info["x_res"] / 2.0),
                grid_info["x_res"],
                0.0,
                grid_info["unique_y"][-1] + (grid_info["y_res"] / 2.0),
                0.0,
                -grid_info["y_res"],
            )
        )
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(DEFAULT_NODATA)
        band.WriteArray(array)
        return dataset

    def _process_regular_grid_input(self, output_file, x, y, z, crs, grid_info):
        native_dataset = self._build_regular_grid_dataset(x, y, z, crs, grid_info)
        native_bounds = (
            grid_info["unique_x"][0] - (grid_info["x_res"] / 2.0),
            grid_info["unique_y"][0] - (grid_info["y_res"] / 2.0),
            grid_info["unique_x"][-1] + (grid_info["x_res"] / 2.0),
            grid_info["unique_y"][-1] + (grid_info["y_res"] / 2.0),
        )

        self.log(
            f"Regular grid detected: {grid_info['x_res']:.2f} x {grid_info['y_res']:.2f} m, "
            f"completeness {grid_info['completeness'] * 100:.1f}%."
        )

        target_resolution_x = self.resolution_x_spin.value()
        target_resolution_y = self.resolution_y_spin.value()
        same_x = abs(target_resolution_x - grid_info["x_res"]) <= 1e-6
        same_y = abs(target_resolution_y - grid_info["y_res"]) <= 1e-6

        if same_x and same_y:
            out_dataset = gdal.GetDriverByName("GTiff").CreateCopy(
                output_file,
                native_dataset,
                options=self._gdal_creation_options(),
            )
        else:
            rows, cols, _ = self._estimate_output_grid(
                native_bounds[0],
                native_bounds[1],
                native_bounds[2],
                native_bounds[3],
                target_resolution_x,
                target_resolution_y,
            )
            self._validate_output_pixels(rows, cols, "Regular-grid export")
            out_dataset = gdal.Warp(
                output_file,
                native_dataset,
                options=gdal.WarpOptions(
                    format="GTiff",
                    xRes=target_resolution_x,
                    yRes=target_resolution_y,
                    resampleAlg=self.regular_grid_method_combo.currentText().lower(),
                    srcNodata=DEFAULT_NODATA,
                    dstNodata=DEFAULT_NODATA,
                    multithread=True,
                    creationOptions=self._gdal_creation_options(),
                    callback=self._gdal_progress_callback,
                ),
            )

        if out_dataset is None:
            raise RuntimeError("Regular-grid export failed.")
        out_dataset.FlushCache()

    def _gdal_progress_callback(self, completion, message, user_data):
        del message, user_data
        self._set_progress(completion * 100)
        return 1

    # -----------------------------
    # Parallel processing
    # -----------------------------
    def _resolve_worker_python_executable(self):
        exec_dir = os.path.dirname(sys.executable)
        candidates = [
            sys.executable if os.path.basename(sys.executable).lower().startswith("python") else None,
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(exec_dir, "python3.exe"),
            os.path.join(exec_dir, "python.exe"),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _effective_worker_count(self, total_tiles):
        if not self.parallel_check.isChecked() or total_tiles <= 1:
            return 1
        requested = max(1, min(self.worker_count_spin.value(), total_tiles))
        if requested <= 1:
            return 1
        if os.name == "nt":
            python_exe = self._resolve_worker_python_executable()
            if not python_exe:
                self.log(
                    "Parallel processing needs a standalone Python interpreter, which could "
                    "not be found next to this QGIS installation. Continuing with a single process.",
                    Qgis.Warning,
                )
                return 1
            multiprocessing.set_executable(python_exe)
        return requested

    def _performance_summary_line(self):
        if self.parallel_check.isChecked():
            return f"Parallel processing: enabled ({self.worker_count_spin.value()} worker processes)."
        return "Parallel processing: disabled (single process)."

    # -----------------------------
    # Summary
    # -----------------------------
    def update_summary(self):
        if self.interpolate_radio.isChecked():
            self.summary_label.setText(self._interpolation_summary())
        elif self.resample_radio.isChecked():
            self.summary_label.setText(self._resample_summary())
        else:
            self.summary_label.setText(self._smooth_summary())

    def _interpolation_summary(self):
        lines = []
        crs = self._selected_interpolation_crs()
        lines.append(f"CRS: {self._crs_label(crs)}")
        lines.append(
            f"Grid cell size: {self.resolution_x_spin.value():.2f} x {self.resolution_y_spin.value():.2f} m"
        )
        lines.append(f"Input structure: {self.grid_structure_combo.currentText()}")
        method_key = self.method_combo.currentText().lower()
        lines.append(f"Method: {self.method_combo.currentText()}")
        lines.append(INTERPOLATION_HINTS.get(method_key, ""))
        lines.append(self._value_conversion_summary())
        lines.append(
            f"Regular-grid resampling: {self.regular_grid_method_combo.currentText()}"
        )

        if self.vector_source_radio.isChecked():
            layer = self.vector_combo.currentData()
            if layer is None:
                lines.append("Input: no point layer selected.")
            else:
                extent = layer.extent()
                rows, cols, pixels = self._estimate_output_grid(
                    extent.xMinimum(),
                    extent.yMinimum(),
                    extent.xMaximum(),
                    extent.yMaximum(),
                    self.resolution_x_spin.value(),
                    self.resolution_y_spin.value(),
                )
                estimated_size = pixels * 4
                lines.append(
                    f"Estimated output: {cols:,} x {rows:,} pixels, about {self._format_bytes(estimated_size)} before compression."
                )
                if pixels > LARGE_OUTPUT_PIXEL_WARNING:
                    lines.append("Large job: interpolation stays tile-based, but runtime and disk usage will be high.")
        else:
            if self.xyz_file_path:
                lines.append("XYZ file selected. Extent and output size will be evaluated during processing.")
            else:
                lines.append("Input: no XYZ file selected.")

        if crs and crs.isValid() and crs.mapUnits() != QgsUnitTypes.DistanceMeters:
            lines.append("Warning: the selected CRS is not metric. This plugin expects meters.")
        if method_key in ADVANCED_METHOD_LIMITS:
            limit = ADVANCED_METHOD_LIMITS[method_key]
            lines.append(
                f"Safety limit for {self.method_combo.currentText()}: "
                f"{limit['max_points']:,} points and {limit['max_pixels']:,} pixels."
            )
        if method_key in ("idw", "nearest"):
            lines.append(self._performance_summary_line())
        return "\n".join(lines)

    def _resample_summary(self):
        layer = self.raster_combo.currentData()
        lines = [f"Target resolution: {self.resample_resolution_spin.value():.2f} m/pixel"]
        if layer is None:
            self.resample_info_label.setText("Select a raster layer to inspect its current resolution.")
            lines.append("Input: no raster selected.")
            return "\n".join(lines)

        try:
            dataset = self._open_raster_dataset(layer)
            source_res_x, source_res_y = self._dataset_resolution(dataset)
            xmin, ymin, xmax, ymax = self._dataset_bounds(dataset)
            rows, cols, pixels = self._estimate_output_grid(
                xmin,
                ymin,
                xmax,
                ymax,
                self.resample_resolution_spin.value(),
                self.resample_resolution_spin.value(),
            )
            self.resample_info_label.setText(
                f"Current raster resolution: {source_res_x:.3f} m x {source_res_y:.3f} m per pixel."
            )
            lines.append(f"Current resolution: {source_res_x:.3f} x {source_res_y:.3f} m/pixel")
            lines.append(
                f"Estimated output: {cols:,} x {rows:,} pixels, about {self._format_bytes(pixels * 4 * max(1, dataset.RasterCount))} before compression."
            )
            if layer.crs().isValid() and layer.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
                lines.append("Warning: the selected raster CRS is not metric.")
            dataset = None
        except Exception as exc:
            self.resample_info_label.setText(str(exc))
            lines.append(f"Input warning: {exc}")

        return "\n".join(lines)

    def _smooth_summary(self):
        layer = self.smooth_raster_combo.currentData()
        method_key = self.smooth_method_combo.currentText().lower()
        lines = [
            f"Smoothing radius: {self.smooth_radius_spin.value():.2f} m",
            SMOOTHING_HINTS.get(method_key, ""),
        ]
        if layer is None:
            self.smooth_info_label.setText("Select a raster layer to inspect its resolution.")
            lines.append("Input: no raster selected.")
            return "\n".join(lines)

        try:
            dataset = self._open_raster_dataset(layer)
            source_res_x, source_res_y = self._dataset_resolution(dataset)
            radius_x = self.smooth_radius_spin.value() / source_res_x if source_res_x else 0.0
            radius_y = self.smooth_radius_spin.value() / source_res_y if source_res_y else 0.0
            self.smooth_info_label.setText(
                f"Current raster resolution: {source_res_x:.3f} m x {source_res_y:.3f} m per pixel."
            )
            lines.append(f"Current resolution: {source_res_x:.3f} x {source_res_y:.3f} m/pixel")
            lines.append(
                f"Effective filter radius: about {radius_x:.1f} x {radius_y:.1f} pixels."
            )
            if layer.crs().isValid() and layer.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
                lines.append("Warning: the selected raster CRS is not metric.")
            lines.append(self._performance_summary_line())
            dataset = None
        except Exception as exc:
            self.smooth_info_label.setText(str(exc))
            lines.append(f"Input warning: {exc}")

        return "\n".join(lines)

    def process(self):
        self.run_btn.setEnabled(False)
        self._set_progress(0)
        try:
            output_file = self._selected_output()
            if self.interpolate_radio.isChecked():
                self._process_interpolation(output_file)
            elif self.resample_radio.isChecked():
                self._process_resample(output_file)
            else:
                self._process_smooth(output_file)
            self._set_progress(100)
            self.log("Processing finished successfully.", Qgis.Success)
        except Exception as exc:
            self.log(f"Error: {exc}", Qgis.Critical)
        finally:
            self.run_btn.setEnabled(True)

    # -----------------------------
    # Interpolation
    # -----------------------------
    def _process_interpolation(self, output_file):
        resolution_x = self.resolution_x_spin.value()
        resolution_y = self.resolution_y_spin.value()
        if resolution_x <= 0 or resolution_y <= 0:
            raise ValueError("Grid cell size must be greater than zero.")

        self.log("Loading interpolation input points...")
        x, y, z, crs = self._collect_interpolation_points()
        z = self._convert_vertical_values(z)

        selected_mode = self._selected_grid_mode()
        regular_grid = None
        if selected_mode == "regular":
            regular_grid = self._detect_regular_grid(x, y, min_completeness=0.0)
            if regular_grid is None:
                raise ValueError(
                    "Regular-grid mode was selected, but the input does not look like a regular XY grid."
                )
        elif selected_mode == "auto":
            regular_grid = self._detect_regular_grid(x, y)

        if regular_grid is not None:
            self._process_regular_grid_input(output_file, x, y, z, crs, regular_grid)
            self._add_output_to_project(output_file)
            return

        grid_spec = self._grid_spec_from_points(x, y, resolution_x, resolution_y)
        self.log(
            f"Interpolating {len(z):,} points to {grid_spec['cols']:,} x {grid_spec['rows']:,} pixels "
            f"with a {resolution_x:.2f} x {resolution_y:.2f} m grid cell size."
        )

        method = self.method_combo.currentText().lower()
        if method in ("idw", "nearest"):
            self._interpolate_idw_or_nearest(output_file, x, y, z, crs, grid_spec)
        else:
            self._interpolate_advanced_method(output_file, x, y, z, crs, grid_spec)

        self._add_output_to_project(output_file)

    def _interpolate_idw_or_nearest(self, output_file, x, y, z, crs, grid_spec):
        method = self.method_combo.currentText().lower()
        points = np.column_stack((x, y))
        hull_equations = None if self.extrap_check.isChecked() else self._build_hull_equations(points)

        dataset = self._create_output_dataset(
            output_file,
            grid_spec["cols"],
            grid_spec["rows"],
            1,
            crs.toWkt(),
        )
        dataset.SetGeoTransform(
            (
                grid_spec["xmin"],
                grid_spec["resolution_x"],
                0.0,
                grid_spec["ymax"],
                0.0,
                -grid_spec["resolution_y"],
            )
        )
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(DEFAULT_NODATA)

        power = self.idw_power_spin.value()
        neighbours = min(self.idw_neighbors_spin.value(), len(z))
        tile_windows = list(self._tile_windows(grid_spec["rows"], grid_spec["cols"]))
        total_tiles = len(tile_windows)
        tasks = [
            (
                xoff, yoff, xsize, ysize,
                grid_spec["xmin"], grid_spec["ymax"],
                grid_spec["resolution_x"], grid_spec["resolution_y"],
                method, power, neighbours, hull_equations, DEFAULT_NODATA,
            )
            for xoff, yoff, xsize, ysize in tile_windows
        ]

        worker_count = self._effective_worker_count(total_tiles)
        if worker_count > 1:
            try:
                self._run_idw_tiles_parallel(tasks, x, y, z, worker_count, band, total_tiles)
            except Exception as exc:
                self.log(f"Parallel interpolation failed ({exc}); retrying with a single process.", Qgis.Warning)
                parallel_workers.init_idw_worker(x, y, z)
                self._run_idw_tiles_sequential(tasks, band, total_tiles)
        else:
            parallel_workers.init_idw_worker(x, y, z)
            self._run_idw_tiles_sequential(tasks, band, total_tiles)

        dataset.FlushCache()
        dataset = None

    def _run_idw_tiles_sequential(self, tasks, band, total_tiles):
        for tile_index, task in enumerate(tasks, start=1):
            xoff, yoff, tile_array = parallel_workers.idw_or_nearest_tile(task)
            band.WriteArray(tile_array, xoff, yoff)
            self._set_progress((tile_index / total_tiles) * 100)

    def _run_idw_tiles_parallel(self, tasks, x, y, z, worker_count, band, total_tiles):
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=parallel_workers.init_idw_worker,
            initargs=(x, y, z),
        ) as executor:
            for tile_index, (xoff, yoff, tile_array) in enumerate(
                executor.map(parallel_workers.idw_or_nearest_tile, tasks), start=1
            ):
                band.WriteArray(tile_array, xoff, yoff)
                self._set_progress((tile_index / total_tiles) * 100)

    def _interpolate_advanced_method(self, output_file, x, y, z, crs, grid_spec):
        method = self.method_combo.currentText().lower()
        limits = ADVANCED_METHOD_LIMITS[method]
        if grid_spec["pixels"] > limits["max_pixels"]:
            raise ValueError(
                f"{method.title()} interpolation is limited to {limits['max_pixels']:,} pixels. "
                "Use IDW or Nearest for larger jobs."
            )
        if len(z) > limits["max_points"]:
            raise ValueError(
                f"{method.title()} interpolation is limited to {limits['max_points']:,} points. "
                "Use IDW or Nearest for larger jobs."
            )

        points = np.column_stack((x, y))
        hull_equations = None if self.extrap_check.isChecked() else self._build_hull_equations(points)

        grid_x = grid_spec["xmin"] + (np.arange(grid_spec["cols"]) + 0.5) * grid_spec["resolution_x"]
        grid_y = grid_spec["ymax"] - (np.arange(grid_spec["rows"]) + 0.5) * grid_spec["resolution_y"]
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        query_coords = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))

        self._set_progress(15)
        if method == "rbf":
            model = RBFInterpolator(points, z, neighbors=min(64, len(z)))
            values = model(query_coords)
        else:
            values = griddata(points, z, (mesh_x, mesh_y), method=method).ravel()

        if hull_equations is not None:
            inside_mask = self._points_inside_hull(query_coords, hull_equations)
            values[~inside_mask] = DEFAULT_NODATA

        values = np.where(np.isfinite(values), values, DEFAULT_NODATA)
        grid_array = values.reshape((grid_spec["rows"], grid_spec["cols"])).astype(np.float32)

        dataset = self._create_output_dataset(
            output_file,
            grid_spec["cols"],
            grid_spec["rows"],
            1,
            crs.toWkt(),
        )
        dataset.SetGeoTransform(
            (
                grid_spec["xmin"],
                grid_spec["resolution_x"],
                0.0,
                grid_spec["ymax"],
                0.0,
                -grid_spec["resolution_y"],
            )
        )
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(DEFAULT_NODATA)
        band.WriteArray(grid_array)
        dataset.FlushCache()
        dataset = None

    # -----------------------------
    # Resample
    # -----------------------------
    def _process_resample(self, output_file):
        layer = self.raster_combo.currentData()
        if layer is None:
            raise ValueError("Please select a raster layer.")
        self._validate_metric_crs(layer.crs(), "Raster layer")

        target_resolution = self.resample_resolution_spin.value()
        if target_resolution <= 0:
            raise ValueError("Target resolution must be greater than zero.")

        dataset = self._open_raster_dataset(layer)
        xmin, ymin, xmax, ymax = self._dataset_bounds(dataset)
        rows, cols, _ = self._estimate_output_grid(xmin, ymin, xmax, ymax, target_resolution, target_resolution)
        self._validate_output_pixels(rows, cols, "Resampling")

        first_band = dataset.GetRasterBand(1)
        source_nodata = first_band.GetNoDataValue() if first_band else None
        target_nodata = source_nodata if source_nodata is not None else DEFAULT_NODATA

        method_name = self.resample_method_combo.currentText().lower()
        self.log(
            f"Resampling raster to {target_resolution:.2f} m/pixel using {self.resample_method_combo.currentText()}."
        )
        options = gdal.WarpOptions(
            format="GTiff",
            xRes=target_resolution,
            yRes=target_resolution,
            resampleAlg=method_name,
            srcNodata=source_nodata,
            dstNodata=target_nodata,
            multithread=True,
            creationOptions=self._gdal_creation_options(),
            callback=self._gdal_progress_callback,
        )
        out_dataset = gdal.Warp(output_file, dataset, options=options)
        if out_dataset is None:
            raise RuntimeError("GDAL warp failed.")
        out_dataset.FlushCache()
        out_dataset = None
        dataset = None
        self._add_output_to_project(output_file)

    # -----------------------------
    # Smooth
    # -----------------------------
    def _process_smooth(self, output_file):
        layer = self.smooth_raster_combo.currentData()
        if layer is None:
            raise ValueError("Please select a raster layer.")
        self._validate_metric_crs(layer.crs(), "Raster layer")

        source_path = self._raster_source_path(layer)
        dataset = self._open_raster_dataset(layer)
        radius_m = self.smooth_radius_spin.value()
        if radius_m <= 0:
            raise ValueError("Smoothing radius must be greater than zero.")

        pixel_x, pixel_y = self._dataset_resolution(dataset)
        if pixel_x <= 0 or pixel_y <= 0:
            raise ValueError("Raster pixel size is invalid.")

        method_key = self.smooth_method_combo.currentText().lower()
        sigma_x = sigma_y = 0.0
        size_x = size_y = 0
        if method_key == "gaussian":
            sigma_x = max(radius_m / pixel_x, 0.01)
            sigma_y = max(radius_m / pixel_y, 0.01)
            halo_x = max(1, int(ceil(sigma_x * 3)))
            halo_y = max(1, int(ceil(sigma_y * 3)))
        else:
            radius_px_x = max(1, int(ceil(radius_m / pixel_x)))
            radius_px_y = max(1, int(ceil(radius_m / pixel_y)))
            size_x = radius_px_x * 2 + 1
            size_y = radius_px_y * 2 + 1
            halo_x = radius_px_x
            halo_y = radius_px_y

        output_dataset = self._create_output_dataset(
            output_file,
            dataset.RasterXSize,
            dataset.RasterYSize,
            dataset.RasterCount,
            dataset.GetProjection(),
        )
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())

        tile_windows = list(self._tile_windows(dataset.RasterYSize, dataset.RasterXSize))
        total_steps = max(1, dataset.RasterCount * len(tile_windows))
        step_index = 0

        self.log(
            f"Smoothing raster with a {radius_m:.2f} m radius using {self.smooth_method_combo.currentText()}."
        )

        for band_number in range(1, dataset.RasterCount + 1):
            input_band = dataset.GetRasterBand(band_number)
            output_band = output_dataset.GetRasterBand(band_number)
            source_nodata = input_band.GetNoDataValue()
            target_nodata = source_nodata if source_nodata is not None else DEFAULT_NODATA
            output_band.SetNoDataValue(target_nodata)

            tasks = [
                (
                    xoff, yoff, xsize, ysize, halo_x, halo_y,
                    dataset.RasterXSize, dataset.RasterYSize,
                    method_key, sigma_x, sigma_y, size_x, size_y,
                    source_nodata, target_nodata,
                )
                for xoff, yoff, xsize, ysize in tile_windows
            ]

            worker_count = self._effective_worker_count(len(tile_windows))
            if worker_count > 1:
                try:
                    step_index = self._run_smooth_tiles_parallel(
                        tasks, source_path, band_number, worker_count, output_band, step_index, total_steps
                    )
                except Exception as exc:
                    self.log(f"Parallel smoothing failed ({exc}); retrying this band with a single process.", Qgis.Warning)
                    parallel_workers.init_smooth_worker(source_path, band_number)
                    step_index = self._run_smooth_tiles_sequential(tasks, output_band, step_index, total_steps)
            else:
                parallel_workers.init_smooth_worker(source_path, band_number)
                step_index = self._run_smooth_tiles_sequential(tasks, output_band, step_index, total_steps)

        output_dataset.FlushCache()
        output_dataset = None
        dataset = None
        self._add_output_to_project(output_file)

    def _run_smooth_tiles_sequential(self, tasks, output_band, step_offset, total_steps):
        for task in tasks:
            xoff, yoff, inner = parallel_workers.smooth_tile(task)
            output_band.WriteArray(inner, xoff, yoff)
            step_offset += 1
            self._set_progress((step_offset / total_steps) * 100)
        return step_offset

    def _run_smooth_tiles_parallel(
        self, tasks, source_path, band_number, worker_count, output_band, step_offset, total_steps
    ):
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=parallel_workers.init_smooth_worker,
            initargs=(source_path, band_number),
        ) as executor:
            for xoff, yoff, inner in executor.map(parallel_workers.smooth_tile, tasks):
                output_band.WriteArray(inner, xoff, yoff)
                step_offset += 1
                self._set_progress((step_offset / total_steps) * 100)
        return step_offset
