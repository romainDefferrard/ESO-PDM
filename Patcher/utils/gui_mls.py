
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PlotWindowMLS(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.crs = self.parent().crs

        # Inherit attributes from parent
        self.superpositions = self.parent().superpositions
        self.raster_mesh = self.parent().raster_mesh
        self.x_mesh, self.y_mesh = self.raster_mesh
        self.raster = self.parent().raster

        self.plot_index = 0
        self.num_plots = len(self.superpositions)

        self.figure, self.ax = plt.subplots()
        self.figure.set_constrained_layout(True)

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.update_plot()

    def generate_plot(self, index):
        superpos = self.superpositions[index]
        Z = np.ma.masked_where(~superpos, superpos)

        if self.crs is not None: # Display OSM
            from pyproj import Transformer
            import contextily as ctx

            xmin, xmax = np.nanmin(self.x_mesh), np.nanmax(self.x_mesh)
            ymin, ymax = np.nanmin(self.y_mesh), np.nanmax(self.y_mesh)

            tr = Transformer.from_crs(self.crs, "EPSG:3857", always_xy=True)
            xmin_m, ymin_m = tr.transform(xmin, ymin)
            xmax_m, ymax_m = tr.transform(xmax, ymax)

            self.ax.set_xlim(xmin_m, xmax_m)
            self.ax.set_ylim(ymin_m, ymax_m)

            try:
                ctx.add_basemap(
                    self.ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    crs="EPSG:3857",
                    attribution=False,
                )
            except Exception as e:
                logging.warning(f"Basemap unavailable: {e}")

            xm, ym = tr.transform(self.x_mesh, self.y_mesh)

            self.ax.pcolormesh(
                xm, ym, Z,
                shading="auto",
                cmap=None,
                facecolor=(0.2, 0.6, 1.0, 0.35),
                edgecolors="none",
                alpha=0.4, 
            )

        
        else:   # No CRS
            self.ax.pcolormesh(
                self.x_mesh,
                self.y_mesh,
                Z,
                shading="auto",
                cmap=None,
                facecolor=(0.2, 0.6, 1.0, 0.35),
                edgecolors="none",
            )

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.tick_params(axis="x", labelrotation=90)

    def update_plot(self):
        self.ax.clear()
        self.generate_plot(self.plot_index)
        self.canvas.draw()


class ControlPanelMLS(QWidget):
    def __init__(self, parent, plot_window):
        super().__init__(parent)
        self.plot_window = plot_window

        self.extraction_state = self.parent().extraction_state
        self.flight_pairs = self.parent().flight_pairs
        self.output_dir = self.parent().output_dir
        self.time_windows = self.parent().time_windows

        self.execute_limatch = False

        self.initUI_panel()
        self.update_labels()

    def initUI_panel(self):
    
        self.setFixedWidth(320)
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Pair label
        self.flight_label = QLabel("")
        layout.addWidget(self.flight_label)

        # Time window labels
        self.layout_dividerLine(layout)

        self.gps_label = QLabel("")
        layout.addWidget(self.gps_label)

        layout.addStretch()

        self.layout_dividerLine(layout)

        # LiMatch checkbox
        self.limatch_checkbox = QCheckBox("Execute LiMatch")
        self.limatch_checkbox.stateChanged.connect(self.toggle_limatch)
        self.execute_limatch = False
        layout.addWidget(self.limatch_checkbox)

        self.layout_dividerLine(layout)

        # Output dir
        output_layout = QHBoxLayout()
        self.output_lineedit = QLineEdit(self.output_dir)
        output_layout.addWidget(self.output_lineedit)

        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_browse_btn)
        layout.addLayout(output_layout)

        layout.addStretch()

        self.layout_dividerLine(layout)

        # Extract button
        self.extract_button = QPushButton("Extract MLS overlap")
        self.extract_button.clicked.connect(self.proceed_extraction)
        layout.addWidget(self.extract_button)

    
        self.layout_dividerLine(layout)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(self.previous_plot)
        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self.next_plot)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        self.setLayout(layout)

    def layout_dividerLine(self, layout):
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

    def toggle_limatch(self, state):
        self.execute_limatch = (state == 2)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if folder:
            self.output_dir = folder
            self.output_lineedit.setText(folder)
            self.parent().output_dir = folder

    def update_labels(self):
        idx = self.plot_window.plot_index
        self.flight_label.setText(f"Flight pairs: {self.flight_pairs[idx]}")

        (tmin_i, tmax_i), (tmin_j, tmax_j) = self.time_windows[idx]

        if not np.isfinite(tmin_i) or not np.isfinite(tmin_j):
            self.gps_label.setText("GPS Times:\n(no spatial overlap)")
            return

        txt = (
            "GPS Times:\n"
            "PC1:\n"
            f"  min: {tmin_i:.3f}\n"
            f"  max: {tmax_i:.3f}\n\n"
            "PC2:\n"
            f"  min: {tmin_j:.3f}\n"
            f"  max: {tmax_j:.3f}"
        )
        self.gps_label.setText(txt)

    def proceed_extraction(self):
        self.execute_limatch = self.limatch_checkbox.isChecked()
        self.extraction_state = True
        self.parent().extraction_state = True
        self.window().close()

    def previous_plot(self):
        self.plot_window.plot_index = (self.plot_window.plot_index - 1) % self.plot_window.num_plots
        self.plot_window.update_plot()
        self.update_labels()

    def next_plot(self):
        self.plot_window.plot_index = (self.plot_window.plot_index + 1) % self.plot_window.num_plots
        self.plot_window.update_plot()
        self.update_labels()


class GUIMainWindowMLS(QMainWindow):
    def __init__(self, superpositions, raster_mesh, raster, time_windows, extraction_state, flight_pairs, output_dir, crs):
        super().__init__()
        self.crs = crs

        self.setWindowTitle("MLS Overlap UI")
        self.setGeometry(100, 100, 950, 500)

        self.superpositions = superpositions
        self.raster_mesh = raster_mesh
        self.raster = raster
        self.time_windows = time_windows
        self.extraction_state = extraction_state
        self.flight_pairs = flight_pairs
        self.output_dir = output_dir

        self.initUI()

    def initUI(self):
        self.plot_window = PlotWindowMLS(self)
        self.control_panel = ControlPanelMLS(self, self.plot_window)

        title_bar = make_title_bar("MLS Patcher")

        panels_layout = QHBoxLayout()
        panels_layout.addWidget(self.control_panel)
        panels_layout.addWidget(self.plot_window)

        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(title_bar)
        main_layout.addLayout(panels_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

def make_title_bar(text: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedHeight(42)
    label.setStyleSheet("""
        QLabel {
            background-color: #2b2b2b;
            color: #e6e6e6;
            font-size: 16px;
            font-weight: bold;
            border-bottom: 1px solid #444;
        }
    """)
    return label