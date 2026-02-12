
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PlotWindowMLS(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.superpositions = self.parent().superpositions
        self.raster_mesh = self.parent().raster_mesh
        self.x_mesh, self.y_mesh = self.raster_mesh
        self.raster = self.parent().raster

        self.plot_index = 0
        self.num_plots = len(self.superpositions)

        self.figure, self.ax = plt.subplots()
        self.figure.set_constrained_layout(True)

        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.update_plot()

    def generate_plot(self, index):
        superpos = self.superpositions[index]

        Z = np.ma.masked_where(~superpos, superpos)

        self.ax.pcolormesh(
            self.x_mesh,
            self.y_mesh,
            Z,
            shading="auto",
            cmap=None,
            facecolor=(0.0, 0.0, 1.0, 0.4),
            edgecolors="none",
        )

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("E [m]")
        self.ax.set_ylabel("N [m]")
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

        tw = self.time_windows[idx]

        if isinstance(tw, (tuple, list)) and len(tw) == 2 and np.isscalar(tw[0]) and np.isscalar(tw[1]):
            t0, t1 = float(tw[0]), float(tw[1])
        else:
            (tmin_i, tmax_i), (tmin_j, tmax_j) = tw
            t0 = max(tmin_i, tmin_j)
            t1 = min(tmax_i, tmax_j)

        if not np.isfinite(t0) or not np.isfinite(t1) or t0 >= t1:
            self.gps_label.setText(
                "GPS time intersection:\n"
                "tmin: NaN\n"
                "tmax: NaN"
            )
            return

        self.gps_label.setText(
            "GPS time intersection:\n"
            f"tmin: {t0:.3f}\n"
            f"tmax: {t1:.3f}"
        )

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
    def __init__(self, superpositions, raster_mesh, raster, time_windows, extraction_state, flight_pairs, output_dir):
        super().__init__()

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

        title_bar = make_title_bar("MLS Overlap Extraction")

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