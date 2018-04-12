import os
import logging

from PyQt5 import QtWidgets, uic, QtCore
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)

import plots

log = logging.getLogger(__name__)


UI_FILE = os.path.join(os.path.dirname(__file__), 'qt_map_window.ui')

# A list of colors that give good contrast with each colormap
CROSSHAIR_COLORS = {
    'plasma': '#20f986',  # wintergreen
    'viridis': '#fd5956',  # grapefruit
    'inferno': '#20f987',  # wintergreen
    'magma': '#20f986',  # wintergreen
    'gray': 'red',
    'bone': 'red',
    'copper': 'green',
    'autumn': 'blue',
    'spring': 'blue',
    'summer': 'red',
    'winter': 'red',
    'spectral': 'black',
    'rainbow': 'black',
    'gnuplot': 'green',
}


class QtMapView(QtCore.QObject):
    """A Qt view for a frameset map. It should be controlled by a
    presenter.
    
    Attributes
    ----------
    cmap_changed : signal
      Fires when the user changes the colormap via the UI
    limits_applied : signal
      Fires when the user requests that data be redrawn with new limits.
    limits_reset : signal
      Fires when the user asks that the norm limits be reset to the data.
    
    """
    window = None
    ui = None
    fig = None
    crosshairs = None
    latest_cmap = "plasma"
    
    # Signals
    cmap_changed = QtCore.pyqtSignal('QString')
    component_changed = QtCore.pyqtSignal('QString')
    edge_mask_toggled = QtCore.pyqtSignal(bool)
    spectrum_fit_toggled = QtCore.pyqtSignal(bool)
    map_vmin_changed = QtCore.pyqtSignal(float)
    map_vmax_changed = QtCore.pyqtSignal(float)
    limits_applied = QtCore.pyqtSignal()
    limits_reset = QtCore.pyqtSignal()
    map_hovered = QtCore.pyqtSignal(object, object)
    map_clicked = QtCore.pyqtSignal(object, object)
    map_moved = QtCore.pyqtSignal(int, int)
    
    def setup_ui(self):  # pragma: no cover
        Ui_FrameWindow, QMainWindow = uic.loadUiType(UI_FILE)
        log.debug("Built map window using uic")
        # Create the UI elements
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_FrameWindow()
        self.ui.setupUi(self.window)
        # Create the matplotlib elements
        self.create_canvas()
        # Connect the UI signals to the view signals
        self.ui.cmbCmap.currentTextChanged.connect(self.cmap_changed)
        self.ui.cmbComponent.currentTextChanged.connect(self.component_changed)
        self.ui.chkEdgeMask.toggled.connect(self.edge_mask_toggled)
        self.ui.chkFitSpectrum.toggled.connect(self.spectrum_fit_toggled)
        self.ui.spnVMin.valueChanged.connect(self.map_vmin_changed)
        self.ui.spnVMax.valueChanged.connect(self.map_vmax_changed)
        self.ui.btnApplyLimits.clicked.connect(self.limits_applied)
        self.ui.btnResetLimits.clicked.connect(self.limits_reset)
        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_in_canvas)
        self.fig.canvas.mpl_connect('button_release_event', self.mouse_clicked_canvas)
        self.fig.canvas.mpl_connect('key_press_event', self.keyboard_nav)
    
    def keyboard_nav(self, event):
        # Holds descriptions of how much to move for each key in pixels
        moves = {
            'up': (-1, 0),
            'w': (-1, 0),
            'down': (1, 0),
            's': (1, 0),
            'left': (0, -1),
            'a': (0, -1),
            'right': (0, 1),
            'd': (0, 1),
            'ctrl+up': (-10, 0),
            'ctrl+w': (-10, 0),
            'ctrl+down': (10, 0),
            'ctrl+s': (10, 0),
            'ctrl+left': (0, -10),
            'ctrl+a': (0, -10),
            'ctrl+right': (0, 10),
            'ctrl+d': (0, 10),
        }
        if event.key in moves.keys():
            self.map_moved.emit(*moves[event.key])
    
    def mouse_in_canvas(self, mouse_event):
        if mouse_event.inaxes is self.map_ax:
            self.map_hovered.emit(mouse_event.xdata, mouse_event.ydata)
        else:
            self.map_hovered.emit(None, None)
    
    def mouse_clicked_canvas(self, mouse_event):
        if mouse_event.inaxes is self.map_ax:
            log.debug("Clicked map: x=%f, y=%f", mouse_event.xdata, mouse_event.ydata)
            self.map_clicked.emit(mouse_event.xdata, mouse_event.ydata)
        else:
            self.map_clicked.emit(None, None)
    
    def update_cursor_labels(self, xy, pixel, value):
        xy_s, pixel_s, value_s = "", "", ""
        if xy is not None:
            xy_s = "{x:.2f}, {y:.2f}".format(x=xy.x, y=xy.y)
        if pixel is not None:
            pixel_s = "[{row}, {col}]".format(row=pixel.vertical, col=pixel.horizontal)
        if value is not None:
            value_s = "{}".format(value)
        self.ui.lblCursorXY.setText(xy_s)
        self.ui.lblCursorPixel.setText(pixel_s)
        self.ui.lblCursorValue.setText(value_s)
    
    def update_crosshair_labels(self, xy, pixel, value):
        xy_s, pixel_s, value_s = "", "", ""
        if xy is not None:
            xy_s = "{x:.2f}, {y:.2f}".format(x=xy.x, y=xy.y)
        if pixel is not None:
            pixel_s = "[{row}, {col}]".format(row=pixel.vertical, col=pixel.horizontal)
        if value is not None:
            value_s = "{}".format(value)
        self.ui.lblCrosshairsXY.setText(xy_s)
        self.ui.lblCrosshairsPixel.setText(pixel_s)
        self.ui.lblCrosshairsValue.setText(value_s)
    
    def redraw_crosshairs(self, xy):  # pragma: no cover
        """Draw a set of crosshairs on the map at location given by `xy`."""
        # Remove the old crosshairs first
        if self.crosshairs is not None:
            for artist in self.crosshairs:
                artist.remove()
            self.crosshairs = None
        # Plot new crosshairs on the map axes
        if xy is not None:
            log.debug("Drawing crosshairs at {}".format(xy))
            color = CROSSHAIR_COLORS[self._latest_cmap]
            xartist = self.map_ax.axvline(xy.x, linestyle="--", color=color, linewidth=1)
            yartist = self.map_ax.axhline(xy.y, linestyle="--", color=color, linewidth=1)
            self.crosshairs = [xartist, yartist]
        
    def show(self):  # pragma: no cover
        self.window.show()
    
    def hide(self):  # pragma: no cover
        if hasattr(self.window, 'hide'):
            self.window.hide()
    
    def redraw_canvas(self):  # pragma: no cover
        self.fig.canvas.draw()
    
    def plot_map_data(self, map_data, norm, cmap, extent):  # pragma: no cover
        # First clear the currently selected cursor
        self.map_clicked.emit(None, None)
        # Now plot the new data
        log.debug("Plotting new map data")
        self._latest_cmap = cmap
        self.map_ax.clear()
        if map_data.ndim == 2:
            # Scalar map showing some sort of metrich
            plots.plot_txm_map(map_data, ax=self.map_ax, norm=norm,
                               cmap=cmap, extent=extent)
        elif map_data.ndim == 3:
            # RGB map showing different components
            plots.plot_composite_map(map_data, ax=self.map_ax, extent=extent)
    
    def plot_histogram_data(self, map_data, norm, cmap, extent):  # pragma: no cover
        log.debug("Plotting new map histogram")
        self.hist_ax.clear()
        plots.plot_txm_histogram(map_data, ax=self.hist_ax, norm=norm,
                                 cmap=cmap, add_cbar=False)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        self.cbar.on_mappable_changed(mappable)
    
    def update_spectrum_ui(self, spectrum, fitted_spectrum, norm, cmap,
                      edge_range):  # pragma: no cover
        # Disable the edge_mask option if no edge range
        if edge_range is None:
            self.ui.chkEdgeMask.setEnabled(False)
        else:
            self.ui.chkEdgeMask.setEnabled(True)
    
    def plot_spectrum(self, spectrum, fitted_spectrum, norm, cmap,
                      edge_range):  # pragma: no cover
        log.debug("Plotting new map spectrum")
        # Clear the old axes
        self.spectrum_ax.clear()
        self.edge_ax.clear()
        # Plot the new spectra
        plots.plot_spectrum(
            spectrum=spectrum.values,
            energies=spectrum.index,
            norm=norm,
            color='x',
            cmap=cmap,
            ax=self.spectrum_ax,
        )
        plots.plot_spectrum(
            spectrum=spectrum.values,
            energies=spectrum.index,
            norm=norm,
            color='x',
            cmap=cmap,
            ax=self.edge_ax,
        )
        # Plot the fitted spectrum for comparison
        if fitted_spectrum is not None:
            self.spectrum_ax.plot(
                fitted_spectrum.index,
                fitted_spectrum.values,
                marker='None', linestyle=":")
            self.edge_ax.plot(
                fitted_spectrum.index,
                fitted_spectrum.values,
                marker='None', linestyle=":")
        # Set axes limits on the edge axes
        self.edge_ax.set_xlim(norm.vmin, norm.vmax)
        self.redraw_canvas()
    
    def connect_presenter(self, presenter):  # pragma: no cover
        """Connect to signals for changed presenter state."""
        presenter.map_data_changed.connect(self.show)
        presenter.map_data_changed.connect(self.plot_map_data)
        presenter.map_data_changed.connect(self.plot_histogram_data)
        presenter.map_data_changed.connect(self.redraw_canvas)
        presenter.map_data_cleared.connect(self.hide)
        presenter.map_spectrum_changed.connect(self.plot_spectrum)
        presenter.map_spectrum_changed.connect(self.update_spectrum_ui)
        presenter.cmap_list_changed.connect(self.set_cmap_list)
        presenter.component_list_changed.connect(self.set_component_list)
        presenter.map_limits_changed.connect(self.set_map_limits)
        presenter.map_cursor_changed.connect(self.update_cursor_labels)
        presenter.map_pixel_changed.connect(self.update_crosshair_labels)
        presenter.map_pixel_changed.connect(self.redraw_crosshairs)
        # Connect to signal for when we can draw the UI elements
        presenter.app_ready.connect(self.setup_ui)
        # `map_view` signals received when data changes
    
    def create_canvas(self):  # pragma: no cover
        self.fig = Figure()
        canvas = FigureCanvas(self.fig)
        self.ui.mainLayout.addWidget(canvas)
        canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        # Add subplots
        gridspec = GridSpec(2, 4)
        self.map_ax = self.fig.add_subplot(
            gridspec.new_subplotspec((0, 0), rowspan=2, colspan=2)
        )
        self.spectrum_ax = self.fig.add_subplot(
            gridspec.new_subplotspec((0, 2), rowspan=1, colspan=2)
        )
        self.hist_ax = self.fig.add_subplot(
            gridspec.new_subplotspec((1, 2), rowspan=1, colspan=1)
        )
        self.edge_ax = self.fig.add_subplot(
            gridspec.new_subplotspec((1, 3), rowspan=1, colspan=1)
        )
        # Create the colorbar on the histogram axes
        self.cbar = plots.draw_histogram_colorbar(ax=self.hist_ax,
                                                  cmap="viridis",
                                                  norm=Normalize(0, 1))
        self.cbar.ax.set_xlabel("Map Value")
        # Adjust the margins
        self.fig.tight_layout(pad=0)
        self.fig.canvas.draw_idle()
    
    def set_cmap_list(self, new_list):
        self.ui.cmbCmap.clear()
        self.ui.cmbCmap.addItems(new_list)
        # Check for, and disable, jet colormaps
        jet_idxs = [i for i, x in enumerate(new_list) if x == "jet"]
        for idx in jet_idxs:
            item = self.ui.cmbCmap.model().item(idx)
            item.setEnabled(False)
            item.setToolTip("Sorry, Brian...")
    
    def set_component_list(self, new_list):
        self.ui.cmbComponent.clear()
        self.ui.cmbComponent.addItems(new_list)
    
    def set_map_limits(self, vmin, vmax, step, decimals):
        log.debug('map view received map limits vmin=%f, vmax=%f, step=%f, decimals=%d',
                  vmin, vmax, step, decimals)
        # Temporarily disconnect signals
        oldStateVMin = self.ui.spnVMin.blockSignals(True)
        oldStateVMax = self.ui.spnVMax.blockSignals(True)
        # Update UI values
        # self.ui.spnVMin.setMinimum(vmin - 2 * step)
        self.ui.spnVMin.setMaximum(vmax - 10**-decimals)
        # self.ui.spnVMax.setMaximum(vmax + 2 * step)
        self.ui.spnVMax.setMinimum(vmin + 10**-decimals)
        self.ui.spnVMin.setDecimals(decimals)
        self.ui.spnVMax.setDecimals(decimals)
        self.ui.spnVMin.setSingleStep(step)
        self.ui.spnVMax.setSingleStep(step)
        self.ui.spnVMin.setValue(vmin)
        self.ui.spnVMax.setValue(vmax)
        # Reconnect any disconnected signals
        self.ui.spnVMin.blockSignals(oldStateVMin)
        self.ui.spnVMax.blockSignals(oldStateVMax)
