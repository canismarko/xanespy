import os
import logging
import math
from time import time
import threading

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from matplotlib import animation, cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.animation import ArtistAnimation
import numpy as np

import plots
from utilities import xycoord


UI_FILE = os.path.join(os.path.dirname(__file__), 'qt_frame_window.ui')
log = logging.getLogger(__name__)


class FrameAnimation(animation.ArtistAnimation):  # pragma: no cover
    """Performs the animation for scrolling through frames arbitarily.
    """

    def __init__(self, fig, artists, *args, **kwargs):
        self.fig = fig
        self.artists = artists
        log.debug("Creating animation in thread %d", threading.get_ident())
        # Call parent init methods
        animation.ArtistAnimation.__init__(self, *args, fig=fig,
                                           artists=artists, **kwargs)

    def _step(self, current_idx):
        log.debug("Animating to frame {}".format(current_idx))
        artists = self.artists[current_idx]
        self._draw_next_frame(artists, self._blit)
        return True

    def stop(self):
        log.debug("Stopping animation in thread %d", threading.get_ident())
        return self._stop()


class FrameChangeSource(QtCore.QObject):
    _is_running = False
    callbacks = []

    def __init__(self, view, *args, **kwargs):
        log.debug('creating source')
        self.callbacks = [] # To avoid a Borg callback list
        self.view = view
        super().__init__(*args, **kwargs)

    def add_callback(self, func, *args, **kwargs):
        self.callbacks.append((func, args, kwargs))

    def remove_callback(self, func, *args, **kwargs):
        if args or kwargs:
            self.callbacks.remove((func, args, kwargs))
        else:
            funcs = [c[0] for c in self.callbacks]
            if func in funcs:
                self.callbacks.pop(funcs.index(func))

    def start(self):
        if not self._is_running:
            log.debug("Starting FrameChangeSource in thread %d", threading.get_ident())
            # Listen to the frame adjustment signals
            self.view.frame_changed.connect(self._on_change)
            self._is_running = True

    def stop(self):
        if self._is_running:
            log.debug("Stopping FrameChangeSource in thread %d", threading.get_ident())
            self.view.frame_changed.disconnect(self._on_change)
            self._is_running = False

    def _on_change(self, new_idx):
        for func, args, kwargs in self.callbacks:
            func(new_idx, *args, **kwargs)


class QtFrameView(QtCore.QObject):  # pragma: no cover
    window = None
    ui = None
    fig = None
    spectrum_ax = None
    hist_ax = None
    hist_cb = None
    edge_ax = None
    _frame_animation = None

    # Signals
    expand_hdf_tree = QtCore.pyqtSignal()
    frame_changed = QtCore.pyqtSignal(int)
    draw_frames = QtCore.pyqtSignal(
        object, np.ndarray, object, 'QString', tuple,
        arguments=('frames', 'energies', 'norm', 'cmap', 'extent'))
    draw_histogram = QtCore.pyqtSignal(object, object, 'QString',
                                       arguments=('data', 'norm', 'cmap'))

    def setup(self):
        # Load the Qt Designer .ui file
        Ui_FrameWindow, QMainWindow = uic.loadUiType(UI_FILE)
        log.debug("Built frame window using uic") 
        # Create the UI elements
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_FrameWindow()
        self.ui.setupUi(self.window)
        # Add labels to the HDF Tree widget
        self.ui.hdfTree.setHeaderLabels(['Name', 'Type'])
        header = self.ui.hdfTree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setStretchLastSection(False)
        # Connect the signal for drawing the frames, histogram, etc.
        self.create_canvas()
        # Create animations from the collected artists
        self.source = FrameChangeSource(view=self)
        self.draw_frames.connect(self._animate_frames)
        self.draw_histogram.connect(self._draw_histogram)

    def create_status_bar(self):
        self.status_layout = QtWidgets.QVboxLayout()
        self.ui.statusbar.layout().addItem(QtWidgets.QSpacerItem(100, 20))
        # Indicator for the current frame shape
        self.ui.statusbar.addWidget(QtWidgets.QLabel("Shape:"))
        self.lblShape = QtWidgets.QLabel()
        self.lblShape.setMinimumWidth(120)
        self.ui.statusbar.addWidget(self.lblShape)
        # Indicator for the current index
        self.ui.statusbar.addWidget(QtWidgets.QLabel("Frame:"))
        self.lblIndex = QtWidgets.QLabel()
        self.lblIndex.setMinimumWidth(30)
        self.ui.statusbar.addWidget(self.lblIndex)
        # Indicator for the current energy
        self.ui.statusbar.addWidget(QtWidgets.QLabel("Energy:"))
        self.lblEnergy = QtWidgets.QLabel()
        self.lblEnergy.setMinimumWidth(100)
        self.ui.statusbar.addWidget(self.lblEnergy)
        # Indicator for the current cursor position
        self.ui.statusbar.addWidget(QtWidgets.QLabel("Cursor x,y:"))
        self.lblCursor = QtWidgets.QLabel()
        self.lblCursor.setMinimumWidth(120)
        self.ui.statusbar.addWidget(self.lblCursor)
        # Indicator for the current pixel position
        self.ui.statusbar.addWidget(QtWidgets.QLabel("Pixel:"))
        self.lblPixel = QtWidgets.QLabel()
        self.lblPixel.setMinimumWidth(100)
        self.ui.statusbar.addWidget(self.lblPixel)
        # Indicator for the current cursor value
        self.ui.statusbar.addWidget(QtWidgets.QLabel("Value:"))
        self.lblValue = QtWidgets.QLabel()
        self.lblValue.setMinimumWidth(100)
        self.ui.statusbar.addWidget(self.lblValue)

    def create_canvas(self):
        # Add the canvas to the UI
        self.fig = Figure()
        canvas = FigureCanvas(self.fig)
        self.ui.mainLayout.addWidget(canvas)
        # Add subplots
        gridspec = GridSpec(2, 4)
        self.img_ax = self.fig.add_subplot(
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
        self.cbar.ax.set_xlabel("Intensity")
        # Adjust the margins
        self.fig.tight_layout(pad=0)
        self.fig.canvas.draw_idle()

    def connect_signals(self, presenter):
        # Connect internal signals and slots
        self.expand_hdf_tree.connect(self.ui.hdfTree.expandAll)
        # Inform the presenter of updates to the UI
        self.ui.spnVMin.valueChanged.connect(presenter.set_frame_vmin)
        self.ui.spnVMax.valueChanged.connect(presenter.set_frame_vmax)
        self.ui.btnResetLimits.clicked.connect(presenter.reset_frame_range)
        self.ui.btnApplyLimits.clicked.connect(presenter.refresh_frames)
        self.ui.cmbTimestep.currentIndexChanged.connect(presenter.set_timestep)
        self.ui.cmbCmap.currentTextChanged.connect(presenter.change_cmap)
        self.ui.sldFrameSlider.valueChanged.connect(presenter.move_slider)
        self.ui.btnRefresh.clicked.connect(presenter.refresh_frames)
        self.ui.hdfTree.currentItemChanged.connect(presenter.change_hdf_group)
        # Use the media control stype buttons to change presenter frames
        self.ui.btnPlay.toggled.connect(presenter.play_frames)
        self.ui.btnForward.clicked.connect(presenter.next_frame)
        self.ui.btnBack.clicked.connect(presenter.previous_frame)
        self.ui.btnFirst.clicked.connect(presenter.first_frame)
        self.ui.btnLast.clicked.connect(presenter.last_frame)
        self.ui.sldPlaySpeed.valueChanged.connect(presenter.set_play_speed)
        # Update the UI from the presenter
        self.frame_changed.connect(self.ui.sldFrameSlider.setValue)
        # Connect a handler for when the user hovers over the frame
        def hover_frame(event):
            if event.inaxes is self.img_ax:
                xy = xycoord(x=event.xdata, y=event.ydata)
            else:
                xy = None
            presenter.hover_frame_pixel(xy)
        self.fig.canvas.mpl_connect('motion_notify_event', hover_frame)

    @property
    def frame_controls(self):
        """Gives a list of all the UI buttons that are associated with
        changing the currently active frame."""
        widgets = [self.ui.btnFirst, self.ui.btnBack, self.ui.btnPlay,
                   self.ui.btnRefresh, self.ui.btnForward, self.ui.btnLast,
                   self.ui.sldPlaySpeed, self.ui.sldFrameSlider]
        return widgets

    @property
    def plotting_controls(self):
        """Gives a list of all the UI elements that are associated with
        changing how the frameset is plotted."""
        widgets = [self.ui.cmbTimestep, self.ui.cmbComponent, self.ui.cmbCmap,
                   self.ui.spnVMin, self.ui.spnVMax,
                   self.ui.btnApplyLimits, self.ui.btnResetLimits]
        return widgets

    def disable_frame_controls(self, status):
        log.debug("Disabling frame controls: %s", status)
        for ctrl in self.frame_controls:
            ctrl.setDisabled(status)

    def disable_plotting_controls(self, status):
        log.debug("Disabling plotting: %s", status)
        for ctrl in self.plotting_controls:
            ctrl.setDisabled(status)

    def use_busy_cursor(self, status):
        if status:
            self.window.setCursor(QtCore.Qt.WaitCursor)
        else:
            self.window.unsetCursor()

    def set_drawing_status(self, status):
        if status:
            self.ui.statusbar.showMessage("Drawing...")
        else:
            self.ui.statusbar.clearMessage()

    def set_ui_enabled(self, enable=True):
        """
        Turn on (default) or off the main interactive elements in the
        frame window. Useful for indicating blocking operations.
        """
        self.ui.cmbCmap.setEnabled(enable)
        self.ui.cmbTimestep.setEnabled(enable)
        self.ui.cmbComponent.setEnabled(enable)
        self.ui.btnFirst.setEnabled(enable)
        self.ui.btnBack.setEnabled(enable)
        self.ui.btnMax.setEnabled(enable)
        self.ui.btnPlay.setEnabled(enable)
        self.ui.btnRefresh.setEnabled(enable)
        self.ui.btnForward.setEnabled(enable)
        self.ui.btnLast.setEnabled(enable)
        self.ui.sldFrameSlider.setEnabled(enable)
        self.ui.spnVMin.setEnabled(enable)
        self.ui.spnVMax.setEnabled(enable)
        self.ui.btnApplyLimits.setEnabled(enable)
        self.ui.btnResetLimits.setEnabled(enable)
        
    def _animate_frames(self, frames, energies, norm, cmap, extent):
        # Disconnect the old animation first
        if self._frame_animation:
            log.debug("Disconnecting old frame animations")
            self._frame_animation.stop()
            del self._frame_animation
        # Prepare the axes for new plotting
        artists = []
        self.img_ax.clear()
        # Create the new artists
        start = time()
        for energy, frame in zip(energies, frames):
            im_artist = plots.plot_txm_map(frame, norm=norm,
                                           cmap=cmap, ax=self.img_ax,
                                           extent=extent)
            line_artist = self.spectrum_ax.axvline(energy,
                                                   linestyle=":",
                                                   animated=True,
                                                   color="gray")
            edge_artist = self.edge_ax.axvline(energy,
                                               linestyle=":",
                                               animated=True,
                                               color="gray")
            artists.append([im_artist, line_artist, edge_artist])
        # Create animations from the collected artists
        self._frame_animation = FrameAnimation(
            fig=self.fig,
            artists=artists,
            event_source=self.source,
            blit=True)
        # Redraw the canvas (this part is slow)
        self.fig.canvas.draw_idle()
        # User feedback and logging
        log.debug("Artist creation took %d sec", time() - start)

    def draw_spectrum(self, spectrum, energies, norm, cmap, edge_range):
        self.spectrum_ax.clear()
        plots.plot_xanes_spectrum(spectrum=spectrum,
                                  energies=energies,
                                  norm=norm,
                                  cmap=cmap,
                                  color="y",
                                  ax=self.spectrum_ax)
        # Now plot a zoomed in version on the edge itself
        self.edge_ax.clear()
        plots.plot_xanes_spectrum(spectrum=spectrum,
                                  energies=energies,
                                  norm=norm,
                                  cmap=cmap,
                                  color="y",
                                  ax=self.edge_ax)
        self.edge_ax.set_xlim(*edge_range)

    def _draw_histogram(self, data, norm, cmap):
        # Update the histogram
        self.hist_ax.clear()
        plots.plot_txm_histogram(data=data, ax=self.hist_ax,
                                 cmap=cmap, norm=norm, add_cbar=False)
        self.hist_ax.xaxis.set_visible(False)
        # Update the colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        self.cbar.on_mappable_changed(mappable)

    def show(self):
        self.window.show()

    def clear_axes(self):
        self.img_ax.clear()
        self.spectrum_ax.clear()
        self.hist_ax.clear()
        self.edge_ax.clear()
        self.fig.canvas.draw()

    def set_slider_max(self, val):
        self.ui.sldFrameSlider.setRange(0, val)

    def set_vmin(self, val):
        self.ui.spnVMin.setValue(val)

    def set_vmax(self, val):
        self.ui.spnVMax.setValue(val)

    def set_vmin_decimals(self, val):
        self.ui.spnVMin.setDecimals(val)

    def set_vmax_decimals(self, val):
        self.ui.spnVMax.setDecimals(val)

    def set_vmin_step(self, val):
        self.ui.spnVMin.setSingleStep(val)

    def set_vmax_step(self, val):
        self.ui.spnVMax.setSingleStep(val)

    def set_vmin_maximum(self, val):
        self.ui.spnVMin.setMaximum(val)

    def set_vmax_minimum(self, val):
        self.ui.spnVMax.setMinimum(val)

    def set_cmap_list(self, cmap_list):
        self.ui.cmbCmap.clear()
        self.ui.cmbCmap.addItems(cmap_list)

    def set_cmap(self, cmap):
        self.ui.cmbCmap.setCurrentText(cmap)

    def add_hdf_tree_item(self, item):
        self.ui.hdfTree.addTopLevelItem(item)

    def select_active_hdf_item(self, item):
        self.ui.hdfTree.setCurrentItem(item)

    def set_timestep_list(self, timestep_list):
        self.ui.cmbTimestep.clear()
        self.ui.cmbTimestep.addItems(timestep_list)

    def set_timestep(self, idx):
        self.ui.cmbTimestep.setCurrentIndex(idx)

    def show_status_message(self, message):
        self.ui.statusbar.showMessage(message)

    def set_status_shape(self, msg):
        self.ui.lblShape.setText(msg)

    def set_status_index(self, msg):
        self.ui.lblIndex.setText(msg)

    def set_status_energy(self, msg):
        self.ui.lblEnergy.setText(msg)

    def set_status_cursor(self, msg):
        self.ui.lblCursor.setText(msg)

    def set_status_unit(self, msg):
        self.ui.lblUnit.setText(msg)

    def set_status_pixel(self, msg):
        self.ui.lblPixel.setText(msg)

    def set_status_value(self, msg):
        self.ui.lblValue.setText(msg)

    def set_window_title(self, title):
        self.window.setWindowTitle(title)
