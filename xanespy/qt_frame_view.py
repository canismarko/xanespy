import os
import logging
import math
from time import time
import threading

from PyQt5 import QtWidgets, uic, QtCore
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
    _hist_animation = None

    # Signals
    frame_changed = QtCore.pyqtSignal(int)
    draw_frames = QtCore.pyqtSignal(np.ndarray, np.ndarray, object, 'QString',
                                    arguments=('frames', 'energies', 'norm', 'cmap'))
    draw_histogram = QtCore.pyqtSignal(np.ndarray, object, 'QString',
                                       arguments=('data', 'norm', 'cmap'))

    def setup(self):
        # Load the Qt Designer .ui file
        starttime = time()
        Ui_FrameWindow, QMainWindow = uic.loadUiType(UI_FILE)
        log.debug("Built UI using uic in %d sec", time() - starttime) 
        # Create the UI elements
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_FrameWindow()
        self.ui.setupUi(self.window)
        self.ui.hdfTree.setHeaderLabels(["Dataset", "Type"])
        # Add some labels to the statusbar
        self.lblStatus = QtWidgets.QLabel()
        self.lblStatus.setText("")
        self.ui.statusbar.addWidget(self.lblStatus)
        # Connect the signal for drawing the frames, histogram, etc.
        self.create_canvas()
        # Create animations from the collected artists
        self.source = FrameChangeSource(view=self)
        self.draw_frames.connect(self._animate_frames)
        self.draw_histogram.connect(self._draw_histogram)

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
        # Inform the presenter of updates to the UI
        self.ui.spnVMin.valueChanged.connect(presenter.set_frame_vmin)
        self.ui.spnVMax.valueChanged.connect(presenter.set_frame_vmax)
        self.ui.btnResetLimits.clicked.connect(presenter.reset_frame_range)
        self.ui.btnApplyLimits.clicked.connect(presenter.refresh_frames)
        self.ui.cmbTimestep.currentIndexChanged.connect(presenter.set_timestep)
        self.ui.cmbCmap.currentTextChanged.connect(presenter.change_cmap)
        self.ui.sldFrameSlider.valueChanged.connect(presenter.move_slider)
        self.ui.btnRefresh.clicked.connect(presenter.refresh_frames)
        # Use the media control stype buttons to change presenter frames
        self.ui.btnPlay.toggled.connect(presenter.play_frames)
        self.ui.btnForward.clicked.connect(presenter.next_frame)
        self.ui.btnBack.clicked.connect(presenter.previous_frame)
        self.ui.btnFirst.clicked.connect(presenter.first_frame)
        self.ui.btnLast.clicked.connect(presenter.last_frame)
        self.ui.sldPlaySpeed.valueChanged.connect(presenter.set_play_speed)
        # Update the UI from the presenter
        self.frame_changed.connect(self.ui.sldFrameSlider.setValue)

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
        
    def _animate_frames(self, frames, energies, norm, cmap):
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
                                           cmap=cmap, ax=self.img_ax)
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
        # source = FrameChangeSource(view=self)
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

    def set_timestep_list(self, timestep_list):
        self.ui.cmbTimestep.clear()
        self.ui.cmbTimestep.addItems(timestep_list)

    def set_timestep(self, idx):
        self.ui.cmbTimestep.setCurrentIndex(idx)

    def show_status_message(self, message):
        self.lblStatus.setText(message)
