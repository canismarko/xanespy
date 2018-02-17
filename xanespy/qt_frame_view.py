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
            self.view.presenter_frame_changed.connect(self._on_change)
            self._is_running = True
    
    def stop(self):
        if self._is_running:
            log.debug("Stopping FrameChangeSource in thread %d", threading.get_ident())
            self.view.presenter_frame_changed.disconnect(self._on_change)
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
    file_opened = QtCore.pyqtSignal('QString', arguments=('filename',))
    expand_hdf_tree = QtCore.pyqtSignal()
    frame_slider_moved = QtCore.pyqtSignal(int)
    presenter_frame_changed = QtCore.pyqtSignal(int)
    new_vrange_requested = QtCore.pyqtSignal(float, float)
    reset_vrange_requested = QtCore.pyqtSignal()
    new_timestep_requested = QtCore.pyqtSignal(int)
    new_cmap_requested = QtCore.pyqtSignal(str)
    new_component_requested = QtCore.pyqtSignal(str)
    play_button_clicked = QtCore.pyqtSignal(bool)
    forward_button_clicked = QtCore.pyqtSignal()
    back_button_clicked = QtCore.pyqtSignal()
    first_button_clicked = QtCore.pyqtSignal()
    last_button_clicked = QtCore.pyqtSignal()
    play_speed_requested = QtCore.pyqtSignal(int)
    
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
        # Connect UI signals to view signals
        self.ui.sldFrameSlider.valueChanged.connect(self.frame_slider_moved)
        self.ui.btnApplyLimits.clicked.connect(self.request_new_vrange)
        self.ui.btnResetLimits.clicked.connect(self.reset_vrange_requested)
        self.ui.cmbTimestep.currentIndexChanged.connect(self.new_timestep_requested)
        self.ui.cmbCmap.currentTextChanged.connect(self.new_cmap_requested)
        self.ui.cmbComponent.currentTextChanged.connect(self.new_component_requested)
        self.ui.btnPlay.toggled.connect(self.play_button_clicked)
        self.ui.btnForward.clicked.connect(self.forward_button_clicked)
        self.ui.btnBack.clicked.connect(self.back_button_clicked)
        self.ui.btnFirst.clicked.connect(self.first_button_clicked)
        self.ui.btnLast.clicked.connect(self.last_button_clicked)
        self.ui.sldPlaySpeed.valueChanged.connect(self.play_speed_requested)
    
    def request_new_vrange(self, bool):
        vmin = self.ui.spnVMin.value()
        vmax = self.ui.spnVMax.value()
        self.new_vrange_requested.emit(vmin, vmax)
    
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
    
    def connect_presenter(self, presenter):
        presenter.frame_data_changed.connect(self._animate_frames)
        presenter.frame_data_changed.connect(self.draw_histogram)
        presenter.mean_spectrum_changed.connect(self.draw_spectrum)
        presenter.active_frame_changed.connect(self.presenter_frame_changed.emit)
        presenter.active_frame_changed.connect(self.move_slider)
        presenter.frame_vrange_changed.connect(self.change_vrange)
        presenter.active_energy_changed.connect(self.set_status_energy)
    
    def connect_signals(self, presenter):
        # Connect internal signals and slots
        self.expand_hdf_tree.connect(self.ui.hdfTree.expandAll)
        # self.ui.sldFrameSlider.valueChanged.connect(presenter.move_slider)
        self.ui.btnRefresh.clicked.connect(presenter.refresh_frames)
        self.ui.hdfTree.currentItemChanged.connect(presenter.change_hdf_group)
        # Use the media control stype buttons to change presenter frames
        # self.ui.btnPlay.toggled.connect(presenter.play_frames)
        # self.ui.btnForward.clicked.connect(presenter.next_frame)
        # self.ui.btnBack.clicked.connect(presenter.previous_frame)
        # self.ui.btnFirst.clicked.connect(presenter.first_frame)
        # self.ui.btnLast.clicked.connect(presenter.last_frame)
        # self.ui.sldPlaySpeed.valueChanged.connect(presenter.set_play_speed)
        # Opening a new HDF file
        self.ui.actionOpen.triggered.connect(self.open_hdf_file)
        # Connect a handler for when the user hovers over the frame
        def hover_frame(event):
            if event.inaxes is self.img_ax:
                xy = xycoord(x=event.xdata, y=event.ydata)
            else:
                xy = None
            presenter.hover_frame_pixel(xy)
        self.fig.canvas.mpl_connect('motion_notify_event', hover_frame)
    
    def open_hdf_file(self):
        # Ask the user for an HDF file to open
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "QFileDialog.getOpenFileName()", "",
            "All Files (*);;HDF5 Files (*.h5 *.hdf *.hdf5)",
            'HDF5 Files (*.h5 *.hdf *.hdf5)')
        self.file_opened.emit(filename)
        return filename
    
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
    
    def move_slider(self, val):
        self.ui.sldFrameSlider.setValue(val)
    
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
    
    def draw_spectrum(self, spectrum, fitted_spectrum, norm, cmap, edge_range=None):
        self.spectrum_ax.clear()
        plots.plot_spectrum(spectrum=spectrum.values,
                            energies=spectrum.index,
                            norm=norm,
                            cmap=cmap,
                            color="y",
                            ax=self.spectrum_ax)
        # Now plot a zoomed in version on the edge itself
        self.edge_ax.clear()
        plots.plot_spectrum(spectrum=spectrum.values,
                                  energies=spectrum.index,
                                  norm=norm,
                                  cmap=cmap,
                                  color="y",
                                  ax=self.edge_ax)
        if edge_range is not None:
            self.edge_ax.set_xlim(*edge_range)
    
    def draw_histogram(self, data, energies, norm, cmap):
        log.debug("Plotting frame histogram")
        # Update the histogram
        self.hist_ax.clear()
        plots.plot_txm_histogram(data=data, ax=self.hist_ax,
                                 cmap=cmap, norm=norm, add_cbar=False)
        self.hist_ax.xaxis.set_visible(False)
        # Update the colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        self.cbar.on_mappable_changed(mappable)
        self.cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, num=5))
    
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

    def change_vrange(self, vmin, vmax, step, decimals):
        """Change the range of frame map colors.
        
        Does not affect the rendered frames."""
        self.ui.spnVMin.setValue(vmin)
        self.ui.spnVMax.setValue(vmax)
        self.ui.spnVMin.setSingleStep(step)
        self.ui.spnVMax.setSingleStep(step)
        self.ui.spnVMin.setDecimals(decimals)
        self.ui.spnVMax.setDecimals(decimals)
        # Set the min and max of the spin boxes so they don't cross
        self.ui.spnVMin.setMaximum(vmax - step)
        self.ui.spnVMax.setMinimum(vmin + step)
    
    # def set_vmin(self, val):
    #     self.ui.spnVMin.setValue(val)
    
    # def set_vmax(self, val):
    #     self.ui.spnVMax.setValue(val)
    
    # def set_vmin_decimals(self, val):
    #     self.ui.spnVMin.setDecimals(val)
    
    # def set_vmax_decimals(self, val):
    #     self.ui.spnVMax.setDecimals(val)
    
    # def set_vmin_step(self, val):
    #     self.ui.spnVMin.setSingleStep(val)
    
    # def set_vmax_step(self, val):
    #     self.ui.spnVMax.setSingleStep(val)
    
    # def set_vmin_maximum(self, val):
    #     self.ui.spnVMin.setMaximum(val)
    
    # def set_vmax_minimum(self, val):
    #     self.ui.spnVMax.setMinimum(val)
    
    def set_cmap_list(self, cmap_list):
        self.ui.cmbCmap.clear()
        self.ui.cmbCmap.addItems(cmap_list)
        self.ui.cmbCmap.setCurrentText('copper')
    
    def set_component_list(self, component_list):
        self.ui.cmbComponent.clear()
        self.ui.cmbComponent.addItems(component_list)
    
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

    def set_status_energy(self, energy):
        s = "{:.2f} eV".format(energy)
        self.ui.lblEnergy.setText(s)
    
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
