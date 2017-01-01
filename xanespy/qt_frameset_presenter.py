import logging
import math

from PyQt5 import QtWidgets, QtCore
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np


CMAPS = ['viridis', 'inferno', 'plasma', 'magma', 'gray', 'bone',
         'copper', 'autumn', 'spring', 'summer', 'winter', 'spectral',
         'rainbow', 'gnuplot']
log = logging.getLogger(__name__)


class QtFramesetPresenter():
    frame_cmap = "copper"
    _frame_vmin = 0
    _frame_vmax = 0
    map_cmap = "plasma"
    active_frame = 0
    active_timestep = 0
    num_frames = 0
    timer = None

    def __init__(self, frameset, map_view, frame_view):
        self.frameset = frameset
        self.num_frames = frameset.num_energies
        self.map_view = map_view
        self.frame_view = frame_view

    @property
    def dirty(self):
        return self._dirty

    @dirty.setter
    def dirty(self, val):
        self._dirty = val
        self.frame_view.show_status_message("Dirty")

    def prepare_ui(self):
        log.debug("Preparing UI from {}".format(self))
        self.app = QtWidgets.QApplication([])
        self.frame_view.setup()
        # Set the colormap lists with valid colormaps
        cmap_list = sorted(list(plt.cm.datad.keys()))
        self.frame_view.set_cmap_list(CMAPS)
        self.frame_view.set_cmap(self.frame_cmap)
        # Set the list of possible timesteps
        with self.frameset.store() as store:
            tslist = ["{} - {}".format(idx, ts)
                      for idx, ts in enumerate(store.timestep_names)]
        self.frame_view.set_timestep_list(tslist)
        self.frame_view.set_timestep(self.active_timestep)
        # Do some preliminary plotting
        self.reset_frame_range()
        self.draw_frame_spectra()
        self.draw_frame_histogram()
        self.animate_frames()
        # Update some widgets
        num_energies = self.frameset.frames(self.active_timestep).shape[0]
        self.frame_view.set_slider_max(num_energies - 1)
        # Connect response signals for the widgets
        self.frame_view.connect_signals(presenter=self)
        # Create a timer for playing through all the frames
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.set_play_speed(15)

    def launch(self):
        # Show the graphs and launch to event loop
        self.frame_view.show()
        return self.app.exec_()

    def set_timestep(self, new_timestep):
        self.active_timestep = new_timestep
        self.reset_frame_range()
        self.draw_frame_histogram()
        self.draw_frame_spectra()
        self.refresh_frames()

    def move_slider(self, new_value):
        new_frame = new_value
        if not self.active_frame == new_frame:
            self.active_frame = new_frame
            self.frame_view.frame_changed.emit(self.active_frame)

    def update_frame_range_limits(self):
        """Check that the (min, max, step) of the frame spinboxes are
        reasonable. This should be called when the spinbox values change."""
        # Round the step to the nearest multiple of ten
        decimals = - math.floor(math.log10(self._frame_vmax - self._frame_vmin) - 1)
        self.frame_view.set_vmax_decimals(decimals)
        self.frame_view.set_vmin_decimals(decimals)
        step = 1 / 10**decimals
        self.frame_view.set_vmin_step(step)
        self.frame_view.set_vmax_step(step)
        # Now set new limits on the view's vmin and vmax
        self.frame_view.set_vmin_maximum(self._frame_vmax)
        self.frame_view.set_vmax_minimum(self._frame_vmin)

    def set_frame_vmin(self, new_value):
        log.debug("Changing frame vmin to %f", new_value)
        assert new_value <= self._frame_vmax, "Vmin must be less than Vmax"
        self._frame_vmin = new_value
        self.update_frame_range_limits()
        self.frame_view.set_vmin(self._frame_vmin)
        self.dirty = True

    def set_frame_vmax(self, new_value):
        log.debug("Changing frame vmax to %f", new_value)
        assert new_value >= self._frame_vmin, "Vmax must be greater than Vmin"
        self._frame_vmax = new_value
        self.update_frame_range_limits()
        self.frame_view.set_vmax(self._frame_vmax)
        self.dirty = True

    def reset_frame_range(self):
        """Reset the frame plotting vmin and vmax based on the currently
        selected data. VMin will be the 2nd percentile and VMax will
        be the 98th percentile."""
        data = self.frameset.frames(self.active_timestep)
        # Calculate the relevant perctile intervals
        p_lower = np.percentile(data, 1)
        p_upper = np.percentile(data, 99)
        # Update the view with the new values
        self._frame_vmin = p_lower
        self._frame_vmax = p_upper
        self.set_frame_vmax(p_upper)
        self.set_frame_vmin(p_lower)
        self.dirty = True

    def frame_norm(self):
        norm = Normalize(vmin=self._frame_vmin, vmax=self._frame_vmax, clip=True)
        return norm

    def draw_frame_spectra(self):
        frame_spectrum = self.frameset.spectrum(edge_jump_filter=True,
                                                index=self.active_timestep)
        self.frame_view.draw_spectrum(frame_spectrum.values,
                                      energies=frame_spectrum.index,
                                      cmap=self.frame_cmap,
                                      norm=self.frame_norm(),
                                      edge_range=self.frameset.edge.edge_range)

    def draw_frame_histogram(self):
        frames = self.frameset.frames(timeidx=self.active_timestep)
        self.frame_view.draw_histogram.emit(frames.flatten(),
                                            self.frame_norm(),
                                            self.frame_cmap)

    def change_cmap(self, new_cmap):
        if not self.frame_cmap == new_cmap:
            log.debug("Changing cmap from %s to %s", self.frame_cmap, new_cmap)
            self.frame_cmap = new_cmap
            self.dirty = True
            self.refresh_frames()

    def refresh_frames(self):
        self.draw_frame_histogram()
        self.draw_frame_spectra()
        self.animate_frames()
        self.dirty = False
                                      
    def animate_frames(self):
        frames = self.frameset.frames(timeidx=self.active_timestep)
        energies = self.frameset.energies(timeidx=self.active_timestep)
        log.debug("Animating frames from presenter")
        self.frame_view.draw_frames.emit(frames,
                                         energies,
                                         self.frame_norm(),
                                         self.frame_cmap)

    def play_frames(self, start):
        if start:
            log.debug("Playing frames")
            # Create a new timer and start it
            self.play_timer.start()
        else:
            # Stop the timer and delete it
            log.debug("Stopped playing frames")
            self.play_timer.stop()

    def set_play_speed(self, new_speed):
        """Change how fast the play timer ticks. Input speeds should be in the
        range of 0, 30 with 30 being the fastest. This is converted to timer
        intervals between 1ms and 1000ms on a exponential scale."""
        # Convert from logarithmic scale to linear scale
        log_interval = (30 - new_speed) / 10
        new_interval = int(10**log_interval)
        log.debug("Changing play speed: Setting %d -> %d", new_speed, new_interval)
        self.play_timer.setInterval(new_interval)
            
    def next_frame(self):
        self.active_frame = (self.active_frame + 1) % self.num_frames
        self.frame_view.frame_changed.emit(self.active_frame)

    def previous_frame(self):
        self.active_frame = (self.active_frame - 1) % self.num_frames
        self.frame_view.frame_changed.emit(self.active_frame)

    def first_frame(self):
        self.active_frame = 0
        self.frame_view.frame_changed.emit(self.active_frame)

    def last_frame(self):
        self.active_frame = self.num_frames - 1
        self.frame_view.frame_changed.emit(self.active_frame)
