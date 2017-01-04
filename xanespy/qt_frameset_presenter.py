import logging
import math

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

import exceptions

from utilities import xy_to_pixel


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
    active_representation = 'intensities'
    frame_pixel = None
    num_frames = 0
    timer = None

    def __init__(self, frameset, map_view, frame_view):
        self.frameset = frameset
        self.num_frames = frameset.num_energies
        self.map_view = map_view
        self.frame_view = frame_view
        # Switch to absorbances representation if it's valid
        if self.frameset.has_representation('absorbances'):
            self.active_representation = 'absorbances'
        else:
            self.active_representation = 'intensities'

    @property
    def dirty(self):
        return self._dirty

    @dirty.setter
    def dirty(self, val):
        self._dirty = val
        self.frame_view.show_status_message("Dirty")

    def prepare_ui(self):
        log.debug("Preparing UI from {}".format(self))
        self.create_app()
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
        self.update_status_shape()
        # Update some widgets
        num_energies = self.active_frames().shape[0]
        self.frame_view.set_slider_max(num_energies - 1)
        # Prepare data for the HDF tree view
        self.build_hdf_tree()
        # Connect response signals for the widgets
        self.frame_view.connect_signals(presenter=self)
        self.frame_view.frame_changed.connect(self.update_status_frame)
        self.frame_view.frame_changed.connect(self.update_status_value)
        # Create a timer for playing through all the frames
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.set_play_speed(15)

    def create_app(self):  # pragma: no cover
        self.app = QtWidgets.QApplication([])
        self.app.setApplicationName("Xanespy")

    def launch(self):  # pragma: no cover
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
        self.frame_view.set_vmin_decimals(decimals)
        self.frame_view.set_vmax_decimals(decimals)
        step = 1 / 10**decimals
        self.frame_view.set_vmin_step(step)
        self.frame_view.set_vmax_step(step)
        # Now set new limits on the view's vmin and vmax
        self.frame_view.set_vmin_maximum(self._frame_vmax)
        self.frame_view.set_vmax_minimum(self._frame_vmin)

    def set_frame_vmin(self, new_value):
        log.debug("Changing frame vmin to %f", new_value)
        # assert new_value <= self._frame_vmax, "Vmin must be less than Vmax"
        self._frame_vmin = new_value
        # self.update_frame_range_limits()
        self.frame_view.set_vmin(self._frame_vmin)
        self.dirty = True

    def set_frame_vmax(self, new_value):
        log.debug("Changing frame vmax to %f", new_value)
        # assert new_value >= self._frame_vmin, "Vmax must be greater than Vmin"
        self._frame_vmax = new_value
        # self.update_frame_range_limits()
        self.frame_view.set_vmax(self._frame_vmax)
        self.dirty = True

    def reset_frame_range(self):
        """Reset the frame plotting vmin and vmax based on the currently
        selected data. VMin will be the 2nd percentile and VMax will
        be the 98th percentile."""
        try:
            data = self.active_frames()
        except exceptions.GroupKeyError:
            # We're not on a valid frameset, so pass
            pass
        else:
            # Calculate the relevant perctile intervals
            p_lower = np.percentile(data, 1)
            p_upper = np.percentile(data, 99)
            # Update the view with the new values
            self._frame_vmin = p_lower
            self._frame_vmax = p_upper
            self.set_frame_vmin(p_lower)
            self.set_frame_vmax(p_upper)
            self.dirty = True

    def frame_norm(self):
        norm = Normalize(vmin=self._frame_vmin, vmax=self._frame_vmax, clip=True)
        return norm

    def draw_frame_spectra(self):
        frame_spectrum = self.frameset.spectrum(
            edge_jump_filter=True,
            representation=self.active_representation,
            index=self.active_timestep)
        self.frame_view.draw_spectrum(frame_spectrum.values,
                                      energies=frame_spectrum.index,
                                      cmap=self.frame_cmap,
                                      norm=self.frame_norm(),
                                      edge_range=self.frameset.edge.edge_range)

    def draw_frame_histogram(self):
        frames = self.active_frames()
        self.frame_view.draw_histogram.emit(frames,
                                            self.frame_norm(),
                                            self.frame_cmap)

    def change_cmap(self, new_cmap):
        if not self.frame_cmap == new_cmap:
            log.debug("Changing cmap from %s to %s", self.frame_cmap, new_cmap)
            self.frame_cmap = new_cmap
            self.dirty = True
            self.refresh_frames()

    def build_hdf_tree(self):
        """Build the items and insert them into the view's HDF tree based on
        the structure of the frameset's HDF file.

        """
        icons = {
            'metadata': QtGui.QIcon.fromTheme('x-office-spreadsheet'),
            'frameset': QtGui.QIcon.fromTheme('emblem-photos'),
            'map': QtGui.QIcon.fromTheme('image-x-generic'),
        }
        active_path = self.frameset.hdf_path(self.active_representation)
        self._active_tree_item = None

        # Recursive helper function for building the tree
        def tree_item(data, parent=None):
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, data['name'])
            item.setText(2, data['path'])
            if data['context'] is not None:
                item.setText(1, data['context'])
                item.setIcon(1, icons[data['context']])
            # Now recursively add children
            for child in data['children']:
                new_item = tree_item(child)
                ret = item.addChild(tree_item(child))
            # See if we're looking at the active representation
            if data['path'] == active_path:
                self._active_tree_item = item
            return item

        # Start building the tree
        for child in self.frameset.data_tree():
            new_root_item = tree_item(child)
            # Disable any items that are outside this parent group
            if child['name'] != self.frameset.parent_name:
                new_root_item.setDisabled(True)
            # Now add the new root to the tree
            self.frame_view.add_hdf_tree_item(new_root_item)

        # Select the currently active item and expand its ancestors
        if self._active_tree_item is not None:
            self._active_tree_item.setSelected(True)
            self.frame_view.select_active_hdf_item(self._active_tree_item)
            ancestor = self._active_tree_item.parent()
            while ancestor is not None:
                ancestor.setExpanded(True)
                ancestor = ancestor.parent()

    def refresh_frames(self):
        if self.active_representation is not None:
            # A valid set of frames is available, so plot them
            self.draw_frame_histogram()
            self.draw_frame_spectra()
            self.animate_frames()
            self.dirty = False
        else:
            # Invalid frame data, so just clear the axes
            self.frame_view.clear_axes()
        # Update the status bar with new frame data
        self.update_status_shape()

    def active_frames(self):
        frames = self.frameset.frames(timeidx=self.active_timestep,
                                      representation=self.active_representation)
        return frames

    def animate_frames(self):
        frames = self.active_frames()
        extent = self.frameset.extent(self.active_representation)
        energies = self.frameset.energies(timeidx=self.active_timestep)
        log.debug("Animating frames from presenter")
        self.frame_view.draw_frames.emit(frames,
                                         energies,
                                         self.frame_norm(),
                                         self.frame_cmap,
                                         extent)

    def hover_frame_pixel(self, xy):
        if xy is None:
            # Update the UI to clear the cursor data
            cursor_s = ""
            pixel_s = ""
            self.frame_pixel = None
        else:
            # Update the UI with the cursor location
            cursor_s = "({x:.2f}, {y:.2f})".format(x=xy.x, y=xy.y)
            # Convert from (x, y) to (row, col)
            extent = self.frameset.extent(self.active_representation)
            shape = self.frameset.frame_shape()
            pixel = xy_to_pixel(xy, extent=extent, shape=shape)
            pixel_s = "[{row}, {col}]".format(row=pixel.vertical,
                                              col=pixel.horizontal)
            self.frame_pixel = pixel
        # Now update the UI
        self.frame_view.set_status_pixel(pixel_s)
        self.frame_view.set_status_cursor(cursor_s)
        self.update_status_value()

    def update_status_value(self):
        px = self.frame_pixel
        if px is not None:
            # Get the value of this pixel from the frame data
            value_s = str(self.active_frames()[self.active_frame][px])
        else:
            value_s = ""
        self.frame_view.set_status_value(value_s)

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
        log.debug("Changing play speed: Setting %d -> %d ms", new_speed, new_interval)
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

    def update_status_shape(self):
        try:
            shape = self.active_frames().shape[-2:]
        except exceptions.GroupKeyError:
            # No valid frames so show a placeholder
            s = "---"
        else:
            s = "({rows}, {cols})"
            s = s.format(rows=shape[0], cols=shape[1])
        # Set the UI's text
        self.frame_view.set_status_shape(s)

    def update_status_frame(self, new_frame):
        """Create a string (and send it to the UI) that indicates the energy
        of the requested frame."""
        self.frame_view.set_status_index(str(new_frame))
        energy = self.frameset.energies(self.active_timestep)[new_frame]
        s = "{:.2f} eV".format(energy)
        self.frame_view.set_status_energy(s)

    def change_hdf_group(self, new_item, old_item):
        is_selectable = new_item.text(1) in ["frameset", "map"]
        old_name = old_item.text(0) if old_item is not None else "None"
        # Figure out the path for this group and set the new data_name
        path = new_item.text(2).split('/')
        if len(path) > 2:
            self.frameset.data_name = path[2]
        # Set the active representation and data groups
        if is_selectable:
            # A valid representation was chosen, so save it for future plotting
            new_representation = new_item.text(0)
        else:
            # A non-leaf node was chosen, so no representation
            new_representation = None
        self.active_representation = new_representation
        log.debug("Changing representation %s -> %s", old_name, new_representation)
        log.debug("New HDF data path: %s", path)
        self.reset_frame_range()
        self.refresh_frames()
