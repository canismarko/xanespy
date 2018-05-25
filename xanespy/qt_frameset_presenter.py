# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of Xanespy.
#
# Xanespy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Xanespy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Xanespy. If not, see <http://www.gnu.org/licenses/>.


import logging
import math
import sys
import warnings

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import exceptions

from utilities import xy_to_pixel, pixel_to_xy, xycoord, Pixel, get_component


CMAPS = ['plasma', 'viridis', 'inferno', 'magma', 'gray', 'bone',
         'copper', 'autumn', 'spring', 'summer', 'winter', 'spectral',
         'rainbow', 'gnuplot', 'jet']
COMPS = ['real', 'imag', 'modulus', 'phase']
log = logging.getLogger(__name__)


class QtFramesetPresenter(QtCore.QObject):
    """Presenter for showing XanesFrameset frames and maps via Qt.
    
    Attributes
    ----------
    app_ready : pyqtSignal
      Emitted when the application has been created and is ready for drawing.
    busy_status_changed : pyqtSignal
      Emitted when processing has started or ended.
    map_data_changed : pyqtSignal
      Emitted when the map data is different and should be re-plotted.
    map_data_cleared : pyqtSignal
      Emitted when no map data is available and plots can be cleared.
    """
    _frame_vmin = 0
    _frame_vmax = 1
    _map_vmin = 0
    _map_vmax = 1
    map_cmap = "plasma"
    frame_cmap = ""
    active_frame = 0
    active_timestep = 0
    active_representation = None
    active_frame_component = 'real'
    active_map_component = 'real'
    frame_pixel = None
    num_frames = 0
    timer = None
    _map_data = None
    _map_pixel = None
    use_edge_mask = False
    show_spectrum_fit = False
    
    # Signals
    app_ready = QtCore.pyqtSignal()
    busy_status_changed = QtCore.pyqtSignal(bool)
    map_data_changed = QtCore.pyqtSignal(
        np.ndarray, object, 'QString', object,
        arguments=['map_data', 'norm', 'cmap', 'extent'])
    map_data_cleared = QtCore.pyqtSignal()
    process_events = QtCore.pyqtSignal()
    cmap_list_changed = QtCore.pyqtSignal(list)
    component_list_changed = QtCore.pyqtSignal(list)
    timestep_list_changed = QtCore.pyqtSignal(list)
    frame_hover_changed = QtCore.pyqtSignal(object, 'QString', object, object,
                                            arguments=('xy', 'xy_unit', 'px', 'val'))
    hdf_tree_changed = QtCore.pyqtSignal(list)
    hdf_path_changed = QtCore.pyqtSignal('QString')
    mean_spectrum_changed = QtCore.pyqtSignal(
        pd.Series, object, object, 'QString', object,
        arguments=['spectrum', 'fitted_spectrum', 'norm', 'cmap', 'edge_range'])
    map_spectrum_changed = QtCore.pyqtSignal(
        object, object, object, 'QString', object,
        arguments=['spectrum', 'fitted_spectrum', 'norm', 'cmap', 'edge_range'])
    map_limits_changed = QtCore.pyqtSignal(
        float, float, float, int,
        arguments=['vmin', 'vmax', 'step', 'decimals'])
    map_cursor_changed = QtCore.pyqtSignal(
        object, object, object,
        arguments=['xy', 'pixel', 'value'])
    map_pixel_changed = QtCore.pyqtSignal(
        object, object, object,
        arguments=['xy', 'pixel', 'value'])
    frame_data_changed = QtCore.pyqtSignal(
        object, np.ndarray, object, 'QString', tuple,
        arguments=('frames', 'energies', 'norm', 'cmap', 'extent'))
    frame_data_cleared = QtCore.pyqtSignal()
    active_frame_changed = QtCore.pyqtSignal(int)
    active_energy_changed = QtCore.pyqtSignal(float)
    frame_vrange_changed = QtCore.pyqtSignal(
        float, float, float, int,
        arguments=('vmin', 'vmax', 'step', 'decimals'))
    
    def __init__(self, frameset, *args, **kwargs):
        self.frameset = frameset
        # Set the list of views to be empty to start
        self.map_views = []
        self.map_threads = []
        self.frame_views = []
        self.frame_threads = []
        super().__init__(*args, **kwargs)
    
    def add_frame_view(self, view, threaded=True):
        """Attach a view to this presenter.
        
        Parameters
        ----------
        view : QObject
          The view will be connected to signals that describe changes in
          frame data.
        threaded : bool, optional
          If true, this view will be added to its own thread before
          signals get connected
        
        """
        self.frame_views.append(view)
        # Let the view connect to this presenters signals
        view.connect_presenter(presenter=self)
        # Get the map_view prepared and running in its own thread
        if threaded:
            thread = QtCore.QThread()
            view.moveToThread(thread)
            thread.start()
            self.frame_threads.append(thread)
        else:
            log.info("Frame view not threaded")
        # Connect to the view's signals
        view.frame_slider_moved.connect(self.change_active_frame)
        view.new_vrange_requested.connect(self.set_frame_vrange)
        view.new_vrange_requested.connect(self.refresh_frames)
        view.reset_vrange_requested.connect(self.reset_frame_range)
        view.reset_vrange_requested.connect(self.refresh_frames)
        view.new_cmap_requested.connect(self.change_frame_cmap)
        view.new_component_requested.connect(self.change_frame_component)
        view.new_timestep_requested.connect(self.set_timestep)
        view.play_button_clicked.connect(self.play_frames)
        view.forward_button_clicked.connect(self.next_frame)
        view.back_button_clicked.connect(self.previous_frame)
        view.first_button_clicked.connect(self.first_frame)
        view.last_button_clicked.connect(self.last_frame)
        view.play_speed_requested.connect(self.set_play_speed)
        view.new_hdf_group_requested.connect(self.change_hdf_group)
        view.figure_hovered.connect(self.hover_frame_pixel)
    
    def add_map_view(self, view, threaded=True):
        """Attach a view to this presenter.
        
        Parameters
        ----------
        view : QObject
          The view will be connected to signals that describe changes in
          map data.
        threaded : bool, optional
          If true, this view will be added to its own thread before
          signals get connected, giving a snapier UI. Disabling makes
          testing more straightforward.
        
        """
        self.map_views.append(view)
        # Let the view connect to this presenters signals
        view.connect_presenter(presenter=self)
        # Get the map_view prepared and running in its own thread
        if threaded:
            thread = QtCore.QThread()
            view.moveToThread(thread)
            thread.start()
            self.map_threads.append(thread)
        else:
            log.info('Map view not threaded.')
        # Connect to the view's signals
        view.cmap_changed.connect(self.change_map_cmap)
        view.component_changed.connect(self.change_map_component)
        view.edge_mask_toggled.connect(self.toggle_edge_mask)
        view.spectrum_fit_toggled.connect(self.toggle_spectrum_fit)
        view.map_vmin_changed.connect(self.set_map_vmin)
        view.map_vmax_changed.connect(self.set_map_vmax)
        view.limits_applied.connect(self.update_maps)
        view.limits_applied.connect(self.update_spectra)
        view.limits_reset.connect(self.reset_map_range)
        view.limits_reset.connect(self.update_maps)
        view.limits_reset.connect(self.update_spectra)
        view.map_hovered.connect(self.set_map_cursor)
        view.map_clicked.connect(self.set_map_pixel)
        view.map_moved.connect(self.move_map_pixel)
    
    def toggle_spectrum_fit(self, state):
        if self.show_spectrum_fit != state:
            self.show_spectrum_fit = state
            self.update_spectra()
        
    def toggle_edge_mask(self, state):
        if self.use_edge_mask != state:
            self.use_edge_mask = state
            self.update_maps()
            self.update_spectra()
    
    def prepare_ui(self, expand_tree=True):
        # self.create_app()
        # Create a timer for playing through all the frames
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.set_play_speed(15)
        # We're done creating the app, so let the views create their UI's
        self.app_ready.emit()
        self.cmap_list_changed.emit(CMAPS)
        self.component_list_changed.emit(COMPS)
        self.update_timestep_list()
        # Prepare data for the HDF tree view
        self.build_hdf_tree(expand_tree=expand_tree)
    
    def update_timestep_list(self):
        # Set the list of possible timesteps
        tslist = ["{} - {}".format(idx, ts)
                  for idx, ts in enumerate(self.frameset.timestep_names)]
        self.timestep_list_changed.emit(tslist)
    
    def create_app(self):  # pragma: no cover
        log.debug("Creating QApplication")
        self.app = QtWidgets.QApplication([])
        self.app.setApplicationName("Xanespy")
        # self.app.setStyle("cleanlooks")
        self.process_events.connect(self.app.processEvents)
    
    def launch(self):  # pragma: no cover
        # Show the graphs and launch to event loop
        ret = self.app.exec()
        # Quit threads after execution
        [thread.quit() for thread in self.map_threads]
        [thread.quit() for thread in self.frame_threads]
        return ret
    
    def set_timestep(self, new_timestep):
        self.active_timestep = new_timestep
        self.refresh_frames()
        self.update_maps()
    
    # def move_slider(self, new_value):
    #     new_frame = new_value
    #     if not self.active_frame == new_frame:
    #         self.active_frame = new_frame
    #         self.frame_view.frame_changed.emit(self.active_frame)
    
    # def update_frame_range_limits(self):
    #     """Check that the (min, max, step) of the frame spinboxes are
    #     reasonable. This should be called when the spinbox values change."""
    #     # Round the step to the nearest multiple of ten
    #     decimals = - math.floor(math.log10(self._frame_vmax - self._frame_vmin) - 1)
    #     self.frame_view.set_vmin_decimals(decimals)
    #     self.frame_view.set_vmax_decimals(decimals)
    #     step = 1 / 10**decimals
    #     self.frame_view.set_vmin_step(step)
    #     self.frame_view.set_vmax_step(step)
    #     # Now set new limits on the view's vmin and vmax
    #     self.frame_view.set_vmin_maximum(self._frame_vmax)
    #     self.frame_view.set_vmax_minimum(self._frame_vmin)
    
    def set_frame_vrange(self, vmin, vmax):
        if vmin < vmax:
            log.debug("Setting frame vrange to (%f, %f)" % (vmin, vmax))
            self._frame_vmin = vmin
            self._frame_vmax = vmax
        else:
            log.info("Invalid frame range {}".format((vmin, vmax)))
        # Calculate new step and decimal values
        decimals = - math.floor(math.log10(self._frame_vmax - self._frame_vmin) - 1)
        step = 1 / 10**decimals
        # Emit all the new values
        self.frame_vrange_changed.emit(self._frame_vmin,
                                       self._frame_vmax, step, decimals)
    
    def set_map_vmin(self, new_value):
        if self._map_vmin != new_value:
            log.debug("Changing map vmin to %f", new_value)
            self._map_vmin = new_value
            self.update_map_limits()
            
    
    def set_map_vmax(self, new_value):
        if self._map_vmax != new_value:
            log.debug("Changing map vmax to %f", new_value)
            self._map_vmax = new_value
            self.update_map_limits()
    
    def update_map_limits(self):
        vmin, vmax = self._map_vmin, self._map_vmax
        decimals = - math.floor(math.log10(vmax - vmin) - 2)
        step = 10 / 10**decimals
        decimals = max(decimals, 0)
        # Call the appropriate signal
        self.map_limits_changed.emit(vmin, vmax, step, decimals)
    
    def reset_frame_range(self):
        """Reset the frame plotting vmin and vmax based on the currently
        selected data. VMin will be the 1nd percentile and VMax will
        be the 99th percentile. The results can be accessed by calling
        the `frame_norm()` method.
        
        Returns
        -------
        (vmin, vmax) : (float, float)
          A 2-tuple with the new min and max range.
        None
          If current representation is not a valid map, None will be
          returned.
        
        """
        try:
            data = self.active_frames()
        except exceptions.GroupKeyError:
            # We're not on a valid frameset, so pass
            pass
        else:
            # Calculate the relevant perctile intervals
            p_lower = np.nanpercentile(data, 1)
            p_upper = np.nanpercentile(data, 99)
            # Update the view with the new values
            self._frame_vmin = p_lower
            self._frame_vmax = p_upper
            self.set_frame_vrange(p_lower, p_upper)
    
    def reset_map_range(self):
        """Reset the map plotting vmin and vmax based on the currently
        selected data. VMin will be the 1st percentile and VMax will
        be the 99th percentile. This method also caches the values so
        they can be retrieved via the `map_norm()` method. If the
        current representation is not valid map data, then the cached
        vmin and vmax will be returned with no other changes.
        
        Returns
        -------
        (vmin, vmax) : (float, float)
          A 2-tuple with the current min and max range.
        
        """
        data = self.active_map()
        if data is not None:
            # Remove any np.nan values
            data = data[~np.isnan(data)]
            # Calculate the relevant perctile intervals
            p_lower = np.percentile(data, 1)
            p_upper = np.percentile(data, 99)
            # Update the view with the new values
            self._map_vmin = p_lower
            self._map_vmax = p_upper
            log.debug("Found new map range values %f, %f", p_lower, p_upper)
        else:
            # We're not on a valid mapset, so just get the current values
            p_lower = self._map_vmin
            p_upper = self._map_vmin
        self.update_map_limits()
        return (p_lower, p_upper)
    
    def set_map_cursor(self, x, y):
        log.debug("New cursor xy: {}, {}".format(x, y))
        if None in [x, y] or self.active_map() is None:
            # No pixel was hovered or no map, so clear all the stuff
            xy = None
            pixel = None
            value = None
        else:
            try:
                extent = self.frameset.extent(self.active_representation)
                shape = self.frameset.frame_shape(self.active_representation)
            except exceptions.GroupKeyError:
                msg = "Could not load extent and shape for {}"
                log.debug(msg.format(self.active_representation))
                xy = None
                pixel = None
                value = None
            else:
                xy = xycoord(x, y)
                pixel = xy_to_pixel(xy, extent=extent, shape=shape)
                value = self.active_map()[pixel]
        self.map_cursor_changed.emit(xy, pixel, value)
    
    def set_map_pixel(self, x, y):
        log.debug("New map pixel xy: {}, {}".format(x, y))
        if None in [x, y]:
            # No pixel was hovered, so clear all the stuff
            xy = None
            pixel = None
            value = None
        else:
            xy = xycoord(x, y)
            extent = self.frameset.extent(self.active_representation)
            shape = self.frameset.frame_shape(self.active_representation)
            pixel = xy_to_pixel(xy, extent=extent, shape=shape)
            value = self.active_map()[pixel]
        self._map_pixel = pixel
        self.map_pixel_changed.emit(xy, pixel, value)
        self.update_spectra()
    
    def move_map_pixel(self, vert, horiz):
        """Move the active pixel by the given amount in each direction.
        
        If the current pixel is not active, this is a no-op.
        """
        curr_px = self._map_pixel
        if self._map_pixel is not None:
            shape = self.frameset.frame_shape(self.active_representation)
            new_pixel = Pixel(vertical=(curr_px[0] + vert) % shape[0],
                              horizontal=(curr_px[1] + horiz) % shape[1])
            self._map_pixel = new_pixel
            # Calculate new xy coords and map value
            extent = self.frameset.extent(self.active_representation)
            xy = pixel_to_xy(pixel=new_pixel, extent=extent, shape=shape)
            val = self.active_map()[new_pixel]
            self.map_pixel_changed.emit(xy, new_pixel, val)
            self.update_spectra()
    
    def frame_norm(self):
        norm = Normalize(vmin=self._frame_vmin, vmax=self._frame_vmax, clip=True)
        return norm
    
    def map_norm(self):
        norm = Normalize(vmin=self._map_vmin, vmax=self._map_vmax, clip=True)
        return norm
    
    def change_active_frame(self, new_idx):
        valid_change = new_idx != self.active_frame
        if valid_change:
            log.debug("Setting index %d -> %d", self.active_frame, new_idx)
            self.active_frame = (new_idx % self.num_frames)
            self.active_frame_changed.emit(self.active_frame)
            energy = self.frameset.energies(self.active_timestep)[self.active_frame]
            self.active_energy_changed.emit(energy)
    
    def change_frame_component(self, new_comp):
        if not self.active_frame_component == new_comp:
            log.debug("Changing frame component from %s to %s", self.active_frame_component, new_comp)
            self.active_frame_component = new_comp
            self.reset_frame_range()
            self.refresh_frames()
    
    def change_map_component(self, new_comp):
        if not self.active_map_component == new_comp:
            log.debug("Changing map component from %s to %s", self.active_map_component, new_comp)
            self.active_map_component = new_comp
            self.reset_map_range()
            self.update_maps()
            self.update_spectra()
    
    def change_frame_cmap(self, new_cmap):
        if not self.frame_cmap == new_cmap:
            log.debug("Changing cmap from %s to %s", self.frame_cmap, new_cmap)
            self.frame_cmap = new_cmap
            self.refresh_frames()
   
    def change_map_cmap(self, new_cmap):
        if new_cmap != self.map_cmap:
            # Cmap has changed, so update stuff
            self.map_cmap = new_cmap
            self.update_maps()
            self.update_spectra()
    
    def build_hdf_tree(self, expand_tree):
        """Build the items and insert them into the view's HDF tree based on
        the structure of the frameset's HDF file.
        
        """
        self.hdf_tree_changed.emit(self.frameset.data_tree())
    
    def refresh_frames(self):
        if self.active_representation is not None:
            # A valid set of frames is available, so plot them
            self.update_spectra()
            frames = self.active_frames()
            extent = self.frameset.extent(self.active_representation)
            energies = self.frameset.energies(timeidx=self.active_timestep)
            log.debug("Changing frames from presenter")
            self.frame_data_changed.emit(frames,
                                         energies,
                                         self.frame_norm(),
                                         self.frame_cmap, extent)
            self.change_active_frame(0)
        else:
            # Invalid frame data, so just clear the axes
            self.frame_data_cleared.emit()
    
    def active_frames(self):
        frames = self.frameset.frames(timeidx=self.active_timestep,
                                      representation=self.active_representation)
        frames = get_component(frames, self.active_frame_component)
        return frames
    
    def active_map(self):
        """Returns the active map array if possible. If it doesn't exist,
        return None.
        
        """
        try:
            map_data = self.frameset.map_data(
                timeidx=self.active_timestep,
                representation=self.active_representation,
            )
        except exceptions.GroupKeyError:
            return None
        if map_data is not None:
            # Retrieve only active complex component
            map_data = get_component(map_data, self.active_map_component)
            # Apply a mask to show only the XAS edge
            if self.use_edge_mask:
                mask = self.frameset.edge_mask()
                map_data = np.ma.array(map_data, mask=mask)
        return map_data
    
    def hover_frame_pixel(self, xy):
        # Validate that the pixel's on the graph, and has data
        pixel_is_valid = xy is not None
        try:
            extent = self.frameset.extent(self.active_representation)
            shape = self.frameset.frame_shape(self.active_representation)
        except exceptions.GroupKeyError:
            pixel_is_valid = False
        if pixel_is_valid:
            # Modify the UI with the cursor location
            cursor_s = "({x:.2f}, {y:.2f})".format(x=xy.x, y=xy.y)
            # Convert from (x, y) to (row, col)
            pixel = xy_to_pixel(xy, extent=extent, shape=shape)
            pixel_s = "[{row}, {col}]".format(row=pixel.vertical,
                                              col=pixel.horizontal)
            self.frame_pixel = pixel
            # Get the frame's value
            value = self.active_frames()[self.active_frame][pixel]
        else:
            # Modify the UI to clear the cursor data
            cursor_s = ""
            pixel_s = ""
            value = None
            pixel = None
            self.frame_pixel = None
        # Now update the UI
        unit = self.frameset.pixel_unit()
        self.frame_hover_changed.emit(xy, unit, pixel, value)
    
    # def update_status_value(self):
    #     px = self.frame_pixel
    #     try:
    #         assert px is not None
    #         # Get the value of this pixel from the frame data
    #         value_s = str(self.active_frames()[self.active_frame][px])
    #     except (IndexError, AssertionError):
    #         value_s = ""
    #     self.frame_view.set_status_value(value_s)
    
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
        log.debug("Going to next frame")
        self.change_active_frame(self.active_frame + 1)
    
    def previous_frame(self):
        log.debug("Going to previous frame")
        self.change_active_frame(self.active_frame - 1)
    
    def first_frame(self):
        log.debug("Going to first frame")
        self.change_active_frame(0)
    
    def last_frame(self):
        log.debug("Going to last frame")
        self.change_active_frame(-1)
    
    # def update_status_shape(self):
    #     try:
    #         shape = self.active_frames().shape
    #     except exceptions.GroupKeyError:
    #         # No valid frames so show a placeholder
    #         s = "---"
    #     else:
    #         s = "{}".format(shape)
    #     # Set the UI's text
    #     self.frame_view.set_status_shape(s)
    
    # def update_status_unit(self):
    #     unit = self.frameset.pixel_unit()
    #     s = "({}):".format(unit)
    #     self.frame_view.set_status_unit(s)
    
    def change_hdf_file(self, filename):
        raise NotImplementedError()
    
    def change_hdf_group(self, new_item, old_item):
        self.busy_status_changed.emit(True)
        self.process_events.emit()
        context = new_item.text(1)
        is_selectable = context in ["frameset", "map"]
        old_name = old_item.text(0) if old_item is not None else "None"
        # Figure out the path for this group and set the new data_name
        path = new_item.text(2)
        path_nodes = path.split('/')
        # Sanity checks for the HDF path
        if len(path_nodes) > 2:
            self.frameset.data_name = path_nodes[2]
        # Change the active datagroup if requested
        if len(path_nodes) > 1:
            self.frameset.parent_name = path_nodes[1]
            # Set the active representation and data groups
        if context in ["frameset", 'map']:
            # A valid representation was chosen, so save it for future plotting
            active_representation = new_item.text(0)
        else:
            # A non-leaf node was chosen, so no representation
            active_representation = None
        log.debug("Changing frame representation %s -> %s",
                  self.active_representation, active_representation)
        self.update_timestep_list()
        self.active_representation = active_representation
        self.num_frames = self.frameset.num_energies
        log.debug("New HDF data path: %s", path)
        self.reset_frame_range()
        self.reset_map_range()
        self.refresh_frames()
        # Update the window title with the new path
        self.hdf_path_changed.emit(path)
        # Get new map data
        map_data = self.active_map()
        # Inform the UI of the new map data if necessary and cache it
        is_new_map = not np.array_equal(self._map_data, map_data)
        if is_new_map:
            self._map_data = map_data
            self.update_maps()
            self.update_spectra()
        # Make the UI usable again
        self.busy_status_changed.emit(False)
    
    def update_spectra(self):
        """Get the most recent data for mean and single-pixel spectra and send
        them out to the signals `mean_spectrum_changed` and
        `pixel_sepctrum_changed`.
        """
        try:
            mean_spectrum = self.frameset.spectrum(
                index=self.active_timestep,
                pixel=None,
                edge_jump_filter=self.use_edge_mask,
                representation=self.active_representation)
            map_spectrum = self.frameset.spectrum(
                index=self.active_timestep,
                pixel=self._map_pixel,
                edge_jump_filter=self.use_edge_mask,
                representation=self.active_representation)
        except exceptions.GroupKeyError:
            pass
        else:
            # Convert to active component
            energies = map_spectrum.index
            map_spectrum = get_component(map_spectrum, self.active_map_component)
            map_spectrum = pd.Series(map_spectrum, index=energies)
            # Get the fitted spectrum if available
            if self.show_spectrum_fit:
                log.warning('Plotting of spectrum fits not implemented.')
                fit = True
            else:
                fit = None
            edge_range = getattr(self.frameset.edge, 'edge_range', None)
            self.mean_spectrum_changed.emit(mean_spectrum,
                                            fit,
                                            self.map_norm(),
                                            self.map_cmap,
                                            edge_range)
            log.debug("Emitting map spectrum")
            self.map_spectrum_changed.emit(map_spectrum,
                                           fit,
                                           self.map_norm(),
                                           self.map_cmap,
                                           edge_range)
    
    def update_maps(self):
        """Send the current mapping data to the `map_data_changed` signal.
        
        This method should be called after anything changes the visual
        representation of the map. Examples:
        
        - Changing the active representation
        - Changing the active complex component (real, imag, etc)
        - Changing the colormap
        - Changing the map normalization limits
        - Changing whether the XAS edge mask is applied
        
        """
        map_data = self.active_map()
        if map_data is not None:
            log.debug("Updating maps: %s", self.active_representation)
            extent = self.frameset.extent(self.active_representation)
            self.map_data_changed.emit(map_data, self.map_norm(),
                                       self.map_cmap, extent)
        else:
            self.map_data_cleared.emit()
