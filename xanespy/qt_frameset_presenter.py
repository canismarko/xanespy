import logging
import math

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import exceptions

from utilities import xy_to_pixel, pixel_to_xy, xycoord, Pixel


CMAPS = ['plasma', 'viridis', 'inferno', 'magma', 'gray', 'bone',
         'copper', 'autumn', 'spring', 'summer', 'winter', 'spectral',
         'rainbow', 'gnuplot']
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
    frame_cmap = "copper"
    _frame_vmin = 0
    _frame_vmax = 1
    _map_vmin = 0
    _map_vmax = 1
    map_cmap = "plasma"
    active_frame = 0
    active_timestep = 0
    active_representation = None
    frame_pixel = None
    num_frames = 0
    timer = None
    _map_data = None
    _map_pixel = None
    use_edge_mask = False
    
    # Signals
    app_ready = QtCore.pyqtSignal()
    busy_status_changed = QtCore.pyqtSignal(bool)
    map_data_changed = QtCore.pyqtSignal(
        np.ndarray, object, 'QString', object,
        arguments=['map_data', 'norm', 'cmap', 'extent'])
    map_data_cleared = QtCore.pyqtSignal()
    process_events = QtCore.pyqtSignal()
    cmap_list_changed = QtCore.pyqtSignal(list)
    mean_spectrum_changed = QtCore.pyqtSignal(
        pd.Series, object, 'QString', object,
        arguments=['spectrum', 'norm', 'cmap', 'edge_range'])
    map_spectrum_changed = QtCore.pyqtSignal(
        pd.Series, object, 'QString', object,
        arguments=['spectrum', 'norm', 'cmap', 'edge_range'])
    map_limits_changed = QtCore.pyqtSignal(
        float, float, float, int,
        arguments=['vmin', 'vmax', 'step', 'decimals'])
    map_cursor_changed = QtCore.pyqtSignal(
        object, object, object,
        arguments=['xy', 'pixel', 'value'])
    map_pixel_changed = QtCore.pyqtSignal(
        object, object, object,
        arguments=['xy', 'pixel', 'value'])
    
    def __init__(self, frameset, frame_view, *args, **kwargs):
        self.frameset = frameset
        self.num_frames = frameset.num_energies
        self.frame_view = frame_view
        # Set the list of views to be empty to start
        self.map_views = []
        self.frame_views = []
        super().__init__(*args, **kwargs)
        self.connect_signals()
    
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
            self.map_thread = QtCore.QThread()
            view.moveToThread(self.map_thread)
            self.map_thread.start()
        # Connect to the view's signals
        view.cmap_changed.connect(self.change_map_cmap)
        view.edge_mask_toggled.connect(self.toggle_edge_mask)
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

    def toggle_edge_mask(self, state):
        if self.use_edge_mask != state:
            self.use_edge_mask = state
            self.update_maps()
            self.update_spectra()
    
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
        raise NotImplementedError("Coming soon!")
    
    def connect_signals(self):
        pass
    
    def prepare_ui(self):
        self.create_app()
        # Get the frame_view prepared and running
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
        # Connect response signals for the widgets in the frameview
        self.frame_view.connect_signals(presenter=self)
        self.frame_view.frame_changed.connect(self.update_status_frame)
        self.frame_view.frame_changed.connect(self.update_status_value)
        # Prepare data for the HDF tree view
        self.build_hdf_tree()
        # Create a timer for playing through all the frames
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.set_play_speed(15)
        # Connect signals for notifying the user of long operations
        self.busy_status_changed.connect(self.frame_view.disable_frame_controls)
        self.busy_status_changed.connect(self.frame_view.disable_plotting_controls)
        self.busy_status_changed.connect(self.frame_view.use_busy_cursor)
        self.busy_status_changed.connect(self.frame_view.set_drawing_status)
        # We're done creating the app, so let the views create their UI's
        self.app_ready.emit()
        self.cmap_list_changed.emit(CMAPS)
    
    def create_app(self):  # pragma: no cover
        self.app = QtWidgets.QApplication([])
        self.app.setApplicationName("Xanespy")
        self.process_events.connect(self.app.processEvents)
    
    def launch(self):  # pragma: no cover
        # Show the graphs and launch to event loop
        self.frame_view.show()
        return self.app.exec_()
    
    def set_timestep(self, new_timestep):
        self.active_timestep = new_timestep
        self.draw_frame_histogram()
        self.draw_frame_spectra()
        self.refresh_frames()
        self.update_maps()
    
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
    
    def set_frame_vmax(self, new_value):
        log.debug("Changing frame vmax to %f", new_value)
        # assert new_value >= self._frame_vmin, "Vmax must be greater than Vmin"
        self._frame_vmax = new_value
        # self.update_frame_range_limits()
        self.frame_view.set_vmax(self._frame_vmax)

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
            p_lower = np.percentile(data, 1)
            p_upper = np.percentile(data, 99)
            # Update the view with the new values
            self._frame_vmin = p_lower
            self._frame_vmax = p_upper
            self.set_frame_vmin(p_lower)
            self.set_frame_vmax(p_upper)
    
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
        if None in [x, y]:
            # No pixel was hovered, so clear all the stuff
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
            self.refresh_frames()
    
    def change_map_cmap(self, new_cmap):
        if new_cmap != self.map_cmap:
            # Cmap has changed, so update stuff
            self.map_cmap = new_cmap
            self.update_maps()
            self.update_spectra()
    
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
                child_item = tree_item(child)
                ret = item.addChild(child_item)
                child_item.setExpanded(True)
            return item
        
        # Start building the tree
        for child in self.frameset.data_tree():
            new_root_item = tree_item(child)
            # Disable any items that are outside this parent group
            if child['name'] != self.frameset.parent_name:
                new_root_item.setDisabled(True)
            # Now add the new root to the tree
            self.frame_view.add_hdf_tree_item(new_root_item)
        # Expand the full tree to make it easier for the user to browse
        self.frame_view.expand_hdf_tree.emit()
    
    def refresh_frames(self):
        if self.active_representation is not None:
            # A valid set of frames is available, so plot them
            self.draw_frame_histogram()
            self.draw_frame_spectra()
            self.animate_frames()
        else:
            # Invalid frame data, so just clear the axes
            self.frame_view.clear_axes()
        # Update the status bar with new frame data
        self.update_status_shape()
        self.update_status_unit()
    
    def active_frames(self):
        frames = self.frameset.frames(timeidx=self.active_timestep,
                                      representation=self.active_representation)
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
        # Apply a mask to show only the XAS edge
        if self.use_edge_mask:
            mask = self.frameset.edge_mask()
            map_data = np.ma.array(map_data, mask=mask)
        return map_data
    
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
        else:
            # Modify the UI to clear the cursor data
            cursor_s = ""
            pixel_s = ""
            self.frame_pixel = None
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
            shape = self.active_frames().shape
        except exceptions.GroupKeyError:
            # No valid frames so show a placeholder
            s = "---"
        else:
            s = "{}".format(shape)
        # Set the UI's text
        self.frame_view.set_status_shape(s)
    
    def update_status_unit(self):
        unit = self.frameset.pixel_unit()
        s = "({}):".format(unit)
        self.frame_view.set_status_unit(s)
    
    def update_status_frame(self, new_frame):
        """Create a string (and send it to the UI) that indicates the energy
        of the requested frame."""
        self.frame_view.set_status_index(str(new_frame))
        energy = self.frameset.energies(self.active_timestep)[new_frame]
        s = "{:.2f} eV".format(energy)
        self.frame_view.set_status_energy(s)
    
    def change_hdf_group(self, new_item, old_item):
        self.busy_status_changed.emit(True)
        self.process_events.emit()
        context = new_item.text(1)
        is_selectable = context in ["frameset", "map"]
        old_name = old_item.text(0) if old_item is not None else "None"
        # Figure out the path for this group and set the new data_name
        path = new_item.text(2)
        path_nodes = path.split('/')
        if len(path_nodes) > 2:
            self.frameset.data_name = path_nodes[2]
        # Set the active representation and data groups
        if context in ["frameset", 'map']:
            # A valid representation was chosen, so save it for future plotting
            active_representation = new_item.text(0)
        else:
            # A non-leaf node was chosen, so no representation
            active_representation = None
        log.debug("Changing frame representation %s -> %s",
                  self.active_representation, active_representation)
        self.active_representation = active_representation
        log.debug("New HDF data path: %s", path)
        self.reset_frame_range()
        self.reset_map_range()
        self.refresh_frames()
        # Update the window title with the new path
        title = "Xanespy Frameset ({})".format(path)
        self.frame_view.set_window_title(title)
        # Update some UI elements with new data
        try:
            num_energies = self.active_frames().shape[0]
        except exceptions.GroupKeyError:
            pass
        else:
            self.frame_view.set_slider_max(num_energies - 1)
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
            mean_spectrum = self.frameset.spectrum(index=self.active_timestep,
                                                  pixel=None,
                                                  edge_jump_filter=self.use_edge_mask,
                                                  representation=self.active_representation)
            map_spectrum = self.frameset.spectrum(index=self.active_timestep,
                                                  pixel=self._map_pixel,
                                                  edge_jump_filter=self.use_edge_mask,
                                                  representation=self.active_representation)
        except exceptions.GroupKeyError:
            pass
        else:
            self.mean_spectrum_changed.emit(mean_spectrum,
                                            self.map_norm(),
                                            self.map_cmap,
                                            self.frameset.edge.edge_range)
            self.map_spectrum_changed.emit(map_spectrum,
                                           self.map_norm(),
                                           self.map_cmap,
                                           self.frameset.edge.edge_range)
    
    def update_maps(self):
        
        """Send the current mapping data to the `map_data_changed` signal.
        
        This method should be called after anything changes the visual
        representation of the map. Examples:
        
        - Changing the active representation
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
