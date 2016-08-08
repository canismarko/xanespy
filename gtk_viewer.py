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

import os
import threading
import time
from collections import namedtuple

import numpy as np
import gi
from gi.repository import Gtk, Gdk, GObject, GLib

from utilities import xycoord
from .frame import Pixel, xy_to_pixel, pixel_to_xy
from .plotter import GtkFramesetPlotter


gi.require_version('Gtk', '3.0')


WATCH_CURSOR = Gdk.Cursor(Gdk.CursorType.WATCH)
ARROW_CURSOR = Gdk.Cursor(Gdk.CursorType.ARROW)

# print(dir(GObject))

class WatchCursor():
    """Factory returns a function that Perform some slow action `target`
    with a watch cursor over the given `windows`. When writing the
    `target` function, make sure to use GLib.idle_add for anything
    that modifies the UI.
    """
    _timer = None

    def __init__(self, target, windows, threading=True):
        self.target = target
        self.windows = windows
        self.threading = threading

    def __call__(self, *args, delay=0):
        def dostuff(*args):
            # Unpack user data
            windows, target, *more = args
            # Start watch cursor
            for window in windows:
                real_window = window.get_window()
                if real_window:
                    real_window.set_cursor(WATCH_CURSOR)
                    window.set_sensitive(False)
            # Call the actual function
            target(*more)
            # Remove watch cursor
            for window in windows:
                real_window = window.get_window()
                if real_window:
                    real_window.set_cursor(ARROW_CURSOR)
                    window.set_sensitive(True)
            self._timer = None
            return False

        # Start target process
        if self.threading:
            # Reset timer
            if self._timer is not None:
                GLib.source_remove(self._timer)
            self._timer = GLib.timeout_add(delay, dostuff,
                                           self.windows, self.target,
                                           *args)
        else:
            dostuff(self.windows, self.target, *args)


class GtkTxmViewer():
    play_mode = False
    display_type = 'corrected'
    active_pixel = None
    active_xy = None
    _current_idx = 0
    animation_delay = 1000 / 15
    show_map = True
    show_map_background = False
    apply_edge_jump = False
    """View a XANES frameset using a Gtk GUI."""
    def __init__(self, frameset, plotter=None):
        if plotter is None:
            plotter = GtkFramesetPlotter(frameset=frameset)
        self.plotter = plotter
        self.plotter.create_axes()
        self.frameset = frameset
        self.builder = Gtk.Builder()
        # Load the GUI from a glade file
        gladefile = os.path.join(os.path.dirname(__file__),
                                 "xanes_viewer.glade")
        self.builder.add_from_file(gladefile)
        self.window = self.builder.get_object('XanesViewerWindow')
        self.image_sw = self.builder.get_object('ImageWindow')
        self.image_sw.add(self.plotter.frame_canvas)
        # Prepare the map window for later
        self.map_sw = self.builder.get_object("MapWindow")
        self.map_sw.add(self.plotter.map_canvas)
        self.map_window = self.builder.get_object('MapViewerWindow')
        self.map_window.maximize()
        # Set initial state for background switch
        switch = self.builder.get_object('BackgroundSwitch')
        switch.set_active(self.show_map_background)
        # Set some initial values
        self.current_adj = self.builder.get_object('CurrentFrame')
        self.current_adj.set_property('upper', len(self.frameset) - 1)
        # Prepare the tree of different subsets
        with self.frameset.hdf_file(mode='r') as f:
            parent_group = f[self.frameset.frameset_group]
            groups = []
            node = namedtuple('node', ('name', 'display', 'path', 'parent'))
            for key in parent_group.keys():
                group = parent_group[key]
                display = " ".join(
                    [word.capitalize() for word in key.split("_")]
                )
                groups.append(node(name=key,
                                   display=display,
                                   path=group.name,
                                   parent=group.attrs.get('parent', None)))
        # Remove non-frameset nodes
        treestore = Gtk.TreeStore(str, str)
        groups = [g for g in groups if g.parent is not None]
        top_level = [g for g in groups if g.parent == ""]
        active_iters = []
        def add_node(parent_iter, node):
            new_iter = treestore.append(parent_iter, (node.display, node.name))
            # Check if this group should be selected
            if node.path == self.frameset.active_group:
                active_iters.append(new_iter)
            groups.pop(groups.index(node))
            # Resursive function that builds the tree
            children = [g for g in groups if g.parent == node.path]
            for child in children:
                add_node(parent_iter=new_iter, node=child)
        # Start at the top and build the tree recursively
        for node in top_level:
            add_node(parent_iter=None, node=node)
        treeview = self.builder.get_object("FramesetTreeView")
        treeview.set_model(treestore)
        columns = ["display"]
        for i in range(0, len(columns)):
            cell = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(columns[i], cell, text=i)
            treeview.append_column(col)
        if active_iters:
            # Set current active group
            active_iter = active_iters[0]
            active_path = treestore.get_path(active_iter)
            selection = treeview.get_selection()
            treeview.expand_to_path(active_path)
            selection.select_path(active_path)
        # Put the non-glade things in the window
        self.plotter.plot_xanes_spectrum()
        # Populate the combobox with list of available representations
        self.rep_combo = self.builder.get_object('ActiveRepresentationCombo')
        self.rep_list = Gtk.ListStore(str, str)
        reps = self.frameset.representations()
        for rep in reps:
            uppercase = " ".join(
                [word.capitalize() for word in rep.split('_')]
            )
            rep_iter = self.rep_list.append([uppercase, rep])
            # Save active group for later initialization
            if rep == self.frameset.default_representation:
                active_rep = rep_iter
        if self.frameset.is_background():
            self.active_group = bg_iter
        self.rep_combo.set_model(self.rep_list)
        if active_rep:
            self.rep_combo.set_active_iter(active_rep)
        # Set event handlers
        both_windows = [self.window, self.map_window]
        handlers = {
            'gtk-quit': Gtk.main_quit,
            'previous-frame': self.previous_frame,
            'create-artists': WatchCursor(self.refresh_artists,
                                          windows=both_windows),
            'max-frame': self.max_frame,
            'next-frame': self.next_frame,
            'play-frames': WatchCursor(self.play_frames,
                                       windows=[self.window]),
            'last-frame': self.last_frame,
            'first-frame': self.first_frame,
            'key-release-main': self.key_pressed_main,
            'key-release-map': WatchCursor(self.navigate_map,
                                           windows=[self.map_window]),
            'toggle-particles': WatchCursor(self.toggle_particles,
                                            windows=both_windows),
            'toggle-normalization': WatchCursor(self.toggle_normalization,
                                                windows=both_windows),
            'update-window': self.update_window,
            'change-active-group': WatchCursor(self.change_active_group,
                                               windows=[self.window]),
            'change-representation': WatchCursor(self.change_representation,
                                               windows=[self.window]),
            'launch-map-window': WatchCursor(self.launch_map_window,
                                             windows=[self.window]),
            'hide-map-window': self.hide_map_window,
            'toggle-map': WatchCursor(self.toggle_map_visible,
                                      windows=[self.map_window]),
            'toggle-map-background': WatchCursor(self.toggle_map_background,
                                                 windows=[self.map_window]),
            'toggle-edge-jump': WatchCursor(self.toggle_edge_jump,
                                            windows=both_windows),
        }
        self.builder.connect_signals(handlers)
        self.window.connect('delete-event', self.quit)
        # Connect handlers for clicking on a pixel
        click_pixel = WatchCursor(self.click_map_pixel,
                                  windows=[self.map_window])
        self.plotter.map_figure.canvas.mpl_connect('button_press_event',
                                                   click_pixel)
        # Connect handler for mousing over the frame image
        self.plotter.frame_figure.canvas.mpl_connect('motion_notify_event',
                                                     self.update_current_location)
        # Prepare animation
        self.event_source = FrameChangeSource(viewer=self)
        # Make everything visible
        self.update_window()
        self.plotter.draw()

    def toggle_map_visible(self, widget, object=None):
        self.show_map = widget.get_active()
        GLib.idle_add(self.draw_map_plots)
        GLib.idle_add(self.update_map_window)

    def toggle_edge_jump(self, widget, object=None):
        self.plotter.apply_edge_jump = widget.get_active()
        if self.frameset.map_name:
            # Only replot the map if necessary
            GLib.idle_add(self.draw_map_plots)
            GLib.idle_add(self.update_map_window)
        self.refresh_artists()
        self.update_window()

    def toggle_map_background(self, widget, object=None):
        self.show_map_background = widget.get_active()
        GLib.idle_add(self.draw_map_plots)
        GLib.idle_add(self.update_map_window)

    def quit(self, widget, object=None):
        self.map_window.destroy()
        self.play_mode = False
        self.window.destroy()
        # Reclaim memory
        self.plotter.destroy()
        Gtk.main_quit()

    def hide_map_window(self, widget, object=None):
        self.map_window.hide()
        return True

    def launch_map_window(self, widget):
        GLib.idle_add(self.draw_map_plots)
        GLib.idle_add(self.map_window.show_all)

    def click_map_pixel(self, event):
        if event.inaxes in [self.plotter.map_ax, self.plotter.image_ax]:
            # Convert xy position to pixel values
            xy = xycoord(x=event.xdata, y=event.ydata)
            self.active_pixel = xy_to_pixel(xy,
                                            extent=self.frameset.extent(),
                                            shape=self.frameset.map_shape())
            self.plotter.active_pixel = self.active_pixel
            # Make sure active_xy is in the center of the pixel
            self.active_xy = pixel_to_xy(self.active_pixel,
                                         extent=self.frameset.extent(),
                                         shape=self.frameset.map_shape())
            self.plotter.active_xy = self.active_xy
        else:
            self.active_pixel = None
            self.active_xy = None
            self.plotter.active_pixel = None
            self.plotter.active_xy = None
        GLib.idle_add(self.draw_map_plots)
        GLib.idle_add(self.update_map_window)

    def update_current_location(self, event):
        x_label = self.builder.get_object('XCursorLabel')
        y_label = self.builder.get_object('YCursorLabel')
        v_label = self.builder.get_object('VCursorLabel')
        h_label = self.builder.get_object('HCursorLabel')
        I_label = self.builder.get_object('ICursorLabel')
        frame = self.current_frame()
        if event.inaxes == self.plotter.image_ax:
            # Convert xy position to pixel values
            xy = xycoord(x=round(event.xdata, 1), y=round(event.ydata, 1))
            pixel = xy_to_pixel(xy, extent=self.frameset.extent(),
                                shape=self.frameset.map_shape())
            x_label.set_text(str(xy.x))
            y_label.set_text(str(xy.y))
            v_label.set_text(str(pixel.vertical))
            h_label.set_text(str(pixel.horizontal))
            row = np.clip(pixel.vertical, 0, frame.image_data.shape[0]-1)
            col = np.clip(pixel.horizontal, 0, frame.image_data.shape[1]-1)
            value = frame.image_data[row][col]
            I_label.set_text(str(round(value, 4)))
        else:
            # Set all the cursor labels to blank values
            s = "--"
            x_label.set_text(s)
            y_label.set_text(s)
            v_label.set_text(s)
            h_label.set_text(s)
            I_label.set_text(s)

    def draw_map_plots(self):
        self.plotter.draw_map(show_map=self.show_map,
                              show_background=self.show_map_background)
        # Show crosshairs to indicate active pixel
        if self.show_map_background:
            color = 'white'
        else:
            color = 'black'
        self.plotter.draw_crosshairs(active_xy=self.active_xy, color=color)
        self.plotter.draw_map_xanes()
        self.plotter.plot_histogram()

    def update_map_window(self):
        # Show position of active pixel
        if self.active_pixel is None:
            s = "None"
        else:
            s = "(V:{v}, H:{h})".format(h=self.active_pixel.horizontal,
                                        v=self.active_pixel.vertical)
        label = self.builder.get_object("ActivePixelLabel")
        label.set_text(s)

    def navigate_map(self, widget, event):
        """Navigate around the map using keyboard."""
        if self.active_pixel is not None:
            horizontal = self.active_pixel.horizontal
            vertical = self.active_pixel.vertical
            if event.keyval == Gdk.KEY_Left:
                horizontal = horizontal - 1
            elif event.keyval == Gdk.KEY_Right:
                horizontal = horizontal + 1
            elif event.keyval == Gdk.KEY_Up:
                vertical = vertical - 1
            elif event.keyval == Gdk.KEY_Down:
                vertical = vertical + 1
            self.active_pixel = Pixel(horizontal=horizontal, vertical=vertical)
            self.active_xy = pixel_to_xy(self.active_pixel,
                                         extent=self.frameset.extent(),
                                         shape=self.frameset.map_shape())
            self.plotter.active_pixel = self.active_pixel
            self.plotter.active_xy = self.active_xy
        GLib.idle_add(self.draw_map_plots)
        GLib.idle_add(self.update_map_window)

    def change_active_group(self, selection, object=None):
        """Update to a new frameset HDF group after user has picked tree entry."""
        model, treeiter = selection.get_selected()
        def disable_map_button():
            # Disable map button until the data are loaded (reset in self.update_window)
            self.builder.get_object("ShowMapButton").set_sensitive(False)
        GLib.idle_add(disable_map_button)
        # Load new group
        new_group = model[treeiter][1]
        self.active_groupname = new_group
        self.frameset.switch_group(new_group)
        # # Update UI elements
        self.refresh_artists()
        self.update_window()

    def change_representation(self, widget, object=None):
        """Update to a new representation of this frameset."""
        # Load new group
        new_rep = self.rep_list[widget.get_active_iter()][1]
        self.plotter.active_representation = new_rep
        # Update UI elements
        self.refresh_artists()
        self.update_window()

    @property
    def current_idx(self):
        value = self.current_adj.get_property('value')
        return int(value)

    @current_idx.setter
    def current_idx(self, value):
        self.current_adj.set_property('value', value)

    def key_pressed_main(self, widget, event):
        if event.keyval == Gdk.KEY_Left:
            self.previous_frame()
        elif event.keyval == Gdk.KEY_Right:
            self.next_frame()

    def toggle_particles(self, widget):
        self.plotter.show_particles = not self.plotter.show_particles
        self.refresh_artists()
        self.update_window()

    def toggle_normalization(self, widget, event):
        self.plotter.normalize_xanes = widget.get_active()
        self.refresh_artists()
        self.update_window()

    def play_frames(self, widget):
        self.play_mode = widget.get_property('active')
        if self.play_mode:
            GLib.timeout_add(self.animation_delay, self.next_frame, None)

    def first_frame(self, widget):
        self.current_idx = 0
        self.update_window()

    def last_frame(self, widget):
        self.current_idx = len(self.frameset) - 1
        self.update_window()

    def previous_frame(self, widget=None):
        """Go to the next frame in the sequence (or wrap around if at the
        end).
        """
        self.current_idx = (self.current_idx - 1) % len(self.frameset)
        self.update_window()

    def max_frame(self, widget=None):
        """Find the frame with the highest intensity and active it"""
        spectrum = self.frameset.xanes_spectrum()
        self.current_idx = spectrum.values.argmax()
        self.update_window()

    def next_frame(self, widget=None):
        self.current_idx = (self.current_idx + 1) % len(self.frameset)
        self.update_window()
        if self.play_mode:
            keep_going = True
        else:
            keep_going = False
        return keep_going

    def remove_artists(self):
        """Remove current artists from the plotting axes."""
        for artist_tuple in self.frame_animation.artists:
            for artist in artist_tuple:
                artist.remove()

    def refresh_artists(self, *args, **kwargs):
        # Redraw xanes spectrum
        self.plotter.plot_xanes_spectrum()
        def connect_animation():
            self.plotter.connect_animation(self.event_source)
        GLib.idle_add(connect_animation)

    def update_window(self, widget=None):
        def change_gui():
            current_frame = self.current_frame()
            # Set labels on the sidepanel
            energy_label = self.builder.get_object('EnergyLabel')
            energy_label.set_text(str(current_frame.energy))
            x_label = self.builder.get_object('XPosLabel')
            x_label.set_text(str(current_frame.sample_position.x))
            y_label = self.builder.get_object('YPosLabel')
            y_label.set_text(str(current_frame.sample_position.y))
            z_label = self.builder.get_object('ZPosLabel')
            z_label.set_text(str(current_frame.sample_position.z))
            particle_label = self.builder.get_object('ActiveParticleLabel')
            particle_label.set_text(str(current_frame.active_particle_idx))
            shape_label = self.builder.get_object('ShapeLabel')
            shape_label.set_text(str(current_frame.image_data.shape))
            norm_label = self.builder.get_object('NormLabel')
            norm = self.frameset.image_normalizer(self.plotter.active_representation)
            norm_text = '[{}, {}]'.format(
                round(norm.vmin, 2),
                round(norm.vmax, 2)
            )
            norm_label.set_text(norm_text)
            # Check if the "show map" button should be active
            map_btn = self.builder.get_object("ShowMapButton")
            if self.frameset.map_name:
                map_btn.set_sensitive(True)
            else:
                map_btn.set_sensitive(False)
        GLib.idle_add(change_gui)

    def progress_modal(self, objs, operation='Working'):
        """
        Display the progress of the current operation via print statements.
        """
        modal = self.builder.get_object("WaitingWindow")
        modal.show_all()
        ctr = 1
        for obj in objs:
            ctr += 1
            yield obj
        modal.hide()

    def show(self):
        self.window.show_all()
        self.plotter.connect_animation(event_source=self.event_source)
        # Initialize threading support
        GObject.threads_init()

        Gtk.main()
        Gtk.main_quit()

    def current_image(self):
        return self.images[self.current_idx]

    def current_frame(self):
        return self.frameset[self.current_idx]


class FrameChangeSource():
    callbacks = []

    def __init__(self, viewer):
        self.viewer = viewer

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
        # Listen to the frame adjustment signals
        if not hasattr(self, 'handler_id'):
            self.handler_id = self.viewer.current_adj.connect('value-changed', self._on_change)

    def stop(self):
        if hasattr(self, 'handler_id'):
            self.viewer.current_adj.disconnect(self.handler_id)
            del self.handler_id

    def _on_change(self, widget=None, object=None):
        for func, args, kwargs in self.callbacks:
            func(self.viewer.current_idx, *args, **kwargs)
