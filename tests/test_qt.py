#!/usr/bin/env python
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
# along with Xanespy.  If not, see <http://www.gnu.org/licenses/>.

# flake8: noqa

"""Tests for the Qt5 viewer."""
import unittest
from unittest import mock
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

from xanespy import QtFrameView, QtMapView, XanesFrameset, QtFramesetPresenter
from xanespy.qt_frame_view import FrameChangeSource
from xanespy import exceptions
from xanespy.utilities import xycoord, Extent, shape


# Define mocked views for the various Qt UI windows
MockFrameView = mock.MagicMock(spec_set=QtFrameView)
MockMapView = mock.MagicMock(spec_set=QtMapView)
MockFrameset = mock.MagicMock(spec_set=XanesFrameset)

class QtTestCase(unittest.TestCase):

    def create_presenter(self, frame_view=None, frameset=None, map_view=None):
        if frame_view is None:
            frame_view = MockFrameView()
        if frameset is None:
            frameset = MockFrameset()
            frameset.num_energies = 20
        if map_view is None:
            map_view = MockMapView()
        p = QtFramesetPresenter(frameset=frameset,
                                frame_view=frame_view,
                                map_view=map_view)
        # Mock create_app method so it can run headless (eg travis-ci.org)
        p.create_app = mock.MagicMock(name="create_app")
        return p

    def dummy_frame_data(self, shape):
        """Create some dummy data with a given shape. It's pretty much just an
        arange."""
        length = np.prod(shape)
        data = np.arange(length)
        data = np.reshape(data, shape)
        return data


class FrameViewerTestcase(QtTestCase):

    def test_init(self):
        """Check that certain values are set properly during __init__."""
        # Switch to "absorbances" representation if possible
        frameset = MockFrameset()
        frameset.has_representation.return_value = True
        presenter = self.create_presenter(frameset=frameset)
        self.assertEqual(presenter.active_representation, "absorbances")
        # Don't switch to "absorbances" representation if it doesn't exist
        frameset = MockFrameset()
        frameset.has_representation.return_value = False
        presenter = self.create_presenter(frameset=frameset)
        self.assertEqual(presenter.active_representation, "intensities")

    def test_prepare_ui(self):
        # Prepare a frameset object to test the presenter
        frameset = MockFrameset()
        data = np.random.rand(10, 128, 128)
        frameset.frames = mock.Mock(return_value=data)
        frameset.num_energies = 20
        # Create the presenter
        presenter = self.create_presenter(frameset=frameset)
        # Check that the number of energies is set properly
        self.assertEqual(presenter.num_frames, 20)
        # Run the actual code
        presenter.prepare_ui()
        presenter.frame_view.set_slider_max.assert_called_with(9)
        self.assertTrue(presenter.frame_view.set_cmap_list.called)
        self.assertTrue(presenter.frame_view.set_timestep_list.called)
        presenter.frame_view.set_timestep.assert_called_with(0)

    def test_set_timestep(self):
        # Set up some fake data
        frameset = MockFrameset()
        data = np.linspace(5, 105, num=10 * 128 * 128)
        data = data.reshape((10, 128, 128))
        frameset.frames = mock.Mock(return_value=data)
        presenter = self.create_presenter(frameset=frameset)
        presenter.active_representation = "absorbances"
        # Now invoke to function to be tested
        presenter.set_timestep(5)
        # Check that all the view elements are updated
        self.assertEqual(presenter.active_timestep, 5)
        frameset.frames.assert_called_with(timeidx=5,
                                           representation='absorbances')
        self.assertTrue(presenter.frame_view.draw_frames.emit.called)
        self.assertTrue(presenter.frame_view.draw_spectrum.called)
        self.assertTrue(presenter.frame_view.draw_histogram.emit.called)
        # Check that the frame range was reset
        self.assertAlmostEqual(presenter._frame_vmin, 6)

    def test_change_vmin_vmax(self):
        presenter = self.create_presenter()
        # Check the vmax
        presenter.set_frame_vmax(5.0)
        self.assertEqual(presenter._frame_vmax, 5.0)
        # Check the vmin
        presenter.set_frame_vmin(2)
        self.assertEqual(presenter._frame_vmin, 2)
        # Check normalizer
        norm = presenter.frame_norm()
        self.assertEqual(norm.vmin, 2)
        self.assertEqual(norm.vmax, 5)

    def test_draw_frame_histogram(self):
        presenter = self.create_presenter()
        presenter.draw_frame_histogram()
        self.assertTrue(presenter.frame_view.draw_histogram.emit.called)

    def test_draw_frame_spectra(self):
        # Set some expected values on a frameset
        frameset = MockFrameset()
        spectrum = pd.Series()
        frameset.spectrum = mock.Mock(return_value=spectrum)
        frameset.edge.edge_range = (8353, 8359)
        # Create the presenter and change the frame range
        presenter = self.create_presenter(frameset=frameset)
        presenter.set_frame_vmax(5.7)
        presenter.set_frame_vmin(1.5)
        # Check that the spectrum was drawn
        presenter.draw_frame_spectra()
        self.assertTrue(frameset.spectrum.called)
        # Check that the normalizer was correct
        called_norm = presenter.frame_view.draw_spectrum.call_args[1]['norm']
        self.assertEqual(called_norm.vmin, 1.5)
        self.assertEqual(called_norm.vmax, 5.7)
        # Check that the edge range is provided
        called_edge = presenter.frame_view.draw_spectrum.call_args[1]['edge_range']
        self.assertEqual(called_edge, (8353, 8359))

    def test_move_slider(self):
        presenter = self.create_presenter()
        presenter.move_slider(5)
        self.assertEqual(presenter.active_frame, 5)

    def test_reset_limits(self):
        frameset = MockFrameset()

        def frames(*args, **kwargs):
            if kwargs.get('representation', '') is None:
                raise exceptions.GroupKeyError()
            else:
                return np.linspace(1.5, 5)

        frameset.frames.side_effect = frames
        presenter = self.create_presenter(frameset=frameset)
        presenter.reset_frame_range()
        P_lower = np.percentile(frameset.frames(), 1)
        P_upper = np.percentile(frameset.frames(), 99)
        self.assertEqual(presenter._frame_vmin, P_lower)
        self.assertEqual(presenter._frame_vmax, P_upper)
        presenter.frame_view.set_vmin.assert_called_with(P_lower)
        presenter.frame_view.set_vmax.assert_called_with(P_upper)
        # Check what happens in the representation is invalid
        presenter.frame_view.set_vmin.reset_mock()
        presenter.active_representation = None
        presenter.reset_frame_range()
        presenter.frame_view.set_vmin.assert_not_called()

    def test_update_frame_range_limits(self):
        presenter = self.create_presenter()
        presenter.set_frame_vmin(1)
        presenter.set_frame_vmax(1.5)
        presenter.update_frame_range_limits()
        # Check that the right things were updated on the view
        presenter.frame_view.set_vmin_decimals.assert_called_with(2)
        presenter.frame_view.set_vmax_decimals.assert_called_with(2)
        presenter.frame_view.set_vmin_step.assert_called_with(10**-2)
        presenter.frame_view.set_vmax_step.assert_called_with(10**-2)
        presenter.frame_view.set_vmin_maximum.assert_called_with(1.5)
        presenter.frame_view.set_vmax_minimum.assert_called_with(1)

    def test_refresh_frames(self):
        presenter = self.create_presenter()
        presenter.refresh_frames()
        self.assertTrue(presenter.frame_view.draw_frames.emit.called)
        self.assertTrue(presenter.frame_view.draw_histogram.emit.called)
        self.assertTrue(presenter.frame_view.draw_spectrum.called)
        # Should just clear the axes if there's no data
        presenter.active_representation = None
        presenter.refresh_frames()
        self.assertTrue(presenter.frame_view.clear_axes.called)

    def test_change_cmap(self):
        presenter = self.create_presenter()
        presenter.frame_cmap = "viridis"
        # Check that it doesn't replot the frames if cmap stays the same
        presenter.change_cmap('viridis')
        presenter.frame_view.animate_frames.assert_not_called()
        # now actually change the color map
        presenter.change_cmap('rainbow')
        self.assertEqual(presenter.frame_cmap, 'rainbow')
        presenter.frame_view.animate_frames.assert_not_called()

    def test_next_frame(self):
        presenter = self.create_presenter()
        self.assertEqual(presenter.active_frame, 0)
        # Move to next frame
        presenter.next_frame()
        self.assertEqual(presenter.active_frame, 1)
        presenter.frame_view.frame_changed.emit.assert_called_with(1)
        # Move to next frame by wrapping around
        presenter.active_frame = 19
        presenter.next_frame()
        self.assertEqual(presenter.active_frame, 0)
        presenter.frame_view.frame_changed.emit.assert_called_with(0)

    def test_previous_frame(self):
        presenter = self.create_presenter()
        self.assertEqual(presenter.active_frame, 0)
        # Move to previous frame (with wrapping)
        presenter.previous_frame()
        self.assertEqual(presenter.active_frame, 19)
        presenter.frame_view.frame_changed.emit.assert_called_with(19)
        # Move to next frame by wrapping around
        presenter.previous_frame()
        self.assertEqual(presenter.active_frame, 18)
        presenter.frame_view.frame_changed.emit.assert_called_with(18)

    def test_first_frame(self):
        presenter = self.create_presenter()
        presenter.active_frame = 10
        presenter.first_frame()
        self.assertEqual(presenter.active_frame, 0)
        presenter.frame_view.frame_changed.emit.assert_called_with(0)

    def test_last_frame(self):
        presenter = self.create_presenter()
        presenter.active_frame = 10
        presenter.last_frame()
        self.assertEqual(presenter.active_frame, 19)
        presenter.frame_view.frame_changed.emit.assert_called_with(19)

    def test_build_hdf_tree(self):
        dummy_tree = [
            {'ndim': 0,
             'context': None,
             'level': 0,
             'name': "active_parent",
             'path': "/test-dir",
             'children': [
                 {'ndim': 2,
                  'context': 'frameset',
                  'level': 1,
                  'name': "Child",
                  'path': "/test-dir/test-dataset",
                  'children': [],
                 }
             ]},
            {'ndim': 0,
             'context': None,
             'level': 0,
             'name': "different_parent",
             'path': "/test-dir",
             'children': [],
            }
        ]
        frameset = MockFrameset()
        frameset.parent_name = "active_parent"
        frameset.data_tree = mock.Mock(return_value=dummy_tree)
        active_path = "/test-dir/test-dataset"
        frameset.hdf_path = mock.Mock(return_value=active_path)
        presenter = self.create_presenter(frameset=frameset)
        presenter.build_hdf_tree()
        # Check that the frame view was updated with the new items
        self.assertTrue(presenter.frame_view.add_hdf_tree_item.called)
        # Check that the tree items given are correct
        item1 = presenter.frame_view.add_hdf_tree_item.call_args_list[0][0][0]
        item2 = presenter.frame_view.add_hdf_tree_item.call_args_list[1][0][0]
        self.assertFalse(item1.isDisabled())
        self.assertTrue(item2.isDisabled())
        # Check that the active item is setApplicationName
        self.assertTrue(presenter.frame_view.select_active_hdf_item.called)
        item = presenter.frame_view.select_active_hdf_item.call_args[0][0]
        self.assertEqual(item.text(2), active_path)
    
    def test_change_hdf_group(self):
        # Create a mocked frameset object to test the presenter
        frameset = MockFrameset()
        data = np.random.rand(10, 128, 128)
        frameset.frames = mock.Mock(return_value=data)
        # Create the presenter object
        presenter = self.create_presenter(frameset=frameset)
        # Make a mock tree item
        new_item = mock.MagicMock(QtWidgets.QTreeWidgetItem)()
        
        def item_text(pos):
            if pos == 0:
                return 'new_groupname'
            elif pos == 1:
                return 'frameset'
            elif pos == 2:
                return '/test-sample/aligned/new_groupname'
    
        new_item.text.side_effect = item_text
        # Call the actual method to change the active tree item
        presenter.change_hdf_group(new_item, old_item=None)
        # Check that the new representation is set and used
        self.assertEqual(presenter.active_representation, 'new_groupname')
        frameset.frames.assert_called_with(timeidx=0, representation='new_groupname')
        # Check that the data_name is set correctly
        self.assertEqual(frameset.data_name, 'aligned')
        # Check that the statusbar text is updated
        presenter.frame_view.set_status_shape.assert_called_with(
            '(128, 128)')

    def test_update_status_shape(self):
        # Prepare a dummy frameset
        frameset = MockFrameset()
        data = np.random.rand(10, 128, 128)
        frameset.frames = mock.Mock(return_value=data)
        # Call the update status shape method
        presenter = self.create_presenter(frameset=frameset)
        presenter.update_status_shape()
        # Assert that the right frame_view calls were made
        presenter.frame_view.set_status_shape.assert_called_with(
            '(128, 128)')
        # Now make it invalid frame data
        presenter.active_representation = None
        def side_effect(*args, **kwargs):
            raise exceptions.GroupKeyError()
        frameset.frames = mock.Mock(side_effect=side_effect)
        presenter.update_status_shape()
        presenter.frame_view.set_status_shape.assert_called_with(
            '---')

    def test_update_status_frame(self):
        # Prepare a dummy frameset
        frameset = MockFrameset()
        data = np.linspace(8250, 8640, 10)
        frameset.energies = mock.Mock(return_value=data)
        # Call the update status shape method
        presenter = self.create_presenter(frameset=frameset)
        presenter.update_status_frame(1)
        # Check that the right things were called
        presenter.frame_view.set_status_energy.assert_called_with(
            '8293.33 eV')
        presenter.frame_view.set_status_index.assert_called_with(
            '1')

    def test_clear_hdf_group(self):
        """What happens when the user picks an HDF group that can't be
        displayed?"""
        presenter = self.create_presenter()
        new_item = mock.MagicMock(QtWidgets.QTreeWidgetItem)()
        
        def item_text(pos):
            if pos == 0:
                return 'aligned'
            elif pos == 1:
                return ''
            elif pos == 2:
                return '/test-sample/aligned'
        
        new_item.text.side_effect = item_text
        presenter.change_hdf_group(new_item, old_item=None)
        presenter.frame_view.clear_axes.assert_called_with()

    def test_dirty_flag(self):
        """Check that the dirty flag gets set so the presenter knows when to refresh."""
        # Check the accessor method
        presenter = self.create_presenter()
        presenter._dirty = 15
        self.assertEqual(presenter.dirty, 15)

    def test_play_frames(self):
        presenter = self.create_presenter()
        presenter.prepare_ui()
        # Now start playing the frames
        presenter.play_timer = mock.MagicMock()
        presenter.play_frames(True)
        presenter.play_timer.start.assert_called_with()
        presenter.play_timer.stop.assert_not_called()
        presenter.play_frames(False)
        presenter.play_timer.stop.assert_called_with()

    def test_hover_frame_pixel(self):
        # Set some dummy data on the frameset
        frameset = MockFrameset()
        data = self.dummy_frame_data((10, 1024, 1024))
        frameset.frames = mock.MagicMock(return_value=data)
        frameset.frame_shape = mock.MagicMock(return_value=(1024, 1024))
        # Set a fake extent on the frameset
        extent = Extent(-10, 10, -10, 10)
        frameset.extent = mock.MagicMock(return_value=extent)
        # Get the presenter value and see how it responds to hover
        presenter = self.create_presenter(frameset=frameset)
        presenter.active_frame = 2  # To avoid confusion with active_timestep
        presenter.hover_frame_pixel(xy=xycoord(3, 7))
        presenter.frame_view.set_status_cursor.assert_called_with(
            "(3.00, 7.00)")
        presenter.frame_view.set_status_pixel.assert_called_with(
            "[870, 666]")
        self.assertEqual(presenter.frame_pixel, (870, 666))
        presenter.frame_view.set_status_value.assert_called_with(
            str(data[2][870, 666]))
        # Now what happens when we leave the canvas
        presenter.hover_frame_pixel(xy=None)
        presenter.frame_view.set_status_cursor.assert_called_with(
            "")
        presenter.frame_view.set_status_pixel.assert_called_with(
            "")
        presenter.frame_view.set_status_value.assert_called_with(
            "")

class FrameSourceTestCase(unittest.TestCase):

    def test_add_callback(self):
        view = MockFrameView()
        source = FrameChangeSource(view=view)
        # Check that callback list starts out empty
        self.assertEqual(source.callbacks, [])
        # Add a callback function with some fake arguments
        func = mock.Mock()
        source.add_callback(func, 5, key="value")
        self.assertEqual(len(source.callbacks), 1)
        self.assertEqual(source.callbacks[0][0], func)
        self.assertEqual(source.callbacks[0][1], (5,))
        self.assertEqual(source.callbacks[0][2], {'key': 'value'})
        # Activate the source and see if the callbacks are executed
        source._on_change(9)
        func.assert_called_once_with(9, 5, key='value')

    def test_remove_callback_with_args(self):
        view = MockFrameView()
        source = FrameChangeSource(view=view)
        func = mock.Mock()
        # Add a callback to test
        source.add_callback(func, 5, key="value")
        self.assertEqual(len(source.callbacks), 1)
        # Remove the callback with arguments
        source.remove_callback(func, 5, key='value')
        self.assertEqual(len(source.callbacks), 0, "Callback not removed")

    def test_remove_callback_without_args(self):
        view = MockFrameView()
        source = FrameChangeSource(view=view)
        func = mock.Mock()
        # Add a callback to test
        source.add_callback(func, 5, key="value")
        self.assertEqual(len(source.callbacks), 1)
        # Remove the callback without arguments
        source.remove_callback(func)
        self.assertEqual(len(source.callbacks), 0, "Callback not removed")

    def test_start_stop(self):
        view = MockFrameView()
        source = FrameChangeSource(view=view)
        # Check that signals get connected
        source.start()
        self.assertTrue(source._is_running)
        view.frame_changed.connect.assert_called_with(source._on_change)
        # Check that signals get disconnected
        source.stop()
        self.assertFalse(source._is_running)
        view.frame_changed.disconnect.assert_called_with(source._on_change)


# Launch the tests if this is run as a script
if __name__ == '__main__':
    unittest.main()

