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
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pandas as pd

from xanespy import QtFrameView, QtMapView, XanesFrameset, QtFramesetPresenter
from xanespy.qt_frame_view import FrameChangeSource


# Define mocked views for the various Qt UI windows
MockFrameView = mock.MagicMock(spec_set=QtFrameView)
MockMapView = mock.MagicMock(spec_set=QtMapView)
MockFrameset = mock.MagicMock(spec_set=XanesFrameset)

class QtTestCase(unittest.TestCase):

    def setUp(self):
        self.frameview = MockFrameView()
        self.frameset = MockFrameset()
        self.frameset.frames.return_value = np.random.rand(10, 128, 128)
        self.frameset.num_energies = 20
        self.mapview = MockMapView()
        p = QtFramesetPresenter(frameset=self.frameset,
                                frame_view=self.frameview,
                                map_view=self.mapview)
        self.presenter = p
        self.presenter.create_app = mock.MagicMock(name="create_app")


class FrameViewerTestcase(QtTestCase):

    def test_prepare_ui(self):
        presenter = self.presenter
        # Check that the number of energies is set properly
        self.assertEqual(presenter.num_frames, 20)
        # Run the actual code
        presenter.prepare_ui()
        self.frameview.set_slider_max.assert_called_with(9)
        self.assertTrue(self.frameview.set_cmap_list.called)
        self.assertTrue(self.frameview.set_timestep_list.called)
        self.frameview.set_timestep.assert_called_with(0)

    def test_set_timestep(self):
        presenter = self.presenter
        # Set up some fake data
        data = np.linspace(5, 105, num=1000)
        self.frameset.frames.return_value = data
        # Now invoke to function to be tested
        presenter.set_timestep(5)
        # Check that all the view elements are updated
        self.assertEqual(presenter.active_timestep, 5)
        self.frameset.frames.assert_called_with(timeidx=5)
        self.assertTrue(self.frameview.draw_frames.emit.called)
        self.assertTrue(self.frameview.draw_spectrum.called)
        self.assertTrue(self.frameview.draw_histogram.emit.called)
        # Check that the frame range was reset
        self.assertAlmostEqual(presenter._frame_vmin, 6)

    def test_change_vmin_vmax(self):
        presenter = self.presenter
        # Check the vmax
        presenter.set_frame_vmax(5.0)
        self.assertEqual(presenter._frame_vmax, 5.0)
        self.frameview.set_vmax_decimals.assert_called_with(1)
        self.frameview.set_vmax_step.assert_called_with(0.1)
        self.frameview.set_vmin_maximum.assert_called_with(5.0)
        # Check the vmin
        presenter.set_frame_vmin(2)
        self.assertEqual(presenter._frame_vmin, 2)
        self.frameview.set_vmin_decimals.assert_called_with(1)
        self.frameview.set_vmin_step.assert_called_with(0.1)
        self.frameview.set_vmax_minimum.assert_called_with(2)
        # Check normalizer
        norm = presenter.frame_norm()
        self.assertEqual(norm.vmin, 2)
        self.assertEqual(norm.vmax, 5)

    def test_draw_frame_histogram(self):
        presenter = self.presenter
        presenter.draw_frame_histogram()
        self.assertTrue(self.frameview.draw_histogram.emit.called)

    def test_draw_frame_spectra(self):
        presenter = self.presenter
        presenter.set_frame_vmax(5.7)
        presenter.set_frame_vmin(1.5)
        # Set some expected values on the frameset
        spectrum = pd.Series()
        self.frameset.spectrum.return_value = spectrum
        self.frameset.edge.edge_range = (8353, 8359)
        # Check that the spectrum was drawn
        presenter.draw_frame_spectra()
        self.assertTrue(self.frameset.spectrum.called)
        # Check that the normalizer was correct
        called_norm = self.frameview.draw_spectrum.call_args[1]['norm']
        self.assertEqual(called_norm.vmin, 1.5)
        self.assertEqual(called_norm.vmax, 5.7)
        # Check that the edge range is provided
        called_edge = self.frameview.draw_spectrum.call_args[1]['edge_range']
        self.assertEqual(called_edge, (8353, 8359))

    def test_move_slider(self):
        presenter = self.presenter
        presenter.move_slider(5)
        self.assertEqual(self.presenter.active_frame, 5)

    def test_reset_limits(self):
        presenter = self.presenter
        frameset = self.frameset
        frameset.frames.return_value = np.linspace(1.5, 5)
        presenter.reset_frame_range()
        P_lower = np.percentile(frameset.frames(), 1)
        P_upper = np.percentile(frameset.frames(), 99)
        self.assertEqual(presenter._frame_vmin, P_lower)
        self.assertEqual(presenter._frame_vmax, P_upper)
        self.frameview.set_vmin.assert_called_with(P_lower)
        self.frameview.set_vmax.assert_called_with(P_upper)

    def test_refresh_frames(self):
        presenter = self.presenter
        presenter.refresh_frames()
        self.assertTrue(self.frameview.draw_frames.emit.called)
        self.assertTrue(self.frameview.draw_histogram.emit.called)
        self.assertTrue(self.frameview.draw_spectrum.called)

    def test_change_cmap(self):
        presenter = self.presenter
        presenter.frame_cmap = "viridis"
        # Check that it doesn't replot the frames if cmap stays the same
        presenter.change_cmap('viridis')
        self.frameview.animate_frames.assert_not_called()
        # now actually change the color map
        presenter.change_cmap('rainbow')
        self.assertEqual(presenter.frame_cmap, 'rainbow')
        self.frameview.animate_frames.assert_not_called()

    def test_next_frame(self):
        presenter = self.presenter
        self.assertEqual(presenter.active_frame, 0)
        # Move to next frame
        presenter.next_frame()
        self.assertEqual(presenter.active_frame, 1)
        self.frameview.frame_changed.emit.assert_called_with(1)
        # Move to next frame by wrapping around
        presenter.active_frame = 19
        presenter.next_frame()
        self.assertEqual(presenter.active_frame, 0)
        self.frameview.frame_changed.emit.assert_called_with(0)

    def test_previous_frame(self):
        presenter = self.presenter
        self.assertEqual(presenter.active_frame, 0)
        # Move to previous frame (with wrapping)
        presenter.previous_frame()
        self.assertEqual(presenter.active_frame, 19)
        self.frameview.frame_changed.emit.assert_called_with(19)
        # Move to next frame by wrapping around
        presenter.previous_frame()
        self.assertEqual(presenter.active_frame, 18)
        self.frameview.frame_changed.emit.assert_called_with(18)

    def test_first_frame(self):
        presenter = self.presenter
        presenter.active_frame = 10
        presenter.first_frame()
        self.assertEqual(presenter.active_frame, 0)
        self.frameview.frame_changed.emit.assert_called_with(0)

    def test_last_frame(self):
        presenter = self.presenter
        presenter.active_frame = 10
        presenter.last_frame()
        self.assertEqual(presenter.active_frame, 19)
        self.frameview.frame_changed.emit.assert_called_with(19)


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

