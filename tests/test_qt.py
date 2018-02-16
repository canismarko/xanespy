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
from unittest import mock, skip, skipUnless
import time
import os
import sys
import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pandas as pd
try:
    from PyQt5 import QtWidgets, QtTest
except ImportError:
    HAS_PYQT = False
else:
    HAS_PYQT = True
    from xanespy.qt_map_view import QtMapView
    from xanespy.qt_frameset_presenter import QtFramesetPresenter
    from xanespy.qt_frame_view import FrameChangeSource, QtFrameView

from xanespy import XanesFrameset, k_edges, exceptions
from xanespy.utilities import xycoord, Extent, shape, Pixel

pp = pprint.PrettyPrinter(indent=2)

if HAS_PYQT:
    # Define mocked views for the various Qt UI windows
    MockFrameView = mock.MagicMock(spec_set=QtFrameView)
    MockMapView = mock.MagicMock(spec_set=QtMapView)

def dummy_frame_data(shape=(10, 128, 128)):
    """Create some dummy data with a given shape. It's pretty much just an
    arange with reshaping."""
    length = np.prod(shape)
    data = np.arange(length)
    data = np.reshape(data, shape)
    return data

def MockFrameset(*args, **kwargs):
    Es = np.linspace(8250, 8650, num=10)
    data = dummy_frame_data((10, 128, 128))
    spectrum = pd.Series(np.mean(data, axis=(1, 2)), Es)
    fs_attrs = {
        'map_data.return_value': kwargs.pop('map_data.return_value', data[0]),
        'frames.return_value': kwargs.pop('frames.return_value', data),
        'energies.return_value': kwargs.pop('energies.return_value', Es),
        'spectrum.return_value': kwargs.pop('spectrum.return_value', spectrum),
        'extent.return_value': kwargs.pop('extent.return_value', Extent(0, 1, 0, 1)),
    }
    return mock.MagicMock(*args, spec_set=XanesFrameset, **fs_attrs, **kwargs)


class QtTestCase(unittest.TestCase):

    def create_presenter(self, frame_view=None, frameset=None, map_view=None):
        if frame_view is None:
            frame_view = MockFrameView()
        if frameset is None:
            frameset = MockFrameset()
            # Create some dummy data to be returned by the frameset
            frameset.num_energies = 20
            frame_data = dummy_frame_data()
            frameset.frames = mock.Mock(return_value=frame_data)
            map_data = self.dummy_map_data()
            frameset.map_data = mock.Mock(return_value=map_data)
            frameset.spectrum = mock.Mock(return_value=pd.Series())
            extent = Extent(-10, 10, -10, 10)
            frameset.extent = mock.Mock(return_value=extent)
            frameset.frame_shape = mock.Mock(return_value=(128, 128))
        if map_view is None:
            map_view = MockMapView()
        p = QtFramesetPresenter(frameset=frameset,
                                frame_view=frame_view)
        p.num_frames = 20
        # Mock create_app method so it can run headless (eg travis-ci.org)
        p.create_app = mock.MagicMock(name="create_app")
        return p

    def dummy_map_data(self, shape=(128, 128)):
        """Create some dummy data with a given shape. It's pretty much just an
        arange with reshaping."""
        length = np.prod(shape)
        data = np.arange(length)
        data = np.reshape(data, shape)
        return data

@skipUnless(HAS_PYQT, "PyQt5 required")
class MapViewTestCase(QtTestCase):

    def test_keyboard_nav(self):
        view = QtMapView()
        event = mock.Mock()
        # First test a non-navigation key
        event.key = 'u'
        spy = QtTest.QSignalSpy(view.map_moved)
        view.keyboard_nav(event)
        self.assertEqual(len(spy), 0)
        # Now test a navigation key
        event.key = 'ctrl+w'
        view.keyboard_nav(event)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0], [-10, 0])

    def test_mouse_in_canvas(self):
        view = QtMapView()
        view.map_ax = "fake_axis"
        event = mock.Mock()
        # Test hovering out of the axis
        spy = QtTest.QSignalSpy(view.map_hovered)
        view.mouse_in_canvas(event)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0], [None, None])
        # Test hovering into the axis
        event.inaxes = view.map_ax
        event.xdata = 3.54
        event.ydata = -12.98
        view.mouse_in_canvas(event)
        self.assertEqual(len(spy), 2)
        self.assertEqual(spy[-1], [3.54, -12.98])

    def test_mouse_clicked(self):
        view = QtMapView()
        view.map_ax = "fake_axis"
        event = mock.Mock()
        # Test hovering out of the axis
        spy = QtTest.QSignalSpy(view.map_clicked)
        view.mouse_clicked_canvas(event)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0], [None, None])
        # Test hovering into the axis
        event.inaxes = view.map_ax
        event.xdata = 3.54
        event.ydata = -12.98
        view.mouse_clicked_canvas(event)
        self.assertEqual(len(spy), 2)
        self.assertEqual(spy[-1], [3.54, -12.98])

    def test_update_crosshairs(self):
        view = QtMapView()
        view.ui = mock.Mock()
        xy = xycoord(1.23, -5.87)
        pixel = Pixel(348, 750)
        value = 3.1415926
        view.update_crosshair_labels(xy, pixel, value)
        # Check that new label text was set
        view.ui.lblCrosshairsXY.setText.assert_called_with(
            "1.23, -5.87")
        view.ui.lblCrosshairsPixel.setText.assert_called_with(
            "[348, 750]")
        view.ui.lblCrosshairsValue.setText.assert_called_with(
            "3.1415926")
        # Repeat the test with no pixel given
        view.update_crosshair_labels(None, None, None)
        view.ui.lblCrosshairsXY.setText.assert_called_with(
            "")
        view.ui.lblCrosshairsPixel.setText.assert_called_with(
            "")
        view.ui.lblCrosshairsValue.setText.assert_called_with(
            "")

    def test_update_cursor_labels(self):
        view = QtMapView()
        view.ui = mock.Mock()
        xy = xycoord(1.23, -5.87)
        pixel = Pixel(348, 750)
        value = 3.1415926
        view.update_cursor_labels(xy, pixel, value)
        # Check that new label text was set
        view.ui.lblCursorXY.setText.assert_called_with(
            "1.23, -5.87")
        view.ui.lblCursorPixel.setText.assert_called_with(
            "[348, 750]")
        view.ui.lblCursorValue.setText.assert_called_with(
            "3.1415926")
        # Repeat the test with no pixel given
        view.update_cursor_labels(None, None, None)
        view.ui.lblCursorXY.setText.assert_called_with(
            "")
        view.ui.lblCursorPixel.setText.assert_called_with(
            "")
        view.ui.lblCursorValue.setText.assert_called_with(
            "")

    def test_set_cmap_list(self):
        view = QtMapView()
        view.ui = mock.Mock()
        view.set_cmap_list(['a', 'b'])
        view.ui.cmbCmap.clear.assert_called_with()
        view.ui.cmbCmap.addItems.assert_called_with(['a', 'b'])
    
    def test_map_limits(self):
        view = QtMapView()
        view.ui = mock.Mock()
        view.set_map_limits(0, 10, 0.1, 3)
        # Check the vmin spin box
        view.ui.spnVMin.setMaximum.assert_called_once_with(9.999)
        view.ui.spnVMin.setSingleStep.assert_called_once_with(0.1)
        view.ui.spnVMin.setDecimals.assert_called_once_with(3)
        view.ui.spnVMin.setValue.assert_called_once_with(0)
        # Check the vmax spin box
        view.ui.spnVMax.setMinimum.assert_called_once_with(0.001)
        view.ui.spnVMax.setSingleStep.assert_called_once_with(0.1)
        view.ui.spnVMax.setDecimals.assert_called_once_with(3)
        view.ui.spnVMax.setValue.assert_called_once_with(10)

@skipUnless(HAS_PYQT, "PyQt5 required")
class PresenterTestCase(QtTestCase):

    def test_set_frame_range(self):
        # Prepare test scaffolding
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.frame_vrange_changed)
        # Execute code under test
        presenter.set_frame_vrange(1.3, 2.7)
        # Check that the right values were saved
        self.assertEqual(presenter._frame_vmin, 1.3)
        self.assertEqual(presenter._frame_vmax, 2.7)
        # Check that the correct signal was emitted
        self.assertEqual(len(spy), 1)
        # Check that absurd values don't get set
        presenter.set_frame_vrange(1, 1)
        self.assertEqual(presenter._frame_vmin, 1.3)
        self.assertEqual(presenter._frame_vmax, 2.7)
        
    #     def test_reset_limits(self):
    #     frameset = MockFrameset()
        
    #     def frames(*args, **kwargs):
    #         if kwargs.get('representation', '') is None:
    #             raise exceptions.GroupKeyError()
    #         else:
    #             return np.linspace(1.5, 5)
        
    #     frameset.frames.side_effect = frames
    #     presenter = self.create_presenter(frameset=frameset)
    #     presenter.active_representation = "optical_depths"
    #     presenter.reset_frame_range()
    #     P_lower = np.percentile(frameset.frames(), 1)
    #     P_upper = np.percentile(frameset.frames(), 99)
    #     self.assertEqual(presenter._frame_vmin, P_lower)
    #     self.assertEqual(presenter._frame_vmax, P_upper)
    #     presenter.frame_view.set_vmin.assert_called_with(P_lower)
    #     presenter.frame_view.set_vmax.assert_called_with(P_upper)
    #     # Check what happens in the representation is invalid
    #     presenter.frame_view.set_vmin.reset_mock()
    #     presenter.active_representation = None
    #     presenter.reset_frame_range()
    #     presenter.frame_view.set_vmin.assert_not_called()
    
    # def test_update_frame_range_limits(self):
    #     presenter = self.create_presenter()
    #     presenter.set_frame_vmin(1)
    #     presenter.set_frame_vmax(1.5)
    #     presenter.update_frame_range_limits()
    #     # Check that the right things were updated on the view
    #     presenter.frame_view.set_vmin_decimals.assert_called_with(2)
    #     presenter.frame_view.set_vmax_decimals.assert_called_with(2)
    #     presenter.frame_view.set_vmin_step.assert_called_with(10**-2)
    #     presenter.frame_view.set_vmax_step.assert_called_with(10**-2)
    #     presenter.frame_view.set_vmin_maximum.assert_called_with(1.5)
    #     presenter.frame_view.set_vmax_minimum.assert_called_with(1)
    
    def test_change_active_frame(self):
        presenter = self.create_presenter()
        presenter.num_frames = 20
        spy = QtTest.QSignalSpy(presenter.active_frame_changed)
        presenter.change_active_frame(1)
        # was the new frame set?
        self.assertEqual(presenter.active_frame, 1)
        # Was the signal emitted for changing the frame
        self.assertEqual(len(spy), 1)
    
    def test_next_frame(self):
        presenter = self.create_presenter()
        presenter.num_frames = 20
        spy = QtTest.QSignalSpy(presenter.active_frame_changed)
        self.assertEqual(presenter.active_frame, 0)
        # Move to next frame
        presenter.next_frame()
        self.assertEqual(presenter.active_frame, 1)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0][0], 1)
        # presenter.active_frame_changed.emit.assert_called_with(1)
        # Move to next frame by wrapping around
        presenter.active_frame = 19
        presenter.next_frame()
        self.assertEqual(presenter.active_frame, 0)
        self.assertEqual(len(spy), 2)
        self.assertEqual(spy[1][0], 0)
    
    def test_previous_frame(self):
        presenter = self.create_presenter()
        presenter.num_frames = 20
        presenter.active_frame = 2
        spy = QtTest.QSignalSpy(presenter.active_frame_changed)
        self.assertEqual(presenter.active_frame, 2)
        # Move to next frame
        presenter.previous_frame()
        self.assertEqual(presenter.active_frame, 1)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0][0], 1)
        # presenter.active_frame_changed.emit.assert_called_with(1)
        # Move to next frame by wrapping around
        presenter.active_frame = 0
        presenter.previous_frame()
        self.assertEqual(presenter.active_frame, 19)
        self.assertEqual(len(spy), 2)
        self.assertEqual(spy[1][0], 19)
    
    def test_first_frame(self):
        presenter = self.create_presenter()
        presenter.num_frames = 20
        presenter.active_frame = 10
        spy = QtTest.QSignalSpy(presenter.active_frame_changed)
        presenter.first_frame()
        self.assertEqual(presenter.active_frame, 0)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0][0], 0)
    
    def test_last_frame(self):
        presenter = self.create_presenter()
        presenter.num_frames = 20
        presenter.active_frame = 10
        spy = QtTest.QSignalSpy(presenter.active_frame_changed)
        presenter.last_frame()
        self.assertEqual(presenter.active_frame, 19)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0][0], 19)
    
    def test_active_map(self):
        presenter = self.create_presenter()
        fs = presenter.frameset
        # Create a random mask to apply
        mask = np.random.choice([True, False], size=fs.map_data().shape)
        masked_map = np.ma.array(fs.map_data(), mask=mask)
        fs.edge_mask = mock.Mock(return_value=mask)
        # Set the flag for applying the mask and the map_ax
        presenter.toggle_edge_mask(True)
        self.assertTrue(presenter.use_edge_mask)
        map_ = presenter.active_map()
        np.testing.assert_equal(map_, masked_map)
        np.testing.assert_equal(map_.mask, mask)

    def test_hover_map_pixel(self):
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.map_cursor_changed)
        presenter.set_map_cursor(1.2, -3.5)
        # Check that the signal was dispatched properly
        self.assertEqual(len(spy), 1)
        xy, pixel, value = spy[0]
        self.assertEqual(xy, (1.2, -3.5))
        self.assertEqual(pixel, (86, 72))
        self.assertEqual(value, presenter.frameset.map_data()[86, 72])
        # What happens if we clear the map cursor
        spy = QtTest.QSignalSpy(presenter.map_cursor_changed)
        presenter.set_map_cursor(None, None)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0], [None, None, None])
        # What happens if the data aren't valid
        spy = QtTest.QSignalSpy(presenter.map_cursor_changed)
        def no_data(*args, **kwargs):
            raise exceptions.GroupKeyError()
        presenter.frameset.frame_shape = mock.Mock(side_effect=no_data)
        presenter.set_map_cursor(1, 3)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0], [None, None, None])
    
    def test_spectrum_fit(self):
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.map_spectrum_changed)
        self.assertFalse(presenter.show_spectrum_fit)
        presenter.toggle_spectrum_fit(True)
        # Check that the state variable is set
        self.assertTrue(presenter.show_spectrum_fit)
        # Check that the right arguments are given to the signal
        self.assertEqual(len(spy), 1)
        fit = spy[0][1]
        self.assertTrue(fit is not None)
        # Now disable fit plotting
        presenter.toggle_spectrum_fit(False)
        # Check that the state variable is set
        self.assertFalse(presenter.show_spectrum_fit)
        # Check that the right arguments are given to the signal
        self.assertEqual(len(spy), 2)
        fit = spy[1][1]
        self.assertIs(fit, None)
    
    def test_click_map_pixel(self):
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.map_pixel_changed)
        spectrum_spy = QtTest.QSignalSpy(presenter.map_spectrum_changed)
        presenter.set_map_pixel(1.2, -3.5)
        # Check that the interal state is set appropriately
        self.assertEqual(presenter._map_pixel, (86, 72))
        # Check that the signal was dispatched properly
        self.assertEqual(len(spectrum_spy), 1)
        self.assertEqual(len(spy), 1)
        xy, pixel, value = spy[0]
        self.assertEqual(xy, (1.2, -3.5))
        self.assertEqual(pixel, (86, 72))
        self.assertEqual(value, presenter.frameset.map_data()[86, 72])
        # What happens if we clear the map cursor
        spy = QtTest.QSignalSpy(presenter.map_cursor_changed)
        presenter.set_map_cursor(None, None)
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0], [None, None, None])

    def test_move_map_pixel(self):
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.map_pixel_changed)
        # First check that nothing happens if no pixel is set
        presenter._map_pixel = None
        presenter.move_map_pixel(1, -3)
        self.assertEqual(len(spy), 0)
        self.assertEqual(presenter._map_pixel, None)
        # Now check that it moves if there's an active pixel
        presenter._map_pixel = (5, 5)
        presenter.move_map_pixel(1, -3)
        self.assertEqual(presenter._map_pixel, (6, 2))
        self.assertEqual(len(spy), 1)

    def test_map_data_changed(self):
        """Check that changing the hdf group to a `map` representation
        launches the map view window.
        
        """
        # Prepare a dummy frameset to test the data
        
        fs = MockFrameset()
        map_data = np.linspace(11, 111)
        fs.map_data = mock.Mock(return_value=map_data)
        fs.frames = mock.Mock(
            return_value=dummy_frame_data((10, 128, 128)))
        fs.spectrum = mock.Mock(return_value=pd.Series())
        fs.extent = mock.Mock(return_value=(0, 1, 2, 4))
        presenter = self.create_presenter(frameset=fs)
        new_item = mock.MagicMock(QtWidgets.QTreeWidgetItem)()
        
        def item_text(pos):
            if pos == 0:
                return 'whiteline_map'
            elif pos == 1:
                return 'map'
            elif pos == 2:
                return '/test-sample/aligned/whiteline_map'
        
        new_item.text.side_effect = item_text
        show_spy = QtTest.QSignalSpy(presenter.map_data_changed)
        # Change the active hdf data group
        presenter.change_hdf_group(new_item, old_item=None)
        # Check that the map data are updated
        np.testing.assert_equal(show_spy[0][0], presenter._map_data)
        new_norm = show_spy[0][1]
        self.assertEqual(new_norm.vmin, 12)
        self.assertEqual(new_norm.vmax, 110)
        self.assertEqual(show_spy[0][2], 'plasma')
    
    def test_refresh_frames(self):
        """Check that asking for the frames to be reshed emits the right signals.
        """
        # Prepare mock data for the presenter to interact with
        Es = np.linspace(8250, 8650, num=10)
        data = dummy_frame_data((10, 128, 128))
        fs_attrs = {
            'frames.return_value': data,
            'energies.return_value': Es,
            'spectrum.return_value': pd.Series(np.mean(data, axis=(1, 2)), Es),
        }
        fs = MockFrameset(**fs_attrs)
        # Create the presenter object and cache some fake map data
        presenter = self.create_presenter(frameset=fs)
        presenter.active_representation = 'optical_depths'
        presenter.num_frames = 2
        # Prepare spies to monitor the signal events
        changed_spy = QtTest.QSignalSpy(presenter.frame_data_changed)
        spectrum_spy = QtTest.QSignalSpy(presenter.mean_spectrum_changed)
        # Run the code under test
        presenter.refresh_frames()
        self.assertEqual(len(changed_spy), 1, '``frame_data_changed`` not emitted')
        self.assertEqual(len(spectrum_spy), 1, '``mean_spectrum_changed`` not emitted')
    
    def test_map_data_cleared(self):
        """Check that changing the hdf group to a non-`map` representation
        hides the map view window.
        
        """
        # Prepare mock data for the presenter to interact with
        fs = MockFrameset()
        def no_data(*args, **kwargs):
            raise exceptions.GroupKeyError()
        fs.map_data = mock.Mock(side_effect=no_data)
        fs.frames = mock.Mock(return_value=dummy_frame_data((10, 128, 128)))
        # Create the presenter object and cache some fake map data
        presenter = self.create_presenter(frameset=fs)
        # Prepare spies to monitor the signal events
        changed_spy = QtTest.QSignalSpy(presenter.map_data_changed)
        cleared_spy = QtTest.QSignalSpy(presenter.map_data_cleared)
        # Update the map data
        presenter.update_maps()
        # Check that the right signals were emitted
        self.assertEqual(len(changed_spy), 0)
        self.assertEqual(len(cleared_spy), 1)
    
    def test_update_spectra(self):
        """Look for new spectra and send them out the appropriate signals."""
        presenter = self.create_presenter()
        fs = presenter.frameset
        # Create fake spectrum and frames arrays to test
        spectrum = pd.Series(np.linspace(750, 760, dtype=np.complex))
        fs.spectrum = mock.Mock(return_value=spectrum)
        frames = np.broadcast_to(spectrum.values, (128, 128, len(spectrum)))
        frames = np.moveaxis(frames, -1, 0)
        fs.frames = mock.Mock(return_value=frames)
        # Launch and monitor the call the `update_spectra`
        mean_spy = QtTest.QSignalSpy(presenter.mean_spectrum_changed)
        map_spy = QtTest.QSignalSpy(presenter.map_spectrum_changed)
        presenter.update_spectra()
        self.assertEqual(len(mean_spy), 1)
        np.testing.assert_equal(mean_spy[0][0].values, spectrum.values)
        self.assertEqual(len(map_spy), 1)
        np.testing.assert_equal(map_spy[0][0].values, spectrum.values)
        # Check that it uses the active pixel for getting the spectrum
        fs.spectrum.reset_mock()
        presenter._map_pixel = (0, 0)
        presenter.update_spectra()
        kwargs = fs.spectrum.call_args[1]
        self.assertEqual(kwargs['pixel'], (0, 0))
    
    def test_change_map_limits(self):
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.map_limits_changed)
        # Check what happens if we set it to the same value
        presenter.set_map_vmin(presenter._map_vmin)
        presenter.set_map_vmax(presenter._map_vmax)
        self.assertEqual(len(spy), 0)
        # Now set some different values
        presenter.set_map_vmax(7)
        presenter.set_map_vmin(3)
        presenter.update_map_limits()
        # Check the signal was called with the right arguments
        vmin, vmax, step, decimals = spy[-1]
        self.assertEqual(vmax, 7)
        self.assertEqual(vmin, 3)
        self.assertEqual(step, 0.1)
        self.assertEqual(decimals, 2)
    
    def test_map_data_caching(self):
        """If the same data is retrieved in successive calls, no plotting
        should take place."""
        Es = np.linspace(8250, 8650, num=10)
        data = dummy_frame_data((10, 128, 128))
        fs_attrs = {
            'map_data.return_value': data[0],
            'frames.return_value': data,
            'energies.return_value': Es,
            'spectrum.return_value': pd.Series(np.mean(data, axis=(1, 2)), Es),
        }
        fs = MockFrameset(**fs_attrs)
        presenter = self.create_presenter(frameset=fs)
        new_item = mock.MagicMock(QtWidgets.QTreeWidgetItem)()
        new_item.text = mock.Mock(return_value='map')
        # new_item.text.side_effect = item_text
        # Make sure it starts off with no cached data
        self.assertIs(presenter._map_data, None)
        # Prepare spies for monitoring the signals
        presenter.change_hdf_group(new_item, old_item=None)
        np.testing.assert_equal(presenter._map_data,
                                self.dummy_map_data())
        # Having the same data should not emit the changed signal
        changed_spy = QtTest.QSignalSpy(presenter.map_data_changed)
        cleared_spy = QtTest.QSignalSpy(presenter.map_data_cleared)
        presenter.change_hdf_group(new_item, old_item=None)
        self.assertEqual(len(changed_spy), 0)
        self.assertEqual(len(cleared_spy), 0)
    
    def test_change_map_cmap(self):
        # Prepare presenter and spies
        presenter = self.create_presenter()
        spy = QtTest.QSignalSpy(presenter.map_data_changed)
        presenter.map_cmap = "viridis"
        # Keep the cmap the same and make sure nothing happens
        presenter.change_map_cmap("viridis")
        self.assertEqual(len(spy), 0)
        # Change the cmap and make sure the map data is refreshed
        presenter.change_map_cmap("copper")
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0][2], 'copper')
        
    
    def test_reset_map_range(self):
        fs = MockFrameset()
        map_data = np.linspace(9, 109, num=101)
        fs.map_data = mock.Mock(return_value=map_data)
        presenter = self.create_presenter(frameset=fs)
        # Check that the initial state is its default
        self.assertEqual(presenter._map_vmin, 0)
        self.assertEqual(presenter._map_vmax, 1)
        # Now change the limits and check again
        vmin, vmax = presenter.reset_map_range()
        self.assertEqual(vmin, 10)
        self.assertEqual(vmax, 108)
        # Check that the cached values were updated
        self.assertEqual(presenter._map_vmin, 10)
        self.assertEqual(presenter._map_vmax, 108)
        # Check what happens if some values are nan
        map_data = np.linspace(8, 109, num=102)
        map_data[0] = np.nan
        fs.map_data = mock.Mock(return_value=map_data)
        vmin, vmax = presenter.reset_map_range()
        self.assertEqual(vmin, 10)
        self.assertEqual(vmax, 108)
        self.assertEqual(presenter._map_vmin, 10)
        self.assertEqual(presenter._map_vmax, 108)

@skipUnless(HAS_PYQT, "PyQt5 required")
class OldFrameViewerTestcase(QtTestCase):
    
    def test_init(self):
        """Check that certain values are set properly during __init__."""
        # Switch to "optical_depths" representation if possible
        frameset = MockFrameset()
        frameset.has_representation.return_value = True
        presenter = self.create_presenter(frameset=frameset)
        self.assertEqual(presenter.active_representation, None)
    
    def test_prepare_ui(self):
        # Prepare a frameset object to test the presenter
        frameset = MockFrameset()
        data = np.random.rand(10, 128, 128)
        frameset.frames = mock.Mock(return_value=data)
        frameset.num_energies = 20
        # Create the presenter
        presenter = self.create_presenter(frameset=frameset)
        # Run the actual code
        presenter.prepare_ui()
        self.assertTrue(presenter.frame_view.set_cmap_list.called)
        self.assertTrue(presenter.frame_view.set_timestep_list.called)
        presenter.frame_view.set_timestep.assert_called_with(0)
    
    def test_set_timestep(self):
        # Set up some fake data
        frameset = MockFrameset()
        frames = dummy_frame_data()
        frameset.frames = mock.Mock(return_value=frames)
        maps = self.dummy_map_data()
        frameset.map_data = mock.Mock(return_value=maps)
        presenter = self.create_presenter(frameset=frameset)
        presenter.active_representation = "optical_depths"
        map_spy = QtTest.QSignalSpy(presenter.map_data_changed)
        frames_spy = QtTest.QSignalSpy(presenter.frame_data_changed)
        spectrum_spy = QtTest.QSignalSpy(presenter.mean_spectrum_changed)
        map_spectrum_spy = QtTest.QSignalSpy(presenter.map_spectrum_changed)
        # Now invoke to function to be tested
        presenter.set_timestep(5)
        # Check that all the view elements are updated
        self.assertEqual(presenter.active_timestep, 5)
        frameset.frames.assert_called_with(timeidx=5,
                                           representation='optical_depths')
        frameset.map_data.assert_called_with(timeidx=5,
                                           representation='optical_depths')
        # self.assertTrue(presenter.frame_view.draw_spectrum.called)
        # self.assertTrue(presenter.frame_view.draw_histogram.emit.called)
        # Check that map and frame data are updated
        self.assertEqual(len(frames_spy), 1)
        self.assertEqual(len(map_spy), 1)
        self.assertEqual(len(spectrum_spy), 1)
        self.assertEqual(len(map_spectrum_spy), 1)
    
    def test_move_slider(self):
        presenter = self.create_presenter()
        presenter.move_slider(5)
        self.assertEqual(presenter.active_frame, 5)
    
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
        presenter.build_hdf_tree(expand_tree=False)
        # Check that the frame view was updated with the new items
        self.assertTrue(presenter.frame_view.add_hdf_tree_item.called)
        # Check that the tree items given are correct
        item1 = presenter.frame_view.add_hdf_tree_item.call_args_list[0][0][0]
        item2 = presenter.frame_view.add_hdf_tree_item.call_args_list[1][0][0]
        self.assertFalse(item1.isDisabled())
        self.assertFalse(item2.isDisabled())
    
    def test_change_hdf_group(self):
        # Create a mocked frameset object to test the presenter
        frameset = MockFrameset()
        data = dummy_frame_data((10, 128, 128))
        frameset.frames = mock.Mock(return_value=data)
        frameset.pixel_unit = mock.Mock(return_value='um')
        # Create the presenter object
        presenter = self.create_presenter(frameset=frameset)
        presenter.prepare_ui()
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
            '(10, 128, 128)')
        presenter.frame_view.set_status_unit.assert_called_with(
            '(um):')
        # Check that the window titles were updated
        presenter.frame_view.set_window_title.assert_called_with(
            'Xanespy Frameset (/test-sample/aligned/new_groupname)'
        )
        # Check that relevant UI elements were updated
        presenter.frame_view.set_slider_max.assert_called_with(9)
        # Check that UI was disabled then re-enabled
        calls = [mock.call(True), mock.call(False)]
        presenter.frame_view.disable_plotting_controls.assert_has_calls(calls)
        presenter.frame_view.disable_frame_controls.assert_has_calls(calls)
        presenter.frame_view.set_drawing_status.assert_has_calls(calls)
    
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
            '(10, 128, 128)')
        # Now make it invalid frame data
        presenter.frame_representation = None
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
        frameset = MockFrameset()
        presenter = self.create_presenter(frameset=frameset)
        new_item = mock.MagicMock(QtWidgets.QTreeWidgetItem)()
        
        def item_text(pos):
            if pos == 0:
                return 'aligned'
            elif pos == 1:
                return ''
            elif pos == 2:
                return '/test-sample/aligned'
        
        new_item.text.side_effect = item_text
        def no_frames(*args, **kwargs):
            raise exceptions.GroupKeyError()
        frameset.frames = mock.Mock(side_effect=no_frames)
        presenter.change_hdf_group(new_item, old_item=None)
        presenter.frame_view.clear_axes.assert_called_with()
    
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
        data = dummy_frame_data((10, 1024, 1024))
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
            "[154, 666]")
        self.assertEqual(presenter.frame_pixel, (154, 666))
        presenter.frame_view.set_status_value.assert_called_with(
            str(data[2][154, 666]))
        # Now what happens when we leave the canvas
        presenter.hover_frame_pixel(xy=None)
        presenter.frame_view.set_status_cursor.assert_called_with(
            "")
        presenter.frame_view.set_status_pixel.assert_called_with(
            "")
        presenter.frame_view.set_status_value.assert_called_with(
            "")

@skipUnless(HAS_PYQT, "PyQt5 required")
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
        view.presenter_frame_changed.connect.assert_called_with(source._on_change)
        # Check That Signals Get Disconnected
        source.stop()
        self.assertFalse(source._is_running)
        view.presenter_frame_changed.disconnect.assert_called_with(source._on_change)


# Launch the tests if this is run as a script
if __name__ == '__main__':
    unittest.main()

