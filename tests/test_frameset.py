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

import datetime as dt
import unittest
from unittest import TestCase, mock
import logging
import math
import os
import shutil
import warnings
from collections import namedtuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import h5py
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter('ignore', PendingDeprecationWarning)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import Normalize
import pytz
from skimage import data, transform

from cases import XanespyTestCase
from xanespy import exceptions, edges
from xanespy.utilities import (xycoord, prog, position, Extent,
                               xy_to_pixel, pixel_to_xy,
                               get_component, Pixel, broadcast_reverse)
from xanespy.xanes_frameset import XanesFrameset
from xanespy.xanes_math import (transform_images, direct_whitelines,
                                particle_labels, k_edge_jump,
                                k_edge_mask, l_edge_mask,
                                apply_references, iter_indices,
                                predict_edge, fit_kedge, kedge_params,
                                KEdgeParams, extract_signals_nmf,
                                guess_kedge, transformation_matrices,
                                apply_internal_reference,
                                register_template, _fit_spectrum)
# from xanespy.edges import KEdge, k_edges, l_edges
from xanespy.importers import (import_ssrl_frameset,
                               import_aps_8BM_frameset,
                               import_nanosurveyor_frameset,
                               _average_frames,
                               magnification_correction,
                               decode_aps_params, decode_ssrl_params,
                               read_metadata, CURRENT_VERSION as IMPORT_VERSION)
from xanespy.xradia import XRMFile
from xanespy.txmstore import TXMStore


TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')


class TXMStoreTest(XanespyTestCase):
    hdfname = os.path.join(SSRL_DIR, 'txmstore-test.h5')
    @classmethod
    def setUpClass(cls):
        # Delete temporary HDF5 files
        if os.path.exists(cls.hdfname):
            os.remove(cls.hdfname)
        # Prepare an HDF5 file that these tests can use.
        import_ssrl_frameset(SSRL_DIR, hdf_filename=cls.hdfname)
    
    @classmethod
    def tearDownClass(cls):
        # Delete temporary HDF5 files
        if os.path.exists(cls.hdfname):
            os.remove(cls.hdfname)
    
    def store(self, mode='r'):
        store = TXMStore(hdf_filename=self.hdfname,
                         parent_name='ssrl-test-data',
                         data_name='imported',
                         mode=mode)
        return store
    
    def test_getters(self):
        store = self.store()
        self.assertEqual(store.intensities.shape, (1, 2, 1024, 1024))
        self.assertEqual(store.references.shape, (1, 2, 1024, 1024))
        self.assertEqual(store.absorbances.shape, (1, 2, 1024, 1024))
        self.assertEqual(store.pixel_sizes.shape, (1, 2,))
        self.assertEqual(store.energies.shape, (1, 2,))
        self.assertEqual(store.timestamps.shape, (1, 2, 2))
        self.assertEqual(store.original_positions.shape, (1, 2, 3))
        # Raises exception for non-existent datasets
        with self.assertRaises(exceptions.GroupKeyError):
            store.get_map('madeup_data')
        with self.assertRaises(exceptions.GroupKeyError):
            store.get_frames('madeup_data')
        with self.assertRaises(exceptions.GroupKeyError):
            store.get_frames(None)
        # Check that `get_frames()` returns the frames associated with a map
    
    def test_data_group(self):
        store = self.store()
        self.assertEqual(store.parent_group().name, '/ssrl-test-data')
        self.assertEqual(store.data_group().name, '/ssrl-test-data/imported')
    
    def test_fork_group(self):
        store = self.store('r+')
        raises = self.assertRaises(exceptions.CreateGroupError)
        logs = self.assertLogs(level=logging.CRITICAL)
        with raises, logs:
            store.fork_data_group(dest=store.data_name, src=store.data_name)
        # Set a marker to see if it changes
        store.parent_group().create_group('new_group')
        store.data_name = 'new_group'
        store.data_group().attrs['test_val'] = 'Hello'
        # Now verify that the previous group was overwritten
        store.data_name = 'imported'
        store.fork_data_group(dest='new_group')
        self.assertNotIn('test_val', list(store.data_group().attrs.keys()))
        # Check that the new group is registered as the "latest"
        self.assertEqual(store.latest_data_name, 'new_group')
        # Check that we can easily fork a non-existent group
        store.fork_data_group(dest='brand_new')
        store.close()
    
    def test_data_tree(self):
        """Check that a data tree can be created showing the possible groups
        to choose from.
        
        """
        store = self.store()
        f = h5py.File(self.hdfname)
        # Check that all top-level groups are accounted for
        tree = store.data_tree()
        self.assertEqual(len(f.keys()), len(tree))
        # Check properties of a specific entry (absorbance data)
        abs_dict = tree[0]['children'][0]['children'][0]
        self.assertEqual(abs_dict['level'], 2)
        self.assertEqual(abs_dict['context'], 'frameset')

    def test_data_name(self):
        store = self.store('r+')
        store.data_name = 'imported'
        self.assertEqual(store.data_name, 'imported')
        # Check that data_name can't be set before the group exists
        with self.assertRaises(exceptions.CreateGroupError):
            store.data_name = 'new_group'
        store.close()

    def test_setters(self):
        store = self.store('r+')
        # Check that the "type" attribute is set
        store.absorbances = np.zeros((2, 1024, 1024))
        self.assertEqual(store.absorbances.attrs['context'], 'frameset')
    
    def test_get_frames(self):
        store = self.store()
        # Check that the method returns data
        self.assertEqual(store.get_frames('absorbances').shape, (1, 2, 1024, 1024))
        

class XrayEdgeTest(unittest.TestCase):
    def setUp(self):
        class DummyEdge(edges.KEdge):
            regions = [
                (8250, 8290, 20),
                (8290, 8295, 1),
            ]
            pre_edge = (8250, 8290)
            post_edge = (8290, 8295)
            map_range = (8291, 8293)

        self.edge = DummyEdge()

    def test_energies(self):
        self.assertEqual(
            self.edge.all_energies(),
            [8250, 8270, 8290, 8291, 8292, 8293, 8294, 8295]
        )

    def test_norm_energies(self):
        self.assertEqual(
            self.edge.energies_in_range(),
            [8291, 8292, 8293]
        )


MockStore = mock.MagicMock(TXMStore)

class XanesFramesetTest(TestCase):
    """Set of python tests that work on full framesets and require data
    from multiple frames to make sense."""

    def dummy_frame_data(self, shape=(5, 5, 128, 128)):
        """Create some dummy data with a given shape. It's pretty much just an
        arange with reshaping."""
        length = np.prod(shape)
        data = np.arange(length)
        data = np.reshape(data, shape)
        return data

    def create_frameset(self, store=None, edge=None):
        if edge is None:
            edge = edges.k_edges['Ni_NCA']
        # Create new frameset object
        fs = XanesFrameset(filename="", edge=edge)
        # Mock out the `store` retrieval so we can control it
        if store is None:
            store = MockStore()
            store.get_frames.return_value = self.dummy_frame_data()
        store.__enter__ = mock.Mock(return_value=store)
        fs.store = mock.Mock(return_value=store)
        self.store = store
        return fs

    def test_fit_spectra(self):
        """This test does not evaluate the quality of the fit, only that the
        method selects the correct data and passes it on to the
        correct math routine.
        """
        FRAME_SHAPE = (16, 16)
        # Prepare some fake data
        fake_As = np.zeros((2, 10, *FRAME_SHAPE))
        fake_Es = np.linspace(8250, 8640, num=10)
        fake_Es = np.broadcast_to(fake_Es, (2, 10))
        store = MockStore()
        store.absorbances = fake_As
        type(store).whiteline_fit = mock.PropertyMock()
        store.energies.value = fake_Es
        # Prepare a fake frameset
        fs = self.create_frameset(store=store)
        fs.frame_shape = mock.Mock(return_value=FRAME_SHAPE)
        # Call the method begin tested
        fs.fit_spectra(edge_mask=False)
        # No results are specified, but at least the function was
        # called.
        
    def test_particle_series(self):
        store = MockStore()
        fake_data = np.random.rand(4, 256, 256)
        store.get_map = mock.Mock(return_value=fake_data)
        particle_map = np.random.choice([0, 1, 2, 3], size=(256,256))
        store.particle_labels = particle_map
        fs = self.create_frameset(store=store)
        particles = fs.particle_series()
        self.assertEqual(particles.shape, (3, 4)) # (3 particles, 4 energies)

    def test_subtract_surroundings(self):
        store = MockStore()
        # Prepare some fake data
        od = np.array(
            [[[[1, 1, 1, 1],
               [1, 4, 4, 1],
               [1, 4, 4, 1],
               [1, 1, 1, 1]]]]
        )
        mask = np.array(
            [[True, True,  True,  True],
             [True, False, False, True],
             [True, False, False, True],
             [True, True,  True,  True]]
        )
        expectation = np.array(
            [[[[0, 0, 0, 0],
               [0, 3, 3, 0],
               [0, 3, 3, 0],
               [0, 0, 0, 0]]]]
        )
        store.absorbances = od
        fs = self.create_frameset(store=store)
        fs.edge_mask = mock.Mock(return_value=mask)
        # Do the calculation
        fs.subtract_surroundings()
        # Check that the subtraction happened properly
        np.testing.assert_equal(store.absorbances, expectation)

    def test_fitted_spectrum(self):
        # Prepare some dummy data
        store = MockStore()
        fit_params = np.random.rand(5, 256, 256, 8)
        store.get_frames = mock.Mock(return_value=fit_params)
        fs = self.create_frameset(store=store)
        fs.energies = mock.Mock(return_value=np.linspace(8250, 8640, num=40))
        pixel = (0, 0)
        # Calculate the predicted spectrum for a single pixel
        result = fs.fitted_spectrum(pixel=pixel)
        store.get_frames.assert_called_with('fit_parameters')
        # Validate the predicted spectrum
        expected_Es = np.linspace(8250, 8640, num=500)
        np.testing.assert_equal(result.index, expected_Es)
        expected_As = predict_edge(expected_Es, *fit_params[0,0,0])
        np.testing.assert_equal(result.values, expected_As)
        # Calculate the predicted spectrum for a whole frame
        result = fs.fitted_spectrum(pixel=None)
        # Validate the predicted spectrum
        np.testing.assert_equal(result.index, expected_Es)
        mean_params = np.mean(fit_params[0], axis=(0, 1))
        assert mean_params.shape == (8,)
        expected_As = predict_edge(expected_Es, *mean_params)
        np.testing.assert_equal(result.values, expected_As)
        # Test with a user-provided energy list
        energies = np.linspace(8300, 8500, num=47)
        result = fs.fitted_spectrum(energies=energies)
        # Validate the predicted spectrum
        np.testing.assert_equal(result.index, energies)
        mean_params = np.mean(fit_params[0], axis=(0, 1))
        assert mean_params.shape == (8,)
        expected_As = predict_edge(energies, *mean_params)
        np.testing.assert_equal(result.values, expected_As)

    def test_spectrum(self):
        store = MockStore()
        # Prepare fake energy data
        energies = np.linspace(8300, 8400, num=51)
        store.energies = np.broadcast_to(energies, (10, 51))
        # Prepare fake spectrum (absorbance) data
        spectrum = np.sin((energies-8300)*4*np.pi/100)
        frames = np.broadcast_to(spectrum, (10, 128, 128, 51))
        frames = np.swapaxes(frames, 3, 1)
        store.get_frames = mock.Mock(return_value=frames)
        fs = self.create_frameset(store=store)
        # Check that the return value is correct
        result = fs.spectrum()
        np.testing.assert_equal(result.index, energies)
        np.testing.assert_almost_equal(result.values, spectrum)

    def test_has_representation(self):
        fs = self.create_frameset()
        fs.has_representation('some data')
        self.store.has_dataset.assert_called_once_with('some data')

    def test_fork_group(self):
        """Tests that the XanesFrameset.fork_group properly hands off to
        TXMStore.fork_data_group.
        
        """
        store = MockStore()
        fs = self.create_frameset(store=store)
        # Call the fork_group method
        fs.fork_data_group(dest="new_group", src="old_group")
        store.fork_data_group.assert_called_once_with(
            dest='new_group', src='old_group'
        )

    def test_align_frames_invalid(self):
        """Check that the `align_frames` method throws the right exceptions on
        bad inputs.
        
        """
        fs = self.create_frameset()
        # Bad blur value
        with self.assertRaises(ValueError):
            fs.align_frames(blur="bad-blur", plot_results=False)
        # Bad method
        with self.assertRaises(ValueError):
            fs.align_frames(method="bad-method", plot_results=False)

    def test_label_particle(self):
        store = MockStore()
        fs = self.create_frameset()
        # Prepare dummy frame data
        num_E = 10
        E_step = 50
        frames = np.random.rand(5, num_E, 128, 128)
        store.absorbances.value = frames
        store.get_frames = mock.Mock(return_value=frames)
        # Prepare fake range of energies
        energies = np.arange(8250, 8250 + E_step * num_E, step=E_step)
        energies = np.broadcast_to(energies, (5, 10))
        store.energies = energies
        # Call the `label_particles` method
        fs.label_particles()
    
    def test_store_accessors(self):
        """Tests the almost-trivial methods the follow the following pattern:
        
        - open the txmstore
        - do something to/with the store
        - close the store
        - return the result.
        
        """
        # Prepare the mocked store
        store = MockStore()
        # Create the frameset
        fs = self.create_frameset(store=store)
        # Test `data_tree()` method
        data_tree = ['1', '2']
        store.data_tree = mock.Mock(return_value=data_tree)
        self.assertEqual(fs.data_tree(), data_tree)
        # Test `has_representation()` method
        store.has_dataset = mock.Mock(return_value=True)
        self.assertTrue(fs.has_representation("absorbances"))
        store.has_dataset.assert_called_with('absorbances')
        # Test `starttime()` and `endtime()` methods
        timestamps = np.array([
            # Index-0 timstep is the *fake* timestamps
            [['2015-02-21 10:47:19', '2015-02-25 10:47:26.500000'],
             ['2015-02-21 10:55:48', '2015-02-25 10:55:55.500000']],
            # Index-1 timestep is the *real* timestamps
            [['2015-02-22 10:47:19', '2015-02-22 10:47:26.500000'],
             ['2015-02-22 10:55:48', '2015-02-22 10:55:55.500000']]
        ])
        store.timestamps = timestamps
        starttime = fs.starttime(timeidx=1)
        self.assertEqual(starttime, np.datetime64('2015-02-22 10:47:19'))
        endtime = fs.endtime(timeidx=1)
        self.assertEqual(endtime, np.datetime64('2015-02-22 10:55:55.500000'))
    
    def test_components(self):
        fs = self.create_frameset()
        self.assertEqual(fs.components(), ['modulus'])
    
    def test_mean_frame(self):
        # Prepare frameset and mock store
        frames = self.dummy_frame_data()
        store = MockStore()
        store.get_frames.return_value = frames
        fs = self.create_frameset(store=store)
        # Call the `mean_frames` method
        result = fs.mean_frame(representation="intensities")
        # Check that the result is correct
        self.assertEqual(result.ndim, 2)
        store.get_frames.assert_called_with('intensities')
        expected = np.mean(frames, axis=(0, 1))
        np.testing.assert_equal(result, expected)
    
    def test_map_data(self):
        store = MockStore()
        frameset = self.create_frameset(store=store)
        # Check on getting data by timeidx
        data = self.dummy_frame_data((10, 128, 128))
        store.get_map.return_value = data
        result = frameset.map_data(timeidx=5)
        np.testing.assert_equal(result, data[5])
        # Check on getting and already 2D map
        data = self.dummy_frame_data((128, 128))
        store.get_map.return_value = data
        frameset.clear_caches()
        result = frameset.map_data(timeidx=5)
        np.testing.assert_equal(result, data)

    def test_frames(self):
        # Make mocked data
        store = MockStore
        data = self.dummy_frame_data()
        store.get_frames.return_value = data
        fs = self.create_frameset(store=store)
        # Check that the method returns the right data
        result = fs.frames(timeidx=3, representation='marbles')
        store.get_frames.assert_called_once_with(name='marbles')
        np.testing.assert_equal(result, data[3])

    def test_energies(self):
        # Make mocked data
        store = MockStore
        data = self.dummy_frame_data((10, 61))
        store.energies = data
        fs = self.create_frameset(store=store)
        # Check that the method returns the right data
        result = fs.energies(timeidx=3)
        np.testing.assert_equal(result, data[3])

    def test_all_extents(self):
        pass
        
    def test_extent(self):
        # Create mock data source
        store = MockStore()
        data = self.dummy_frame_data((5, 8, 128, 128))
        store.get_frames.return_value = data
        px_sizes = np.linspace(0.0315783 * 8, 0.0335783 * 8, num=8)
        px_sizes = np.broadcast_to(px_sizes, (5, 8))
        store.pixel_sizes = px_sizes
        fs = self.create_frameset(store=store)        
        # Check that passing multi-frame index gives the median
        actual = fs.extent('absorbances')
        expected = (-16.6800896, 16.6800896, -16.6800896, 16.6800896)
        np.testing.assert_almost_equal(actual, expected)
        # Check that passing an index gives that frame
        actual = fs.extent('absorbances', idx=(0, 0))
        expected = (-16.1680896, 16.1680896, -16.1680896, 16.1680896)
        np.testing.assert_almost_equal(actual, expected)

    def test_extent_array(self):
        # Create mock data source
        store = MockStore()
        data = self.dummy_frame_data((5, 8, 128, 128))
        store.get_frames.return_value = data
        px_sizes = np.linspace(0.0315783 * 8, 0.0335783 * 8, num=8)
        px_sizes = np.broadcast_to(px_sizes, (5, 2, 8))
        px_sizes = np.moveaxis(px_sizes, 1, 2)
        store.pixel_sizes = px_sizes
        fs = self.create_frameset(store=store)        
        # Check that passing multi-frame index gives the median
        actual = fs.extent('absorbances')
        expected = (-16.6800896, 16.6800896, -16.6800896, 16.6800896)
        np.testing.assert_almost_equal(actual, expected)
        # Check that passing an index gives that frame
        actual = fs.extent('absorbances', idx=(0, 0))
        expected = (-16.1680896, 16.1680896, -16.1680896, 16.1680896)
        np.testing.assert_almost_equal(actual, expected)
    
    def test_active_path(self):
        store = MockStore()
        class DummyDataGroup():
            name = '/ssrl-test-data/imported'
        store.data_group.return_value = DummyDataGroup()
        fs = self.create_frameset(store=store)
        # Test the parent path
        self.assertEqual(
            fs.hdf_path(),
            '/ssrl-test-data/imported'
        )
        store.data_group.assert_called_with()
        store.frames.assert_not_called()
        # Test a specific representation's path
        class DummyDataGroup():
            name = '/ssrl-test-data/imported/absorbances'
        store.get_frames.return_value = DummyDataGroup()
        self.assertEqual(
            fs.hdf_path('absorbances'),
            '/ssrl-test-data/imported/absorbances'
        )
        store.get_frames.assert_called_with(representation="absorbances")
    
    def test_switch_groups(self):
        """Test that switching between HDF5 groups works robustly."""
        # Without the `src` argument
        fs = self.create_frameset()
        fs.fork_data_group('new_group')
        self.assertEqual(fs.data_name, 'new_group')

    def test_repr(self):
        fs = XanesFrameset(filename=None, edge=edges.k_edges['Ni_NCA'],
                           groupname="ssrl-test-data")
        expected = "<XanesFrameset: 'ssrl-test-data'>"
        self.assertEqual(fs.__repr__(), expected)


class OldXanesFramesetTest(XanespyTestCase):
    """Set of python tests that work on full framesets and require data
    from multiple frames to make sense. These "old" tests work
    directly with imported HDF files. The newer tests above mock out
    the TXMStore class so we can properly isolate the XanesFrameset
    functionality.

    """
    originhdf = os.path.join(SSRL_DIR, 'txmstore-test.h5')
    temphdf = os.path.join(SSRL_DIR, 'txmstore-test-tmp.h5')

    @classmethod
    def setUpClass(cls):
        # Prepare an HDF5 file that these tests can use.
        if os.path.exists(cls.originhdf):
            os.remove(cls.originhdf)
        import_ssrl_frameset(SSRL_DIR, hdf_filename=cls.originhdf)

    def setUp(self):
        # Copy the HDF5 file so we can safely make changes
        shutil.copy(self.originhdf, self.temphdf)
        self.frameset = XanesFrameset(filename=self.temphdf,
                                      groupname='ssrl-test-data',
                                      edge=edges.k_edges['Ni_NCA'])

    def tearDown(self):
        if os.path.exists(self.temphdf):
            os.remove(self.temphdf)

    @classmethod
    def tearDownClass(cls):
        # Delete temporary HDF5 files
        if os.path.exists(cls.originhdf):
            os.remove(cls.originhdf)
    
    def test_align_frames(self):
        # Perform an excessive translation to ensure data are correctable
        with self.frameset.store(mode='r+') as store:
            Ts = np.identity(3)
            Ts = np.copy(np.broadcast_to(Ts, (*store.absorbances.shape[0:2], 3, 3)))
            Ts[0, 1, 0, 2] = 100
            Ts[0, 1, 1, 2] = 100
            transform_images(store.absorbances,
                             transformations=Ts,
                             out=store.absorbances)
            old_imgs = store.absorbances.value
        # Check that reference_frame arguments of the wrong shape are rejected
        with self.assertRaisesRegex(Exception, "does not match shape"):
            self.frameset.align_frames(commit=False,
                                       reference_frame=0, plot_results=False)
        # Perform an alignment but don't commit to disk
        self.frameset.align_frames(commit=False, reference_frame=(0, 0),
                                   plot_results=False)
        # Check that the translations weren't applied yet
        with self.frameset.store() as store:
            hasnotchanged = np.all(np.equal(old_imgs, store.absorbances.value))
        self.assertTrue(hasnotchanged)
        # Apply the translations
        self.frameset.apply_transformations(crop=True, commit=True)
        with self.frameset.store() as store:
            new_shape = store.absorbances.shape
        # Test for inequality by checking shapes
        self.assertEqual(old_imgs.shape[:-2], new_shape[:-2])
        self.assertNotEqual(old_imgs.shape[-2:], new_shape[-2:])
   
    def test_deferred_transformations(self):
        """Test that the system properly stores data transformations for later
        processing."""
        with self.frameset.store() as store:
            data_shape = store.absorbances.shape
        # Check that staged transforms are initially None
        self.assertTrue(self.frameset._transformations is None)
        # Stage some transformations
        self.frameset.stage_transformations(
            translations=np.array([[[0, 0],[1, 1]]]),
            scales=np.array([[[1, 1], [0.5, 0.5]]]),
            rotations=np.array([[[0], [3]]])
        )
        # Check that the transformations have been saved
        self.assertFalse(self.frameset._transformations is None)
        self.assertEqual(
            self.frameset._transformations.shape,
            (1, 2, 3, 3)
        )
        t1 = transform.AffineTransform(scale=(0.5, 0.5), rotation=3, translation=(1, 1))
        np.testing.assert_allclose(
            self.frameset._transformations[0, 1],
            t1.params
        )
        # # Check that transformations accumulated
        # Stage some transformations
        self.frameset.stage_transformations(
            translations=np.array([[[0, 0],[1, 1]]]),
            scales=np.array([[[1, 1], [0.5, 0.5]]]),
            rotations=np.array([[[0], [1.5]]])
        )
        t2 = transform.AffineTransform(scale=(0.5, 0.5), rotation=1.5, translation=(1, 1))
        cumulative = t1.params @ t2.params
        np.testing.assert_allclose(
            self.frameset._transformations[0, 1],
            cumulative
        )
        # Check that transformations are reset after being applied
        self.frameset.apply_transformations(commit=True, crop=True)
        self.assertEqual(self.frameset._transformations, None)
        # # Check that cropping was successfully applied
        with self.frameset.store() as store:
            new_shape = store.absorbances.shape
        self.assertEqual(new_shape, (1, 2, 1023, 1023))

    def test_calculate_clusters(self):
        """Check the the data are separated into signals and discretized by
        k-means clustering."""
        N_COMPONENTS = 3
        with self.assertLogs(level=logging.WARNING):
            # Post/pre-edge identification is known to fail
            self.frameset.calculate_signals(n_components=N_COMPONENTS,
                                            method="nmf")
        # Check that nmf signals and weights are saved
        with self.frameset.store() as store:
            n_energies = store.absorbances.shape[1]  # Expecting: 2
            good_shape = (N_COMPONENTS, n_energies)
            self.assertEqual(store.signals.shape, good_shape)
            self.assertEqual(
                store.signal_method,
                "Non-Negative Matrix Factorization")
            # Check for shape of weights
            good_shape = (1, *self.frameset.frame_shape(), N_COMPONENTS)
            self.assertEqual(store.signal_weights.shape, good_shape)
        # Check that a composite RGB map is saved
        with self.frameset.store() as store:
            good_shape = (1, *self.frameset.frame_shape(), 3)
            self.assertEqual(store.signal_map.shape, good_shape)
        # Check that k-means cluster map is saved
        with self.frameset.store() as store:
            good_shape = (1, *self.frameset.frame_shape())
            self.assertEqual(store.cluster_map.shape, good_shape)


# Launch the tests if this is run as a script
if __name__ == '__main__':
    unittest.main()
