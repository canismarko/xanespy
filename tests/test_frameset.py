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

import glob
import datetime as dt
import unittest
from unittest import TestCase, mock
import logging
import math
import os
import shutil
import warnings
from collections import namedtuple
from functools import partial
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage

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
                                extract_signals_nmf,
                                transformation_matrices,
                                apply_internal_reference,
                                register_template)
# from xanespy.edges import KEdge, k_edges, l_edges
from xanespy.importers import (import_ssrl_xanes_dir,
                               import_nanosurveyor_frameset,
                               _average_frames,
                               magnification_correction,
                               decode_aps_params, decode_ssrl_params,
                               read_metadata, CURRENT_VERSION as IMPORT_VERSION)
from xanespy.xradia import XRMFile
from xanespy.txmstore import TXMStore, TXMDataset


TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')

try:
    import mpi4py
except ImportError:
    mpi_support = False
else:
    mpi_support = True


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


class TXMDatasetTest(TestCase):
    class StoreStub(TXMStore):
        replace_dataset = mock.Mock()
        open_file = mock.Mock()
        latest_data_name = mock.Mock()
        dtype_dataset = TXMDataset(name="dtype_dataset", dtype=np.int32)
    
    def store(self):
        return self.StoreStub(hdf_filename='/dev/null', parent_name="None")
    
    def test_setter(self):
        store = self.store()
        store.dtype_dataset = [3, 5]
        store.replace_dataset.assert_called_with(name='dtype_dataset',
                                                 data=[3, 5],
                                                 context=None,
                                                 dtype=np.int32)


MockStore = mock.MagicMock(TXMStore)
# For testing things that need a pickleable function
x = np.linspace(0, 1, num=6)
def _line(a, b, x):
    return a * x + b
line = partial(_line, x=x)

class XanesFramesetTest(TestCase):
    """Set of python tests that work on full framesets and require data
    from multiple frames to make sense.
    
    """
    beamer_basename = os.path.join(TEST_DIR, 'beamer_output_test')
    hdf_filename = os.path.join(TEST_DIR, 'txmstore-test.h5')
    
    def tearDown(self):
        # Remove temporary beamer output file
        if os.path.exists(self.beamer_basename + "_maps.tex"):
            os.remove(self.beamer_basename + "_maps.tex")
            os.remove(self.beamer_basename + "_spectra.tex")
        # Remove temporary pgffile output
        pgffiles = glob.glob('%s*.pgf' % self.beamer_basename)
        [os.remove(f) for f in pgffiles]
        pngfiles = glob.glob('%s*.png' % self.beamer_basename)
        [os.remove(f) for f in pngfiles]
    
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
        fs = XanesFrameset(filename=self.hdf_filename, edge=edge)
        # Mock out the `store` retrieval so we can control it
        if store is None:
            store = MockStore()
            store.get_frames.return_value = self.dummy_frame_data()
        store.__enter__ = mock.Mock(return_value=store)
        fs.store = mock.Mock(return_value=store)
        self.store = store
        return fs
    
    # @unittest.skipIf(not mpi_support, 'No support for MPI.')
    # def test_fit_spectra(self):
    #     """This test does not evaluate the quality of the fit, only that the
    #     method selects the correct data and passes it on to the
    #     correct math routine.
        
    #     """
    #     FRAME_SHAPE = (16, 16)
    #     # Prepare some fake data
    #     fake_As = np.zeros((2, 10, *FRAME_SHAPE))
    #     fake_Es = np.linspace(8250, 8640, num=10)
    #     fake_Es = np.broadcast_to(fake_Es, (2, 10))
    #     store = MockStore()
    #     store.optical_depths = fake_As
    #     type(store).whiteline_fit = mock.PropertyMock()
    #     store.energies.value = fake_Es
    #     # Prepare a fake frameset
    #     fs = self.create_frameset(store=store)
    #     fs.frame_shape = mock.Mock(return_value=FRAME_SHAPE)
    #     # Call the method begin tested
    #     fs.fit_spectra(edge_mask=False)
    #     # No results are specified, but at least the function was
    #     # called.
    
    def test_segment_materials(self):
        # Prepare dummy data
        store = MockStore()
        store.energies = np.broadcast_to(np.linspace(850, 860, num=3), shape=(8, 3))
        store.optical_depths = np.random.rand(8, 3, 16, 16)
        store.get_dataset = mock.MagicMock(return_value=np.random.rand(8, 3, 16, 16))
        store.get_frames = store.get_dataset
        store.pixel_sizes = np.ones((8, 3, 2))
        fs_names = ['fs1', 'fs2']
        store.frameset_names = mock.MagicMock(return_value=fs_names)
        fs = self.create_frameset(store=store)
        fs.segment_materials(thresholds=(0.3, 0.67))
        # Check that it was turned into three sgements data
        self.assertEqual(np.min(store.segments), 0)
        self.assertEqual(np.max(store.segments), 2)
    
    def test_plot_map(self):
        store = MockStore()
        store.get_dataset = mock.MagicMock(return_value=np.random.rand(1, 16, 16))
        store.pixel_sizes = np.array([[5, 5]])
        fs = self.create_frameset(store=store)
        artists = fs.plot_map()
        self.assertIsInstance(artists[0], mpl.image.AxesImage)
    
    def test_plot_histogram(self):
        store = MockStore()
        n_energies = 10
        data = np.random.rand(10, 16, 16) * 10 + 850
        store.get_dataset = mock.MagicMock(return_value=data)
        store.pixel_sizes = np.array([[5, 5]])
        store.energies = np.broadcast_to(np.linspace(850, 860, num=n_energies), shape=(1, n_energies))
        fs = self.create_frameset(store=store)
        artists = fs.plot_histogram()
        self.assertIsInstance(artists[0], mpl.patches.Rectangle)
    
    def test_plot_beamer_spectra(self):
        # Prepare dummy data
        store = MockStore()
        store.energies = np.broadcast_to(np.linspace(850, 860, num=3), shape=(8, 3))
        store.optical_depths = np.random.rand(8, 3, 16, 16)
        store.get_dataset = mock.MagicMock(return_value=np.random.rand(8, 3, 16, 16))
        store.get_frames = store.get_dataset
        store.pixel_sizes = np.ones((8, 3, 2))
        fs_names = ['fs1', 'fs2']
        store.frameset_names = mock.MagicMock(return_value=fs_names)
        fs = self.create_frameset(store=store)
        # Do the actual plotting
        with warnings.catch_warnings() as w:
            warnings.filterwarnings('ignore', message=".*Is LaTeX installed.*",
                                    category=RuntimeWarning)
            context = fs.plot_beamer_spectra(basename=self.beamer_basename)
        # Check that the PGF files were created
        pgffiles = glob.glob('%s*.pgf' % self.beamer_basename)
        self.assertEqual(len(pgffiles), 16)
        # Check that the context was returned properly
        self.assertIsInstance(context, dict)
        self.assertIn('frames', context.keys())
        self.assertEqual(context['frames'][0]['title'], 'ssrl-test-data - fs1')
        self.assertEqual(len(context['frames'][0]['plots']), 6)
        # Check that using a bad groupby argument throws exception
        with self.assertRaises(ValueError):
            context = fs.plot_beamer_spectra(basename=self.beamer_basename,
                                          groupby='map_name')
        # Do the plotting by timestep
        with warnings.catch_warnings() as w:
            warnings.filterwarnings('ignore', message=".*Is LaTeX installed.*",
                                    category=RuntimeWarning)
            context = fs.plot_beamer_spectra(basename=self.beamer_basename, groupby='timestep')
    
    def test_plot_beamer_maps(self):
        store = MockStore()
        store.optical_depths = np.random.rand(8, 3, 16, 16)
        store.get_dataset = mock.MagicMock(return_value=np.random.rand(8, 16, 16))
        store.pixel_sizes = np.ones((8, 3, 2))
        map_names = ['map1', 'map2']
        store.map_names = mock.MagicMock(return_value=map_names)
        fs = self.create_frameset(store=store)
        with warnings.catch_warnings() as w:
            warnings.filterwarnings('ignore', message=".*Is LaTeX installed.*",
                                    category=RuntimeWarning)
            context = fs.plot_beamer_maps(basename=self.beamer_basename, tight_layout=False)
        # Check that the PGF files were created
        pgffiles = glob.glob('%s*.pgf' % self.beamer_basename)
        self.assertEqual(len(pgffiles), 16)
        # Check that the context was returned properly
        self.assertIsInstance(context, dict)
        self.assertIn('frames', context.keys())
        self.assertEqual(context['frames'][0]['title'], 'ssrl-test-data - map1')
        self.assertEqual(len(context['frames'][0]['plots']), 6)
        # Check that using a bad groupby argument throws exception
        with self.assertRaises(ValueError):
            context = fs.plot_beamer_maps(basename=self.beamer_basename,
                                          groupby='gibberish')
        # Do the plotting by timestep
        with warnings.catch_warnings() as w:
            warnings.filterwarnings('ignore', message=".*Is LaTeX installed.*",
                                    category=RuntimeWarning)
            context = fs.plot_beamer_maps(basename=self.beamer_basename,
                                          groupby='timestep', tight_layout=False)
   
    def test_export_beamer(self):
        fs = self.create_frameset()
        maps_fname = self.beamer_basename + '_maps.tex'
        # Check that the file was created
        self.assertFalse(os.path.exists(maps_fname))
        fs.export_beamer_file(basename=self.beamer_basename)
        self.assertTrue(os.path.exists(maps_fname))
    
    def test_lc_fitting(self):
        # Prepare stubbed data
        store = MockStore()
        od_data = np.random.rand(1, 6, 16, 16)
        store.get_frames = mock.MagicMock(return_value=od_data)
        store.get_dataset = mock.MagicMock(return_value=od_data)
        Es = [np.linspace(840, 862, num=6)]
        store.energies = Es
        fs = self.create_frameset(store=store)
        spectrum = fs.spectrum()
        # DO the actual fitting
        weights, residuals = fs.fit_linear_combinations(
            sources=[spectrum], quiet=True, edge_filter=False)
        self.assertEqual(weights.shape, (1, 2, 16, 16))
        self.assertEqual(residuals.shape, (1, 16, 16))
        # Check that the data were saved
        # np.testing.assert_equal(store.linear_combination_parameters, weights)
        expected_names = "('c0', 'c1', 'offset')"
        args, kwargs = store.replace_dataset.call_args_list[0]
        store.replace_dataset.assert_any_call(
            'linear_combination_parameters', weights,
            context='frameset', attrs={'parameter_names': expected_names}
        )
        store.replace_dataset.assert_any_call(
            'linear_combination_residuals', residuals, context='map',
            attrs={'frame_source': 'optical_depths'})
    
    def test_median_filter(self):
        store = MockStore()
        od_data = np.random.rand(1, 6, 16, 16)
        store.get_dataset = mock.MagicMock(return_value=od_data)
        fs = self.create_frameset(store=store)
        store.replace_dataset.reset_mock()
        fs.apply_median_filter(size=(1, 3, 3, 3))
        new_data = ndimage.median_filter(od_data, size=(1, 3, 3, 3))
        self.assertEqual(store.replace_dataset.call_count, 1)
        saved_data = store.replace_dataset.call_args[1]['data']
        np.testing.assert_equal(new_data, saved_data)
    
    def test_fit_spectra(self):
        store = MockStore()
        od_data = np.random.rand(1, 6, 16, 16)
        store.get_dataset = mock.MagicMock(return_value=od_data)
        Es = np.array([np.linspace(840, 862, num=6, dtype=np.float32)])
        store.energies = Es
        fs = self.create_frameset(store=store)
        params, residuals = fs.fit_spectra(line, p0=np.zeros((1, 2, 16, 16)),
                                           nonnegative=True, edge_filter=False,
                                           pnames=('slope', 'intercept'), quiet=True)
        self.assertFalse(np.any(params<0))
        self.assertEqual(params.shape, (1, 2, 16, 16))
        self.assertEqual(residuals.shape, (1, 16, 16))
        # Check that the data were saved
        store.replace_dataset.assert_any_call(
            'fit_parameters', params, context='frameset',
            attrs={'parameter_names': str(('slope', 'intercept'))})
        store.replace_dataset.assert_any_call('fit_residuals', residuals, context='map',
                                              attrs={'frame_source': 'optical_depths'})
    
    def test_particle_series(self):
        store = MockStore()
        fake_data = np.random.rand(4, 256, 256)
        store.get_dataset = mock.Mock(return_value=fake_data)
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
        store.optical_depths = od
        fs = self.create_frameset(store=store)
        fs.edge_mask = mock.Mock(return_value=mask)
        # Do the calculation
        fs.subtract_surroundings()
        # Check that the subtraction happened properly
        np.testing.assert_equal(store.optical_depths, expectation)
    
    def test_edge_mask(self):
        store = MockStore()
        store.has_dataset = mock.MagicMock(return_value=False)
        store.intensities = np.random.rand(128, 128)
        fs = self.create_frameset(store=store)
        # Check that the new edge mask has same shape as intensities
        np.testing.assert_equal(fs.edge_mask(), np.zeros(shape=(128, 128)))
    
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
        # Check that multiple spectra can be acquired simultaneously
        result = fs.spectrum(index=slice(0, 2))
        result = np.array([ser.values for ser in result])
        spectras = np.array([fs.spectrum(index=0), fs.spectrum(index=0)])
        np.testing.assert_equal(result, spectras)
        # Check that the derivative is calculated correctly
        derivative = 4*np.pi/100 * np.cos((energies-8300)*4*np.pi/100)
        result = fs.spectrum(derivative=1)
        np.testing.assert_almost_equal(result.values, derivative, decimal=3)
    
    def test_nonenergy_spectrum(self):
        """If the frames aren't in energy order"""
        store = MockStore()
        # Prepare fake energy data
        energies = np.linspace(8300, 8400, num=51)
        store.energies = np.broadcast_to(energies, (10, 51))
        # Prepare fake spectrum (absorbance) data
        x_params = np.linspace(0, 2*np.pi, num=7)
        spectrum = np.sin(x_params)
        frames = np.broadcast_to(spectrum, (10, 128, 128, 7)) # 7, not 51
        frames = np.swapaxes(frames, 3, 1)
        store.get_frames = mock.Mock(return_value=frames)
        fs = self.create_frameset(store=store)
        # Check that the return value is correct
        result = fs.spectrum()
        np.testing.assert_equal(result.index, np.arange(0, 7))
        np.testing.assert_almost_equal(result.values, spectrum)
    
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
    
    def test_crop_frames(self):
        store = MockStore()
        ODs = self.dummy_frame_data()
        store.optical_depths = ODs
        store.get_dataset.return_value = ODs
        store.frameset_names.return_value = ('optical_depths',)
        store.map_names.return_value = ()
        # Crop down to a smaller frame-size
        fs = self.create_frameset(store=store)
        slices = (slice(31, 95), slice(31, 95))
        store.replace_dataset.reset_mock()
        fs.crop_frames(slices)
        # Check that the data-set was updated
        self.assertEqual(store.replace_dataset.call_count, 1)
        call_args = store.replace_dataset.call_args[0]
        self.assertEqual(call_args[0], 'optical_depths')
        np.testing.assert_array_equal(call_args[1], ODs[:,:,31:95,31:95])
        # Check that the wrong number of dimensions raises an exception
        bad_slices = (slice(31, 95), slice(31, 95))
    
    def test_align_frames(self):
        # Prepare mismatched data to test
        store = MockStore()
        ODs = np.zeros(shape=(1, 2, 64, 64))
        # Make two mismatched squares
        ODs[0,0,30:34,30:34] = 1
        ODs[0,1,32:36,32:36] = 1
        store.optical_depths = ODs
        store.get_dataset.return_value = ODs
        store.get_frames.return_value = ODs
        fs = self.create_frameset(store=store)
        # Check that reference_frame arguments of the wrong shape are rejected
        with self.assertRaisesRegex(Exception, "does not match shape"):
            fs.align_frames(commit=False, reference_frame=0,
                            plot_results=True)
        # Perform an alignment but don't commit to disk
        fs.align_frames(commit=False, reference_frame=(0, 0),
                        plot_results=False, quiet=True)
        # Check that the translations weren't applied yet
        hasnotchanged = np.all(np.equal(ODs, store.optical_depths))
        self.assertTrue(hasnotchanged)
        # Apply the translations
        fs.apply_transformations(crop=True, commit=True, quiet=True)
        # Check that the right data were written back to disk
        self.assertEqual(store.replace_dataset.call_count, 3)
        ds_names = tuple(c[0][0] for c in store.replace_dataset.call_args_list)
        self.assertEqual(ds_names, ('intensities', 'references', 'optical_depths'))
        new_ODs = store.replace_dataset.call_args[0][1]
        new_shape = store.optical_depths.shape
        # Test for inequality by checking shapes
        self.assertNotEqual(ODs.shape[-2:], new_ODs.shape[-2:])
        self.assertEqual(new_ODs.shape, (1, 2, 62, 62))
        # Test with a median filter
        store.replace_dataset.reset_mock()
        fs.align_frames(commit=True, plot_results=False, quiet=True,
                        median_filter_size=(5, 5))
        # Check that we can align to the max frame
        fs.align_frames(commit=False, reference_frame='max', quiet=True)
    
    def test_align_frames_invalid(self):
        """Check that the `align_frames` method throws the right exceptions on
        bad inputs.
        
        """
        fs = self.create_frameset()
        # Bad method
        with self.assertRaises(ValueError):
            fs.align_frames(method="bad-method", plot_results=False)
    
    def test_label_particle(self):
        store = MockStore()
        store.optical_depths = mock.MagicMock()
        fs = self.create_frameset()
        # Prepare dummy frame data
        num_E = 10
        E_step = 50
        frames = np.random.rand(5, num_E, 128, 128)
        store.optical_depths.value = frames
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
        self.assertTrue(fs.has_representation("optical_depths"))
        store.has_dataset.assert_called_with('optical_depths')
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
        # Without timeindex
        starttime = fs.starttime(timeidx=None)
        self.assertEqual(starttime, np.datetime64('2015-02-21 10:47:19'))
        # Test the end times
        endtime = fs.endtime(timeidx=1)
        self.assertEqual(endtime, np.datetime64('2015-02-22 10:55:55.500000'))
    
    def test_components(self):
        fs = self.create_frameset()
        self.assertEqual(fs.components(), ['modulus'])
    
    def test_mean_frame(self):
        # Prepare frameset and mock store
        frames = self.dummy_frame_data()
        store = MockStore()
        store.get_dataset.return_value = frames
        fs = self.create_frameset(store=store)
        # Call the `mean_frames` method
        result = fs.mean_frame(representation="intensities")
        # Check that the result is correct
        self.assertEqual(result.ndim, 2)
        store.get_dataset.assert_called_with('intensities')
        expected = np.mean(frames, axis=(0, 1))
        np.testing.assert_equal(result, expected)
    
    def test_map_data(self):
        store = MockStore()
        frameset = self.create_frameset(store=store)
        # Check on getting data by timeidx
        data = self.dummy_frame_data((10, 128, 128))
        store.get_dataset.return_value = data
        result = frameset.map_data(timeidx=5)
        np.testing.assert_equal(result, data[5])
        # Check on getting and already 2D map
        data = self.dummy_frame_data((128, 128))
        store.get_dataset.return_value = data
        frameset.clear_caches()
        result = frameset.map_data(timeidx=5)
        np.testing.assert_equal(result, data)
        # Check that a non-map returns None
        class Data():
            attrs = {'context': 'frameset'}
            ndim = 4
        data = Data()
        store.get_dataset.return_value = data
        result = frameset.map_data()
        self.assertIs(result, None)
    
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
    
    def test_extent(self):
        # Create mock data source
        store = MockStore()
        data = self.dummy_frame_data((5, 8, 128, 128))
        store.get_dataset.return_value = data
        px_sizes = np.linspace(0.0315783 * 8, 0.0335783 * 8, num=8)
        px_sizes = np.broadcast_to(px_sizes, (5, 8))
        store.pixel_sizes = px_sizes
        fs = self.create_frameset(store=store)        
        # Check that passing multi-frame index gives the median
        actual = fs.extent('optical_depths')
        expected = (-16.6800896, 16.6800896, -16.6800896, 16.6800896)
        np.testing.assert_almost_equal(actual, expected)
        # Check that passing an index gives that frame
        actual = fs.extent('optical_depths', idx=(0, 0))
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
        actual = fs.extent('optical_depths')
        expected = (-16.6800896, 16.6800896, -16.6800896, 16.6800896)
        np.testing.assert_almost_equal(actual, expected)
        # Check that passing an index gives that frame
        actual = fs.extent('optical_depths', idx=(0, 0))
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
            name = '/ssrl-test-data/imported/optical_depths'
        store.get_dataset.return_value = DummyDataGroup()
        self.assertEqual(
            fs.hdf_path('optical_depths'),
            '/ssrl-test-data/imported/optical_depths'
        )
        store.get_dataset.assert_called_with(representation="optical_depths")
    
    def test_switch_groups(self):
        """Test that switching between HDF5 groups works robustly."""
        # Without the `src` argument
        fs = self.create_frameset()
        fs.fork_data_group('new_group')
        self.assertEqual(fs.data_name, 'new_group')
    
    def test_repr(self):
        fs = XanesFrameset(filename=self.hdf_filename, edge=edges.k_edges['Ni_NCA'],
                           groupname="ssrl-test-data")
        expected = "<XanesFrameset: 'ssrl-test-data'>"
        self.assertEqual(fs.__repr__(), expected)
    
    def test_str(self):
        fs = XanesFrameset(filename=self.hdf_filename, edge=edges.k_edges['Ni_NCA'],
                           groupname="ssrl-test-data")
        self.assertEqual(str(fs), 'ssrl-test-data')


class OldXanesFramesetTest(XanespyTestCase):
    """Set of python tests that work on full framesets and require data
    from multiple frames to make sense. These "old" tests work
    directly with imported HDF files. The newer tests above mock out
    the TXMStore class so we can properly isolate the XanesFrameset
    functionality.
    
    """
    originhdf = os.path.join(TEST_DIR, 'txmstore-test.h5')
    temphdf = os.path.join(TEST_DIR, 'txmstore-test-tmp.h5')
    
    def setUp(self):
        # Copy the HDF5 file so we can safely make changes
        shutil.copy(self.originhdf, self.temphdf)
        self.frameset = XanesFrameset(filename=self.temphdf,
                                      groupname='ssrl-test-data',
                                      edge=edges.k_edges['Ni_NCA'])
    
    def tearDown(self):
        if os.path.exists(self.temphdf):
            os.remove(self.temphdf)
    
    def test_deferred_transformations(self):
        """Test that the system properly stores data transformations for later
        processing."""
        with self.frameset.store() as store:
            data_shape = store.optical_depths.shape
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
        # Check that the translations can be applied without commiting
        out = self.frameset.apply_transformations(commit=False, crop=True, quiet=True)
        # Check that transformations accumulated by staging some more
        # transformations
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
        self.frameset.apply_transformations(commit=True, crop=True, quiet=True)
        self.assertEqual(self.frameset._transformations, None)
        # Check that cropping was successfully applied
        with self.frameset.store() as store:
            new_shape = store.optical_depths.shape
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
            n_energies = store.optical_depths.shape[1]  # Expecting: 2
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
            self.assertEqual(store.cluster_fit.shape, good_shape)


# Launch the tests if this is run as a script
if __name__ == '__main__':
    unittest.main()
