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

import logging
import datetime as dt
import unittest
from unittest import TestCase, mock
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import warnings
import contextlib

import pytz
import numpy as np
import pandas as pd
import h5py
from skimage import data
import matplotlib.pyplot as plt


from xanespy import exceptions, utilities
from xanespy.xradia import XRMFile, TXRMFile
from xanespy.nanosurveyor import CXIFile, HDRFile
from xanespy.sxstm import SxstmDataFile
from xanespy.importers import (magnification_correction,
                               decode_aps_params, decode_ssrl_params,
                               import_ssrl_xanes_dir, CURRENT_VERSION,
                               import_nanosurveyor_frameset,
                               import_cosmic_frameset,
                               import_aps4idc_sxstm_files,
                               import_aps8bm_xanes_dir,
                               import_aps8bm_xanes_file,
                               import_aps32idc_xanes_files,
                               import_aps32idc_xanes_file,
                               read_metadata, minimum_shape,
                               rebin_image, )
from xanespy.txmstore import TXMStore


# logging.basicConfig(level=logging.DEBUG)


TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
APS32_DIR = os.path.join(TEST_DIR, 'txm-data-32-idc')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')
COSMIC_DIR = os.path.join(TEST_DIR, 'ptycho-data-cosmic')
SXSTM_DIR = os.path.join(TEST_DIR, "sxstm-data-4idc/")


class APS32IDCImportTest(TestCase):
    """Check that the program can import a collection of SSRL frames from
    a directory."""
    src_data = os.path.join(APS32_DIR, 'nca_32idc_xanes.h5')
    
    def setUp(self):
        self.hdf = os.path.join(APS32_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def test_imported_hdf(self):
        # Run the import function
        import_aps32idc_xanes_file(self.src_data,
                                   hdf_filename=self.hdf, hdf_groupname='experiment1',
                                   downsample=1)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            self.assertNotIn('experiment2', list(f.keys()))
            parent_group = f['experiment1']
            data_group = f['experiment1/imported']
            # Check metadata about beamline
            self.assertEqual(parent_group.attrs['technique'], 'Full-field TXM')
            self.assertEqual(parent_group.attrs['xanespy_version'], CURRENT_VERSION)
            self.assertEqual(parent_group.attrs['beamline'], "APS 32-ID-C")
            self.assertEqual(parent_group.attrs['original_directory'],
                             os.path.dirname(self.src_data))
            self.assertEqual(parent_group.attrs['latest_data_name'], 'imported')
            # Check h5 data structure
            keys = list(data_group.keys())
            self.assertIn('intensities', keys)
            self.assertTrue(np.any(data_group['intensities']))
            self.assertEqual(data_group['intensities'].shape, (1, 3, 256, 256))
            self.assertEqual(data_group['intensities'].attrs['context'], 'frameset')
            self.assertIn('flat_fields', keys)
            self.assertTrue(np.any(data_group['flat_fields']))
            self.assertEqual(data_group['flat_fields'].attrs['context'], 'frameset')
            self.assertIn('dark_fields', keys)
            self.assertEqual(data_group['dark_fields'].shape, (1, 2, 256, 256))
            self.assertTrue(np.any(data_group['dark_fields']))
            self.assertEqual(data_group['dark_fields'].attrs['context'], 'frameset')
            self.assertIn('optical_depths', keys)
            self.assertEqual(data_group['optical_depths'].shape, (1, 3, 256, 256))
            self.assertTrue(np.any(data_group['optical_depths']))
            self.assertEqual(data_group['optical_depths'].attrs['context'], 'frameset')
            self.assertEqual(data_group['pixel_sizes'].attrs['unit'], 'µm')
            self.assertEqual(data_group['pixel_sizes'].shape, (1, 3))
            # Original pixel size is 29.99nm but we have downsampling factor 1
            self.assertTrue(np.all(data_group['pixel_sizes'].value == 0.02999 * 2))
            self.assertEqual(data_group['energies'].shape, (1, 3))
            expected_Es = np.array([[8340, 8350, 8360]])
            np.testing.assert_array_almost_equal(data_group['energies'].value,
                                                 expected_Es, decimal=3)
            self.assertIn('timestamps', keys)
            expected_timestamp = np.empty(shape=(1, 3, 2), dtype="S32")
            expected_timestamp[...,0] = b'2016-10-07 18:24:42'
            expected_timestamp[...,1] = b'2016-10-07 18:37:42'
            np.testing.assert_equal(data_group['timestamps'].value,
                                    expected_timestamp)
            self.assertIn('timestep_names', keys)
            self.assertEqual(data_group['timestep_names'][0], bytes("soc000", 'ascii'))
            self.assertIn('filenames', keys)
            self.assertEqual(data_group['filenames'].shape, (1, 3))
            self.assertEqual(data_group['filenames'][0, 0], self.src_data.encode('ascii'))
            # self.assertIn('original_positions', keys)
    
    def test_exclude_frames(self):
        # Run the import function
        import_aps32idc_xanes_file(self.src_data,
                                   hdf_filename=self.hdf, hdf_groupname='experiment1',
                                   downsample=1, exclude=(1,))
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            self.assertNotIn('experiment2', list(f.keys()))
            parent_group = f['experiment1']
            data_group = f['experiment1/imported']
            self.assertEqual(data_group['intensities'].shape, (1, 2, 256, 256))
    
    def test_limited_dark_flat(self):
        # Only import some of the flat and dark field images
        import_aps32idc_xanes_file(self.src_data,
                                   hdf_filename=self.hdf, hdf_groupname='experiment1',
                                   downsample=0, dark_idx=slice(0, 1))
        # Check that the right number of files were imported
        with h5py.File(self.hdf, mode='r') as f:
            grp = f['experiment1/imported']
            self.assertEqual(grp['dark_fields'].shape[1], 1)
    
    def test_import_multiple_hdfs(self):
        import_aps32idc_xanes_files([self.src_data, self.src_data],
                                    hdf_filename=self.hdf, hdf_groupname='experiment1',
                                    square=False, downsample=0)
        with h5py.File(self.hdf, mode='r') as f:
            g = f['/experiment1/imported']
            self.assertEqual(g['intensities'].shape, (2, 3, 512, 612))
            self.assertTrue(np.any(g['intensities'][0]))
            self.assertTrue(np.any(g['intensities'][1]))
            # They should be equal since we have import the same data twice
            np.testing.assert_equal(g['intensities'][0], g['intensities'][1])
    
    def test_import_second_hdf(self):
        # Run the import function
        import_aps32idc_xanes_file(self.src_data,
                                   hdf_filename=self.hdf, hdf_groupname='experiment1',
                                   total_timesteps=2, square=False, downsample=0)
        import_aps32idc_xanes_file(self.src_data,
                                   hdf_filename=self.hdf, hdf_groupname='experiment1',
                                   total_timesteps=2, timestep=1, append=True, square=False,
                                   downsample=0)
        with h5py.File(self.hdf, mode='r') as f:
            g = f['/experiment1/imported']
            self.assertEqual(g['intensities'].shape, (2, 3, 512, 612))
            self.assertTrue(np.any(g['intensities'][0]))
            self.assertTrue(np.any(g['intensities'][1]))
            # They should be equal since we have import the same data twice
            np.testing.assert_equal(g['intensities'][0], g['intensities'][1])


class CosmicTest(TestCase):
    """Test for importing STXM and ptychography data.
    
    From ALS Cosmic beamline. Test data taken from beamtime on
    2018-11-09. The cxi file is a stripped down version of the
    original (to save space). Missing crucial data should be added to
    the cxi as needed.
    
    Data
    ====
    ptycho-scan-856eV.cxi : NS_181110188_002.cxi
    stxm-scan-a003.xim : NS_181110203_a003.xim
    stxm-scan-a019.xim : NS_181110203_a019.xim
    stxm-scan.hdr : NS_181110203.hdr
    
    """
    stxm_hdr = os.path.join(COSMIC_DIR, 'stxm-scan.hdr')
    ptycho_cxi = os.path.join(COSMIC_DIR, 'ptycho-scan-856eV.cxi')
    hdf_filename = os.path.join(COSMIC_DIR, 'cosmic-test-import.h5')
    
    def tearDown(self):
        if os.path.exists(self.hdf_filename):
            os.remove(self.hdf_filename)
    
    def test_import_partial_data(self):
        """Check if the cosmic importer works if only hdr or cxi files are
        given."""
        # Import only STXM images
        import_cosmic_frameset(stxm_hdr=[self.stxm_hdr],
                               ptycho_cxi=[],
                               hdf_filename=self.hdf_filename)
        with TXMStore(self.hdf_filename, parent_name='stxm-scan') as store:
            self.assertEqual(store.data_name, 'imported')
        # Import only ptycho images
        import_cosmic_frameset(stxm_hdr=[],
                               ptycho_cxi=[self.ptycho_cxi],
                               hdf_filename=self.hdf_filename)
        with TXMStore(self.hdf_filename, parent_name='ptycho-scan-856eV') as store:
            self.assertEqual(store.data_name, 'imported')
    
    def test_import_cosmic_data(self):
        # Check that passing no data raises and exception
        with self.assertRaises(ValueError):
            import_cosmic_frameset(hdf_filename=self.hdf_filename)
        import_cosmic_frameset(stxm_hdr=[self.stxm_hdr],
                               ptycho_cxi=[self.ptycho_cxi],
                               hdf_filename=self.hdf_filename)
        # Does the HDF file exist
        self.assertTrue(os.path.exists(self.hdf_filename),
                        "%s doesn't exist" % self.hdf_filename)
        hdf_kw = dict(hdf_filename=self.hdf_filename,
                      parent_name='ptycho-scan-856eV',
                      mode='r')
        # Open ptychography TXM store and check its contents
        with TXMStore(**hdf_kw, data_name='imported_ptychography') as store:
            # Make sure the group exists
            self.assertEqual(store.data_group().name,
                             '/ptycho-scan-856eV/imported_ptychography')
            # Check the data structure
            self.assertEqual(store.filenames.shape, (1, 1))
            stored_filename = store.filenames[0,0].decode('utf-8')
            self.assertEqual(stored_filename, os.path.basename(self.ptycho_cxi))
            np.testing.assert_equal(store.energies.value, [[855.9056362433222]])
            np.testing.assert_equal(store.pixel_sizes.value, [[6.0435606480754585]])
            np.testing.assert_equal(store.pixel_unit, 'nm')
            self.assertEqual(store.intensities.shape, (1, 1, 285, 285))
            self.assertEqual(store.optical_depths.shape, (1, 1, 285, 285))
            self.assertEqual(store.timestep_names[0].decode('utf-8'), 'ex-situ')
        # Open STXM TXM store and check its contents
        with TXMStore(**hdf_kw, data_name='imported_stxm') as store:
            # Make sure the group exists
            self.assertEqual(store.data_group().name,
                             '/ptycho-scan-856eV/imported_stxm')
            # Check the data structure
            self.assertEqual(store.filenames.shape, (1, 2))
            stored_filename = store.filenames[0,0].decode('utf-8')
            expected_filename = os.path.join(COSMIC_DIR, 'stxm-scan_a003.xim')
            self.assertEqual(stored_filename, expected_filename)
            np.testing.assert_equal(store.energies.value, [[853, 857.75]])
            np.testing.assert_equal(store.pixel_sizes.value, [[27.2, 27.2]])
            self.assertEqual(store.intensities.shape, (1, 2, 120, 120))
            self.assertEqual(store.optical_depths.shape, (1, 2, 120, 120))
            self.assertEqual(store.timestep_names[0].decode('utf-8'), 'ex-situ')
        # Open imported TXMStore to check its contents
        with TXMStore(**hdf_kw, data_name='imported') as store:
            self.assertEqual(store.filenames.shape, (1, 3))
            self.assertEqual(store.timestep_names.shape, (1,))
            real_px_size = 6.0435606480754585
            np.testing.assert_equal(store.pixel_sizes.value,
                                    [[real_px_size, real_px_size, real_px_size]])
            self.assertEqual(store.pixel_unit, 'nm')


class CosmicFileTest(TestCase):
    stxm_hdr = os.path.join(COSMIC_DIR, 'stxm-scan.hdr')
    ptycho_cxi = os.path.join(COSMIC_DIR, 'ptycho-scan-856eV.cxi')
    
    def setUp(self):
        self.hdr = HDRFile(self.stxm_hdr)
        self.cxi = CXIFile(self.ptycho_cxi)
    
    def test_hdr_filenames(self):
        real_filenames = [os.path.join(COSMIC_DIR, f) for f in
                          ('stxm-scan_a003.xim', 'stxm-scan_a019.xim')]
        self.assertEqual(self.hdr.filenames(), real_filenames)
    
    def test_cxi_filenames(self):
        self.assertEqual(self.cxi.filenames(), ['ptycho-scan-856eV.cxi'])
    
    def test_cxi_image_data(self):
        with self.cxi:
            self.assertEqual(self.cxi.num_images(), 1)
            self.assertEqual(self.cxi.image_frames().shape, (1, 285, 285))
    
    def test_cxi_image_shape(self):
        with self.cxi:
            self.assertEqual(self.cxi.image_shape(), (285, 285))
    
    def test_cxi_energies(self):
        with self.cxi:
            self.assertAlmostEqual(self.cxi.energies()[0], 855.9056, places=3)
    
    def test_cxi_pixel_size(self):
        real_px_size = 6.0435606480754585
        with self.cxi:
            self.assertAlmostEqual(self.cxi.pixel_size(), real_px_size)
    
    def test_hdr_pixel_size(self):
        with self.hdr:
            self.assertEqual(self.hdr.pixel_size(), 27.2)
            
    def test_hdr_image_data(self):
        self.assertEqual(self.hdr.num_images(), 2)
        self.assertEqual(self.hdr.image_frames().shape, (2, 120, 120))
    
    def test_hdr_image_shape(self):
        self.assertEqual(self.hdr.image_shape(), (120, 120))
    
    def test_hdr_energies(self):
        with self.hdr:
            self.assertAlmostEqual(self.hdr.energies()[0], 853., places=3)

    def test_specific_hdr_files(self):
        """This test check specific HDR files that did not succeed at first.
        
        """
        # This one has a negative sign in front of the x-position
        filename1 = os.path.join(COSMIC_DIR, 'NS_181111148.hdr')
        hdr1 = HDRFile(filename1)
        self.assertAlmostEqual(hdr1.pixel_size(), 66.7)


class XradiaTest(TestCase):
    txrm_filename = os.path.join(TEST_DIR, "aps-8BM-sample.txrm")
    def test_pixel_size(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        with XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl") as xrm:
            self.assertAlmostEqual(xrm.um_per_pixel(), 0.03287, places=4)
    
    def test_timestamp_from_xrm(self):
        pacific_tz = pytz.timezone("US/Pacific")
        chicago_tz = pytz.timezone('US/Central')
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        with XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl") as xrm:
            # Check start time
            start = pacific_tz.localize(dt.datetime(2016, 5, 29, 15, 2, 37))
            start = start.astimezone(pytz.utc).replace(tzinfo=None)
            self.assertEqual(xrm.starttime(), start)
            self.assertEqual(xrm.starttime().tzinfo, None)
            # Check end time (offset determined by exposure time)
            end = pacific_tz.localize(dt.datetime(2016, 5, 29, 15, 2, 37, 500000))
            end = end.astimezone(pytz.utc).replace(tzinfo=None)
            self.assertEqual(xrm.endtime(), end)
            xrm.close()
        
        # Test APS frame
        sample_filename = "fov03_xanesocv_8353_0eV.xrm"
        with XRMFile(os.path.join(APS_DIR, sample_filename), flavor="aps") as xrm:
            # Check start time
            start = chicago_tz.localize(dt.datetime(2016, 7, 2, 17, 50, 35))
            start = start.astimezone(pytz.utc).replace(tzinfo=None)
            self.assertEqual(xrm.starttime(), start)
            # Check end time (offset determined by exposure time)
            end = chicago_tz.localize(dt.datetime(2016, 7, 2, 17, 51, 25))
            end = end.astimezone(pytz.utc).replace(tzinfo=None)
            self.assertEqual(xrm.endtime(), end)
    
    def test_str_and_repr(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        with XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl") as xrm:
            self.assertEqual(repr(xrm), "<XRMFile: '{}'>".format(sample_filename))
            self.assertEqual(str(xrm), "<XRMFile: '{}'>".format(sample_filename))
    
    def test_binning(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        with XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl") as xrm:
            self.assertEqual(xrm.binning(), (2, 2))
    
    def test_frame_stack(self):
        with TXRMFile(self.txrm_filename, flavor="aps") as txrm:
            self.assertEqual(txrm.image_stack().shape, (3, 1024, 1024))
            self.assertEqual(txrm.energies().shape, (3,))
    
    def test_num_images(self):
        with TXRMFile(self.txrm_filename, flavor="aps") as txrm:
            self.assertEqual(txrm.num_images(), 3)
    
    def test_starttimes(self):
        with TXRMFile(self.txrm_filename, flavor="aps") as txrm:
            result = txrm.starttimes()
            expected_start = dt.datetime(2017, 7, 9, 0, 49, 2)
            self.assertEqual(result[0], expected_start)


class PtychographyImportTest(TestCase):
    def setUp(self):
        self.hdf = os.path.join(PTYCHO_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def test_directory_names(self):
        """Tests for checking some of the edge cases for what can be passed as
        a directory string."""
        import_nanosurveyor_frameset(PTYCHO_DIR + "/", hdf_filename=self.hdf)
    
    def test_imported_hdf(self):
        import_nanosurveyor_frameset(PTYCHO_DIR, hdf_filename=self.hdf)
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            dataset_name = 'NS_160406074'
            parent = f[dataset_name]
            # Check metadata about the sample
            self.assertEqual(parent.attrs['latest_data_name'], "imported")
            group = parent['imported']
            keys = list(group.keys())
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'], 'ptychography STXM')
            # Check data is structured properly
            self.assertEqual(group['timestep_names'].value[0], bytes(dataset_name, 'ascii'))
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].shape, (1, 3, 228, 228))
            self.assertEqual(group['intensities'].attrs['context'], 'frameset')
            self.assertIn('stxm', keys)
            self.assertEqual(group['stxm'].shape, (1, 3, 20, 20))
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'nm')
            self.assertTrue(np.all(group['pixel_sizes'].value == 4.16667),
                            msg=group['pixel_sizes'].value)
            self.assertEqual(group['pixel_sizes'].shape, (1, 3))
            expected_Es = np.array([[843.9069591, 847.90651815,
                                     850.15627011]])
            np.testing.assert_allclose(group['energies'].value, expected_Es)
            self.assertEqual(group['energies'].shape, (1, 3))
            ## NB: Timestamps not available in the cxi files
            # self.assertIn('timestamps', keys)
            # expected_timestamp = np.array([
            #     [[b'2016-07-02 16:31:36-05:51', b'2016-07-02 16:32:26-05:51'],
            #      [b'2016-07-02 17:50:35-05:51', b'2016-07-02 17:51:25-05:51']],
            #     [[b'2016-07-02 22:19:23-05:51', b'2016-07-02 22:19:58-05:51'],
            #      [b'2016-07-02 23:21:21-05:51', b'2016-07-02 23:21:56-05:51']],
            # ], dtype="S32")
            # self.assertTrue(np.array_equal(group['timestamps'].value,
            #                                expected_timestamp))
            self.assertIn('filenames', keys)
            self.assertEqual(group['filenames'].shape, (1, 3))
            self.assertIn('relative_positions', keys)
            self.assertEqual(group['relative_positions'].shape, (1, 3, 3))
            ## NB: It's not clear exactly what "original positions"
            ## means for STXM data
            self.assertIn('original_positions', keys)
            self.assertEqual(group['original_positions'].shape, (1, 3, 3))

    def test_frame_shape(self):
        """In some cases, frames are different shapes. Specifying a shape in
        the importer can fix this.
        
        """
        expected_shape = (220, 220)
        import_nanosurveyor_frameset(PTYCHO_DIR,
                                     hdf_filename=self.hdf,
                                     frame_shape=expected_shape)
        with h5py.File(self.hdf, mode='r') as f:
            real_shape = f['NS_160406074/imported/intensities'].shape
        self.assertEqual(real_shape, (1, 3, *expected_shape))
    
    def test_partial_import(self):
        """Sometimes the user may want to specify that only a subset of
        ptychographs be imported.
        """
        energy_range = (843, 848)
        import_nanosurveyor_frameset(PTYCHO_DIR,
                                     energy_range=energy_range,
                                     hdf_filename=self.hdf, quiet=True)
        with h5py.File(self.hdf, mode='r') as f:
            dataset_name = 'NS_160406074'
            parent = f[dataset_name]
            group = parent['imported']
            self.assertEqual(group['intensities'].shape[0:2],
                             (1, 2))
            self.assertEqual(group['filenames'].shape, (1, 2))
    
    def test_exclude_re(self):
        """Allow the user to exclude specific frames that are bad."""
        import_nanosurveyor_frameset(PTYCHO_DIR,
                                     exclude_re="(/009/|/100/)",
                                     hdf_filename=self.hdf, quiet=True)
        with h5py.File(self.hdf, mode='r') as f:
            dataset_name = 'NS_160406074'
            parent = f[dataset_name]
            group = parent['imported']
            self.assertEqual(group['intensities'].shape[0:2],
                             (1, 2))
    
    def test_multiple_import(self):
        """Check if we can import multiple different directories of different
        energies ranges."""
        # Import two data sets (order is important to test for sorting)
        import_nanosurveyor_frameset("{}-low-energy".format(PTYCHO_DIR),
                                     hdf_filename=self.hdf, quiet=True,
                                     hdf_groupname="merged")
        import_nanosurveyor_frameset("{}-high-energy".format(PTYCHO_DIR),
                                     hdf_filename=self.hdf, quiet=True,
                                     hdf_groupname="merged",
                                     append=True)
        # Check resulting HDF5 file
        with h5py.File(self.hdf) as f:
            self.assertIn('merged', f.keys())
            # Check that things are ordered by energy
            saved_Es = f['/merged/imported/energies'].value
            np.testing.assert_array_equal(saved_Es, np.sort(saved_Es))
            # Construct the expected path relative to the current directory
            relpath = "ptycho-data-als/NS_160406074-{}-energy/160406074/{}/NS_160406074.cxi"
            toplevel = os.getcwd().split('/')[-1]
            if toplevel == "tests":
                test_dir = ''
            else:
                test_dir = 'tests'
            relpath = os.path.join(test_dir, relpath)
            # Compare the expeected file names
            sorted_files = [[bytes(relpath.format("low", "001"), 'ascii'),
                             bytes(relpath.format("low", "009"), 'ascii'),
                             bytes(relpath.format("high", "021"), 'ascii'),]]
            saved_files = f['/merged/imported/filenames']
            np.testing.assert_array_equal(saved_files, sorted_files)


class APS8BMFileImportTest(TestCase):
    txrm_file = os.path.join(TEST_DIR, 'aps-8BM-sample.txrm')
    txrm_ref = os.path.join(TEST_DIR, 'aps-8BM-reference.txrm')
    def setUp(self):
        self.hdf = os.path.join(APS_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def test_imported_hdf(self):
        import_aps8bm_xanes_file(self.txrm_file,
                                  ref_filename=self.txrm_ref, hdf_filename=self.hdf,
                                  quiet=True)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            group = f['aps-8BM-sample/imported']
            parent = f['aps-8BM-sample']
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'], 'Full-field TXM')
            self.assertEqual(parent.attrs['xanespy_version'], CURRENT_VERSION)
            self.assertEqual(parent.attrs['beamline'], "APS 8-BM-B")
            self.assertEqual(parent.attrs['original_file'], self.txrm_file)
            # Check h5 data structure
            keys = list(group.keys())
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].shape, (1, 3, 1024, 1024))
            self.assertIn('references', keys)
            self.assertIn('optical_depths', keys)
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'µm')
            self.assertEqual(group['pixel_sizes'].shape, (1, 3))
            self.assertTrue(np.any(group['pixel_sizes'].value > 0))
            expected_Es = np.array([[8312.9287109,  8363.0078125,  8412.9541016]])
            np.testing.assert_almost_equal(group['energies'].value, expected_Es)
            self.assertIn('timestamps', keys)
            expected_timestamp = np.array([
                [b'2017-07-09 00:49:02', b'2017-07-09 00:49:30', b'2017-07-09 00:49:58'],
            ], dtype="S32")
            np.testing.assert_equal(group['timestamps'].value,
                                    expected_timestamp)
            self.assertIn('filenames', keys)
            self.assertIn('original_positions', keys)
            self.assertEqual(group['original_positions'].shape, (1, 3, 3))


class APS8BMDirImportTest(TestCase):
    """Check that the program can import a collection of SSRL frames from
    a directory."""
    def setUp(self):
        self.hdf = os.path.join(APS_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def test_import_empty_directory(self):
        """Check that the proper exception is raised if the directory has no
        TXM files in it."""
        EMPTY_DIR = 'temp-empty-dir'
        os.mkdir(EMPTY_DIR)
        try:
            with self.assertRaisesRegex(exceptions.DataNotFoundError,
                                        '/temp-empty-dir'):
                import_aps8bm_xanes_dir(EMPTY_DIR,
                                         hdf_filename="test-file.hdf",
                                         quiet=True)
        finally:
            # Clean up by deleting any temporary files/directories
            if os.path.exists('test-file.hdf'):
                os.remove('test-file.hdf')
            os.rmdir(EMPTY_DIR)
    
    def test_imported_references(self):
        import_aps8bm_xanes_dir(APS_DIR, hdf_filename=self.hdf, quiet=True)
        with h5py.File(self.hdf, mode='r') as f:
            self.assertIn('references', f['fov03/imported'].keys())
    
    def test_groupname_kwarg(self):
        """The groupname keyword argument needs some special attention."""
        with self.assertRaisesRegex(exceptions.CreateGroupError, 'Invalid groupname'):
            import_aps8bm_xanes_dir(APS_DIR, hdf_filename=self.hdf,
                                     quiet=True, groupname="Wawa")
        # Now does it work with the {} inserted
        import_aps8bm_xanes_dir(APS_DIR, hdf_filename=self.hdf,
                                 quiet=True, groupname="Wawa{}")
        
    
    def test_imported_hdf(self):
        import_aps8bm_xanes_dir(APS_DIR, hdf_filename=self.hdf, quiet=True)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            group = f['fov03/imported']
            parent = f['fov03']
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'], 'Full-field TXM')
            self.assertEqual(parent.attrs['xanespy_version'], CURRENT_VERSION)
            self.assertEqual(parent.attrs['beamline'], "APS 8-BM-B")
            self.assertEqual(parent.attrs['original_directory'], APS_DIR)
            # Check h5 data structure
            keys = list(group.keys())
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].shape, (2, 2, 1024, 1024))
            self.assertIn('references', keys)
            self.assertIn('optical_depths', keys)
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'µm')
            self.assertEqual(group['pixel_sizes'].shape, (2,2))
            self.assertTrue(np.any(group['pixel_sizes'].value > 0))
            expected_Es = np.array([[8249.9365234375, 8353.0322265625],
                                    [8249.9365234375, 8353.0322265625]])
            self.assertTrue(np.array_equal(group['energies'].value, expected_Es))
            self.assertIn('timestamps', keys)
            expected_timestamp = np.array([
                [[b'2016-07-02 21:31:36', b'2016-07-02 21:32:26'],
                 [b'2016-07-02 22:50:35', b'2016-07-02 22:51:25']],
                [[b'2016-07-03 03:19:23', b'2016-07-03 03:19:58'],
                 [b'2016-07-03 04:21:21', b'2016-07-03 04:21:56']],
            ], dtype="S32")
            np.testing.assert_equal(group['timestamps'].value,
                                    expected_timestamp)
            self.assertIn('filenames', keys)
            self.assertIn('original_positions', keys)
            # self.assertIn('relative_positions', keys)
            # self.assertEqual(group['relative_positions'].shape, (2, 3))
    
    def test_params_from_aps(self):
        """Check that the new naming scheme is decoded properly."""
        ref_filename = "ref_xanesocv_8250_0eV.xrm"
        result = decode_aps_params(ref_filename)
        expected = {
            'timestep_name': 'ocv',
            'position_name': 'ref',
            'is_background': True,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)

    def test_file_metadata(self):
        filenames = [os.path.join(APS_DIR, 'fov03_xanessoc01_8353_0eV.xrm')]
        df = read_metadata(filenames=filenames, flavor='aps', quiet=True)
        self.assertIsInstance(df, pd.DataFrame)
        row = df.iloc[0]
        self.assertIn('shape', row.keys())
        self.assertIn('timestep_name', row.keys())
        # Check the correct start time
        chicago_tz = pytz.timezone('US/Central')
        realtime = chicago_tz.localize(dt.datetime(2016, 7, 2, 23, 21, 21))
        realtime = realtime.astimezone(pytz.utc).replace(tzinfo=None)
        # Convert to unix timestamp
        self.assertIsInstance(row['starttime'], pd.Timestamp)
        self.assertEqual(row['starttime'], realtime)


class SSRLImportTest(TestCase):
    """Check that the program can import a collection of SSRL frames from
    a directory."""
    def setUp(self):
        self.hdf = os.path.join(SSRL_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
    
    def test_minimum_shape(self):
        shape_list = [(1024, 512), (1024, 1024), (2048, 2048)]
        min_shape = minimum_shape(shape_list)
        self.assertEqual(min_shape, (1024, 512))
        # Check with incompatible shape dimensions
        shape_list = [(1024, 1024), (1024, 1024), (2048, 2048, 2048)]
        with self.assertRaises(exceptions.ShapeMismatchError):
            minimum_shape(shape_list)
        # Check that non-power-of-two shapes raise an exception
        shape_list = [(5, 1024), (1024, 1024), (2048, 2048)]
        with self.assertRaises(exceptions.ShapeMismatchError):
            minimum_shape(shape_list)
        # Check with using named tuples
        shape_list = [utilities.shape(1024, 1024), utilities.shape(1024, 1024)]
        min_shape = minimum_shape(shape_list)
        print(min_shape)
    
    def test_rebin_image(self):
        my_list = [1, 2, 2, 3, 3, 3]
        # Test a symmetrical reshape
        img = np.ones((64, 64))
        new_img = rebin_image(img, (32, 32))
        self.assertEqual(new_img.shape, (32, 32))
        # Test an asymmetrical reshape
        img = np.ones((64, 64))
        new_img = rebin_image(img, (32, 16))
        self.assertEqual(new_img.shape, (32, 16))
            
    def test_imported_hdf(self):
        with warnings.catch_warnings() as w:
            # warnings.simplefilter('ignore', RuntimeWarning, 104)
            warnings.filterwarnings('ignore',
                                    message='Ignoring invalid file .*',
                                    category=RuntimeWarning)
            import_ssrl_xanes_dir(SSRL_DIR, hdf_filename=self.hdf, quiet=True)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            group = f['ssrl-test-data/imported']
            parent = f['ssrl-test-data']
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'], 'Full-field TXM')
            self.assertEqual(parent.attrs['xanespy_version'], CURRENT_VERSION)
            self.assertEqual(parent.attrs['beamline'], "SSRL 6-2c")
            self.assertEqual(parent.attrs['original_directory'], SSRL_DIR)
            # Check imported data structure
            keys = list(group.keys())
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].attrs['context'], 'frameset')
            self.assertEqual(group['intensities'].shape, (1, 2, 1024, 1024))
            self.assertIn('references', keys)
            self.assertEqual(group['references'].attrs['context'], 'frameset')
            self.assertIn('optical_depths', keys)
            self.assertEqual(group['optical_depths'].attrs['context'], 'frameset')
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'µm')
            self.assertEqual(group['pixel_sizes'].attrs['context'], 'metadata')
            isEqual = np.array_equal(group['energies'].value,
                                     np.array([[8324., 8354.]]))
            self.assertTrue(isEqual, msg=group['energies'].value)
            self.assertEqual(group['energies'].attrs['context'], 'metadata')
            self.assertIn('timestamps', keys)
            self.assertEqual(group['timestamps'].attrs['context'], 'metadata')
            self.assertIn('filenames', keys)
            self.assertEqual(group['filenames'].attrs['context'], 'metadata')
            self.assertIn('original_positions', keys)
            self.assertEqual(group['original_positions'].attrs['context'], 'metadata')
            self.assertIn('relative_positions', keys)
            self.assertEqual(group['relative_positions'].attrs['context'], 'metadata')
            self.assertIn('timestep_names', keys)
            self.assertEqual(group['relative_positions'].attrs['context'], 'metadata')
            self.assertEqual(group['timestep_names'][0], "rep01")
   
    def test_params_from_ssrl(self):
        # First a reference frame
        ref_filename = "rep01_000001_ref_201511202114_NCA_INSITU_OCV_FOV01_Ni_08250.0_eV_001of010.xrm"
        result = decode_ssrl_params(ref_filename)
        expected = {
            'timestep_name': 'rep01',
            'position_name': 'NCA_INSITU_OCV_FOV01_Ni',
            'is_background': True,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)
        # Now a sample field of view
        sample_filename = "rep01_201511202114_NCA_INSITU_OCV_FOV01_Ni_08250.0_eV_001of010.xrm"
        result = decode_ssrl_params(sample_filename)
        expected = {
            'timestep_name': 'rep01',
            'position_name': 'NCA_INSITU_OCV_FOV01_Ni',
            'is_background': False,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)
        # This one was a problem, from 2017-04-05
        sample_filename = (
            "NCA_Pg71-5/Pg71-5_NCA_charge2_XANES_170405_1515/"
            "rep01_Pg71-5_NCA_charge2_08250.0_eV_001of005.xrm")
        result = decode_ssrl_params(sample_filename)
        expected = {
            'timestep_name': 'rep01',
            'position_name': 'Pg71-5_NCA_charge2',
            'is_background': False,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)
        # This reference was also a problem
        ref_filename = 'rep01_000001_ref_Pg71-5_NCA_charge2_08250.0_eV_001of010.xrm'
        result = decode_ssrl_params(ref_filename)
        expected = {
            'timestep_name': 'rep01',
            'position_name': 'Pg71-5_NCA_charge2',
            'is_background': True,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)
        # Another bad reference file
        ref_filename = 'rep02_000182_ref_201604061951_Pg71-8_NCA_charge1_08400.0_eV_002of010.xrm'
        result = decode_ssrl_params(ref_filename)
        expected = {
            'timestep_name': 'rep02',
            'position_name': 'Pg71-8_NCA_charge1',
            'is_background': True,
            'energy': 8400.0,
        }
        self.assertEqual(result, expected)
    
    def test_magnification_correction(self):
        # Prepare some fake data
        img1 = [[1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,1]]
        img2 = [[0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,0,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]]
        imgs = np.array([[img1, img2], [img1, img2]], dtype=np.float)
        pixel_sizes = np.array([[1, 2], [1, 2]])
        scales, translations = magnification_correction(imgs, pixel_sizes)
        # Check that the right shape result is returns
        self.assertEqual(scales.shape, (2, 2, 2))
        np.testing.assert_equal(scales[..., 0], scales[..., 1])
        # Check that the first result is not corrected
        np.testing.assert_equal(scales[0, 0], (1., 1.))
        np.testing.assert_equal(translations[0, 0], (0, 0))
        # # Check the values for translation and scale for the changed image
        np.testing.assert_equal(scales[0, 1], (0.5, 0.5))
        np.testing.assert_equal(translations[0,1], (1., 1.))
    
    def test_bad_file(self):
        # One specific file is not saved properly
        filenames = [
            # No image data nor timestamp
            'rep02_000072_ref_20161456_ssrl-test-data_08348.0_eV_002of010.xrm',
            # Valid file
            'rep01_000351_ref_20161456_ssrl-test-data_08354.0_eV_001of010.xrm',
            # Malformed image data
            # 'rep02_000182_ref_20161456_ssrl-test-data_08400.0_eV_002of010.xrm',
        ]
        filenames = [os.path.join(SSRL_DIR, f) for f in filenames]
        # Check that the importer warns the user of the bad file
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            result = read_metadata(filenames, flavor='ssrl', quiet=True)
            self.assertTrue(len(ws) > 0)
            self.assertTrue(any([w.category == RuntimeWarning for w in ws]))
            self.assertTrue(any(['Ignoring invalid file' in str(w.message) for w in ws]))
        # Check that the bad entries was excluded from the processed list
        self.assertEqual(len(result), 1)


class SxstmFileTestCase(unittest.TestCase):
    """Tests for soft x-ray tunneling microscope data from APS 4-ID-C."""
    def test_header(self):
        filename = os.path.join(SXSTM_DIR, 'XGSS_UIC_JC_475v_60c_001_001_001.3ds')
        sxstm_data = SxstmDataFile(filename=filename)
        header = sxstm_data.header_lines()
        self.assertEqual(len(header), 33)
        data = sxstm_data.dataframe()
        sxstm_data.close()


class SxstmImportTestCase(unittest.TestCase):
    """Tests for importing a set of X-ray tunneleing microscopy data from
    APS 4-ID-C. 
    
    """
    hdf_filename = os.path.join(SXSTM_DIR, 'sxstm_imported.h5')
    parent_groupname = 'sxstm-test-data'    
    def tearDown(self):
        if os.path.exists(self.hdf_filename):
            os.remove(self.hdf_filename)
    
    def test_hdf_file(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='X and Y pixel sizes')
            import_aps4idc_sxstm_files(filenames=os.path.join(TEST_DIR, 'sxstm-data-4idc'),
                                       hdf_filename=self.hdf_filename,
                                       hdf_groupname=self.parent_groupname,
                                       shape=(2, 2),
                                       energies=[8324., 8354.])
        # Check that the file exists with the data group
        self.assertTrue(os.path.exists(self.hdf_filename))
        with h5py.File(self.hdf_filename, mode='r') as f:
            # Check that the group structure is correct
            self.assertIn(self.parent_groupname, list(f.keys()))
            parent = f[self.parent_groupname]
            self.assertIn('imported', list(parent.keys()),
                          "Importer didn't create '/%s/imported'" % self.parent_groupname)
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'],
                             'Synchrotron X-ray Scanning Tunneling Microscopy')
            self.assertEqual(parent.attrs['xanespy_version'], CURRENT_VERSION)
            self.assertEqual(parent.attrs['beamline'], "APS 4-ID-C")
            self.assertEqual(parent.attrs['latest_data_name'], 'imported')
            full_path = os.path.abspath(SXSTM_DIR)
            self.assertEqual(parent.attrs['original_directory'], full_path)
            # Check that the datasets are created
            group = parent['imported']
            keys = list(group.keys())
            columns = ['bias_calc', 'current', 'LIA_tip_ch1',
                       'LIA_tip_ch2', 'LIA_sample', 'LIA_shielding',
                       'LIA_topo', 'shielding', 'flux', 'bias',
                       'height']
            for col in columns:
                self.assertIn(col, list(group.keys()),
                              "Importer didn't create '/%s/imported/%s'"
                              "" % (self.parent_groupname, col))
                self.assertEqual(group[col].attrs['context'], 'frameset')
                self.assertEqual(group[col].dtype, 'float32')
                self.assertEqual(group[col].shape, (1, 2, 2, 2))
                self.assertTrue(np.any(group[col]))
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'µm')
            self.assertEqual(group['pixel_sizes'].attrs['context'], 'metadata')
            isEqual = np.array_equal(group['energies'].value,
                                     np.array([[8324., 8354.]]))
            self.assertTrue(isEqual, msg=group['energies'].value)
            self.assertEqual(group['energies'].attrs['context'], 'metadata')
            self.assertIn('filenames', keys)
            self.assertEqual(group['filenames'].attrs['context'], 'metadata')
            self.assertIn('timestep_names', keys)
            self.assertEqual(group['timestep_names'].attrs['context'], 'metadata')
            self.assertEqual(group['timestep_names'][0], b"ex-situ")
            # self.assertIn('timestamps', keys)
            # self.assertEqual(group['timestamps'].attrs['context'], 'metadata')
            # self.assertIn('original_positions', keys)
            # self.assertEqual(group['original_positions'].attrs['context'], 'metadata')
            # self.assertIn('relative_positions', keys)
            # self.assertEqual(group['relative_positions'].attrs['context'], 'metadata')

    def test_file_list(self):
        """See if a file list can be passed instead of a directory name."""
        filelist = [os.path.join(SXSTM_DIR, f) for f in os.listdir(SXSTM_DIR)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='X and Y pixel sizes')
            import_aps4idc_sxstm_files(filenames=filelist,
                                       hdf_filename=self.hdf_filename,
                                       hdf_groupname=self.parent_groupname,
                                       shape=(2, 2),
                                       energies=[8324., 8354.])
