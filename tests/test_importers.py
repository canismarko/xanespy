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
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import warnings

import pytz
import numpy as np
import pandas as pd
import h5py

from xanespy import exceptions
from xanespy.xradia import XRMFile, TXRMFile
from xanespy.sxstm import SxstmDataFile
from xanespy.importers import (magnification_correction,
                               decode_aps_params, decode_ssrl_params,
                               import_ssrl_xanes_dir, CURRENT_VERSION,
                               import_nanosurveyor_frameset,
                               import_aps_8BM_xanes_dir,
                               import_aps_8BM_xanes_file,
                               read_metadata)


TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')
SXSTM_DIR = os.path.join(TEST_DIR, "sxstm-data-4idc/")


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


class APSFileImportTest(TestCase):
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
        import_aps_8BM_xanes_file(self.txrm_file,
                                  ref_filename=self.txrm_ref, hdf_filename=self.hdf)
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


class APSDirImportTest(TestCase):
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
                import_aps_8BM_xanes_dir(EMPTY_DIR, hdf_filename="test-file.hdf")
        finally:
            # Clean up by deleting any temporary files/directories
            if os.path.exists('test-file.hdf'):
                os.remove('test-file.hdf')
            os.rmdir(EMPTY_DIR)
    
    def test_imported_references(self):
        import_aps_8BM_xanes_dir(APS_DIR, hdf_filename=self.hdf, quiet=True)
        with h5py.File(self.hdf, mode='r') as f:
            self.assertIn('references', f['fov03/imported'].keys())
    
    def test_groupname_kwarg(self):
        """The groupname keyword argument needs some special attention."""
        with self.assertRaisesRegex(exceptions.CreateGroupError, 'Invalid groupname'):
            import_aps_8BM_xanes_dir(APS_DIR, hdf_filename=self.hdf,
                                     quiet=True, groupname="Wawa")
        # Now does it work with the {} inserted
        import_aps_8BM_xanes_dir(APS_DIR, hdf_filename=self.hdf,
                                 quiet=True, groupname="Wawa{}")
        
    
    def test_imported_hdf(self):
        import_aps_8BM_xanes_dir(APS_DIR, hdf_filename=self.hdf, quiet=True)
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
        df = read_metadata(filenames=filenames, flavor='aps')
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
    
    def test_imported_hdf(self):
        with warnings.catch_warnings() as w:
            warnings.simplefilter('ignore', RuntimeWarning, 103)
            import_ssrl_xanes_dir(SSRL_DIR, hdf_filename=self.hdf)
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
            result = read_metadata(filenames, flavor='ssrl')
            self.assertEqual(len(ws), 1)
            [self.assertEqual(RuntimeWarning, w.category) for w in ws]
        # Check that the bad entries was excluded from the processed list
        self.assertEqual(len(result), 1)


class SxstmTestCase(unittest.TestCase):
    """Tests for soft x-ray tunneling microscope data from APS 4-ID-C."""
    def test_header(self):
        filename = os.path.join(SXSTM_DIR, 'XGSS_UIC_JC_475v_60c_001_001_001.3ds')
        sxstm_data = SxstmDataFile(filename=filename)
        header = sxstm_data.header_lines()
        self.assertEqual(len(header), 33)
        data = sxstm_data.dataframe()
        sxstm_data.close()

