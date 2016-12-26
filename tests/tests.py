#!/bin/python3
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
import math
import os
import shutil
import warnings
from collections import namedtuple

import h5py
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter('ignore', PendingDeprecationWarning)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import Normalize
import pytz
from skimage import data

from cases import XanespyTestCase
from xanespy import exceptions
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
                                guess_kedge)
from xanespy.edges import KEdge, k_edges, l_edges
from xanespy.importers import (import_ssrl_frameset,
                               import_aps_8BM_frameset,
                               import_nanosurveyor_frameset,
                               _average_frames,
                               magnification_correction,
                               decode_aps_params, decode_ssrl_params,
                               read_metadata, CURRENT_VERSION as IMPORT_VERSION)
from xanespy.xradia import XRMFile
from xanespy.beamlines import (sector8_xanes_script, ssrl6_xanes_script,
                               Zoneplate, ZoneplatePoint, Detector)
from xanespy.txmstore import TXMStore

TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')


class SSRLScriptTest(unittest.TestCase):
    """Verify that a script is created for running an operando TXM
    experiment at SSRL beamline 6-2c. These tests conform to the
    results of the beamline's in-house script generator. They could be
    changed but the effects on the beamline operation should be
    checked first.
    """

    def setUp(self):
        self.output_path = os.path.join(TEST_DIR, 'ssrl_script.txt')
        self.scaninfo_path = os.path.join(TEST_DIR, 'ScanInfo_ssrl_script.txt')
        # Check to make sure the file doesn't already exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        assert not os.path.exists(self.output_path)
        # Values taken from SSRL 6-2c beamtime on 2015-02-22
        self.zp = Zoneplate(
            start=ZoneplatePoint(x=-7.40, y=-2.46, z=-1255.46, energy=8250),
            end=ZoneplatePoint(x=4.14, y=1.38, z=703.06, energy=8640),
        )

    def tearDown(self):
        os.remove(self.output_path)
        os.remove(self.scaninfo_path)

    def test_scaninfo_generation(self):
        """Check that the script writes all the filenames to a ScanInfo file
        for TXM Wizard."""
        with open(self.output_path, 'w') as f:
            ssrl6_xanes_script(dest=f,
                               edge=k_edges["Ni_NCA"](),
                               binning=2,
                               zoneplate=self.zp,
                               iterations=["Test0", "Snorlax"],
                               frame_rest=0,
                               repetitions=8,
                               ref_repetitions=15,
                               positions=[position(3, 4, 5)],
                               reference_position=position(0, 1, 2),
                               abba_mode=False)
        scaninfopath = os.path.join(TEST_DIR, 'ScanInfo_ssrl_script.txt')
        self.assertTrue(os.path.exists(scaninfopath))
        with open(scaninfopath) as f:
            self.assertEqual(f.readline(), 'VERSION 1\n')
            self.assertEqual(f.readline(), 'ENERGY 1\n')
            self.assertEqual(f.readline(), 'TOMO 0\n')
            self.assertEqual(f.readline(), 'MOSAIC 0\n')
            self.assertEqual(f.readline(), 'MULTIEXPOSURE 4\n')
            self.assertEqual(f.readline(), 'NREPEATSCAN   1\n')
            self.assertEqual(f.readline(), 'WAITNSECS   0\n')
            self.assertEqual(f.readline(), 'NEXPOSURES   8\n')
            self.assertEqual(f.readline(), 'AVERAGEONTHEFLY   0\n')
            self.assertEqual(f.readline(), 'REFNEXPOSURES  15\n')
            self.assertEqual(f.readline(), 'REF4EVERYEXPOSURES   8\n')
            self.assertEqual(f.readline(), 'REFABBA 0\n')
            self.assertEqual(f.readline(), 'REFAVERAGEONTHEFLY 0\n')
            self.assertEqual(f.readline(), 'MOSAICUP   1\n')
            self.assertEqual(f.readline(), 'MOSAICDOWN   1\n')
            self.assertEqual(f.readline(), 'MOSAICLEFT   1\n')
            self.assertEqual(f.readline(), 'MOSAICRIGHT   1\n')
            self.assertEqual(f.readline(), 'MOSAICOVERLAP 0.20\n')
            self.assertEqual(f.readline(), 'MOSAICCENTRALTILE   1\n')
            self.assertEqual(f.readline(), 'FILES\n')
            self.assertEqual(f.readline(), 'ref_Test0_08250.0_eV_000of015.xrm\n')


    def test_script_generation(self):
        """Check that the script first moves to the first energy point and location."""
        ref_repetitions = 10
        with open(self.output_path, 'w') as f:
            ssrl6_xanes_script(dest=f,
                               edge=k_edges["Ni_NCA"](),
                               binning=2,
                               zoneplate=self.zp,
                               iterations=["Test0", "Snorlax"],
                               frame_rest=0,
                               ref_repetitions=ref_repetitions,
                               positions=[position(3, 4, 5), position(6, 7, 8)],
                               reference_position=position(0, 1, 2),
                               abba_mode=True)
        with open(self.output_path, 'r') as f:
            # Check that the first couple of lines set up the correct data
            self.assertEqual(f.readline(), ';; 2D XANES ;;\n')
            # Sets up the first energy correctly
            self.assertEqual(f.readline(), ';;;; set the MONO and the ZP\n')
            self.assertEqual(f.readline(), 'sete 8250.00\n')
            self.assertEqual(f.readline(), 'moveto zpx -7.40\n')
            self.assertEqual(f.readline(), 'moveto zpy -2.46\n')
            self.assertEqual(f.readline(), 'moveto zpz -1255.46\n')
            self.assertEqual(f.readline(), ';;;; Move to reference position\n')
            self.assertEqual(f.readline(), 'moveto x 0.00\n')
            self.assertEqual(f.readline(), 'moveto y 1.00\n')
            self.assertEqual(f.readline(), 'moveto z 2.00\n')
            # Collects the first set of references frames
            self.assertEqual(f.readline(), ';;;; Collect reference frames\n')
            self.assertEqual(f.readline(), 'setexp 0.50\n')
            self.assertEqual(f.readline(), 'setbinning 2\n')
            self.assertEqual(f.readline(), 'collect ref_Test0_08250.0_eV_000of010.xrm\n')
            # Read-out the rest of the "collect ..." commands
            [f.readline() for i in range(1, ref_repetitions)]
            # Moves to and collects first sample frame
            self.assertEqual(f.readline(), ';;;; Move to sample position 0\n')
            self.assertEqual(f.readline(), 'moveto x 3.00\n')
            self.assertEqual(f.readline(), 'moveto y 4.00\n')
            self.assertEqual(f.readline(), 'moveto z 5.00\n')
            self.assertEqual(f.readline(), ';;;; Collect frames sample position 0\n')
            self.assertEqual(f.readline(), 'setexp 0.50\n')
            self.assertEqual(f.readline(), 'setbinning 2\n')
            self.assertEqual(f.readline(), 'collect Test0_fov0_08250.0_eV_000of005.xrm\n')


@unittest.skip("Fix these tests before next APS beamtime")
class ApsScriptTest(unittest.TestCase):
    """Verify that a script is created for running an operando
    TXM experiment at APS beamline 8-BM-B."""

    def setUp(self):
        self.output_path = os.path.join(TEST_DIR, 'aps_script.txt')
        # Check to make sure the file doesn't already exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        assert not os.path.exists(self.output_path)
        # Values taken from APS beamtime on 2015-11-11
        self.zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            z_step=9.9329 / 2 # Original script assumed 2eV steps
        )
        self.det = Detector(
            start=ZoneplatePoint(x=0, y=0, z=389.8, energy=8313),
            z_step=0.387465 / 2 # Original script assumed 2eV steps
        )

    def tear_down(self):
        pass
    # os.remove(self.output_path)

    def test_file_created(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges["Ni"](),
                                 zoneplate=self.zp, detector=self.det,
                                 names=["test_sample"], sample_positions=[])
        # Check that a file was created
        self.assertTrue(
            os.path.exists(self.output_path)
        )

    def test_binning(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges["Ni"](),
                                 binning=2, zoneplate=self.zp,
                                 detector=self.det, names=[],
                                 sample_positions=[])
        with open(self.output_path, 'r') as f:
            firstline = f.readline().strip()
        self.assertEqual(firstline, "setbinning 2")

    def test_exposure(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges["Ni"](),
                                 exposure=44, zoneplate=self.zp,
                                 detector=self.det, names=["test_sample"],
                                 sample_positions=[])
        with open(self.output_path, 'r') as f:
            f.readline()
            secondline = f.readline().strip()
        self.assertEqual(secondline, "setexp 44")

    def test_energy_approach(self):
        """This instrument can behave poorly unless the target energy is
        approached from underneath (apparently)."""
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges['Ni_NCA'](),
                                 zoneplate=self.zp, detector=self.det,
                                 names=[], sample_positions=[])
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        # Check that the first zone plate is properly set
        assert False, "Test output of lines"

    def test_first_frame(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=k_edges['Ni_NCA'](),
                sample_positions=[position(x=1653, y=-1727, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                names=["test_sample"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        # Check that x, y are set
        self.assertEqual(lines[2].strip(), "moveto x 1653.00")
        self.assertEqual(lines[3].strip(), "moveto y -1727.00")
        self.assertEqual(lines[4].strip(), "moveto z 0.00")
        # Check that the energy approach lines are in tact
        self.assertEqual(lines[5].strip(), "moveto energy 8150.00")
        self.assertEqual(lines[54].strip(), "moveto energy 8248.00")
        # Check that energy is set
        self.assertEqual(lines[55].strip(), "moveto energy 8250.00")
        # Check that zone-plate and detector are set
        self.assertEqual(lines[56].strip(), "moveto zpz 2797.81")
        self.assertEqual(lines[57].strip(), "moveto detz 377.59")
        # Check that collect command is sent
        self.assertEqual(
            lines[58].strip(),
            "collect test_sample_xanes0_8250_0eV.xrm"
        )

    def test_second_location(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=k_edges['Ni_NCA'](),
                sample_positions=[position(x=1653, y=-1727, z=0),
                                  position(x=1706.20, y=-1927.20, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                names=["test_sample", "test_reference"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[247], "moveto x 1706.20\n")
        self.assertEqual(lines[248], "moveto y -1927.20\n")
        self.assertEqual(lines[250].strip(), "moveto energy 8150.00")

    def test_multiple_iterations(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=k_edges['Ni_NCA'](),
                sample_positions=[position(x=1653, y=-1727, z=0),
                                  position(x=1706.20, y=-1927.20, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                iterations=["ocv"] + ["{:02d}".format(soc) for soc in range(1, 10)],
                names=["test_sample", "test_reference"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(
            lines[58].strip(),
            "collect test_sample_xanesocv_8250_0eV.xrm"
        )
        self.assertEqual(
            lines[1090].strip(),
            "collect test_sample_xanes02_8342_0eV.xrm"
        )

class PtychographyImportTest(XanespyTestCase):
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

    def test_redundant_energies(self):
        """Test that a warning is triggered when importing the same energy
        multiple times."""
        # Do it once the normal way
        import_nanosurveyor_frameset(PTYCHO_DIR,
                                     hdf_filename=self.hdf, quiet=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(len(w), 0)
            import_nanosurveyor_frameset(PTYCHO_DIR,
                                         hdf_filename=self.hdf, quiet=True,
                                         append=False)
            self.assertEqual(len(w), 1)
            self.assertIn('Overwriting', str(w[0].message))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(len(w), 0)
            # Import again to see if a warning is triggered
            import_nanosurveyor_frameset(PTYCHO_DIR,
                                         hdf_filename=self.hdf, quiet=True,
                                         append=True)
            self.assertEqual(len(w), 1)
            self.assertIn('redundant energies', str(w[0].message))

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
            relpath = "ptycho-data-als/NS_160406074-{}-energy/160406074/{}/NS_160406074.cxi"
            sorted_files = [[bytes(relpath.format("low", "001"), 'ascii'),
                             bytes(relpath.format("low", "009"), 'ascii'),
                             bytes(relpath.format("high", "021"), 'ascii'),]]
            saved_files = f['/merged/imported/filenames']
            np.testing.assert_array_equal(saved_files, sorted_files)

class APSImportTest(XanespyTestCase):
    """Check that the program can import a collection of SSRL frames from
    a directory."""
    def setUp(self):
        self.hdf = os.path.join(APS_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)
        prog.quiet = True

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
                                        'xanespy-tests/temp-empty-dir'):
                import_aps_8BM_frameset(EMPTY_DIR, hdf_filename="test-file.hdf")
        finally:
            # Clean up by deleting any temporary files/directories
            if os.path.exists('test-file.hdf'):
                os.remove('test-file.hdf')
            os.rmdir(EMPTY_DIR)

    def test_imported_references(self):
        import_aps_8BM_frameset(APS_DIR, hdf_filename=self.hdf, quiet=True)
        with h5py.File(self.hdf, mode='r') as f:
            self.assertIn('references', f['fov03/imported'].keys())

    def test_imported_hdf(self):
        import_aps_8BM_frameset(APS_DIR, hdf_filename=self.hdf, quiet=True)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            group = f['fov03/imported']
            parent = f['fov03']
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'], 'Full-field TXM')
            self.assertEqual(parent.attrs['xanespy_version'], IMPORT_VERSION)
            self.assertEqual(parent.attrs['beamline'], "APS 8-BM-B")
            self.assertEqual(parent.attrs['original_directory'], APS_DIR)
            # Check h5 data structure
            keys = list(group.keys())
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].shape, (2, 2, 1024, 1024))
            self.assertIn('references', keys)
            self.assertIn('absorbances', keys)
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'µm')
            self.assertEqual(group['pixel_sizes'].shape, (2,2))
            self.assertTrue(np.any(group['pixel_sizes'].value > 0))
            expected_Es = np.array([[8249.9365234375, 8353.0322265625],
                                    [8249.9365234375, 8353.0322265625]])
            self.assertTrue(np.array_equal(group['energies'].value, expected_Es))
            self.assertIn('timestamps', keys)
            expected_timestamp = np.array([
                [[b'2016-07-02 16:31:36-05:51', b'2016-07-02 16:32:26-05:51'],
                 [b'2016-07-02 17:50:35-05:51', b'2016-07-02 17:51:25-05:51']],
                [[b'2016-07-02 22:19:23-05:51', b'2016-07-02 22:19:58-05:51'],
                 [b'2016-07-02 23:21:21-05:51', b'2016-07-02 23:21:56-05:51']],
            ], dtype="S32")
            self.assertTrue(np.array_equal(group['timestamps'].value,
                                           expected_timestamp))
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
        row = df.ix[0]
        self.assertIn('shape', row.keys())
        self.assertIn('timestep_name', row.keys())
        # Check the correct start time
        realtime = dt.datetime(2016, 7, 2, 23, 21, 21,
                               tzinfo=pytz.timezone('US/Central'))
        realtime = realtime.astimezone(pytz.utc).replace(tzinfo=None)
        # Convert to unix timestamp
        realtime = (realtime - dt.datetime(1970, 1, 1)) / dt.timedelta(seconds=1)
        self.assertIsInstance(row['starttime'], float)
        self.assertEqual(row['starttime'], realtime)


class SSRLImportTest(XanespyTestCase):
    """Check that the program can import a collection of SSRL frames from
    a directory."""
    def setUp(self):
        prog.quiet = True
        self.hdf = os.path.join(SSRL_DIR, 'testdata.h5')
        if os.path.exists(self.hdf):
            os.remove(self.hdf)

    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)

    def test_imported_hdf(self):
        import_ssrl_frameset(SSRL_DIR, hdf_filename=self.hdf)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            group = f['ssrl-test-data/imported']
            parent = f['ssrl-test-data']
            # Check metadata about beamline
            self.assertEqual(parent.attrs['technique'], 'Full-field TXM')
            self.assertEqual(parent.attrs['xanespy_version'], IMPORT_VERSION)
            self.assertEqual(parent.attrs['beamline'], "SSRL 6-2c")
            self.assertEqual(parent.attrs['original_directory'], SSRL_DIR)
            # Check imported data structure
            keys = list(group.keys())
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].attrs['context'], 'frameset')
            self.assertEqual(group['intensities'].shape, (1, 2, 1024, 1024))
            self.assertIn('references', keys)
            self.assertEqual(group['references'].attrs['context'], 'frameset')
            self.assertIn('absorbances', keys)
            self.assertEqual(group['absorbances'].attrs['context'], 'frameset')
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
        self.assertEqual(scales.shape, (2, 2))
        # Check that the first result is not corrected
        self.assertEqual(scales[0,0], 1.)
        self.assertEqual(list(translations[0, 0]), [0, 0])
        # Check the values for translation and scale for the changed image
        self.assertEqual(scales[0,1], 0.5)
        self.assertEqual(list(translations[0,1]), [1., 1.])


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

    def test_data_group(self):
        store = self.store()
        self.assertEqual(store.parent_group().name, '/ssrl-test-data')
        self.assertEqual(store.data_group().name, '/ssrl-test-data/imported')

    def test_fork_group(self):
        store = self.store('r+')
        with self.assertRaises(exceptions.CreateGroupError):
            store.fork_data_group(store.data_name)
        # Set a marker to see if it changes
        store.parent_group().create_group('new_group')
        store.data_name = 'new_group'
        store.data_group().attrs['test_val'] = 'Hello'
        # Now verify that the previous group was overwritten
        store.data_name = 'imported'
        store.fork_data_group('new_group')
        self.assertNotIn('test_val', list(store.data_group().attrs.keys()))
        # Check that the new group is registered as the "latest"
        self.assertEqual(store.latest_data_name, 'new_group')
        # Check that we can easily fork a non-existent group
        store.fork_data_group('brand_new')
        store.close()

    def test_data_tree(self):
        """Check that a data tree can be created showing the possible groups to choose from."""
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


class ZoneplateTest(XanespyTestCase):
    def setUp(self):
        # Values taken from APS beamtime on 2015-11-11
        self.aps_zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            z_step=9.9329 / 2 # Original script assumed 2eV steps
        )
        # Values taken from SSRL 6-2c on 2015-02-22
        self.ssrl_zp = Zoneplate(
            start=ZoneplatePoint(x=-7.40, y=-2.46, z=-1255.46, energy=8250),
            end=ZoneplatePoint(x=4.14, y=1.38, z=703.06, energy=8640),
        )

    def test_constructor(self):
        with self.assertRaises(ValueError):
            # Either `step` or `end` must be passed
            Zoneplate(start=None)
        with self.assertRaises(ValueError):
            # Passing both step and end is confusing
            Zoneplate(start=None, z_step=1, end=1)
        # Check that step is set if not expicitely passed
        zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            end=ZoneplatePoint(x=0, y=0, z=3120.6329, energy=8315)
        )
        self.assertApproximatelyEqual(zp.step.z, 9.9329 / 2)

    def test_z_from_energy(self):
        result = self.aps_zp.position(energy=8315).z
        self.assertApproximatelyEqual(result, 3120.6329)

    def test_position(self):
        result = self.aps_zp.position(energy=8315)
        self.assertApproximatelyEqual(result, (0, 0, 3120.6329))
        result = self.ssrl_zp.position(energy=8352)
        self.assertApproximatelyEqual(result, (-4.38, -1.46, -743.23))
        # self.assertApproximatelyEqual(result.x, 0)
        # self.assertApproximatelyEqual(result.y, 0)
        # self.assertApproximatelyEqual(result.z, 3120.6329)


class XrayEdgeTest(unittest.TestCase):
    def setUp(self):
        class DummyEdge(KEdge):
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

    def test_post_edge_xs(self):
        x = np.array([1, 2, 3])
        X = self.edge._post_edge_xs(x)
        expected = np.array([[1, 1], [2, 4], [3, 9]])
        self.assertTrue(np.array_equal(X, expected))
        # Test it with a single value
        x = 5
        X = self.edge._post_edge_xs(x)
        self.assertTrue(np.array_equal(X, [[5, 25]]))
        # Test with a single value but first order
        x = 5
        self.edge.post_edge_order = 1
        X = self.edge._post_edge_xs(x)
        self.assertTrue(np.array_equal(X, [[5]]))


class TXMFramesetTest(XanespyTestCase):
    """Set of python tests that work on full framesets and require data
    from multiple frames to make sense."""
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
                                      edge=k_edges['Ni_NCA'])

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
            transform_images(store.absorbances,
                             translations=np.array([[0, 0],[100, 100]]),
                             out=store.absorbances)
            old_imgs = store.absorbances.value
        # Check that reference_frame arguments of the wrong shape are rejected
        with self.assertRaisesRegex(Exception, "does not match shape"):
            self.frameset.align_frames(commit=False, reference_frame=0)
        # Perform an alignment but don't commit to disk
        self.frameset.align_frames(commit=False, reference_frame=(0, 0))
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

    def test_extent(self):
        expected = (-16.828125, 16.828125, -16.828125, 16.828125)
        self.assertEqual(self.frameset.extent('absorbances'), expected)

    def test_deferred_transformations(self):
        """Test that the system properly stores data transformations for later
        processing."""
        # Check that staged transforms are initially None
        self.assertTrue(self.frameset._translations is None)
        self.assertTrue(self.frameset._scales is None)
        self.assertTrue(self.frameset._rotations is None)
        # Stage some transformations
        self.frameset.stage_transformations(
            translations=np.array([[0, 0],[1, 1]]),
            scales=np.array([[1, 0.5]]),
            rotations=np.array([[0, 3]])
        )
        # Check that the transformations have been saved
        self.assertFalse(self.frameset._translations is None)
        self.assertFalse(self.frameset._scales is None)
        self.assertFalse(self.frameset._rotations is None)
        # Check that transformations accumulated
        self.frameset.stage_transformations(
            translations=np.array([[0, 0],[1, 1]]),
            scales=np.array([[1, 0.5]]),
            rotations=np.array([[0, 3]])
        )
        self.assertTrue(np.array_equal(self.frameset._translations,
                                       np.array([[0, 0],[2, 2]])))
        self.assertTrue(np.array_equal(self.frameset._scales,
                                       np.array([[1., 0.25]])))
        self.assertTrue(np.array_equal(self.frameset._rotations,
                                       np.array([[0, 6]])))
        # Check that transformations are reset after being applied
        self.frameset.apply_transformations(commit=True, crop=True)
        self.assertEqual(self.frameset._translations, None)
        self.assertEqual(self.frameset._scales, None)
        self.assertEqual(self.frameset._rotations, None)
        # Check that cropping was successfully applied
        with self.frameset.store() as store:
            new_shape = store.absorbances.shape
        self.assertEqual(new_shape, (1, 2, 1022, 1022))

    def test_spectrum(self):
        spectrum = self.frameset.spectrum()
        self.assertEqual(spectrum.shape, (2,))

    def test_calculate_clusters(self):
        """Check the the data are separated into signals and discretized by
        k-means clustering."""
        N_COMPONENTS = 3
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

    def test_switch_groups(self):
        """Test that switching between HDF5 groups works robustly."""
        old_group = self.frameset.data_name
        self.frameset.fork_data_group('new_group')
        self.frameset.data_name = old_group
        self.assertEqual(self.frameset.data_name, old_group)
        self.frameset.fork_data_group('new_group')



class XanesMathTest(XanespyTestCase):

    def setUp(self):
        # Prepare energies of the right dimensions
        self.KEdge = k_edges['Ni_NCA']
        K_Es = np.linspace(8250, 8640, num=61)
        K_Es = np.repeat(K_Es.reshape(1, 61), repeats=3, axis=0)
        self.K_Es = K_Es
        # Prepare L-edge energies of the right dimensions
        self.LEdge = l_edges['Ni_NCA']
        L_Es = np.linspace(844, 862, num=61)
        L_Es = np.repeat(L_Es.reshape(1, 61), repeats=3, axis=0)
        self.L_Es = L_Es
        prog.quiet = True

    def coins(self):
        """Prepare some example frames using images from the skimage
        library."""
        coins = np.array([data.coins() for i in range(0, 3*61)])
        coins = coins.reshape(3, 61, *data.coins().shape)
        # Adjust each frame to mimic an X-ray edge with a sigmoid
        S = 1/(1+np.exp(-(self.K_Es-8353))) + 0.1*np.sin(4*self.K_Es-4*8353)
        coins = (coins * S.reshape(3, 61,1,1))
        return coins

    def test_iter_indices(self):
        """Check that frame_indices method returns the right slices."""
        indata = np.zeros(shape=(3, 13, 256, 256))
        indices = iter_indices(indata, leftover_dims=1)
        self.assertEqual(len(list(indices)), 3*13*256)
        # Test with two leftover dimensions
        indices = iter_indices(indata, leftover_dims=2)
        self.assertEqual(len(list(indices)), 3*13)

    def test_guess_kedge_params(self):
        """Given an arbitrary K-edge spectrum, can we guess reasonable
        starting parameters for fitting?"""
        # Load spectrum
        spectrum = pd.read_csv(os.path.join(SSRL_DIR, 'NCA_xanes.csv'),
                               index_col=0, sep=' ', names=['Absorbance'])
        Es = np.array(spectrum.index)
        As = np.array(spectrum.values)[:,0]
        edge = k_edges['Ni_NCA']
        # Do the guessing
        result = guess_kedge(spectrum=As, energies=Es, edge=edge)
        # Check resultant guessed parameters
        self.assertAlmostEqual(result.scale, 0.2, places=2)
        self.assertAlmostEqual(result.voffset, 0.45, places=2)
        self.assertEqual(result.E0, edge.E_0)
        self.assertAlmostEqual(result.ga, 0.97, places=2)
        self.assertAlmostEqual(result.gb, 17, places=1)
        self.assertAlmostEqual(result.pre_m, 0, places=5)
        self.assertAlmostEqual(result.pre_b, 0, places=2)

    def test_apply_references(self):
        # Create some fake frames. Reshaping is to mimic multi-dim dataset
        Is, refs = self.coins()[:2], self.coins()[2:4]
        # Is = Is.reshape(1, 2, 303, 384)
        Is = [[0.1, 0.01],
              [0.001, 1]]
        Is = np.array([Is, Is])
        Is = Is.reshape(1, 2, 2, 2)
        refs = [[1, 1],
                [1, 1]]
        refs = np.array([refs, refs])
        refs = refs.reshape(1, 2, 2, 2)
        out = np.zeros_like(Is)
        # Apply actual reference function
        As = apply_references(Is, refs, out)
        self.assertEqual(As.shape, Is.shape)
        calculated = -np.log(Is/refs)
        self.assertTrue(np.array_equal(As, calculated))

    def test_direct_whiteline(self):
        """Check the algorithm for calculating the whiteline position of a
        XANES spectrum using the maximum value."""
        # Load some test data
        spectrum = pd.read_csv(os.path.join(SSRL_DIR, 'NCA_xanes.csv'),
                               index_col=0, sep=' ', names=['Absorbance'])
        # Calculate the whiteline position
        intensities = np.array([[spectrum['Absorbance'].values]])
        results = direct_whitelines(spectra=intensities,
                                    energies=np.array([spectrum.index]),
                                    edge=k_edges['Ni_NCA'])
        self.assertEqual(results, [8350.])

    def test_fit_kedge(self):
        prog.quiet = True
        # Load some test data
        spectrum = pd.read_csv(os.path.join(SSRL_DIR, 'NCA_xanes.csv'),
                               index_col=0, sep=' ', names=['Absorbance'])
        Is = np.array([spectrum.Absorbance.values])
        Es = np.array([spectrum.index])
        # Give an intial guess
        guess = KEdgeParams(1/5, -0.4, 8333,
                            1,
                            -0.0008, 0,
                            1, 14, 1)
        out = fit_kedge(Is, energies=Es, p0=guess)
        self.assertEqual(out.shape, (1, len(kedge_params)))
        out_params = KEdgeParams(*out[0])
        # Uncomment this plotting to check the fit if the test fails
        # import matplotlib.pyplot as plt
        # plt.plot(Es, Is, marker=".")
        # plt.plot(Es, predict_edge(Es, *out_params), marker="+")
        # plt.show()
        self.assertAlmostEqual(out_params.E0 + out_params.gb, 8350.50, places=2)

    def test_particle_labels(self):
        """Check image segmentation on a set of frames. These tests just check
        that input and output are okay and datatypes are correct; the
        accuracy of the results is not tested, this should be done in
        the jupyter-notebook.
        """
        # Prepare some images for segmentation
        coins = self.coins()
        result = particle_labels(frames=coins, energies=self.K_Es, edge=self.KEdge())
        expected_shape = coins.shape[-2:]
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.dtype, np.int)

    def test_edge_jump(self):
        """Check image masking based on the difference between the pre-edge
        and post-edge."""
        frames = self.coins()
        ej = k_edge_jump(frames, energies=self.K_Es, edge=self.KEdge())
        # Check that frames are reduced to a 2D image
        self.assertEqual(ej.shape, frames.shape[-2:])
        self.assertEqual(ej.dtype, np.float)

    def test_k_edge_mask(self):
        """Check that the edge jump filter can be successfully turned into a
        boolean."""
        frames = self.coins()
        ej = k_edge_mask(frames, energies=self.K_Es, edge=self.KEdge(), min_size=0)
        # Check that frames are reduced to a 2D image
        self.assertEqual(ej.shape, frames.shape[-2:])
        self.assertEqual(ej.dtype, np.bool)

    def test_l_edge_mask(self):
        """Check that the edge jump filter works for l edges."""
        frames = self.coins()  # NB: This mimics a K-edge, not an L-edge
        ej = l_edge_mask(frames, energies=self.L_Es,
                         edge=self.LEdge(), min_size=1)
        # Check that frames are reduced to a 2D image
        self.assertEqual(ej.shape, frames.shape[-2:])
        self.assertEqual(ej.dtype, np.bool)

    def test_transform_images(self):
        data = self.coins().astype('int')
        ret = transform_images(data)
        self.assertEqual(ret.dtype, np.float)
        # Test complex images
        data = self.coins().astype(np.complex)
        ret = transform_images(data)
        self.assertEqual(ret.dtype, np.complex)

    # def test_extract_signals(self):
    #     # Prepare some testing data
    #     x = np.linspace(0, 2*np.pi, num=2*np.pi*50)
    #     signal0 = np.sin(2*x)
    #     signal1 = np.sin(3*x)
    #     in_weights = np.array([[0, 0.25, 0.5, 0.75, 1],
    #                         [1, 0.75, 0.5, 0.25, 0]])
    #     features = np.outer(in_weights[0], signal0)
    #     features += np.outer(in_weights[1], signal1)
    #     self.assertEqual(features.shape, (5, 314))
    #     # Extract the signals
    #     comps, weights = extract_signals_nmf(spectra=features, energies=x,
    #                                      n_components=2)
    #     # Check the results
    #     np.testing.assert_allclose(comps, [signal0, signal1])


class UtilitiesTest(XanespyTestCase):

    def test_broadcast_reverse(self):
        orig = np.zeros(shape=(7, 48))
        target_shape = (7, 48, 958, 432)
        response = broadcast_reverse(orig, shape=target_shape)
        self.assertEqual(response.shape, target_shape)

    def test_interpret_complex(self):
        j = complex(0, 1)
        cmplx = np.array([[0+1j, 1+2j],
                          [2+3j, 3+4j]])
        mod = get_component(cmplx, 'modulus')
        np.testing.assert_array_equal(mod,
                                      np.array([[1, np.sqrt(5)],
                                                [np.sqrt(13), 5]]))
        # Check if real data works ok
        real = np.array([[0, 1],[1, 2]])
        np.testing.assert_array_equal(get_component(real, "modulus"), real)

    def test_xy_to_pixel(self):
        extent = Extent(
            left=-1000, right=-900,
            top=300, bottom=250
        )
        # Try an x-y value in the middle of a pixel
        result = xy_to_pixel(
            xy=xycoord(x=-975, y=272.5),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=4, horizontal=2))
        # Try an x-y value right on the edge of a pixel
        result = xy_to_pixel(
            xy=xycoord(x=-950, y=250),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=0, horizontal=5))

    def test_pixel_to_xy(self):
        extent = Extent(
            left=-1000, right=-900,
            top=300, bottom=250
        )
        result = pixel_to_xy(
            pixel=Pixel(vertical=9, horizontal=4),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, xycoord(x=-955, y=297.5))


class XradiaTest(XanespyTestCase):

    def test_pixel_size(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        xrm = XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl")
        self.assertAlmostEqual(xrm.um_per_pixel(), 0.03287, places=4)

    def test_timestamp_from_xrm(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        xrm = XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl")
        # Check start time
        start = dt.datetime(2016, 5, 29,
                            15, 2, 37,
                            tzinfo=pytz.timezone('US/Pacific'))
        self.assertEqual(xrm.starttime(), start)
        # Check end time (offset determined by exposure time)
        end = dt.datetime(2016, 5, 29,
                          15, 2, 37, 500000,
                          tzinfo=pytz.timezone('US/Pacific'))
        self.assertEqual(xrm.endtime(), end)
        xrm.close()

        # Test APS frame
        sample_filename = "fov03_xanesocv_8353_0eV.xrm"
        xrm = XRMFile(os.path.join(APS_DIR, sample_filename), flavor="aps")
        # Check start time
        start = dt.datetime(2016, 7, 2, 17, 50, 35, tzinfo=pytz.timezone('US/Central'))
        self.assertEqual(xrm.starttime(), start)
        # Check end time (offset determined by exposure time)
        end = dt.datetime(2016, 7, 2, 17, 51, 25, tzinfo=pytz.timezone('US/Central'))
        self.assertEqual(xrm.endtime(), end)
        xrm.close()

@unittest.skip("Probably not the right way to go about this")
class MultipleFramesetTest(XanespyTestCase):
    """Tests for how we can work with and combine multiple framesets."""
    hdf = os.path.join(PTYCHO_DIR, 'mock-ptycho-data.h5')

    def setUp(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)

    def tearDown(self):
        if os.path.exists(self.hdf):
            os.remove(self.hdf)

    def create_hdf_file(self):
        # Create some fake HDF5 file group for testing
        with h5py.File(self.hdf, mode="w"):
            pass

    def test_combine_framesets(self):
        """Test the method that merges mutliple framesets into one."""
        self.create_hdf_file()
        fs_list = [
            XanesFrameset(filename=self.hdf, groupname='group1', edge=None),
            XanesFrameset(filename=self.hdf, groupname='group1', edge=None),
        ]
        # Check that it fails if the group exists
        with self.assertRaises(exceptions.GroupKeyError):
            merge_framesets(fs_list, new_group="group1")
        # import_nanosurveyor_frameset(PTYCHO_DIR + "/", hdf_filename=self.hdf, quiet=True)



# Launch the tests if this is run as a script
if __name__ == '__main__':
    unittest.main()
