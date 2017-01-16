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
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        if os.path.exists(self.scaninfo_path):
            os.remove(self.scaninfo_path)

    def test_bad_arguments(self):
        """Check that incompatible arguments to the script generator results
        in exceptions being raised.
        """
        # Abba_mode and frame_rest are incompatible
        with self.assertRaisesRegex(ValueError, "frame_rest.+abba_mode"):
            ssrl6_xanes_script(dest=None,
                               edge=edges.k_edges["Ni_NCA"],
                               binning=2,
                               zoneplate=self.zp,
                               iterations=["Test0", "Snorlax"],
                               frame_rest=5,
                               repetitions=8,
                               ref_repetitions=15,
                               positions=[position(3, 4, 5)],
                               reference_position=position(0, 1, 2),
                               abba_mode=True)

    def test_scaninfo_generation(self):
        """Check that the script writes all the filenames to a ScanInfo file
        for TXM Wizard."""
        with open(self.output_path, 'w') as f:
            ssrl6_xanes_script(dest=f,
                               edge=edges.k_edges["Ni_NCA"],
                               binning=2,
                               zoneplate=self.zp,
                               iterations=["Test0", "Snorlax"],
                               frame_rest=0,
                               repetitions=8,
                               ref_repetitions=15,
                               iteration_rest=5,
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
                               edge=edges.k_edges["Ni_NCA"],
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
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
    
    def test_file_created(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=edges.k_edges["Ni_NCA"],
                                 zoneplate=self.zp, detector=self.det,
                                 names=["test_sample"], sample_positions=[])
        # Check that a file was created
        self.assertTrue(
            os.path.exists(self.output_path)
        )
    
    def test_binning(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=edges.k_edges["Ni_NCA"],
                                 binning=2, zoneplate=self.zp,
                                 detector=self.det, names=[],
                                 sample_positions=[])
        with open(self.output_path, 'r') as f:
            firstline = f.readline().strip()
        self.assertEqual(firstline, "setbinning 2")
    
    def test_exposure(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=edges.k_edges["Ni_NCA"],
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
            sector8_xanes_script(dest=f, edge=edges.k_edges['Ni_NCA'],
                                 zoneplate=self.zp, detector=self.det,
                                 names=[], sample_positions=[])
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        # Check that several energies are scanned through
        self.assertEqual(lines[2], 'moveto energy 8150.00\n')
        self.assertEqual(lines[3], 'moveto energy 8152.00\n')
        self.assertEqual(lines[51], 'moveto energy 8248.00\n')
        self.assertEqual(lines[52], 'moveto energy 8250.00\n')
        # Check that the first zone plate is properly set
        self.assertEqual(lines[53], 'moveto zpz 2797.81\n')
        self.assertEqual(lines[54], 'moveto detz 377.59\n')
    
    def test_first_frame(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=edges.k_edges['Ni_NCA'],
                sample_positions=[position(x=1653, y=-1727, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                names=["test_sample"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        # Check that x, y are set
        self.assertEqual(lines[55].strip(), "moveto x 1653.00")
        self.assertEqual(lines[56].strip(), "moveto y -1727.00")
        self.assertEqual(lines[57].strip(), "moveto z 0.00")
        # Check that the energy approach lines are in tact
        self.assertEqual(lines[2].strip(), "moveto energy 8150.00")
        self.assertEqual(lines[51].strip(), "moveto energy 8248.00")
        # Check that energy is set
        self.assertEqual(lines[52].strip(), "moveto energy 8250.00")
        # Check that zone-plate and detector are set
        self.assertEqual(lines[53].strip(), "moveto zpz 2797.81")
        self.assertEqual(lines[54].strip(), "moveto detz 377.59")
        # Check that collect command is sent
        self.assertEqual(
            lines[58].strip(),
            "collect test_sample_xanes0_8250_0eV.xrm"
        )
    
    def test_second_location(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=edges.k_edges['Ni_NCA'],
                sample_positions=[position(x=1653, y=-1727, z=0),
                                  position(x=1706.20, y=-1927.20, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                names=["test_sample", "test_reference"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[242], "moveto x 1706.20\n")
        self.assertEqual(lines[243], "moveto y -1927.20\n")
    
    def test_multiple_iterations(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=edges.k_edges['Ni_NCA'],
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
            lines[1361].strip(),
            "collect test_sample_xanes02_8342_0eV.xrm"
        )


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
        with self.assertRaises(exceptions.CreateGroupError):
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
            fs.align_frames(blur="bad-blur")
        # Bad method
        with self.assertRaises(ValueError):
            fs.align_frames(method="bad-method")

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
    
    def test_has_representation(self):
        self.assertTrue(
            self.frameset.has_representation('intensities'))
        self.assertFalse(
            self.frameset.has_representation('not-real-data'))
        self.assertFalse(
            self.frameset.has_representation(None))

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


class XanesMathTest(XanespyTestCase):

    def setUp(self):
        # Prepare energies of the right dimensions
        self.KEdge = edges.NCANickelKEdge
        K_Es = np.linspace(8250, 8640, num=61)
        K_Es = np.repeat(K_Es.reshape(1, 61), repeats=3, axis=0)
        self.K_Es = K_Es
        # Prepare L-edge energies of the right dimensions
        self.LEdge = edges.NCANickelLEdge
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

    def test_predict_edge(self):
        params = KEdgeParams(
            scale=1, voffset=0, E0=8353,
            sigw=1,
            pre_m=0, pre_b=0,
            ga=0, gb=1, gc=1,
        )
        x = np.linspace(8330, 8440)
        result = predict_edge(x, *params)
        # Predict the expected result for arctan function
        expected = np.arctan(x-8353) / np.pi + 1/2
        np.testing.assert_equal(result, expected)

    def test_register_template(self):
        # Prepare some sample data arrays for registration
        frame0 = np.array([
            [1, 1, 1, 0, 0],
            [1, 3, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        frame1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 3, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        frames = np.array([frame0, frame1])
        template = np.array([
            [2, 2, 2],
            [2, 6, 2],
            [2, 2, 2],
        ])
        # Do the actual registration
        results = register_template(frames, reference=frame0, template=template)
        # Compare results
        expected = np.array([[0, 0], [1, 1]])
        np.testing.assert_equal(results, expected)


    def test_fit_spectrum(self):
        params = KEdgeParams(
            scale=1, voffset=0, E0=8353,
            sigw=1,
            pre_m=0, pre_b=0,
            ga=0, gb=1, gc=1,
        )
        Es = np.linspace(8250, 8640)
        predicted = predict_edge(Es, *params)
        # Fit against the initial parameters and check that it returns
        # the same params
        func = _fit_spectrum(params)
        result = func(predicted, Es)
        np.testing.assert_equal(result, params)
        # Test ridiculously bad fits return `nan`
        params = KEdgeParams(
            scale=0, voffset=0, E0=0,
            sigw=1,
            pre_m=0, pre_b=0,
            ga=0, gb=1, gc=1,
        )
        func = _fit_spectrum(params)
        result = func(predicted, Es)
        np.testing.assert_equal(result, [np.nan] * 9)


    def test_apply_internal_reference(self):
        # With real data
        d = np.array([[
            [1., 1.,  1.,   1.,  1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 0.5, 0.05, 0.5, 1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 1.,  1.,   1.,  1.],
        ]], dtype='float64')
        result = apply_internal_reference(d)
        expected = np.array([[
            [0., 0.,  0.,   0.,  0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.7, 3,    0.7, 0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.,  0.,   0.,  0.],
        ]], dtype='float64')
        np.testing.assert_almost_equal(result, expected, decimal=2)
        # With complex data
        d = np.array([[
            [1., 1.,  1.,   1.,  1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 0.5, 0.05, 0.5, 1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 1.,  1.,   1.,  1.],
        ]], dtype='complex128')
        result = apply_internal_reference(d)
        expected = np.array([[
            [0., 0.,  0.,   0.,  0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.7, 3,    0.7, 0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.,  0.,   0.,  0.],
        ]], dtype='float64')
        expected = np.zeros_like(expected) + complex(0, 1) * expected
        np.testing.assert_almost_equal(result, expected, decimal=2)

    def test_transformation_matrices(self):
        r = np.array([math.pi/2])
        s = np.array([0.5, 0.75])
        t = np.array([20, 25])
        # Rotation only
        res = transformation_matrices(rotations=r)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])
        np.testing.assert_almost_equal(res, expected)
        # Translation only
        res = transformation_matrices(translations=t)
        expected = np.array([
            [1, 0, 20],
            [0, 1, 25],
            [0, 0,  1],
        ])
        np.testing.assert_almost_equal(res, expected)
        # Scaling only
        res = transformation_matrices(scales=s)
        expected = np.array([
            [0.5, 0,    0],
            [0,   0.75, 0],
            [0,   0,    1],
        ])
        np.testing.assert_almost_equal(res, expected)
        # No transformations should raise a value error
        with self.assertRaises(ValueError):
            transformation_matrices()

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
        edge = edges.NCANickelKEdge()
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
        # out = np.zeros_like(Is)
        # Apply actual reference function
        As = apply_references(Is, refs)
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
                                    edge=edges.k_edges['Ni_NCA'])
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
        # Check that it fails with imcomplatible input shapes
        with self.assertRaises(ValueError):
            ej = k_edge_jump(frames[0:-1], energies=self.K_Es, edge=self.KEdge())
        # Check that it fails with no post-edge energies
        with self.assertRaisesRegex(exceptions.XanesMathError, "post-edge"):
            ej = k_edge_jump(frames[..., 0:20],
                             energies=self.K_Es[..., 0:20],
                             edge=self.KEdge())

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
        Ts = np.identity(3)
        # Check that datatypes are handled properly
        Ts = np.broadcast_to(Ts, shape=(*data.shape[0:2], 3, 3))
        ret = transform_images(data, transformations=Ts)
        self.assertEqual(ret.dtype, np.float)
        # Test complex images
        data = self.coins().astype(np.complex)
        ret = transform_images(data, transformations=Ts)
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


# Launch the tests if this is run as a script
if __name__ == '__main__':
    unittest.main()
