#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 Mark Wolf
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

import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from xanespy import edges
from xanespy.beamlines import (sector8_xanes_script, ssrl6_xanes_script,
                               Zoneplate, ZoneplatePoint, Detector)
from xanespy.utilities import (xycoord, prog, position, Extent,
                               xy_to_pixel, pixel_to_xy,
                               get_component, Pixel, broadcast_reverse)

TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')


class ZoneplateTest(unittest.TestCase):
    def setUp(self):
        # Values taken from APS beamtime on 2015-11-11
        self.aps_zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            x_step=0, y_step=0,
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
        self.assertAlmostEqual(zp.step.z, 9.9329 / 2)
    
    def test_z_from_energy(self):
        result = self.aps_zp.position(energy=8315).z
        self.assertAlmostEqual(result, 3120.6329)
    
    def test_position(self):
        result = self.aps_zp.position(energy=8315)
        np.testing.assert_almost_equal(result, (0, 0, 3120.6329))
        # for actual, expected in zip(result, ):
        #     self.assertAlmostEqual(actual, expected)
        result = self.ssrl_zp.position(energy=8352)
        np.testing.assert_almost_equal(result, (-4.38, -1.46, -743.23), decimal=2)
        # self.assertAlmostEqual(result, (-4.38, -1.46, -743.23))
        # self.assertApproximatelyEqual(result.x, 0)
        # self.assertApproximatelyEqual(result.y, 0)
        # self.assertApproximatelyEqual(result.z, 3120.6329)


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
            x_step=1,
            y_step=1,
            z_step=9.9329 / 2 # Original script assumed 2eV steps
        )
        self.det = Detector(
            start=ZoneplatePoint(x=0, y=0, z=389.8, energy=8313),
            x_step=1,
            y_step=1,
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
        self.assertEqual(lines[54], 'moveto detx -63.00\n')
        self.assertEqual(lines[55], 'moveto dety -63.00\n')
        self.assertEqual(lines[56], 'moveto detz 377.59\n')
    
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
        self.assertEqual(lines[57].strip(), "moveto x 1653.00")
        self.assertEqual(lines[58].strip(), "moveto y -1727.00")
        self.assertEqual(lines[59].strip(), "moveto z 0.00")
        # Check that the energy approach lines are in tact
        self.assertEqual(lines[2].strip(), "moveto energy 8150.00")
        self.assertEqual(lines[51].strip(), "moveto energy 8248.00")
        # Check that energy is set
        self.assertEqual(lines[52].strip(), "moveto energy 8250.00")
        # Check that zone-plate and detector are set
        self.assertEqual(lines[53].strip(), "moveto zpz 2797.81")
        self.assertEqual(lines[54].strip(), "moveto detx -63.00")
        self.assertEqual(lines[55].strip(), "moveto dety -63.00")
        self.assertEqual(lines[56].strip(), "moveto detz 377.59")
        # Check that collect command is sent
        self.assertEqual(
            lines[60].strip(),
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
        self.assertEqual(lines[243], "moveto x 1706.20\n")
        self.assertEqual(lines[244], "moveto y -1927.20\n")
    
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
            lines[60].strip(),
            "collect test_sample_xanesocv_8250_0eV.xrm"
        )
        self.assertEqual(
            lines[1581].strip(),
            "collect test_sample_xanes02_8342_0eV.xrm"
        )
    
    def test_file_list(self):
        """Check that we can save a tab separated values file with all the
        filenames and meta data in it.
        
        """
        # Run the function to create the Xradia script
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
        # See if a file was created
        base, ext = os.path.splitext(self.output_path)
        tsv_filename = base + '.tsv'
        self.assertTrue(os.path.exists(tsv_filename), tsv_filename + " not created")
