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

import pytz

from xanespy.xradia import XRMFile



TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')


class XradiaTest(TestCase):
    
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

    def test_str_and_repr(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        xrm = XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl")
        self.assertEqual(repr(xrm), "<XRMFile: '{}'>".format(sample_filename))
        self.assertEqual(str(xrm), "<XRMFile: '{}'>".format(sample_filename))

    def test_binning(self):
        sample_filename = "rep01_20161456_ssrl-test-data_08324.0_eV_001of003.xrm"
        xrm = XRMFile(os.path.join(SSRL_DIR, sample_filename), flavor="ssrl")
        self.assertEqual(xrm.binning(), (2, 2))
