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
from unittest import TestCase, mock

import numpy as np

from xanespy.fitting import LinearCombination, L3Curve, prepare_p0


class FittingTestCase(TestCase):
    def test_linear_combination(self):
        # Prepare test sources
        x = np.linspace(0, 2*np.pi, num=361)
        sources = [np.sin(x), np.sin(2*x)]
        # Produce a combo with 0.5*sin(x) + 0.25*sin(2x) + 2
        lc = LinearCombination(sources=sources)
        out = lc(0.5, 0.25, 2)
        expected = 0.5*sources[0] + 0.25*sources[1] + 2
        np.testing.assert_equal(out, expected)
        # Test param_names property
        pnames = ('weight_0', 'weight_1', 'offset')
        self.assertEqual(lc.param_names, pnames)
    
    def test_L3_curve(self):
        # Prepare input data
        Es = np.arange(855, 871, 0.25)
        l3 = L3Curve(energies=Es, num_peaks=1)
        # confirm right number of param names
        names = ('height_0', 'center_0', 'sigma_0',
                 'sig_height', 'sig_center', 'sig_sigma',
                 'offset')
        self.assertEqual(l3.param_names, names)
        # Fit a 1-peak curve
        params = (1, 860, 0.1, 0, 862.5, 1, -2)
        out = l3(*params)
        self.assertEqual(np.argmax(out), 20)
    
    def test_prepare_p0(self):
        # Run the function with known inputs
        p0 = (5, 3, 1)
        out = prepare_p0(p0, num_timesteps=4, frame_shape=(256, 128))
        # Prepare expected value
        expected = np.empty(shape=(4, 3, 256, 128))
        expected[:,0,:,:] = 5
        expected[:,1,:,:] = 3
        expected[:,2,:,:] = 1
        # Check that the arrays match
        np.testing.assert_equal(out, expected)
