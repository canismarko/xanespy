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

from unittest import TestCase, mock
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from xanespy.utilities import (Extent, xy_to_pixel, xycoord, Pixel,
                               pixel_to_xy, get_component,
                               broadcast_reverse, is_kernel)


class UtilitiesTest(TestCase):

    def test_broadcast_reverse(self):
        orig = np.zeros(shape=(7, 48))
        target_shape = (7, 48, 958, 432)
        response = broadcast_reverse(orig, shape=target_shape)
        self.assertEqual(response.shape, target_shape)

    def test_interpret_complex(self):
        j = complex(0, 1)
        cmplx = np.array([[0+1j, 1+2j],
                          [2+3j, 3+4j]])
        # Check modulus
        result = get_component(cmplx, 'modulus')
        mod = np.array([[1, np.sqrt(5)],
                        [np.sqrt(13), 5]])
        np.testing.assert_array_equal(result, mod)
        # Check phase
        result = get_component(cmplx, 'phase')
        phase = np.array([[math.atan2(1, 0), math.atan2(2, 1)],
                          [math.atan2(3, 2), math.atan2(4, 3)]])
        np.testing.assert_array_equal(result, phase)
        # Check real component
        result = get_component(cmplx, 'real')
        real = np.array([[0, 1],
                         [2, 3]])
        np.testing.assert_array_equal(result, real)
        # Check imaginary component
        result = get_component(cmplx, 'imag')
        imag = np.array([[1, 2],
                         [3, 4]])
        np.testing.assert_array_equal(result, imag)
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
        self.assertEqual(result, Pixel(vertical=6, horizontal=2))
        # Try an x-y value right on the edge of a pixel
        result = xy_to_pixel(
            xy=xycoord(x=-950, y=250),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=9, horizontal=5))
        # Try an x-y value at the edge of the image
        result = xy_to_pixel(
            xy=xycoord(x=-900, y=250),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=9, horizontal=9))
        result = xy_to_pixel(
            xy=xycoord(x=-1000, y=300),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=0, horizontal=0))
        
    
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
        self.assertEqual(result, xycoord(x=-955., y=252.5))
