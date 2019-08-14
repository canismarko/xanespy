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

from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from xanespy import plots


class PlotSpectrumTest(TestCase):

    def setUp(self):
        self.fig = plt.figure()
    
    def tearDown(self):
        plt.close(self.fig)
    
    def test_basic_plot(self):
        Es = np.linspace(8353, 8560, num=20)
        ODs = np.sin(Es /2)
        artist = plots.plot_spectrum(ax=self.fig.gca(), energies=Es, spectrum=ODs)
        self.assertIsInstance(artist[0], plt.Line2D)

    def test_pandas_index(self):
        Es = pd.Float64Index([8324.0, 8354.0])
        ODs =  [0.5837463, 0.690377]
        artist = plots.plot_spectrum(ax=self.fig.gca(), energies=Es, spectrum=ODs)
    
    def test_colors(self):
        Es = np.linspace(8353, 8560, num=20)
        ODs = np.sin(Es /2)
        cmap = plt.cm.get_cmap('plasma')
        blue = cmap(0)
        yellow = cmap(256)
        # Colors by x-coordinate
        artists = plots.plot_spectrum(ax=self.fig.gca(), energies=Es, spectrum=ODs, color='x')
        line, scatter = artists
        colors = scatter.get_facecolor()
        np.testing.assert_array_equal(colors[0], blue)
        np.testing.assert_array_equal(colors[-1], yellow)
        # Colors by y-coordinate
        artists = plots.plot_spectrum(ax=self.fig.gca(), energies=Es, spectrum=ODs, color='y')
        line, scatter = artists
        colors = scatter.get_facecolor()
        real_norm = plt.Normalize()
        real_norm.autoscale(ODs)
        np.testing.assert_array_equal(colors[0], cmap(real_norm(ODs[0])))
        np.testing.assert_array_equal(colors[-1], cmap(real_norm(ODs[-1])))


class ScaleNormalizerTest(TestCase):
    def test_scale_none(self):
        values = (2, 5)
        new_norm = plots.scale_normalizer(None, values)
        self.assertIsInstance(new_norm, plt.Normalize)
        self.assertEqual(new_norm.vmin, 2)
        self.assertEqual(new_norm.vmax, 5)
    
    def test_scale_partial_normalizer(self):
        values = (2, 5)
        old_norm = plt.Normalize(1, None)
        new_norm = plots.scale_normalizer(old_norm, values)
        self.assertIsInstance(new_norm, plt.Normalize)
        self.assertEqual(new_norm.vmin, 1)
        self.assertEqual(new_norm.vmax, 5)
