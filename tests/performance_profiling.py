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
# along with Xanespy. If not, see <http://www.gnu.org/licenses/>.

"""Performance Profiling Check how fast some of the
performance-sensitive parts of the Xanespy run. These are generally
looping through some stack of arrays or spectra.
"""

from timeit import timeit
import random
import os

import numpy as np
from skimage import data, transform

import xanespy
from xanespy.xanes_math import register_correlations

# Prepare test image stack
stack = []
im = data.camera()
for i in range(0, 60):
    # Generate a random shift
    d = 15
    shift = (random.randint(-d, d), random.randint(-d, d))
    matrix = transform.SimilarityTransform(translation=shift)
    new_im = transform.warp(im, matrix, mode='symmetric')
    stack.append(new_im)

# Register Translations
# ---------------------
# Prepare test image stack
# stack = []
# camera = data.camera()
# for i in range(0, 60):
#     # Generate a random shift
#     d = 15
#     shift = (random.randint(-d, d), random.randint(-d, d))
#     matrix = transform.SimilarityTransform(translation=shift)
#     new_im = transform.warp(camera, matrix, mode='symmetric')
#     stack.append(new_im)
# stack = np.array(stack)

# # Execute registration
# num = 3
# def run():
#     res = register_correlations(stack, reference=camera)
# time = timeit(run, number=num)
# print("{:.2f} s/run".format(time/num))

# Import files
TEST_DIR = os.path.dirname(__file__)
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
hdfname = os.path.join(APS_DIR, 'testdata.h5')
xanespy.import_aps_8BM_frameset(APS_DIR, hdf_filename=hdfname, quiet=False)
