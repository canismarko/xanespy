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

import math
import os
import warnings
import multiprocessing as mp
from functools import partial

import matplotlib
import unittest
import numpy as np
from skimage import data
import pandas as pd

from xanespy import exceptions, edges
from xanespy.utilities import prog
from xanespy.xanes_math import (transform_images, direct_whitelines,
                                particle_labels, k_edge_jump,
                                k_edge_mask, l_edge_mask,
                                apply_references, iter_indices,
                                extract_signals_nmf,
                                transformation_matrices,
                                apply_internal_reference,
                                apply_mosaic_reference,
                                register_template, register_correlations,
                                downsample_array, FramesPool,
                                resample_image, crop_image)

TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')


# Function for testing multiprocessing
# Must be here so it is pickle-able
def add_one(x):
    return x + 1


class XanesMathTest(unittest.TestCase):
    
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
        # Add some noise otherwise some functions div by zero.
        coins = coins * (0.975 + np.random.rand(*coins.shape)/20)
        coins = coins.astype(np.int32)
        return coins
    
    def test_resample_image(self):
        original = data.horse()
        # Test simple cropping with no resampling
        new_shape = (int(original.shape[0] * 0.5), int(original.shape[1] * 0.5))
        resized = resample_image(original, new_shape=new_shape,
                                 src_dims=(1, 1), new_dims=(0.5, 0.5))
        self.assertEqual(resized.shape, new_shape)
        # Test sample resizing with no cropping
        new_shape = (int(original.shape[0] * 2), int(original.shape[1] * 2))
        resized = resample_image(original, new_shape=new_shape,
                                 src_dims=(1, 1), new_dims=(1, 1))
        self.assertEqual(resized.shape, new_shape)
    
    def test_crop_image(self):
        original = data.horse()
        # Test simple cropping
        cropped = crop_image(original, (64, 64), center=(164, 200))
        expected = original[132:196,168:232]
        np.testing.assert_equal(cropped, expected)
        # Test with a center outside the window
        cropped = crop_image(original, (64, 64), center=(30, 380))
        expected = original[:64,336:]
        np.testing.assert_equal(cropped, expected)
        # Test what happens if the target is bigger than the destination
        with self.assertRaises(exceptions.XanesMathError):
            cropped = crop_image(original, (600, 600))
    
    def test_downsample_array(self):
        """Check that image downsampling works as expected."""
        coins = data.coins()
        # Test for exception on negative factors
        with self.assertRaises(ValueError):
            downsample_array(coins, factor=-1)
        # Test for exception on invalid method
        with self.assertRaises(ValueError):
            downsample_array(coins, factor=1, method='ritual sacrifice')
        # Test simple downsampling
        output = downsample_array(coins, factor=1)
        self.assertEqual(output.shape, (151, 192))
        # Test no downsampling (factor=0) returns original array
        output = downsample_array(coins, factor=0)
        self.assertIs(output, coins)
        # Test a 3D array
        coins3d = np.broadcast_to(coins, (5, *coins.shape))
        output = downsample_array(coins3d, factor=1)
        self.assertEqual(output.shape, (2, 151, 192))
        # Test a 3D array with only certain axes
        coins3d = np.broadcast_to(coins, (5, *coins.shape))
        output = downsample_array(coins3d, factor=1, axis=(1, 2))
        self.assertEqual(output.shape, (5, 151, 192))
    
    def test_mosiac_reference(self):
        """Check that a repeated mosaic of images is converted to optical
        depth.
        
        """
        # Prepare data
        coins = data.coins()
        mosaic = np.tile(coins, (4, 4))
        ref = np.random.rand(*coins.shape) + 1
        od = -np.log(coins/ref)
        expected = np.tile(od, (4, 4))
        # Call the reference correction function
        result = apply_mosaic_reference(mosaic, ref)
        np.testing.assert_almost_equal(result, expected)
    
    def test_register_correlation(self):
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
        frames = np.array([[frame0, frame1]])
        # Do the actual registration
        results = register_correlations(frames, reference=frame0, desc=None)
        # Compare results
        expected = np.array([[[0, 0], [1, 1]]])
        np.testing.assert_equal(results, expected)
        # Try again but with a median filter size
        results = register_correlations(frames, reference=frame0, desc=None,
                                        median_filter_size=3)
        expected = np.array([[[0.0, 0.0],
                              [1.2, 1.2]]])
        np.testing.assert_equal(results, expected)
    
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
        frames = np.array([[frame0, frame1]])
        template = np.array([
            [2, 2, 2],
            [2, 6, 2],
            [2, 2, 2],
        ])
        # Do the actual registration
        results = register_template(frames, reference=frame0,
                                    desc=None, template=template)
        # Compare results
        expected = np.array([[[0, 0], [1, 1]]])
        np.testing.assert_equal(results, expected)
        # Try again but with a median filter size
        template = np.array([
            [1, 1],
            [1, 1],
        ])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Median filtering')
            results = register_template(frames, reference=frame0, desc=None,
                                        template=template, median_filter_size=2)
        expected = np.array([[[0.0, 0.0],
                              [0.0, 0.0]]]) # This doesn't work great for some reason
        np.testing.assert_equal(results, expected)
    
    def test_frame_pool(self):
        # Create a (1, 2, 2, 2) array to test
        a = np.array([[
            [[1, 2],
             [3, 4],
             [5, 6]],
            [[7, 8],
             [9, 10],
             [11, 12]]
        ]])
        # Should produce a (1, 2) array of results
        expected = np.sum(a, axis=(2, 3))
        with FramesPool() as pool:
            out = pool.map(np.sum, a)
        # Check the result
        np.testing.assert_array_equal(out, expected)
    
    def test_apply_internal_reference(self):
        # With real data
        d = np.array([[
            [1., 1.,  1.,   1.,  1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 0.5, 0.05, 0.5, 1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 1.,  1.,   1.,  1.],
        ]], dtype='float64')
        result = apply_internal_reference(d, desc=None)
        expected = np.array([[
            [0., 0.,  0.,   0.,  0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.7, 3,    0.7, 0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.,  0.,   0.,  0.],
        ]], dtype='float64')
        np.testing.assert_almost_equal(result, expected, decimal=2)
        # With complex data
        j = complex(0, 1)
        pi = np.pi
        in_modulus = np.array([[
            [1., 1.,  1.,   1.,  1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 0.5, 0.05, 0.5, 1.],
            [1., 0.5, 0.5,  0.5, 1.],
            [1., 1.,  1.,   1.,  1.],
        ]])
        in_phase = np.pi * np.array([[
            [1/4, 1/4, 1/4, 1/4, 1/4,],
            [1/4, 1/2, 1/2, 1/2, 1/4,],
            [1/4, 3/4, 3/4, 3/4, 1/4,],
            [1/4, 1/2, 1/2, 1/2, 1/4,],
            [1/4, 1/4, 1/4, 1/4, 1/4,],
        ]])
        in_data = in_modulus * (np.cos(in_phase) + j * np.sin(in_phase))
        result = apply_internal_reference(in_data, desc=None)
        OD_expected = np.array([[
            [0., 0.,  0.,   0.,  0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.7, 3,    0.7, 0.],
            [0., 0.7, 0.7,  0.7, 0.],
            [0., 0.,  0.,   0.,  0.],
        ]], dtype='float64')
        np.testing.assert_almost_equal(np.real(result), OD_expected, decimal=2)
    
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
        # Rotation not around a non-zero center
        res = transformation_matrices(rotations=r, center=(512, 512))
        expected = np.array([
            [0, -1, 1024],
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
        indices = iter_indices(indata, leftover_dims=1, quiet=True)
        self.assertEqual(len(list(indices)), 3*13*256)
        # Test with two leftover dimensions
        indices = iter_indices(indata, leftover_dims=2, quiet=True)
        self.assertEqual(len(list(indices)), 3*13)
    
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
        As = apply_references(Is, refs, quiet=True)
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
                                    edge=edges.k_edges['Ni_NCA'], quiet=True)
        self.assertEqual(results, [8350.])
    
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
        bad_edge = self.KEdge()
        bad_edge.post_edge = (np.max(self.K_Es) + 1, np.max(self.K_Es) + 10)
        with self.assertRaisesRegex(exceptions.XanesMathError, "post-edge"):
            ej = k_edge_jump(frames[..., 0:20],
                             energies=self.K_Es[..., 0:20],
                             edge=bad_edge)
    
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
        ret = transform_images(data, transformations=Ts, quiet=True)
        self.assertEqual(ret.dtype, np.float)
        # Test complex images
        coins = self.coins()
        data_imag = coins + coins * complex(0, 1)
        ret = transform_images(data_imag, transformations=Ts, quiet=True)
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
