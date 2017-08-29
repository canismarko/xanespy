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

import matplotlib
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning, 1405)
    matplotlib.use('Agg')
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
                                predict_edge, fit_kedge, kedge_params,
                                KEdgeParams, extract_signals_nmf,
                                guess_kedge, transformation_matrices,
                                apply_internal_reference,
                                apply_mosaic_reference,
                                register_template, _fit_spectrum,
                                fit_linear_combinations)

TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')


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
        return coins

    def test_predict_edge(self):
        params = KEdgeParams(
            scale=1, voffset=0, E0=8353,
            sigw=1,
            bg_slope=0,
            ga=0, gb=1, gc=1,
        )
        x = np.linspace(8330, 8440)
        result = predict_edge(x, *params)
        # Predict the expected result for arctan function
        expected = np.arctan(x-8353) / np.pi + 1/2
        np.testing.assert_equal(result, expected)
    
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
            bg_slope=0,
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
            bg_slope=0,
            ga=0, gb=1, gc=1,
        )
        func = _fit_spectrum(params)
        result = func(predicted, Es)
        np.testing.assert_equal(result, [np.nan] * 8)


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
        result = apply_internal_reference(in_data)
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
    
    def test_lc_fitting(self):
        """Can we fit a linear combination of functions to the given spectra."""
        # Prepare a combination of component spectra
        t = np.linspace(0, 2*np.pi, num=360)
        s1 = np.sin(t)
        s2 = np.sin(4*t)
        noise = 0.3 * np.random.rand(360)
        spectra = np.array([
            1*s1 + 0.5*s2 + noise + 4,
            0.3*s1 + 0.1*s2 + noise + 2,
        ])
        # Execute the test
        weights = fit_linear_combinations(spectra, np.array([s1, s2]))
        # Check that the calculated weights are correct
        self.assertEqual(weights.shape, (2, 3))
        expected = np.array([[1, 0.5],[0.3, 0.1]])
        np.testing.assert_equal(weights, expected)
    
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
        self.assertAlmostEqual(result.scale, 0.214, places=2)
        self.assertAlmostEqual(result.voffset, 0.45, places=2)
        self.assertEqual(result.E0, edge.E_0)
        self.assertAlmostEqual(result.ga, 0.9, places=2)
        self.assertAlmostEqual(result.gb, 17, places=1)
        self.assertAlmostEqual(result.bg_slope, 0, places=5)
    
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
        guess = KEdgeParams(1/5, 0, 8333, # Scale, v_offset, E_0
                            1, # Sigmoid sharpness
                            0, # Slope
                            1, 17, 4) # Gaussian height, center and width
        out = fit_kedge(Is, energies=Es, p0=guess)
        self.assertEqual(out.shape, (1, len(kedge_params)))
        out_params = KEdgeParams(*out[0])
        # Uncomment these plotting commands to check the fit if the test fails
        # guess0 = guess_kedge(spectrum=Is[0], energies=Es[0],
        #                      edge=edges.k_edges['Ni_NCA'])
        # print(guess0)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(Es, Is, marker="+")
        # new_Es = np.linspace(8250, 8640, num=200)
        # plt.plot(new_Es, predict_edge(new_Es, *out_params), marker="+", linestyle="-")
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
        self.assertEqual(result.dtype, np.int32)
    
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
        coins = self.coins()
        data_imag = coins + coins * complex(0, 1)
        ret = transform_images(data_imag, transformations=Ts)
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
