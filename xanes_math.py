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

"""Module containing all the computationally demanding functions. This
allows for easy optimization of parallelizable algorithms. Most
functions will operate on large arrays of data, so this file can be
compiled to C code using Cython.
"""

import warnings

import numpy as np

from skimage import transform, feature


# Helpers for parallelizable code

import sys
import time
import threading
import multiprocessing
from itertools import count

CPU_COUNT = multiprocessing.cpu_count()

def foreach(f,l,threads=CPU_COUNT,return_=False):
    """
    Apply f to each element of l, in parallel
    """

    if threads>1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = zip(count(),l.__iter__())
        else:
            i = l.__iter__()


        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = next(i)
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n,x = v
                        d[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()
        threadlist = [threading.Thread(target=runall) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise (a, b, c)
        if return_:
            r = list(d.items())
            r.sort()
            return [v for (n,v) in r]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return

def parallel_map(f,l,threads=CPU_COUNT):
    return foreach(f,l,threads=threads,return_=True)


def normalize_Kedge(spectra, energies, E_0, pre_edge, post_edge, post_edge_order=2):
    """Normalize the K-edge XANES spectrum so that the pre-edge is linear
    and the EXAFS region projects to an absorbance of 1 and the edge
    energy (E_0). Algorithm is taken mostly from the Atena 4 manual.

    Returns: Normalized spectra as a numpy array of the same shape as input spectra.

    Arguments
    ---------

    - spectra : Numpy array where the last dimension represents
      energy, which should be the same length as `energies` argument.

    - energies : 1-D numpy array with all the energies in electron-volts.

    - E_0 : Energy of the actual edge in electron-volts.

    - pre_edge : 2-tuple with range of energies that represent the
      pre-edge region.

    - post_edge : 2-tuple with range of energies that represent the
      post-edge region.

    - post_edge_order : The order of polynomial to use for fitting the
      post-edge region. Ex: 2 (default) means a quadratic function is
      used.
    """
    warnings.warn(UserWarning("xanes_math.normalize_Kedge not implemented"))
    # Flatten the spectra to be a two-dimensional array for looping
    orig_shape = spectra.shape
    spectra = spectra.reshape(-1, spectra.shape[-1])
    # Fit the pre-edge region
    ## TODO
    # Fit the post-edge region
    ## TODO
    # Restore original shape
    spectra = spectra.reshape(orig_shape)
    return spectra


def direct_whitelines(spectra, energies, edge):
    """Takes an array of X-ray absorbance spectra and calculates the
    positions of maximum intensities over the near-edge region.

    Arguments
    ---------

    - spectra : 2D numpy array of absorbance spectra where the last
      dimension is energy.

    - energies : 1D array of X-ray energies in electron-volts.

    - edge : An XAS Edge object that describes the absorbance edge in
      question.
    """
    # Cut down to only those values on the edge
    edge_mask = (energies > edge.edge_range[0]) & (energies < edge.edge_range[1])
    spectra = spectra[...,edge_mask]
    energies = energies[edge_mask]
    # Calculate the whiteline positions
    whiteline_indices = np.argmax(spectra, axis=-1)
    map_energy = np.vectorize(lambda idx: energies[idx],
                              otypes=[np.float])
    whitelines = map_energy(whiteline_indices)
    # Return results
    return whitelines

def transform_images(data, translations=None, rotations=None,
                     scales=None, out=None, mode='constant'):
    """Takes an array of images and applies each translation, rotation and
    scale. It is assumed that the first dimension of data is the same
    as the length of translations, rotations and scales. Data will be
    written to `out` if given, otherwise returned as a new array.
    """
    # Create a new array if one is not given
    if out is None:
        out = np.zeros_like(data)
    # Define a function to pass into threads
    def apply_transform(idx):
        # Get transformation parameters if given
        scale = scales[idx] if scales is not None else None
        translation = translations[idx] if translations is not None else None
        rot = rotations[idx] if rotations is not None else None
        # Prepare and execute the transformation
        transformation = transform.SimilarityTransform(
            scale=scale,
            translation=translation,
            rotation=rot,
        )
        out[idx] = transform.warp(data[idx], transformation,
                                    order=3, mode=mode)
    # Loop through the images and apply each transformation
    foreach(apply_transform, range(data.shape[0]))
    return out


def register_correlations(frames, reference, upsample_factor=10):
    """Calculate the relative translation between the reference image and
    each image in `frames` using a modified cross-correlation algorithm.

    Returns: Array with same dimensions as 0th axis of `frames`
    containing (x, y) translations for each frame.

    """
    def get_translation(frm):
        results = feature.register_translation(reference,
                                               frm,
                                               upsample_factor=upsample_factor)
        shift, error, diffphase = results
        # Convert (row, col) to (x, y)
        return (shift[1], shift[0])
    translations = np.array(parallel_map(get_translation, frames))
    # Negative in order to properly register with transform_images method
    translations = -translations
    return translations


def register_template(frames, reference, template):
    """Calculate the relative translation between the reference image and
    each image in `frames` using a template matching algorithm.

    Returns: Array with same dimensions as 0th axis of `frames`
    containing (x, y) translations for each frame.
    """
    return None
