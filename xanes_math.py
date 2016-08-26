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
import sys
import threading
import multiprocessing
from itertools import count, product

from scipy import ndimage
import numpy as np
from skimage import transform, feature, filters, morphology, exposure, measure

from utilities import prog

# Helpers for parallelizable code


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
            exception = a(b)
            exception.with_traceback(c)
            raise exception
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


def iter_indices(data, leftover_dims=1, desc=None):
    """Accept an array of frames, indices, etc. and generate slices for
    each frame. Assumes the last two dimensions of `data` are rows and
    columns. All other dimensions will be iterated over.

    - leftover_dims : Integer describing which dimensions should not
      be iterated over. Eg. if data is 3D array and leftover_dims == 1,
      only first two dimenions will be iterated.

    - desc : String to put in the progress bar.
    """
    fs_shape = np.asarray(data.shape[:-leftover_dims])
    length = np.product(fs_shape)
    ranges = [range(0, s) for s in fs_shape]
    indices = product(*ranges)
    return prog(indices, desc=desc, total=length)


def apply_references(intensities, references, out=None):

    """Apply a reference correction to convert intensity values to
    optical depth (-ln(I/I0)) values. Arrays `intensities`, `references` and `out`
    must all have the same shape where the last two dimensions are
    image rows and columns.
    """
    # Create an empty array to hold the results
    if out is None:
        out = np.zeros_like(intensities)
    assert intensities.shape == references.shape
    # Reshape to be an array of images
    image_shape = (intensities.shape[-2], intensities.shape[-1])
    Is = np.reshape(intensities, (-1, *image_shape))
    refs = np.reshape(references, (-1, *image_shape))
    out_shape = out.shape[:-2]
    # Function for parallelization
    def calc(idx):
        out_idx = np.unravel_index(idx, out_shape)
        out[out_idx] = -np.log(np.divide(Is[idx], refs[idx]))
    foreach(calc, range(0, Is.shape[0]), threads=1)
    return out


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


def edge_mask(frames: np.ndarray, energies: np.ndarray, edge,
              sensitivity: float=1, min_size=0):
    """Calculate a mask for what is likely active material at this
    edge. This is done by comparing the edge-jump to the standard
    deviation. Foreground material will be identified when the
    edge-jump accounts for most of the standard deviation.

    Arguments
    ---------
    - frames : Array with images at different energies.

    - energies : X-ray energies corresponding to images in
      `frames`. Must have the same shape along the first dimenion as
      `frames`.

    - sensitivity : A multiplier for the otsu value to determine
      the actual threshold.

    - min_size : Objects below this size (in pixels) will be
      removed. Passing zero (default) will result in no effect. The
      value "auto" will cause the function to estimate a size at
      1/100th the average image size.

    """
    if min_size == "auto":
        # Guess the best min_size from image shape
        min_size = (frames.shape[-1] + frames.shape[-2]) / 200
    # dividing the edge jump by the standard deviation provides sharp constrast
    ej = edge_jump(frames=frames, energies=energies, edge=edge)
    stdev = np.std(frames, axis=tuple(range(0, len(frames.shape)-2)))
    edge_ratio = ej / stdev
    # Thresholding  to separate background from foreground
    img_bottom = edge_ratio.min()
    threshold = filters.threshold_otsu(edge_ratio)
    threshold = img_bottom + sensitivity * (threshold - img_bottom)
    mask = edge_ratio > threshold
    # Remove left-over background fuzz
    if min_size > 0:
        mask = morphology.opening(mask, selem=morphology.disk(min_size))
    # Invert so it removes background instead of particles
    mask = np.logical_not(mask)
    return mask


def edge_jump(frames: np.ndarray, energies: np.ndarray, edge):
    # Check that dimensions match
    if not frames.shape[0] == energies.shape[0]:
        msg = "First dimenions of frames and energies do not match ({} vs {})"
        raise ValueError(msg.format(frames.shape[0], energies.shape[0]))
    pre_edge = edge.pre_edge
    post_edge = edge.post_edge
    # Prepare masks for the post-edge and the pre-edge
    pre_edge_mask = np.logical_and(np.greater_equal(energies, pre_edge[0]),
                                   np.less_equal(energies, pre_edge[1]))
    post_edge_mask = np.logical_and(np.greater_equal(energies, post_edge[0]),
                                   np.less_equal(energies, post_edge[1]))
    # Compare the post-edges and pre-edges
    mean_pre = np.mean(frames[pre_edge_mask, ...], axis=0)
    mean_post = np.mean(frames[post_edge_mask, ...], axis=0)
    ej = mean_post - mean_pre
    return ej


def particle_labels(frames: np.ndarray, energies: np.ndarray, edge,
                    min_distance=20):
    """Prepare a map by segmenting the images into particles.

    Arguments
    ---------

    - frames : An array of images, each one at a different
      energy. These will be merged and used for segmentation.
    """
    # Get edge-jump mask
    mask = ~edge_mask(frames, energies=energies, edge=edge, min_size="auto")
    # Create "edge-distance" image
    distances = ndimage.distance_transform_edt(mask)
    in_range = (distances.min(), distances.max())
    distances = exposure.rescale_intensity(distances,
                                  in_range=in_range,
                                  out_range=(0, 1))
    # Use the local distance maxima as peak centers and compute labels
    local_maxima = feature.peak_local_max(
        distances,
        indices=False,
        min_distance=min_distance,
        labels=mask.astype(np.int)
    )
    markers = measure.label(local_maxima)
    labels = morphology.watershed(-distances, markers, mask=mask)
    return labels


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
    # spectra = spectra[...,edge_mask]
    # energies = energies[edge_mask]
    # # Calculate the whiteline positions
    # whiteline_indices = np.argmax(spectra, axis=-1)
    # map_energy = np.vectorize(lambda idx: energies[idx],
    #                           otypes=[np.float])
    # whitelines = map_energy(whiteline_indices)
    # Convert energies to be same shape as spectra
    mask_shape = (spectra.shape[0],
                  *[1 for i in range(0, spectra.ndim-edge_mask.ndim)],
                  spectra.shape[-1])
    edge_mask = np.broadcast_to(edge_mask.reshape(mask_shape), spectra.shape)
    # Set values outside the mapping range to be negative infinity
    subspectra = np.copy(spectra)
    subspectra[...,~edge_mask] = -np.inf
    # Prepare a new array to hold results
    outshape = spectra.shape[:-1]
    out = np.empty(shape=outshape, dtype=energies.dtype)
    # Iterate over spectra and perform calculation
    def get_whiteline(idx):
        whiteline_idx = (*idx[:energies.ndim-1], np.argmax(subspectra[idx]))
        whiteline = energies[whiteline_idx]
        out[idx] = whiteline
    indices = iter_indices(spectra, desc="Direct whiteline", leftover_dims=1)
    foreach(get_whiteline, indices)
    # Return results
    return out

def transform_images(data, translations=None, rotations=None,
                     scales=None, out=None, mode='constant'):
    """Takes an array of images and applies each translation, rotation and
    scale. It is assumed that the first dimension of data is the same
    as the length of translations, rotations and scales. Data will be
    written to `out` if given, otherwise returned as a new array.

    Returns: A new array similar dimensions to `data` but with
      transformations applied and converted to float32 datatype.
    """
    # Create a new array if one is not given
    if out is None:
        out = np.zeros_like(data, dtype=np.float32)
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
        # (Temporarily rescale intensities so the warp function is happy)
        realrange = (np.min(data[idx]), np.max(data[idx]))
        indata = exposure.rescale_intensity(data[idx],
                                            in_range=realrange,
                                            out_range=(0, 1))
        indata = indata.astype('float32')
        outdata = transform.warp(indata, transformation,
                                  order=3, mode=mode)
        out[idx] = exposure.rescale_intensity(outdata,
                                              in_range=(0, 1),
                                              out_range=realrange)
    # Loop through the images and apply each transformation
    indices = iter_indices(data, desc='Applying', leftover_dims=2)
    foreach(apply_transform, indices)
    return out


def register_correlations(frames, reference, upsample_factor=10):
    """Calculate the relative translation between the reference image and
    each image in `frames` using a modified cross-correlation algorithm.

    Returns: Array with same dimensions as 0th axis of `frames`
    containing (x, y) translations for each frame.

    """
    t_shape = (*frames.shape[:-2], 2)
    translations = np.empty(shape=t_shape, dtype=np.float)
    def get_translation(idx):
        frm = frames[idx]
        results = feature.register_translation(reference,
                                               frm,
                                               upsample_factor=upsample_factor)
        shift, error, diffphase = results
        # Convert (row, col) to (x, y)
        translations[idx] = (shift[1], shift[0])
    indices = iter_indices(frames, desc='Registering', leftover_dims=2)
    foreach(get_translation, indices)
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
