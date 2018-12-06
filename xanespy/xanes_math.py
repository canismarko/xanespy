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
allows for easy optimization of parallelizable algorithms and better
test isolation. Most functions will operate on large arrays of data.

"""

import warnings
import logging
from time import time
import multiprocessing as mp
from itertools import product
from collections import namedtuple
from functools import partial

from scipy import ndimage, linalg, stats
from scipy.optimize import curve_fit, leastsq
from scipy.ndimage import center_of_mass
import numpy as np
from skimage import transform, feature, filters, morphology, exposure, measure
from sklearn import decomposition
import matplotlib.pyplot as plt
from tqdm import tqdm

from utilities import prog, foreach, get_component
from exceptions import XanesMathError


log = logging.getLogger(__name__)


class FramesPool(mp.pool.Pool):
    """A multiprocessing pool that iterates over frames even if there are
    extra dimensions.
    
    Parameters
    ----------
    processes : int, optional
      How many processes to spawn. If omitted, twice the number of
      CPUs will be used.
    
    """
    def map(self, func, frames, spatial_dims=2, chunksize=None, desc=None):
        """Apply the function to each frame in ``frames`` and return the result.
        
        Parameters
        ----------
        func : callable
          Function to be applied to each frame.
        frames : np.ndarray
          Array of frames to operate on. The last dimensions are
          considered spatial dimensions.
        spatial_dims : int, optional
          How many of the last dimensions are spatial frame
          dimensions, 2D, 3D, etc.
        chunksize : int, optional
          How many frames to process at a time.
        desc : str, optional
          If provided, a progress bar will be displayed.
        
        Returns
        -------
        result : np.ndarray
          The result of all calculations. Similar in shape to
          ``frames`` except that the spatial dimensions are replaced by
          whatever is returned by ``func``.
        
        """
        # Calculate the correct shapes to use
        extra_dims = frames.ndim - spatial_dims
        extra_shape = frames.shape[:extra_dims]
        frame_shape = frames.shape[extra_dims:]
        # Reshape frames so they're a flat array of frames
        iter_shape = (-1, *frame_shape)
        iterable = np.reshape(frames, iter_shape)
        # Execute the actual mapping
        result = super().map(func, iterable, chunksize)
        result = np.array(result)
        # Convert shape back to that what the frames shape implies
        new_shape = (*extra_shape, *result.shape[1:])
        result = np.reshape(result, new_shape)
        return result


def crop_image(img, shape, center=None):
    """Return a cropped image with shape around center
    
    Parameters
    ==========
    img
      The 2-D image data to crop.
    shape
      The shape of the data to crop to.
    center
      The point around which the cropping should occur. If this point
      results in cropping outside the given image, the center will be
      moved to ensure proper image shape. If omitted, the center of
      the image will be used.
    
    Returns
    =======
    new_img
      The cropped image.
    
    """
    # Make sure image is 2-dimensionsal
    if img.ndim != 2:
        raise XanesMathError('Image must be 2-dimensional')
    # Make sure the target is smaller than the current image size
    for src_dim, new_dim in zip(img.shape, shape):
        if new_dim > src_dim:
            raise XanesMathError(
                'Invalid image cropping dimensions: {} -> {}'
                ''.format(img.shape, shape))
    # Get default center if neeeded
    if center is None:
        center = (int(img.shape[0]/2), int(img.shape[1]/2))
    # Crop the image
    offset = np.array(shape) / 2
    # Make sure the center position for columns is reasonable
    if center[1] + offset[1] > img.shape[1]:
        center_c = img.shape[1] - offset[1]
    elif center[1] - offset[1] < 0:
        center_c = offset[1]
    else:
        center_c = center[1]
    # Make sure the center position for rows is reasonable
    if center[0] + offset[0] > img.shape[0]:
        center_r = img.shape[0] - offset[0]
    elif center[0] - offset[0] < 0:
        center_r = offset[0]
    else:
        center_r = center[0]
    # Do the actual cropping
    rows = (int(center_r - offset[0]), int(center_r + offset[0]))
    cols = (int(center_c - offset[1]), int(center_c + offset[1]))
    new_img = img[rows[0]:rows[1],cols[0]:cols[1]]
    assert np.array_equal(new_img.shape, shape)
    return new_img


def resample_image(img, new_shape, src_dims, new_dims):
    """Resample and crop an image to match a given parameters.
    
    Based on the values of ``src_dims`` and ``new_dims`` the image
    will be cropped, this cropping will occur around the center of
    weight for the image. For this to work well, it is necessary to
    first have the images in the optical depth domain.
    
    Resampling is done using ``skimage.transform.resize``
    
    Parameters
    ==========
    img : np.ndarray
      The image to be transformed.
    new_shape : 2-tuple
      The shape the image should be when it is returned.
    src_dims : 2-tuple
      The (x, y) dimensions of the ``img`` in physical units (eg µm)
    new_dims : 2-tuple
      The (x, y) dimensions of the target image in physical units (eg
      µm). If ``new_dims`` is larger than ``src_dims``, an exception
      will be raised.
    
    Returns
    =======
    new_img : np.ndarray
      The re-sampled and cropped image.
    
    """
    # Convert everything to numpy arrays
    new_dims = np.array(new_dims)
    src_dims = np.array(src_dims)
    new_shape = np.array(new_shape)
    # Determine how big to make the new image
    size_ratio = new_dims / src_dims
    full_shape = new_shape / size_ratio
    full_shape = tuple(round(v) for v in full_shape)
    # Resample the image
    new_img = np.copy(img)
    resize_needed = not np.array_equal(full_shape, new_img.shape)
    if resize_needed:
        new_img = transform.resize(new_img, full_shape, order=3, mode='reflect')
    # Crop the image
    center = center_of_mass(new_img)
    new_img = crop_image(new_img, new_shape, center=center)
    return new_img


# Helpers for parallelizable code
def iter_indices(data, leftover_dims=1, desc=None, quiet=False, ):
    """Accept an array of frames, indices, etc. and generate slices for
    each frame.
    
    Assumes the last two dimensions of `data` are rows and
    columns. All other dimensions will be iterated over.
    
    Parameters
    ----------
    data : np.ndarray
      Data to iterate over.
    leftover_dims : int, optional
      Integer describing which dimensions should not be iterated
      over. Eg. if data is 3D array and leftover_dims == 1, only first
      two dimenions will be iterated.
    desc : str, optional
      String to put in the progress bar.
    quiet : bool, optional
      Whether to suppress the progress bar, etc.
    
    """
    fs_shape = np.asarray(data.shape[:-leftover_dims])
    ranges = [range(0, s) for s in fs_shape]
    indices = product(*ranges)
    if not quiet:
        indices = prog(indices, desc=desc, total=np.product(fs_shape))
    return indices


def downsample_array(arr, factor, method='mean', axis=None):
    """Reduced the shape of an image in powers of 2.
    
    Increases in the signal-to-noise can be achieved by sacrificing
    spatial resolution: a (1024, 1024) image can be converted to (512,
    512) by taking the mean of each (2, 2) block. The ``factor``
    parameter controls how agressive this conversion is. Extra pixels
    get dropped: eg (1025, 1024) -> (512, 512).
    
    +--------------+--------+--------+--------------+
    | arr          | factor | block  | out          |
    +--------------+--------+--------+--------------+
    | (1024, 1024) | 0      | N/A    | (1024, 1024) |
    | (1024, 1024) | 1      | (2, 2) | (512, 512)   |
    | (1024, 1024) | 2      | (4, 4) | (256, 256)   |
    | (1024, 1024) | 3      | (8, 8) | (128, 128)   |
    +--------------+--------+--------+--------------+
    
    Parameters
    ----------
    arr : np.ndarray
      Input array to be downsampled.
    factor : int
      Downsampling factor. A value of 0 returns the original array.
    method : str, optional
      How to convert a (eg 2x2) block to a single value. Valid choices
      are 'mean' (default), 'median', 'sum'.
    axis : tuple, optional
      Which axes to use for reduction. If omitted, all axes will be
      used.
    
    Raises
    ------
    ValueError
      Negative downsampling factors.
    ValueError
      ``method`` parameter is not one of the valid options.
    
    Returns
    -------
    out : np.ndarray
      The downsample array.

    """
    # Check that the method parameter is valid
    methods = {
        'mean': np.mean,
        'sum': np.sum,
        'median': np.median,
    }
    if method not in methods.keys():
        raise ValueError("Invalid method '{}'. Choices are {}"
                         "".format(method, list(methods.keys())))
    # Prepare a list of axes to operate on if not given
    if axis is None:
        axis = tuple(range(arr.ndim))
    # Check that the downsampling factor is sane.
    if factor < 0:
        raise ValueError("Downsampling factor must be non-negative: {}"
                         "".format(factor))
    elif factor == 0:
        # No resampling necessary
        out = arr
    else:
        # Calculate new shapes and block sizes
        blocksize = 2**factor
        def subshape(arr, bs, axis):
            for idx in range(arr.ndim):
                if idx in axis:
                    yield int(arr.shape[idx] / bs) * bs
                else:
                    yield arr.shape[idx]
        # subshape = tuple(int(s / blocksize) * blocksize for s in arr.shape)
        subshape = tuple(subshape(arr, blocksize, axis))
        # Remove any extra pixels that don't fit into a block
        slices = tuple(slice(0, s, 1) for s in subshape)
        arr = arr[slices]
        # Create extra axes for reducing
        def tmpshape(arr, bs, axis):
            for idx in range(arr.ndim):
                if idx in axis:
                    # Yes downsample
                    yield int(arr.shape[idx] / bs)
                    yield bs
                else:
                    # No downsample
                    yield arr.shape[idx]
                    yield 1
        tmpshape = tuple(tmpshape(arr, blocksize, axis))
        arr = arr.reshape(tmpshape)
        # Reduce the extra (odd) axes to produce a downsampled image
        axis = tuple(range(1, arr.ndim, 2))
        out = methods[method](arr, axis=axis)
    return out


def apply_mosaic_reference(intensity, reference):
    """Use a single reference frame to calculate the reference for a
    mosaic of intensity frames.
    
    Returns
    -------
    - out : ndarray
      Same shape as ``intensity`` (I) but with optical depth: ln(I/I0)
      where I0 is the reference frame.
    
    Parameters
    ----------
    - intensity : ndarray
      Intensity data image of the sample. It's shape must be a whole
      multiple of the shape of ``reference``.
    - reference : ndarray
      A reference image with no sample that will be applied.
    
    """
    reps = np.divide(intensity.shape, reference.shape).astype(int)
    mosaic_ref = np.tile(reference, reps)
    od = np.log(mosaic_ref / intensity)
    return od


def _apply_internal_reference(frame):
    # Calculate background intensity using thresholding
    log.debug("Applying reference for frame")
    # Handle complex data
    is_complex = np.iscomplexobj(frame)
    j = complex(0, 1)
    if is_complex:
        frame = frame.astype(np.complex128)
        direct_img = get_component(frame, 'modulus')
    else:
        frame = frame.astype(np.float64)
        direct_img = get_component(frame, 'real')
    # Do the thresholding
    threshold = filters.threshold_yen(direct_img)
    graymask = direct_img > threshold
    background = frame[graymask]
    # Calculate an I_0 based on median modulus and phase
    if is_complex:
        mod = np.median(np.abs(background))
        angle = np.median(np.angle(background))
        j = complex(0, 1)
        I_0 = mod * np.cos(angle) + j * mod * np.sin(angle)
    else:
        I_0 = np.median(background)
    log.debug("I0 for frame = %f", I_0)
    # Calculate optical depth based on background
    od = np.log(I_0 / frame)
    log.debug("Applied internal reference for frame")
    return od


def apply_internal_reference(intensities, desc="Applying reference"):
    """Apply a reference correction to complex data.
    
    Convert intensities into refractive index. :math:`I_0` is
    determined by separating the pixels into background and foreground
    using Otsu's method.
    
    Parameters
    ==========
    intensities : 
      3+ dimensional array Intensity frames, may be complex-valued.
    desc : str
      Description for the progress bar. ``None`` suppresses output.
    
    Arrays `intensities` and `out` must have the same 3-D shape where
    the last two dimensions are image rows and column.
    
    """
    logstart = time()
    log.debug("Starting internal reference correction")
    # Call the actual function for each frame
    with FramesPool() as pool:
        out = pool.map(_apply_internal_reference, intensities, desc=desc)
    log.info("Internal reference applied in %f seconds",
             (time() - logstart))
    return out


def extract_signals_nmf(spectra, n_components, nmf_kwargs=None, mask=None):
    """Extract the signal components present in the given spectra using
    non-negative matrix factorization. Input data can be negative, but
    it will be shifted up, processed, then shifted down again.
    
    Arguments
    ---------
    spectra : numpy.ndarray
      A numpy array of observations where the last axis is energy.
    n_components : int
      How many components to extract from the data.
    nmf_kwargs : dict, optional
      Keyword arguments to be passed to the constructor of the
      estimator.
    
    Returns
    -------
    components, weights : numpy.ndarray
      Extracted components and weights for each pixel-component
    combination.
    
    """
    if nmf_kwargs is None:
        _nmf_kwargs = {}
    max_iter = _nmf_kwargs.pop('max_iter', 200)
    # Make sure all the values are non-negative
    _spectra = np.abs(spectra)
    # Perform NMF fitting
    nmf = decomposition.NMF(n_components=n_components,
                            max_iter=max_iter,
                            **_nmf_kwargs)
    weights = nmf.fit_transform(_spectra)
    # Extract results and calculate weights
    signals = nmf.components_
    # Log the results
    log.info("NMF of %d samples in %d of %d iterations.", len(spectra),
             nmf.n_iter_+1, max_iter)
    log.info("NMF reconstruction error = %f", nmf.reconstruction_err_)
    return signals, weights


def apply_references(intensities, references, out=None, quiet=False):
    """Apply a reference correction to convert intensity values to optical
    depth.
    
    The formula :math:`-ln\\frac{intensities}{references}` is used to
    calculate the new values. Arrays ``intensities``, ``references``
    and ``out`` must all have the same shape where the last two
    dimensions are image rows and columns.
    
    Parameters
    ----------
    intensities : np.ndarray
      Sample input signal data.
    references : np.ndarray
      Background input signal data. Must be the same shape as
      intensities.
    out : np.ndarray, optional
      Array to receive the results.
    quiet : bool, optional
      Whether to suppress progress bar, etc.
    """
    logstart = time()
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
    prog_iter = prog(range(0, Is.shape[0]), desc="Calc'ing OD",
                     disable=quiet)
    foreach(calc, prog_iter)
    return out


def l_edge_mask(frames: np.ndarray, energies: np.ndarray, edge,
                sensitivity: float=1, frame_dims=2, min_size=0):
    """Calculate a mask for what is likely active material at this
    edge. This is done by comparing each spectrum to the overall
    spectrum using the dot product. A normalization is first applied
    to mitigate differences in total intensity.
    
    Parameters
    ---------
    frames : np.ndarray
      Array with images at different energies.
    energies : np.ndarray
      X-ray energies corresponding to images in `frames`. Must have
      the same shape along the first dimenion as `frames`.
    edge : xanespy.edges.Edge
      An Edge object that contains a description of the elemental edge
      being studied.
    sensitivity : float, optional
      A multiplier for the otsu value to determine the actual
      threshold.
    frame_dims : int, optional
      The number of dimensions that each frame has. Eg, 2 means each
      frame is a two-dimensional image.
    min_size : float, optional
      Objects below this size (in pixels) will be
      removed. Passing zero (default) will result in no effect.
    
    Returns
    -------
    mask : np.ndarray
      A boolean mask with the same shape as the last two dimensions of
      `frames` where True pixels are likely to be background material.
    
    """
    # Take the mean over all timesteps
    ODs = frames
    if frames.ndim > 3:
        E_dim = frame_dims + 1
        ODs = np.mean(np.reshape(frames, (-1, *frames.shape[-E_dim:])), axis=0)
        Es = np.mean(np.reshape(energies, (-1, frames.shape[-E_dim])), axis=0)
    elif frames.ndim == 3:
        Es = energies
    # Convert optical depths into spectra -> (spectrum, energy) shape
    frame_shape = frames.shape[-frame_dims:]
    spectra = ODs.reshape(-1, np.prod(frame_shape))
    spectra = np.moveaxis(spectra, 0, 1)
    assert spectra.ndim == 2
    # Compute normalized global average spectrum
    spectrum = np.mean(spectra, axis=0)
    global_min = np.min(spectrum)
    global_max = np.max(spectrum)
    spectrum = (spectrum - global_min) / (global_max - global_min)
    # Compute some statistical values for normalization
    minima = np.min(spectra, axis=1)
    maxima = np.max(spectra, axis=1)
    # Compute the spectrum baseline
    pre_edge_mask = np.logical_and(
        np.min(edge.pre_edge) <= Es,
        Es <= np.max(edge.pre_edge),
    )
    post_edge_mask = np.logical_and(
        np.min(edge.post_edge) <= Es,
        Es <= np.max(edge.post_edge),
    )
    # Check the pre- and post-edge regions are in the data
    if not (np.any(pre_edge_mask) and np.any(post_edge_mask)):
        msg = "Cannot find pre and post edges for L-edge: {}"
        msg = msg.format(edge)
        warnings.warn(msg, RuntimeWarning)
    baseline_idx = np.logical_or(pre_edge_mask, post_edge_mask)
    baseline = np.mean(spectrum[baseline_idx])
    # Normalize all the pixel spectra based on the baselines and maxima
    pixels = np.moveaxis(spectra, 0, 1)
    pixels = (pixels - baseline) / (maxima - minima)
    # Compute the overlap between the global and pixel spectra
    overlap = np.dot(spectrum, pixels)
    # Threshold the pixel overlap to find foreground vs background
    threshold = filters.threshold_otsu(overlap) * sensitivity
    mask = overlap > threshold
    # Recreate the original image shape as a boolean array
    mask = np.reshape(mask, frame_shape)
    # Remove left-over background fuzz
    if min_size > 0:
        mask = morphology.opening(mask, selem=morphology.disk(min_size))
    mask = np.logical_not(mask)
    return mask


def k_edge_mask(frames: np.ndarray, energies: np.ndarray, edge,
                sensitivity: float=1, min_size=0):
    """Calculate a mask for what is likely active material at this
    edge. This is done by comparing the edge-jump to the standard
    deviation. Foreground material will be identified when the
    edge-jump accounts for most of the standard deviation.
    
    Arguments
    ---------
    frames : numpy.ndarray
      Array with images at different energies.
    energies : numpy.ndarray
      X-ray energies corresponding to images in ``frames``. Must have
      the same shape along the first dimenion as ``frames``.
    edge : KEdge
      A :py:class:`xanespy.edges.KEdge` object that contains a
      description of the elemental edge being studied.
    sensitivity : float, optional
      A multiplier for the otsu value to determine the actual
      threshold.
    min_size : int, optional
      Objects below this size (in pixels) will be removed. Passing
      zero (default) will result in no effect.
    
    Returns
    -------
    mask : numpy.ndarray
      A boolean mask with the same shape as the last two dimensions of
      ``frames`` where True pixels are likely to be background
      material.
    
    """
    # Dividing the edge jump by the standard deviation provides sharp constrast
    ej = k_edge_jump(frames=frames, energies=energies, edge=edge)
    stdev = np.std(frames, axis=tuple(range(0, len(frames.shape)-2)))
    edge_ratio = ej / stdev
    # Thresholding to separate background from foreground
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


def k_edge_jump(frames: np.ndarray, energies: np.ndarray, edge):
    """Determine what the difference is between the post_edge and the
    pre_edge."""
    # Check that dimensions match
    if not frames.shape[0] == energies.shape[0]:
        msg = "First dimenions of frames and energies do not match ({} vs {})"
        raise ValueError(msg.format(frames.shape[0], energies.shape[0]))
    pre_edge = edge.pre_edge
    post_edge = edge.post_edge
    # Prepare masks for the post-edge and the pre-edge
    pre_edge_mask = np.logical_and(np.greater_equal(energies, pre_edge[0]),
                                   np.less_equal(energies, pre_edge[1]))
    if not np.any(pre_edge_mask):
        raise XanesMathError("Could not find pre-edge "
                             "{} in {}".format(pre_edge, energies))
    post_edge_mask = np.logical_and(np.greater_equal(energies, post_edge[0]),
                                    np.less_equal(energies, post_edge[1]))
    if not np.any(post_edge_mask):
        raise XanesMathError("Could not find post-edge "
                             "{} in {}".format(post_edge, energies))
        
    # Compare the post-edges and pre-edges
    mean_pre = np.mean(frames[pre_edge_mask], axis=0)
    mean_post = np.mean(frames[post_edge_mask], axis=0)
    ej = mean_post - mean_pre
    return ej


def particle_labels(frames: np.ndarray, energies: np.ndarray, edge,
                    min_distance=20):
    """Prepare a map by segmenting the images into particles.
    
    Arguments
    ---------
    frames : numpy.ndarray
      An array of images, each one at a different energy. These will
      be merged and used for segmentation.
    
    """
    # Get edge-jump mask
    mask = ~edge.mask(frames, energies=energies, min_size=0)
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


def direct_whitelines(spectra, energies, edge, quiet=False):
    """Takes an array of X-ray optical_depth spectra and calculates the
    positions of maximum intensities over the near-edge region.
    
    Arguments
    ---------
    spectra : np.array
      2D numpy array of optical_depth spectra where the last dimension is
      energy.
    energies : np.array
      Array of X-ray energies in electron-volts. Must be broadcastable
      to the shape of spectra.
    edge
      An XAS Edge object that describes the absorbtion edge in
      question.
    quiet : bool, optional
      Whether to suppress the progress bar, etc.
    
    Returns
    -------
    out : np.ndarray
      Array with the whiteline position of each spectrum.
    
    """
    # Broadcast energies to be same shape as spectra
    # Cut down to only those values on the edge
    edge_mask = (energies > edge.edge_range[0]) & (energies < edge.edge_range[1])
    edge_mask = np.broadcast_to(edge_mask, spectra.shape)
    # Convert energies to be same shape as spectra
    Es = np.broadcast_to(energies, spectra.shape)
    # Set values outside the mapping range to be negative infinity
    subspectra = np.copy(spectra)
    subspectra[...,~edge_mask] = -np.inf
    # Prepare a new array to hold results
    outshape = spectra.shape[:-1]
    out = np.empty(shape=outshape, dtype=energies.dtype)
    # Iterate over spectra and perform calculation

    def get_whiteline(frame_idx):
        whiteline_idx = np.argmax(subspectra[frame_idx])
        whiteline = Es[frame_idx][whiteline_idx]
        out[frame_idx] = whiteline
    indices = iter_indices(spectra, desc="Finding maxima",
                           leftover_dims=1, quiet=quiet)
    foreach(get_whiteline, indices)
    # Return results
    return out


def transform_images(data, transformations, out=None, mode='median', quiet=False):
    """Takes image data and applies the given translation matrices.
    
    It is assumed that the first dimension of `data` is the same as
    the length of `transformations`. The transformation matrices can
    be generated from translation, rotation and scale parameters via
    the :py:meth:`xanespy.xanes_math.transformation_matrics`
    function. Data will be written to `out` if given, otherwise
    returned as a new array.
    
    Arguments
    ---------
    data : np.ndarray
      Numeric array with frames to transform. Last two dimensions are
      assumed to be (row, columns).
    transformations : np.ndarray
      A numeric array shaped compatibally with `data`. The last two
      dimensions are assumed to be (3, 3) and each (3, 3) encodes a
      transformation matrix for the corresponding frame in `data`.
    out : np.ndarray, optional
      A numeric array with same shape as `data` that will hold the
      transformed data.
    mode : str, optional
      Describes how to deal with edges. See scikit-image documentation
      for options. Special value "median" (default), takes the median
      pixel intensity of that frame and uses it as the constant value.
    quiet : bool, optional
      Whether to suppress the progress bar, etc.
    
    Returns
    -------
    out : np.ndarray
      A new array with similar dimensions to `data` but with
      transformations applied and converted to float datatype.
    
    """
    logstart = time()
    # Create a new array if one is not given
    if out is None:
        out_type = np.complex if np.iscomplexobj(data) else np.float
        out = np.zeros_like(data, dtype=out_type)
    # Define a function to pass into threads
    def apply_transform(idx):
        # Get transformation parameters if given
        tmatrix = transformations[idx]
        # Prepare and execute the transformation
        transformation = transform.AffineTransform(
            matrix=tmatrix,
        )
        def do_transform(a, transformation):
            """Helper function, takes one array and transforms it."""
            realrange = (np.min(a), np.max(a))
            # Convert to float so the warp function is happy
            if not np.iscomplexobj(a):
                indata = a.astype(np.float64)
            assert realrange[0] != realrange[1], realrange
            indata = exposure.rescale_intensity(indata,
                                                in_range=realrange,
                                                out_range=(0, 1))
            # Log the anticipated transformation
            msg = "Transforming {idx} by scl={scale}; trn={trans}; rot={rot:.2f} rad"
            msg = msg.format(idx=idx, scale=transformation.scale,
                             trans=transformation.translation,
                             rot=transformation.rotation)
            log.debug(msg)
            # Perform the actual transformation
            if mode == "median":
                realmode = "constant"
                cval = np.median(indata)
            else:
                realmode = mode
                cval = 0.
            outdata = transform.warp(indata, transformation, order=3,
                                     mode=realmode, cval=cval)
            # Return to the original intensity range
            outdata = exposure.rescale_intensity(outdata,
                                                 in_range=(0, 1),
                                                 out_range=realrange)
            return outdata
        frame = data[idx]
        if np.iscomplexobj(frame):
            # Special handling for complex numbers
            j = complex(0, 1)
            out[idx] = (do_transform(np.real(frame), transformation) +
                        j * do_transform(np.imag(frame), transformation))
        else:
            # Only real numbers here
            out[idx] = do_transform(frame, transformation)
    # Loop through the images and apply each transformation
    log.debug("Starting transformations")
    indices = iter_indices(data, desc='Transforming', leftover_dims=2, quiet=quiet)
    foreach(apply_transform, indices)
    log.debug("Finished transformations in %d sec", time() - logstart)
    return out


def transformation_matrices(translations=None, rotations=None,
                            scales=None, center=(0, 0)):
    """Takes array of operations and calculates (3, 3) transformation
    matrices.
    
    This function operates by calculating an AffineTransform similar
    to that described in the scikit-image package.
    
    All three arguments (`translations`, `rotations`, and `scales`,
    should have shapes that are compatible with the frame data, though
    this is not strictly enforced for now. Rotation will necessarily
    have one less degree of freedom than translation/scale values.
    
    Example Shapes:
    
    +----------------------------+--------------+-------------+-------------+
    | Frames                     | Translations | Rotations   | Scales      |
    +============================+==============+=============+=============+
    | (10, 48, 1024, 1024)       | (10, 48, 2)  | (10, 48, 1) | (10, 48, 2) |
    +----------------------------+--------------+-------------+-------------+
    | (10, 48, 1024, 1024, 1024) | (10, 48, 3)  | (10, 48, 2) | (10, 48, 3) |
    +----------------------------+--------------+-------------+-------------+
    
    Parameters
    ----------
    translations : np.ndarray, optional
      How much to move each axis (x, y[, z]).
    
    rotations : np.ndarray, optional
      How much to rotate around the origin (0, 0) pixel.
    
    center : np.ndarray, optional
      Where to set the origin of rotation. Default is the
      first pixel (0, 0).
    
    scales : np.ndarray, optional
      How much to scale the image by in each dimension
      (x, y[, z]).
    
    Returns
    -------
    new_transforms : np.ndarray
      Resulting transformation matrices. Will have the same shape as the
      input arrays but with the last dimension replaced by (3, 3).
    
    """
    spatial_dims = 2
    assert spatial_dims == 2 # Only can handle 2D data for now
    # Get the shape of the final matrix by inspecting the inputs' shapes
    if translations is not None:
        master = translations
    elif rotations is not None:
        master = rotations
    elif scales is not None:
        master = scales
    else:
        raise ValueError("No transformations specified.")
    matrix_shape = (*master.shape[0:-1], spatial_dims+1, spatial_dims+1)
    # Prepare dummy arrays if some are missing
    if translations is None:
        translations = np.zeros((*matrix_shape[0:-2], spatial_dims))
    if rotations is None:
        rotations = np.zeros((*matrix_shape[0:-2], spatial_dims - 1))
    if scales is None:
        scales = np.ones((*matrix_shape[0:-2], spatial_dims))
    # Calculate values for transformation matrix
    #   Values taken from skimage documentations for transform.AffineTransform
    sx, sy = np.moveaxis(scales, -1, 0)
    tx, ty = np.moveaxis(translations, -1, 0)
    r, = np.moveaxis(rotations, -1, 0)
    # Calculate new eigenvectors for this transformation
    ihat = [ sx * np.cos(r), sx * np.sin(r), np.zeros_like(sx)]
    jhat = [-sy * np.sin(r), sy * np.cos(r), np.zeros_like(sx)]
    khat = [ tx,             ty,             np.ones_like(sx) ]
    # Make the eigenvectors into column vectored matrix
    new_transforms = np.array([ihat, jhat, khat]).swapaxes(0, 1)
    # Move the transformations so they're in frame order
    new_transforms = np.moveaxis(np.moveaxis(new_transforms, 0, -1), 0, -1)
    return new_transforms


def _correlation_translation(frame, reference, upsample_factor, median_filter_size):
    if median_filter_size is not None:
        frame = ndimage.median_filter(frame, size=median_filter_size)
    results = feature.register_translation(reference,
                                           frame,
                                           upsample_factor=upsample_factor)
    shift, error, diffphase = results
    # Convert (row, col) to (x, y)
    return (shift[1], shift[0])


def register_correlations(frames, reference, upsample_factor=10,
                          desc="Registering", median_filter_size=None):
    """Calculate the relative translation between the reference image and
    a series of frames.
    
    This uses phase correlation through scikit-image's
    `register_translation` function.
    
    Parameters
    ----------
    frames : np.ndarray
      Array where the last two dimensions are (column, row) of images
      to be registered.
    reference : np.ndarray
      Image frame against which to align the entries in `frames`.
    upsample_factor : int, optional
      Factor controls subpixel registration via scikit-image.
    desc : str, optional
      Description for putting in the progress bar. ``None`` will
      suppress output.
    median_filter_size : int, optional
      If provided, a median filter will be applied to each
      frame. The value of this parameter determines how large the
      kernel is: ``3`` creates a (3, 3) kernel; ``(3, 5)`` creates
      a (3, 5) kernel; etc.
    
    Returns
    -------
    translations : np.ndarray
      Array with same dimensions as 0-th axis of `frames` containing
      (x, y) translations for each frame.
    
    """
    # Apply the median filter to reference (frames are done later)
    if median_filter_size is not None:
        reference = ndimage.median_filter(reference, size=median_filter_size)
    # Prepare a partial with the common data
    func = partial(_correlation_translation,
                   reference=reference, upsample_factor=upsample_factor,
                   median_filter_size=median_filter_size)
    # Execute the translations with multiprocessing
    with FramesPool() as pool:
        translations = pool.map(func, frames, desc=desc)
    # Negative in order to properly register with transform_images method
    translations = -translations
    return translations


def _template_translation(frame, template, ref_center, median_filter_size):
    if median_filter_size is not None:
        frame = ndimage.median_filter(frame, size=median_filter_size)
    match = feature.match_template(frame, template)
    center = np.unravel_index(np.argmax(match), match.shape)
    shift = ref_center - np.array(center)
    # Convert (row, col) to (x, y)
    return (shift[1], shift[0])


def register_template(frames, reference, template, desc="Registering",
                      median_filter_size=None):
    """Calculate the relative translation between the reference image and
    a series of frames.
    
    This uses template cross correlation through scikit-image's
    `match_template` function.
    
    The `register_correlations` algorithm is simpler to use in most
    cases but sometimes results in unreasonable results; in those
    cases, this method can be more reliable to achieve a first
    approximation.
    
    Parameters
    ----------
    frames : np.ndarray
      Array where the last two dimensions are (column, row) of images
      to be registered.
    reference : np.ndarray
      Image frame against which to align the entries in `frames`.
    template : np.ndarray
      A 2D array (smaller than frames and reference) that will be
      identified in each frame and used for alignment.
    desc : str, optional
      Description for putting in the progress bar. ``None`` will
      suppress output.
    median_filter_size : int, optional
      If provided, a median filter will be applied to each
      frame. The value of this parameter determines how large the
      kernel is: ``3`` creates a (3, 3) kernel; ``(3, 5)`` creates
      a (3, 5) kernel; etc.
    
    Returns
    -------
    translations : np.ndarray
      Array with same dimensions as 0-th axis of `frames` containing
      (x, y) translations for each frame.
    
    """
    # Apply median filter to the template and reference (frames done later)
    if median_filter_size is not None:
        reference = ndimage.median_filter(reference, size=median_filter_size)
        template = ndimage.median_filter(template, size=median_filter_size)
        # This functionality doesn't always work great in unit-tests
        warnings.warn('Median filtering with template matching'
                      'can lead to poor registration. Use at your'
                      'own risk.', RuntimeWarning)
    ref_match = feature.match_template(reference, template)
    ref_center = np.unravel_index(np.argmax(ref_match), ref_match.shape)
    ref_center = np.array(ref_center)
    # Prepare the particle with the translation function
    func = partial(_template_translation, template=template, ref_center=ref_center,
                   median_filter_size=median_filter_size)
    # Execute the registration with multiprocessing
    with FramesPool() as pool:
        translations = pool.map(func, frames)
    # Negative in order to properly register with transform_images method
    translations = -translations
    return translations
