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

"""Class definitions for working with a whole stack of X-ray
microscopy frames. Each frame is a micrograph at a different energy. A
frameset then is a three-dimenional dataset with of dimenions (energy,
row, column).

"""

import datetime as dt
import functools
from typing import Callable, Optional
import os
from time import time
import logging
from collections import namedtuple
import math
import subprocess
import warnings
import multiprocessing as mp
import inspect
from typing import Any, List, Dict, Mapping, Sequence, Iterable, Tuple

import pandas as pd
from matplotlib import pyplot, cm, pyplot as plt
from matplotlib.colors import Normalize
import h5py
import numpy as np
from scipy.ndimage import median_filter
from skimage import morphology, filters, transform,  measure
from sklearn import linear_model, cluster
import jinja2 as j2
import tqdm

from .utilities import (prog, xycoord, Pixel, Extent, pixel_to_xy,
                        get_component, broadcast_reverse, xy_to_pixel, nproc)
from .txmstore import TXMStore
from . import plots
from .fitting import LinearCombination, fit_spectra, prepare_p0, guess_p0, KCurve, find_whiteline
from . import exceptions
from . import xanes_math as xm
from .edges import Edge


log = logging.getLogger(__name__)


guess_params = lambda x: x


class XanesFrameset():
    """A collection of TXM frames at different energies moving across an
    absorption edge. Iterating over this object gives the individual
    Frame() objects. The class assumes that the data have been
    imported into an HDF file.
    
    """
    active_group = ''
    parent_name = None
    cmap = 'plasma'
    edge = Edge()
    _data_name = None
    # Places to store staged image transformations
    _transformations = None
    
    def __init__(self, hdf_filename, edge, groupname=None):
        """Parameters
        ----------
        hdf_filename : str
          Path to the HDF file that holds these data.
        edge
          An Edge object (or class) describing the meterial's X-ray
          energy response characteristics. Can be ``None`` but not
          recommended.
        groupname : str, optional
          Top level HDF group corresponding to this frameset. This
          argument is required if there is more than one top-level
          group.
        
        """
        self.hdf_filename = hdf_filename
        # Validate the edge object
        if edge is None:
            warnings.warn('``edge`` set to ``None``.'
                          'Some operations will fail.')
        elif inspect.isclass(edge):
            self.edge = edge()
        else:
            self.edge = edge
        # Validate the parent dataname
        store = TXMStore(hdf_filename=self.hdf_filename,
                         parent_name=groupname,
                         data_name=None,
                         mode='r')
        with store:
            self.parent_name = store.validate_parent_group(groupname)

    def __str__(self):
        s = "{name}"
        return s.format(cls=self.__class__.__name__, name=self.parent_name)

    def __repr__(self):
        s = "<{cls}: '{name}'>"
        return s.format(cls=self.__class__.__name__, name=self.parent_name)

    def hdf_path(self, representation: Optional[str]=None):
        """Return the hdf path for the active group.

        Parameters
        ----------
        representation
          Name of third-level group to use. If omitted, the path to
          the parent group will be given.

        Returns
        -------
        path : str
          The path to the current group in the HDF5 file. Returns an
          empty string if the representation does not exists.

        """
        with self.store() as store:
            if representation is None:
                group = store.data_group()
            else:
                group = store.get_dataset(representation=representation)
            path = group.name
        return path

    def has_representation(self, representation):
        with self.store() as store:
            result = store.has_dataset(representation)
        return result

    @property
    def data_name(self):
        return self._data_name

    @data_name.setter
    def data_name(self, val):
        self._data_name = val
        # Clear any cached values since the data are probably different
        self.clear_caches()

    def data_tree(self):
        """Wrapper around the TXMStore.data_tree() method."""
        with self.store() as store:
            tree = store.data_tree()
        return tree

    def store(self, mode='r'):
        """Get a TXM Store object that saves and retrieves data from the HDF5
        file. The mode argument is passed to h5py as is. This method
        should be used as a context manager, especially if mode is
        something writeable:

        .. code:: python

           # The 'r+' creates the file or appends if one exists
           with self.store(mode='r+') as store:
               # Do stuff with the store...
               img = store.optical_depths[0,0]
        """
        return TXMStore(hdf_filename=self.hdf_filename,
                        parent_name=self.parent_name,
                        data_name=self.data_name,
                        mode=mode)

    def clear_caches(self):
        """Clear cached function values so they will be recomputed with fresh
        data

        """
        self.frame_mask.cache_clear()
        self.frames.cache_clear()
        self.map_data.cache_clear()
        self.energies.cache_clear()
        self.extent.cache_clear()

    def starttime(self, timeidx=None):
        """Determine the earliest timestamp amongst all of the frames.

        Parameters
        ----------
        timeidx : int, optional
          Which timestep to use for finding the start time. If
          omitted or None (default), all timesteps will be checked.

        Returns
        -------
        start_time : np.datetime64
          Naive datetime representing the earliest known frame for
          this timeidx. Timezone will always be UTC no matter where
          the data were collected.

        """
        now = np.datetime64(dt.datetime.now())
        with self.store() as store:
            Ts = store.timestamps
            # Get only the requested time index
            if timeidx is not None:
                Ts = Ts[timeidx]
            else:
                Ts = Ts[:]
            # Figure out which timestamp is the earliest
            Ts = Ts.astype('datetime64').ravel()
            start_idx = np.argmax(now - Ts)
            start_time = Ts[start_idx]
        # Return the earliest timestamp
        return start_time

    def endtime(self, timeidx=None):
        """Determine the latest timestamp amongst all of the frames.

        Parameters
        ----------
        timeidx : int, optional
          Which timestep to use for finding the end time. If
          omitted or None (default), all timesteps will be checked.

        Returns
        -------
        start_time : np.datetime64
          Naive datetime representing the latest known frame for
          this time index. Timezone will always be UTC no matter where
          the data were collected.

        """
        now = np.datetime64(dt.datetime.now())
        with self.store() as store:
            Ts = store.timestamps
            if timeidx is not None:
                Ts = Ts[timeidx]
            Ts = Ts.astype('datetime64').flatten()
            end_idx = np.argmin(now - Ts)
            end_time = Ts[end_idx]
        return end_time

    def components(self):
        """Retrieve a list of valid representations for these data."""
        comps = ['modulus']
        return comps

    def fork_data_group(self, dest, src=None):
        """Turn on different active data for this frameset's store
        object. Similar to `switch_data_group` except that this method
        deletes the existing group and copies symlinks from the current one.

        Arguments
        ---------
        dest : str
          Name for the data group.
        src : str
          String with the name of the data group to copy from. If None
          (default), the current data group will be used.
        """
        # Fork the current group to the new one
        with self.store(mode='r+') as store:
            store.fork_data_group(dest=dest, src=src)
        self.data_name = dest

    def apply_transformations(self, crop=True, commit=True, quiet=False):
        """Take any transformations staged with `self.stage_transformations()`
        and apply them. If commit is truthy, the staged
        transformations are reset.

        Arguments
        ---------
        crop : bool, optional
          If truthy, the images will be cropped after being
          translated, so there are not edges. If falsy, the images
          will be padded with zeros.
        commit : bool, optional
          If truthy, the changes will be saved to the HDF5 store for
          optical depths, intensities and references, and the staged
          transformations will be cleared. Otherwise, only the
          optical_depth data will be transformed and returned.
        quiet : bool, optional
          Whether to suppress the progress bar, etc.

        Returns
        -------
        out : np.ndarray
          Transformed array of the optical depth frames.

        """
        # First, see if there's anything to do
        if self._transformations is None:
            not_actionable = True
        else:
            not_actionable = np.all(self._transformations == 0)
        if not_actionable:
            # Nothing to apply, so no-op
            log.debug("No transformations to apply, skipping.")
            with self.store() as store:
                out = store.get_dataset('optical_depths')[:]
        else:
            if commit:
                names = ['intensities', 'references', 'optical_depths'] # Order matters
            else:
                names = ['optical_depths']
            # Apply the transformations
            for frames_name in names:
                with self.store() as store:
                    if not store.has_dataset(frames_name):
                        continue
                    # Prepare an array to hold results
                    out = np.zeros_like(store.get_dataset(frames_name))
                    # Apply transformation
                    xm.transform_images(data=store.get_dataset(frames_name),
                                        transformations=self._transformations,
                                        out=out, quiet=quiet)
                # Calculate and apply cropping bounds for the image stack
                if crop:
                    tx = self._transformations[..., 0, 2]
                    ty = self._transformations[..., 1, 2]
                    new_rows = out.shape[-2] - (np.max(ty) - np.min(ty))
                    new_cols = out.shape[-1] - (np.max(tx) - np.min(tx))
                    rlower = int(np.ceil(-np.min(ty)))
                    rupper = int(np.floor(new_rows + rlower))
                    clower = int(np.ceil(-np.min(tx)))
                    cupper = int(np.floor(clower + new_cols))
                    log.debug('Cropping "%s" to [%d:%d,%d:%d]',
                              frames_name, rlower, rupper, clower, cupper)
                    out = out[..., rlower:rupper, clower:cupper]
                # Save result and clear saved transformations if appropriate
                if commit:
                    log.debug('Committing applied transformations for "%s"', frames_name)
                    with self.store('r+') as store:
                        store.replace_dataset(frames_name, out, context='frameset')
        # Clear the staged transformations
        if commit:
            log.debug("Clearing staged transformations")
            self._transformations = None
        return out

    def stage_transformations(self, translations=None, rotations=None, center=(0, 0),
                              scales=None):
        """Allows for deferred transformation of the frame data.

        Since each transformation introduces interpolation error, the
        best results occur when the translations are saved up and then
        applied all in one shot. Takes a combination of arrays of
        translations (x, y), rotations and/or scales and saves them
        for later application. This method should be used in
        conjunction apply_transformations().

        All three arguments should have shapes that are compatible
        with the frame data, though this is not strictly enforced for
        now. Rotation will necessarily have one less degree of freedom
        than translation/scale values.

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
        translations : np.ndarray
          How much to move each axis (x, y[, z]).
        rotations : np.ndarray
          How much to rotate around the origin (0, 0) pixel.
        center : 2-tuple
          Where to set the origin of rotation. Default is the first
          pixel (0, 0).
        scales : np.ndarray
          How much to scale the image by in each dimension (x, y[,
          z]).

        """
        # Compute the new transformation matrics for the given transformations
        new_transforms = xm.transformation_matrices(scales=scales,
                                                    rotations=rotations,
                                                    center=center,
                                                    translations=translations)
        # Combine with any previously saved transformations
        if self._transformations is not None:
            new_transforms =  self._transformations @ new_transforms
        # Save transformation matrix for later
        self._transformations = new_transforms

    def align_frames(self,
                     reference_frame="mean",
                     method: str="cross_correlation",
                     template=None,
                     passes=1,
                     median_filter_size=None,
                     commit=True,
                     component="modulus",
                     plot_results=True,
                     results_ax=None,
                     quiet=False):
        """Use cross correlation algorithm to line up the frames.

        All frames will have their sample position set to (0, 0) since
        we don't know which one is the real position. This operation
        will interpolate between pixels so introduces error. If
        multiple passes are performed, the translations are saved and
        combined at the end so this error is only introduced
        once. Using the `commit=False` argument allows for multiple
        different types of registration to be performed in sequence,
        since uncommitted translations will be applied before the next
        round of registration.

        Parameters
        ----------
        reference_frame : 2-tuple, str, optional
          The index of the frame to which all other frames should be
          aligned. If None, the frame of highest intensity will be
          used. If "mean" (default) or "median", the average or median
          of all frames will be used. If "max", the frame with highest
          optical_depth is used. Otherwise, a 2-tuple with (timestep,
          energy) should be provided. This attribute has no effect if
          template matching is used.
        method : str, optional
          Which technique to use to calculate the translation

            - "cross_correlation" (default)
            - "template_match"

          (If "template_match" is used, the `template` argument should
          also be provided.)
        template : np.ndarray, optional
          Image data that should be matched if the `template_match`
          method is used.
        passes : int, optional
          How many times this alignment should be done. Default: 1.
        median_filter_size : int, optional
          If provided, a median filter will be applied to each
          frame. The value of this parameter determines how large the
          kernel is: ``3`` creates a (3, 3) kernel; ``(3, 5)`` creates
          a (3, 5) kernel; etc.
        commit : bool, optional
          If truthy (default), the final translation will be applied
          to the data stored on disk by calling
          `self.apply_transformations(crop=True)` after all passes have
          finished.  component : What component of the data to use:
          'modulus', 'phase', 'imag' or 'real'.  plot_results : If
          truthy (default), plot the root-mean-square of the
          translation distance for each pass.  results_ax : optional
          If ``plot_results`` is true, this axes will be used to
          receive the plot.  quiet : bool, optional Whether to
          suppress the progress bar, etc.
        component : str, optional
          For complex data, which component to use: modulus, phase,
          real, imag.
        plot_results : bool, optional
          If true, a boxplot will be plot with the RMS shifts for
          each pass.
        results_ax : mpl.Axes, optional
          If ``plot_results`` is true, this Axes will receive the
          plot. If omitted, a new axes will be created.
        quiet : bool, optional
          Suppress the progress bar
        
        Returns
        =======
        pass_distances : np.ndarray
          An array with the RMS translations applied to each
          frame. Shape is (pass, frames) where frames is flattened
          across energy and timestep.
        
        """
        logstart = time()
        log.info("Aligning frames with %s algorithm over %d passes",
                 method, passes)
        pass_distances = np.zeros(shape=(passes, self.num_energies*self.num_timesteps), dtype=float)
        # Sanity check on `method` argument
        valid_methods = ['cross_correlation', 'template_match']
        if method not in valid_methods:
            msg = "Unknown method {}. Choices are {}"
            msg = msg.format(method, valid_methods)
            raise ValueError(msg)
        # Guess best reference frame to use
        if reference_frame is "max":
            spectra = self.spectrum(representation='optical_depths', index=slice(None))
            spectra = np.array([s.values for s in spectra])
            reference_frame = np.argmax(spectra)
            reference_frame = np.unravel_index(reference_frame, shape=spectra.shape)
        # Keep track of how many passes and where we started
        for pass_ in range(0, passes):
            log.debug("Starting alignment pass %d of %d", pass_, passes)
            # Get data from store
            frames = self.apply_transformations(crop=True, commit=False, quiet=quiet)
            frames = get_component(frames, component)
            # Calculate axes to use for proper reference image
            if reference_frame == 'mean':
                ref_image = np.mean(frames, axis=(0, 1))
            elif reference_frame == 'median':
                ref_image = np.median(frames, axis=(0, 1))
            else:
                # User has requested a specific reference frame
                ref_image = frames[reference_frame]
                # Check that the argument results in 2D image_data
                if ref_image.ndim != 2:
                    msg = "refrence_frame ({}) does not match"
                    msg += " shape of frameset {}."
                    msg += " Please provide a {}-tuple."
                    msg = msg.format(reference_frame,
                                     frames.shape,
                                     len(frames.shape) - 2)
                    raise IndexError(msg)
            # Calculate translations for each frame
            if not quiet:
                desc = "Registering pass {}".format(pass_) # type: Optional[str]
            else:
                desc = None
            if method == "cross_correlation":
                translations = xm.register_correlations(frames=frames,
                                                        reference=ref_image,
                                                        desc=desc,
                                                        median_filter_size=median_filter_size)
            elif method == "template_match":
                template_ = get_component(template, component)
                translations = xm.register_template(frames=frames,
                                                    reference=ref_image,
                                                    template=template_,
                                                    desc=desc,
                                                    median_filter_size=median_filter_size)
            # Add the root-mean-square to the list of distances translated
            rms = np.sqrt((translations**2).sum(axis=-1).mean())
            log.info("RMS of translations for pass %d = %f", pass_, rms)
            pass_distances[pass_] = np.sqrt((translations**2).sum(-1)).flatten()
            # Save translations for deferred calculation
            self.stage_transformations(translations=translations)
            log.debug("Finished alignment pass %d of %d", pass_, passes)
        # Plot the results if requested
        if plot_results:
            x = range(0, passes)
            if results_ax is None:
                results_ax = plots.new_axes()
            ax = results_ax
            ax.boxplot(pass_distances.swapaxes(0, 1))
            ax.set_xlabel('Pass')
            ax.set_ylabel("Distance (µm)")
            ax.set_title("$L^2$ Norms of Calculated Translations")
        # Apply result of calculations to disk (if requested)
        if commit:
            log.info("Committing final translations to disk")
            self.apply_transformations(crop=True, commit=True)
        log.info("Aligned %d passes in %d seconds", passes, time() - logstart)
        return pass_distances

    def crop_frames(self, slices):
        """Reduce the image size for all frame-data in this group.

        This operation will destructively crop all data-sets that
        contain image data, namely "framesets" and "maps". The
        argument ``slices`` controls which data is kept. For example,
        if the current frames are 128x128, the central 64x64 region
        can be kept by doing the following:

        .. code:: python

            slices = [slice(31, 95), slice(31, 95)]
            fs.crop_frames(slices=slices)

        Parameters
        ----------
        slices : tuple or list
          A slice object for each image dimension. The shape of this
          tuple should match the number of dimensions in the frame
          data to be cropped, starting with the last (columns).

        """
        slc_idx = (..., *slices)
        with self.store(mode='r+') as store:
            # Crop all the framesets and maps
            ds_names = store.frameset_names() + store.map_names()
            for ds_name in ds_names:
                new_ds = store.get_dataset(ds_name).__getitem__(slc_idx)
                store.replace_dataset(ds_name, new_ds)

    def apply_median_filter(self, size, representation='optical_depths'):
        """Permanently apply a median filter to a frameset.

        Parameters
        ----------
        size : 4-tuple
          Dimensions of the kernel to use for filtering in order of
          (time, energy, row, col).
        representation : str, optional
          Which frameset representation to use for filtering.

        """
        with self.store(mode='r+') as store:
            ds = store.get_dataset(representation)
            new_data = median_filter(ds, size=size)
            store.replace_dataset(representation, data=new_data)

    def segment_materials(self, thresholds,
                          representation='optical_depths',
                          component='real'):
        """Split the frames into different materials based on mean values.

        This is most useful with a metric that is unique to a given
        material. For example, the phase representation of
        ptychography data can be used to determine which material is
        which. Results will be stored in
        ``XanesFrameset().store().segments``.

        Parameters
        ----------
        thresholds : tuple
          The boundary values to use for segmentation. If N tresholds
          are given, the data will be split into N+1 segments.
        representation : str, optional
          What kind of data the thresholds represent.
        component : str, optional
          Complex-value representation to use. Default is real
          value. For ptychography data, 'phase' is probably better.

        """
        # Load the intensity data
        with self.store() as store:
            Is = store.get_dataset(representation)
            Is = np.mean(Is, axis=1)
        Is = get_component(Is, component)
        out = np.empty_like(Is, dtype='uint16')
        # Segment values below the first treshold
        out[Is<thresholds[0]] = 0
        # Segment values between two thresholds
        for idx, th in enumerate(thresholds[:-1]):
            next_th = thresholds[idx+1]
            is_th = np.logical_and(th <= Is, Is < next_th)
            out[is_th] = idx + 1
        # Segment values above the last threshold
        out[Is > thresholds[-1]] = len(thresholds)
        # Save data to data store
        with self.store(mode='r+') as store:
            store.segments = out

    def label_particles(self, min_distance=20):
        """Use watershed segmentation to identify particles.

        Parameters
        ----------
        min_distance : int, optional
          Controls how selective the algorithm is at grouping areas
          into particles. Lower numbers means more particles, but
          might split large particles into two.

        """
        with self.store('r+') as store:
            logstart = time()
            frames = store.optical_depths[()]
            # Average across all timesteps
            frames = np.median(frames, axis=0)
            Es = np.median(store.energies, axis=0)
            particles = xm.particle_labels(frames=frames, energies=Es,
                                           edge=self.edge,
                                           min_distance=min_distance)
            store.particle_labels = particles
            # Save metadata about where the data came from
            getattr(store.particle_labels, 'attrs', {})['frame_source'] = 'optical_depths'
        # Logging
        log.info("Calculated particle labels in %d sec", time() - logstart)

    def particle_series(self, map_name="whiteline_max"):
        """Generate median values from map_name across each particle.

        Returns: A 2D array where the first dimension is particles and
        the second is the first dimension of the map dataset (usually time).

        """
        steps = []
        with self.store() as store:
            data = store.get_dataset(map_name)
            for stepdata in data:
                particles = self.particle_regions(intensity_image=stepdata)
                def particle_value(particle):
                    im = particle.intensity_image
                    mask = particle.image
                    return np.nanmedian(im[mask])
                vals = list(map(particle_value, particles))
                steps.append(vals)
        # Convert from (steps, particles) to (particles, steps)
        steps = np.array(steps)
        steps = np.transpose(steps)
        return steps

    def particle_regions(self, intensity_image=None, labels=None):
        """Return a list of regions (1 for each particle) sorted by area.
        (largest first). This requires that the `label_particles`
        method be called first.

        Arguments
        ---------
        intensity_image : np.ndarray, optional
          2D array passed on to the skimage `regionprops` function to
          determine what shows up in the image for each particle.
        labels : np.ndarray, optional
          Array of the same shape as the map, with the particles
          segmented. If None (default), the `particle_labels`
          attribute of the TXM store will be used.
        
        """
        with self.store() as store:
            if labels is None:
                labels = store.particle_labels
            regions = measure.regionprops(labels[()],
                                          intensity_image=intensity_image)
            # Put in order of descending area
            regions.sort(key=lambda p: p.area, reverse=True)
        return np.array(regions)
    
    def plot_mean_frame(self, ax=None, component="modulus",
                        representation="optical_depths",
                        cmap="bone", timeidx=..., *args, **kwargs):
        """Plot the mean image from the selected frames.
        
        Parameters
        ==========
        ax : mpl.Axes, optional
          An axes object to receive the plot. If ommitted, new Axes
          will be created.
        component : str, optional
          Which component (real, imag, modulus, phase) to plot. Only
          relevant for complex-valued data.
        representation : str, optional
          Which dataset representation to use.
        cmap : str, optional
          Matplotlib colormap to use for the image.
        timeidx : tuple or int, optional
          Numpy index for which timestep to include. May include
          slices for either (timestep, energy), eg. ``timeidx=(1,
          slice(10, 15))`` will only select 5 energies in the first
          timestep.
        *args, **kwargs :
          Passed to the matplotlib ``imshow`` function.

        Returns
        =======
        artist
          The imshow ImageArtist.
        """
        if ax is None:
            ax = plots.new_image_axes()
        with self.store() as store:
            data = store.get_dataset(representation)[timeidx]
            mean_axis = tuple(range(data.ndim - 2))
            data = np.mean(data, axis=mean_axis)
            ax_unit = store.pixel_unit
        data = get_component(data, component)
        artist = ax.imshow(data,
                           extent=self.extent(representation=representation),
                           cmap=cmap, *args, **kwargs)
        # Decorate the axes
        ax.set_xlabel(ax_unit)
        ax.set_ylabel(ax_unit)
        return artist

    def mean_frame(self, representation="optical_depths"):
        """Return the mean value with the same shape as an individual
        frame.

        """
        with self.store() as store:
            Is = store.get_dataset(representation)
            frame_shape = self.frame_shape(representation)
            Is = np.reshape(Is, (-1, *frame_shape))
            mean = np.mean(Is, axis=0)
        return mean

    def frame_shape(self, representation="optical_depths"):
        """Return the shape of the individual energy frames."""
        with self.store() as store:
            imshape = store.get_dataset(representation).shape[-2:]
        return imshape
    
    def pixel_size(self, representation='optical_depths', timeidx=0):
        """Return the size of the pixel (with units set by ``pixel_unit``)."""
        with self.store() as store:
            # Filter to only the requested frame
            pixel_size = store.pixel_sizes[timeidx]
            # Take the median across all pixel sizes (except the xy dim)
            pixel_size = np.median(pixel_size)
        return pixel_size

    def pixel_unit(self):
        """Return the unit of measure for the size of a pixel."""
        with self.store() as store:
            unit = store.pixel_unit
        return unit

    def spectra(self):
        """Return a two-dimensional array of spectra for all the pixels in
        shape of (pixel, energy).

        """
        with self.store() as store:
            E_axis = 1
            ODs = store.optical_depths
            ODs = np.moveaxis(ODs, E_axis, -1)
            spectra = np.reshape(ODs, (-1, self.num_energies))
        return spectra
    
    def line_spectra(self, xy0: Tuple[int, int], xy1: Tuple[int, int],
                     representation="optical_depths",
                     timeidx=0, frame_filter=False, frame_filter_kw={}):
        """Return an array of spectra on a line between two points.
        
        This is effectively nearest neighbor interpolation between two (x,
        y) pairs on the frames.
        
        Returns
        -------
        spectra : np.ndarray
          A 2D array of spectra, one for each point on the line.
        
        Parameters
        ----------
        xy0 : 2-tuple
          Starting point for the line.
        xy1 : 2-tuple
          Ending point for the line.
        representation : str, optional
          Which type of data to use for extracting line profiles.
        timeidx : int, optional
          Which time step to use for extracting line profiles.
        frame_filter : bool, str, optional
          Whether to first apply an edge filter mask to the data
          before calculating line profile.
        frame_filter_kw : dict, optional
          Extra keyword arguments to pass to the edge_mask() method.
        
        """
        xy0 = xycoord(*xy0)
        xy1 = xycoord(*xy1)
        # Convert xy to pixels
        shape = self.frame_shape(representation=representation)
        px0 = xy_to_pixel(xy0, extent=self.extent(representation), shape=shape)
        px1 = xy_to_pixel(xy1, extent=self.extent(representation), shape=shape)
        # Make a line with the right number of points
        length = int(np.hypot(px1.horizontal-px0.horizontal, px1.vertical-px0.vertical))
        x = np.linspace(px0.horizontal, px1.horizontal, length)
        y = np.linspace(px0.vertical, px1.vertical, length)
        # Check if an edge mask is needed
        mask = self.frame_mask(mask_type=frame_filter, **frame_filter_kw)
        # Extract the values along the line
        with self.store(mode='r') as store:
            frames = store.get_dataset(representation)[timeidx]
            frames = np.ma.array(frames, mask=np.broadcast_to(mask, frames.shape))
            spectra = frames[:, y.astype(np.int), x.astype(np.int)]
            spectra = np.swapaxes(spectra, 0, 1)
        # And we're done
        return spectra
    
    def fitting_param_names(self, representation="fit_parameters"):
        """Get the human-readable names of the fit parameters."""
        with self.store() as store:
            name_str = store.get_dataset(representation).attrs['parameter_names']
        # Convert string to tuple
        names = name_str[1:-1].split(',')
        names = [n.strip() for n in names]
        return names

    def spectrum(self, pixel=None, frame_filter=False, frame_filter_kw: Mapping={},
                 normalize=False,
                 representation="optical_depths", index=0,
                 derivative=0):
        """Collapse the frameset down to an energy spectrum.

        The x and y dimensions will be averaged to give the final
        intensity at each energy. The ``index`` parameter will be used
        to select a timepoint. If index is a slice(), or something
        similar, you can retrieve mutliple spectra as a list.

        Parameters
        ----------
        pixel : tuple, optional
          A 2-tuple that causes the returned series to represent
          the spectrum for only 1 pixel in the frameset. If None, a
          larger part of the frame will be used, depending on the
          other arguments.
        frame_filter : str or bool, optional
          Allow the user to define which type of mask to apply.
          (e.g. 'edge', 'contrast', None)
        normalize : bool, optional
          If true, the spectrum will be normalized based on the
          nature of the *edge*.
        frame_filter_kw
           Additional arguments to be used for producing an
           frame_mask.  See
           :meth:`~xanespy.xanes_frameset.XanesFrameset.frame_mask`
           for possible values.
        representation : str, optional
          What kind of data to use for creating the spectrum. This
          will be passed to TXMstore.get_dataset()
        index : int or slice, optional
          Which step in the frameset to use. When used to index
          store().optical_depths, this should return a 3D or 4D array
          like (energy, rows, columns).
        derivative : int, optional
          Calculate a derivative of the spectrum before returning
          it. If less than 1 (default), no derivative is calculated.

        Returns
        -------
        spectrum : pd.Series
          A pandas Series with the spectrum, or a list of pandas
          Series if ``index`` parameter is a slice.

        """
        # Retrieve data
        with self.store() as store:
            energies = store.energies[index]
            if pixel is not None:
                pixel = Pixel(*pixel)
                # Get a spectrum for a single pixel
                spectrum_idx = (index, ..., pixel.vertical, pixel.horizontal)
                spectrum = store.get_frames(representation)[spectrum_idx]
            else:
                frames = store.get_frames(representation)[index]
                frames_shape = frames.shape[:-2]
                mask = self.frame_mask(mask_type=frame_filter, **frame_filter_kw)
                frames = np.ma.array(frames, mask=np.broadcast_to(array=mask, shape=(*frames_shape, *mask.shape)))
                # Take average of all pixel frames
                flat = (*frames.shape[:frames.ndim-2], -1) # Collapse image dimension
                spectrum = np.mean(np.reshape(frames, flat), axis=-1)
            # Combine into a series
            if spectrum.shape == energies.shape:
                index = energies
            else:
                index = None
            # Calculate the derivative (gradient) if requested
            if derivative > 0 and index is not None:
                for i in range(derivative):
                    spectrum = np.gradient(spectrum, np.array(index))
            # Normalize
            if normalize:
                # Adjust the limits of the spectrum to be between 0 and 1
                spectrum = self.edge.normalize(spectrum, energies=index)
            # Convert to pandas series
            if spectrum.ndim == 1:
                series = pd.Series(spectrum, index=index)
            elif spectrum.ndim == 2:
                series = [pd.Series(s) for s in spectrum]
            else:
                series = spectrum
        return series
    
    def plot_spectrum(self, ax=None, pixel=None, norm_range=None,
                      normalize=False, representation:
                      str="optical_depths", show_fit=False,
                      frame_filter=False, frame_filter_kw: Mapping =
                      {}, linestyle=":", timeidx: int=0, voffset=0,
                      *args: Any, **kwargs: Any):
        """Calculate and plot the xanes spectrum for this field-of-view.
        
        Arguments
        ---------
        ax : optional
          matplotlib axes object on which to draw
        pixel : 2-tuple, optional
          Coordinates of a specific pixel on the image to plot.
        normalize : bool, optional
          If truthy, will normalize the spectrum based on the behavior
          of the XAS absorbance edge.
        show_fit : bool, optional
          If truthy, will use the edge object to fit the data and plot
          the resulting fit line.
        frame_filter : bool, optional
          If truthy, will allow the user to define a type of mask
          to apply to the data (e.g  'edge', 'contrast', None)
        frame_filter_kw : dict, optional
          **kwargs to be passed into xp.XanesFrameset.frame_mask()
        timeidx
          Which timestep index to use for retrieving data.
        voffset
          Vertical offset for this plot.
        args, kwargs : optional
          Passed to plotting functions.

        """
        if show_fit:
            raise NotImplementedError("`show_fit` parameter coming soon")
        if norm_range is not None:
            norm = Normalize(*norm_range)
        else:
            norm = None
        spectrum = self.spectrum(pixel=pixel,
                                 frame_filter=frame_filter,
                                 normalize=normalize,
                                 frame_filter_kw=frame_filter_kw,
                                 representation=representation,
                                 index=timeidx)
        edge = self.edge
        if ax is None:
            ax = plots.new_axes()
        scatter = plots.plot_spectrum( # type: ignore # https://github.com/python/mypy/issues/2582
            spectrum=spectrum + voffset, ax=ax, energies=spectrum.index,
            norm=norm, *args, **kwargs)
        # Plot lines at edge of normalization range or indicate peak positions
        if edge is not None:
            edge.annotate_spectrum(ax=ax)
        return scatter

    def edge_mask(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("edge_mask is deprecated, use frame_mask() instead."))
        return self.frame_mask(mask_type='edge', *args, **kwargs)

    @functools.lru_cache()
    def frame_mask(self, mask_type=None, sensitivity: float = 1, min_size: int = 0, frame_idx='mean',
                   representation='optical_depths') -> np.ndarray:
        """Calculate a mask for what is likely active material based on either
        the edge or the contrast of the first time index.
        
        Parameters
        ----------
        mask_type : str, bool
          Sets which type of mask to apply to the frames:
          'edge_mask', 'contrast_mask', or ``None``
        sensitivity : optional
          A multiplier for the otsu value to determine the actual
          threshold. Higher values give better statistics, but might
          include the active material, lower values give worse
          statistics but are less likely to include active material.
        min_size : optional
          Objects below this size (in pixels) will be removed. Passing
          zero (default) will result in no effect.
        frame_idx : tuple, optional
          A tuple of (time_index, energy_index) used to pass into
          xp.xanes_math.contrast_mask(). Allows User to create a
          contrast map from an individual (timestep - energy) rather
          than the mean image.
        representation : str, optional
          What dataset to use as input for calculating the frame mask.
        
        Returns
        -------
        mask
          An array matching the frame shape where true values indicate
          pixels that should be considered background.
        
        """
        with self.store() as store:
            # Check for complex values and return representation data
            frames = get_component(store.get_dataset(representation), 'real')
            #ODs = np.real(store.optical_depths[()])
            if mask_type == 'contrast':
                # Create mask based on contrast maps
                mask = xm.contrast_mask(frames=frames,
                                        sensitivity=sensitivity,
                                        min_size=min_size,
                                        frame_idx=frame_idx)
            elif mask_type == 'edge':
                # Create mask based on edge jump
                mask = self.edge.mask(frames=frames,
                                      energies=store.energies,
                                      sensitivity=sensitivity,
                                      min_size=min_size)
            elif not mask_type:
                # Create blank mask array
                mask = np.zeros(shape=store.intensities.shape[-2:], dtype='bool')
            else:
                # Show warning if all of these fail
                raise ValueError('Incorrect *mask_type*. Valid values are `edge`, `contrast` or None.')
        return mask
    
    def fit_linear_combinations(self, sources, component='real', name='linear_combination',
                                representation="optical_depths", *args, **kwargs):
        """Take a set of sources and fit the spectra with them.
        
        Saves to the representation "linear_combinations". Also
        creates "linear_combination_sources" and
        "linear_combination_residuals" datasets.
        
        Parameters
        ----------
        sources : numpy.ndarray
          Sources to use for fitting the combinations.
        component : str, optional
          Complex component to use before fitting.
        name : str, optional
          What to call the resulting dataset in the hdf file.
        representation : str, optional
          What dataset to use as input for fitting.
        args, kwargs : optional
          Passed on to ``self.fit_spectra``.
        
        Returns
        -------
        fits : numpy.ndarray
          The weights (as frames) for each source.
        residuals : numpy.ndarray
          Residual error after fitting, as maps.
        
        """
        # Convert from complex number
        sources = get_component(sources, component)
        # Prepare the fitting callable
        func = LinearCombination(sources=sources)
        # Guess initial guess: equal weights with a zero offset
        p0 = np.ones(shape=(self.num_timesteps, len(sources)+1, *self.frame_shape()))
        # Prepare parameter names
        pnames = ['c%d' % idx for idx in range(len(sources)+1)]
        pnames.append('offset')
        pnames = tuple(pnames)
        # Perform fitting
        results = self.fit_spectra(func=func, p0=p0, name=name, pnames=pnames, *args, **kwargs)
        # Save sources as metadata for due diligence
        with self.store('r+') as store:
            store.replace_dataset("%s_sources" % name, sources,
                                  context='metadata')
        return results

    def fit_kedge(self, quiet=False, ncore=None):
        """Fit all spectra with a K-Edge curve.

        Parameters
        ==========
        quiet : bool, optional
          If true, no progress bar will be displayed.
        ncore : int, optional
          How many processes to use in the pool. See
          :func:`~xanespy.utilities.nproc` for more details.

        """
        # Prepare intial guess at parameters
        k_edge = KCurve(x=self.energies())
        # Fit all the spectra
        self.fit_spectra(k_edge, quiet=quiet, ncore=ncore)
        # Use higher energy precision to calculate whitelines
        new_Es = np.linspace(*self.edge.edge_range, num=200)
        kcurve = KCurve(new_Es)
        with self.store() as store:
            params = store.get_dataset('{}_parameters'.format(kcurve.name))[()]
        # Reshape to be flat
        params = np.moveaxis(params, 1, -1)
        map_shape = params.shape[:-1]
        p_shape = params.shape[-1]
        params = params.reshape(-1, p_shape)
        if not quiet:
            params = tqdm.tqdm(params, desc="Calculating whitelines", unit='px', total=params.shape[0])
        # Process all the spectra
        with mp.Pool(nproc(ncore)) as pool:
            _find_whiteline = functools.partial(find_whiteline, curve=kcurve)
            whitelines = pool.map(_find_whiteline, params, chunksize=1000)
        # Return to the original shape
        whitelines = np.array(whitelines)
        whitelines = whitelines.reshape(map_shape)
        # Save data to disk
        with self.store(mode='a') as store:
            store.whiteline_fit = whitelines
            try:
                store.whiteline_fit.attrs['frame_source'] = 'optical_depths'
            except AttributeError:
                pass
    
    def fit_spectra(self, func, p0=None, pnames=None, name=None,
                    frame_filter='edge', frame_filter_kw: Mapping={},
                    nonnegative=False, component='real',
                    representation='optical_depths', dtype=None,
                    quiet=False, ncore=None):
        """Fit a given function to the spectra at each pixel.
        
        The fit parameters will be saved in the HDF dataset
        "{name}_params" based on the parameter ``name``. RMS residuals
        for each pixel will be saved in "{name}_residuals".
        
        Parameters
        ----------
        func : callable, optional
          The function that will be used for fitting. It should match
          ``func(p0, p1, ...)`` where p0, p1, etc are the fitting
          parameters. Some useful functions can be found in the
          ``xanespy.fitting`` module. If not given, a default curve
          based on the XAS edge will be used.
        p0 : np.ndarray, optional
          Initial guess for parameters, with similar dimensions to a
          frameset. Example, fitting 3 sources (plus offset) for a (1,
          40, 256, 256) 40-energy frameset requires p0 to be (1, 4,
          256, 256). If not given, default parameters will be guessed
          based on the curve if the curve has a *guess_params* method
          matching the call signature of
          :meth:`~xanespy.fitting.Curve.guess_params`.
        pnames : str, optional
          An object with __str__ that will be saved as metadata giving
          the parameters' names.
        name : str, optional
          What to call this fit in the HDF5 file. Use this to allow
          subsequent fits against the same dataset to be saved. If
          ``None``, we will attempt look for ``func.name``, then
          lastly we'll use "fit".
        frame_filter : str or bool, optional
          Allow the User to define which type of mask to apply.
          (e.g 'edge', 'contrast', None)
        frame_filter_kw
          Additional arguments to be used for producing an frame_mask.
           See :meth:`~xanespy.xanes_frameset.XanesFrameset.frame_mask`
           for possible values.
        nonnegative : bool, optional
          If true (default), negative parameters will be avoided. This
          can also be a tuple to allow for fine-grained control. Eg:
          (True, False) will only punish negative values in the first
          of the two parameters.
        component : str, optional
          What to use for complex-valued functions.
        representation : str, optional
          Which set of frames to use for fitting.
        dtype : numpy.dtype, optional
          Specify a datatype to convert all values to. This helps
          avoid fit failure due to precision errors. If omitted, the
          function will also check for ``func.dtype``.
        quiet : bool, optional
          Whether to suppress the progress bar, etc.
        ncore : int, optional
          How many processes to use in the pool. See
          :func:`~xanespy.utilities.nproc` for more details.
        
        Returns
        -------
        params : numpy.ndarray
          The fit parameters (as frames) for each source.
        residuals : numpy.ndarray
          Residual error after fitting, as maps.
        
        Raises
        ------
        GuessParamsError
          If the *func* callable doesn't have a *guess_params*
          method. This can be solved by either using a callable with a
          *guess_params()* method, or explicitly supplying *p0*.
        
        """
        # Get data
        with self.store() as store:
            frames = get_component(store.get_dataset(representation), component)
        if frame_filter:
            frames[..., self.frame_mask(mask_type=frame_filter, **frame_filter_kw)] = np.nan
        # Get the default curve name if necessary
        if name is None:
            name = getattr(func, 'name', 'fit')
        # Make a default set of guess for params
        if p0 is None:
            spectra = self.spectra()
            try:
                p0 = guess_p0(func, spectra, edge=self.edge, quiet=quiet, ncore=ncore)
            except NotImplementedError:
                raise exceptions.GuessParamsError(
                    "Fitting function {} has no ``guess_params`` method. "
                    "Initial parameters *p0* is required.".format(func)) from None
            p0 = p0.reshape((self.num_timesteps, *self.frame_shape(), -1))
            p0 = np.moveaxis(p0, -1, 1)
        # Make sure p0 is the right shape
        if p0.ndim < 4:
            p0 = prepare_p0(p0, self.frame_shape(), self.num_timesteps)
        # Reshape to have energy last
        spectra = np.moveaxis(frames, 1, -1)
        map_shape = spectra.shape[:-1]
        spectra = spectra.reshape((-1, spectra.shape[-1]))
        p0 = np.moveaxis(p0, 1, -1)
        p0 = p0.reshape((-1, p0.shape[-1]))
        # Make sure data-types match to avoid precision errors
        if dtype is None:
            dtype = getattr(func, 'dtype', None)
        if dtype is not None:
            spectra = spectra.astype(dtype)
            p0 = p0.astype(dtype)
        # Perform the actual fitting
        params, residuals = fit_spectra(observations=spectra,
                                        func=func, p0=p0,
                                        nonnegative=nonnegative,
                                        quiet=quiet, ncore=ncore)
        # Reshape to have maps of LC source weight
        params = params.reshape((*map_shape, -1))
        params = np.moveaxis(params, -1, 1)
        residuals = residuals.reshape(map_shape)
        # Ensure correct datatype
        if dtype is not None:
            params = params.astype(dtype)
            residuals = residuals.astype(dtype)
        # Save data to disk
        if pnames is None:
            pnames = getattr(func, 'param_names', '')
        with self.store('r+') as store:
            store.replace_dataset("%s_parameters" % name, params,
                                  context='frameset',
                                  attrs={'parameter_names': str(pnames)})
            store.replace_dataset("%s_residuals" % name, residuals,
                                  context='map',
                                  attrs={'frame_source': representation})
            # Logging
            msg = "Saving fit data to '{name}_parameters' and '{name}_residuals'."
            msg = msg.format(name=name)
            log.info(msg)
        # Return the results to the user
        return params, residuals

    def calculate_whitelines(self, edge_mask=False):
        """Calculate and save a map of the whiteline position of each pixel by
        calculating the energy of simple maximum optical_depth.

        Arguments
        ---------
        - edge_mask : If true, only pixels passing the edge_mask will
          be fit and the remaning pixels will be set to a default
          value. This can help reduce computing time.

        """
        with self.store() as store:
            frames = store.optical_depths
            # Insert two axes into energies for image row/cols
            energies = store.energies[()][:, np.newaxis, np.newaxis, :]
            # Convert numpy axes to be in (pixel, energy) form
            spectra = np.moveaxis(frames, 1, -1)
            # Calculate whiteline positions
            whitelines = xm.direct_whitelines(spectra=spectra,
                                              energies=energies,
                                              edge=self.edge)
        # Save results to disk
        with self.store(mode='r+') as store:
            store.whiteline_max = whitelines
            store.whiteline_max.attrs['frame_source'] = 'optical_depths'

    def calculate_maps(self):
        """Generate a set of maps based on pixel-wise Xanes spectra: whiteline
        position, particle labels.
        
        This method does not do any advanced analysis, so something
        like ``~xanespy.xanes_frameset.XanesFrameset.fit_spectra`` may
        be necessary.
        
        """
        self.calculate_whitelines()
        self.calculate_mean_frames()
        # Calculate particle_labels
        self.label_particles()
    
    def calculate_mean_frames(self):
        # Calculate the mean and median maps
        with self.store(mode="r+") as store:
            energy_axis = -3
            for fs_name in store.frameset_names():
                mean = np.mean(store.get_frames(fs_name), axis=energy_axis)
                mean_name = fs_name + '_mean'
                log.info('Creating new map %s', mean_name)
                store.replace_dataset(mean_name, data=mean, context='map')
                store.get_dataset(mean_name).attrs['frame_source'] = fs_name
    
    def plot_signals(self, cmap="viridis"):
        """Plot the signals from the previously extracted data. Requires that
        self.store().signals and self.store().signal_weights be set.
        
        """
        with self.store() as store:
            signals = store.signals[()]
            n_signals = signals.shape[0]
            weights = store.signal_weights[()]
            energies = store.energies[0]
            px_unit = store.pixel_unit
        figsize = (10, 3*signals.shape[0])
        fig, ax_list = pyplot.subplots(signals.shape[0], 2,
                                       figsize=figsize, squeeze=False)
        # Get min and max values for the plots
        Range = namedtuple('Range', ('min', 'max'))
        imrange = Range(min=np.min(weights), max=np.max(weights))
        norm = Normalize(imrange.min, imrange.max)
        # specrange = Range(min=np.min(signals), max=np.max(signals))
        # Get predicted complex-valued spectra
        w_inv = np.linalg.pinv(weights.reshape((-1, n_signals)))
        predicted_signals = np.dot(w_inv, self.spectra())
        # Plot each signal and weight
        extent = self.extent(representation="optical_depths")
        for idx, signal in enumerate(signals):
            ax1, ax2 = ax_list[idx]
            plots.remove_extra_spines(ax2)
            plots.plot_txm_map(data=weights[0, ..., idx], ax=ax1,
                               norm=norm, edge=self.edge,
                               extent=extent)
            ax1.set_xlabel(px_unit)
            ax1.set_ylabel(px_unit)
            ax1.set_title("Signal {idx} Weights".format(idx=idx))
            # Add a colorbar to the image axes
            mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)
            mappable.set_array(weights[0, ..., idx])
            pyplot.colorbar(mappable=mappable, ax=ax1)
            # Plot the extracted signal and predicted complex signal
            ax2.plot(energies, signal, marker='x', linestyle="None")
            cmplx_signal = predicted_signals[idx]
            cmplx_kwargs = {
                'linestyle': '--',
                'marker': "None",
            }
            ax2.plot(energies, np.abs(cmplx_signal), **cmplx_kwargs)
            ax2.plot(energies, np.angle(cmplx_signal), **cmplx_kwargs)
            ax2.plot(energies, np.real(cmplx_signal), **cmplx_kwargs)
            ax2.plot(energies, np.imag(cmplx_signal), **cmplx_kwargs)
            ax2.legend(["Refined Mod", "Predicted Mod", "Phase", "Real", "Imag"], fontsize=8)
            # ax2.set_ylim(*specrange)
            ax2.set_title("Signal Component {idx}".format(idx=idx))

    def plot_signal_map(self, ax=None, signals_idx=None, interpolation=None):
        """Plot the map of signal strength for signals extracted from
        self.calculate_signals().

        Arguments
        ---------
        - ax : A matplotlib Axes object. If "None" (default) a new
          axes object is created.

        - signals_idx : Indices of which signals to plot. This will be
        passed as a numpy array index. Special value None (default)
        means first three signals will be plotted.

        - interpolation : str
          How to smooth the image when plotting.
        """
        # Plot the composite signal map data
        with self.store() as store:
            composite = store.signal_map[0]
            total_signals = store.signals.shape[0]
            px_unit = store.pixel_unit
        # Determine how many signals to plot
        if signals_idx is None:
            n_signals = min(total_signals, 3)
            signals_idx = slice(0, n_signals, 1)
        composite = composite[..., signals_idx]
        # Do the plotting
        extent = self.extent(representation="optical_depth")
        artist = plots.plot_composite_map(composite, extent=extent,
                                          ax=ax, interpolation=interpolation)
        ax = artist.axes
        ax.set_xlabel(px_unit)
        ax.set_ylabel(px_unit)
        ax.set_title("Composite of signals {}".format(signals_idx))
    
    def calculate_signals(self, n_components=2, method="nmf",
                          frame_source='optical_depths',
                          frame_filter='edge', frame_filter_kw: Mapping={}):
        """Extract signals and assign each pixel to a group, then save the
        resulting RGB cluster map.
        
        Parameters
        ==========
        n_components : int, optional
          The number of signals and number of clusters into which the
          data will be separated.
        method : str, optional
          The technique to use for extracting signals. Available options are
          'nmf' and 'pca'.
        frame_source : str, optional
          Name of the frame-set to use as the input data.
        frame_filter : str or bool, optional
          Allow the User to define which type of mask to apply.
          (e.g 'edge', 'contrast', None)
        frame_filter_kw
          Additional arguments to be used for producing an frame_mask.
           See :meth:`~xanespy.xanes_frameset.XanesFrameset.frame_mask`
           for possible values.
        
        Returns
        =======
        signals : np.
        """
        frame_source = "optical_depths"
        msg = "Performing {} signal extraction with {} components"
        log.info(msg.format(method, n_components))
        frame_shape = self.frame_shape()
        # Retrieve the optical_depths and convert to spectra
        with self.store() as store:
            As = store.get_dataset(frame_source)
            # Collapse to (frame, pixel) shape
            n_timesteps = As.shape[0]
            assert n_timesteps == 1  # We can't currently handle timesteps
            num_spectra = np.prod(frame_shape)
            As = np.reshape(As, (-1, num_spectra))
            spectra = np.moveaxis(As, 0, 1)
        # Get the edge mask so only active material is included
        dummy_mask = np.ones(frame_shape, dtype=np.bool)
        # See if we need a mask
        if frame_filter:
            # Clear caches to make sure we don't use stale mask data
            self.clear_caches()
            # Now retrieve the mask - redundant, but makes it easier to place into code
            try:
                mask = ~self.frame_mask(mask_type=frame_filter,  **frame_filter_kw)
            except exceptions.XanesMathError as e:
                log.error(e)
                log.warning("Failed to calculate mask, using all pixels.")
                warnings.warn('Failed to calculate mask, using all pixels')
                mask = dummy_mask
            else:
                log.debug("Using edge mask for signal extraction")
        else:
            log.debug("No edge mask for signal extraction")
            mask = dummy_mask
        # Separate the data into signals
        if method.lower() == 'nmf':
            signals, weights = xm.extract_signals_nmf(
                spectra=spectra[mask.flatten()], n_components=n_components)
        elif method.lower() == 'pca':
            signals, weights = xm.extract_signals_pca(
                spectra=spectra[mask.flatten()], n_components=n_components)
        else:
            raise ValueError('Recieved Invalid Method : {method}'.format(method=method))
        # Reshape weights into frame dimensions
        weight_shape = (n_timesteps, *frame_shape, n_components)
        weight_frames = np.zeros(shape=weight_shape)
        weight_frames[:, mask, :] = weights
        # Move the signal axis so it's in the same place as the energy axis
        weight_frames = np.moveaxis(weight_frames, -1, 1)
        # Save the calculated signals and weights
        method_names = {
            "nmf": "Non-Negative Matrix Factorization",
            "pca": "Principal Component Analysis",
        }
        with self.store(mode="r+") as store:
            store.signals = signals
            store.signal_method = method_names[method.lower()]
            store.signal_weights = weight_frames
            try:
                store.signals.attrs['frame_source'] = frame_source
            except AttributeError:
                pass
        # ## Construct a composite RGB signal map
        # Calculate a mean frame to normalize the weights
        mean = np.imag(self.mean_frame())
        mean = np.reshape(mean, (*mean.shape, 1))
        mean = np.repeat(mean, n_components, axis=-1)
        composite = weight_frames / np.max(weight_frames)
        # composite = weight_frames / mean
        # Make sure the composite has three color components (RGB)
        with self.store(mode="r+") as store:
            store.signal_map = composite
            try:
                store.signal_map.attrs['frame_source'] = frame_source
            except AttributeError:
                pass
        # ## Perform k-means clustering
        k_results = cluster.k_means(weights, n_clusters=n_components)
        centroid, labels, intertia = k_results
        label_frame = np.zeros(shape=frame_shape)
        label_frame[mask] = labels
        label_frame = label_frame.reshape(1, *frame_shape)
        # Save the resulting k-means cluster map
        with self.store(mode="r+") as store:
            store.cluster_fit = label_frame
        # Return the signals and weights
        return signals, weight_frames

    @functools.lru_cache(maxsize=2)
    def map_data(self, *, timeidx=0, representation="optical_depths"):
        """Return map data for the given time index and representation.
        
        If `representation` is not really mapping data, then the
        result will have more dimensions than expected.
        
        Parameters
        ----------
        timeidx
          Index for the first dimension of the combined data array. If
          the underlying map data has only 2 dimensions, this
          parameter is ignored.
        representation
          The group name for these data. Eg "optical_depths",
          "whiteline_map", "intensities"

        Returns
        -------
        map_data : np.ndarray
          A 2-dimensional array with the form (row, col).

        """
        with self.store() as store:
            map_data = store.get_dataset(representation)
            # Validate map data if it's an actual HDF file
            is_map_data = (not hasattr(map_data, 'attrs') or
                           map_data.attrs.get('context') == 'map')
            if not is_map_data:
                # Not actually a map, so return none
                map_data = None
            elif map_data.ndim > 2:
                map_data = map_data[timeidx]
            else:
                map_data = np.array(map_data)
        return map_data

    def timestamps(self, relative=False, t0=None):
        """Retrieve an array with the timestamp for each scan.

        This will be a numpy array with the ``datetime64`` (absolute)
        or ``float64`` (relative) dtype. Each frame has both a start
        and end time-stamp. Depending on the beamline, the individual
        energy frames might have timestamps based on the whole XANES
        scan. These timestamps are timezone naive, with a timezone
        based on the specifics of the beamline.

        Parameters
        ----------
        relative : bool, optional
          If true, the timestamps will be in seconds from ``t0``.
        t0 : datetime, optional
          If ``relative`` is truthy, this value be used as the start
          time for the frameset. If omitted, the earliest timestamp in
          the frameset will be used.

        Returns
        -------
        timestamps : np.ndarray
          The timestamps for each energy frame with a shape described
          as (num_timesteps, num_energies, 2), where the last axis is
          for the start and end of the frame.

        """
        datetime_dtype = 'datetime64[ms]'
        with self.store() as store:
            timestamps = store.timestamps[()]
        timestamps = timestamps.astype(datetime_dtype)
        # Convert to relative timestamps if necessary
        if relative:
            if t0 is None:
                t0 = np.min(timestamps)
            else:
                t0 = np.datetime64(t0)
            t0 = t0.astype(datetime_dtype)
            timestamps = (timestamps - t0).astype(np.float64) / 1000
        # Return calculated timestamps
        return timestamps

    @functools.lru_cache(maxsize=2)
    def frames(self, representation="optical_depths", timeidx=0):
        """Return the frames for the given time index.

        If `representation` is really mapping data, then the source
        frames will be returned.

        Parameters
        ----------
        timeidx : int
          Index for the first dimension of the combined data array.
        representation : str
          The group name for these data. Eg "optical_depths",
          "whiteline_map", "intensities"


        Returns
        -------
        frames : np.ndarray
          A 3-dimensional array with the form (energy, row, col).

        """
        with self.store() as store:
            frames = store.get_frames(name=representation)[timeidx]
        return frames

    @functools.lru_cache(maxsize=2)
    def energies(self, timeidx=0):
        """Return the array of beam energies for the given time index.

        Returns
        -------
        energies: np.ndarray
          A 1-dimensional array with the energy for each frame.

        """
        with self.store() as store:
            energies = store.energies[timeidx]
        return energies

    def subtract_surroundings(self, sensitivity: float=1.):
        """Use the edge mask to separate "surroundings" from "sample", then
        subtract the average surrounding optical_depth from each
        frame. This effectively removes effects where the entire frame
        is brighter from one energy to the next.

        Parameters
        ----------
        sensitivity : optional
          A multiplier for the otsu value to determine the actual
          threshold. Higher values give better statistics, but might
          include the active material, lower values give worse
          statistics but are less likely to include active material.

        """
        self.clear_caches()
        with self.store(mode='r+') as store:
            log.debug('Subtracting surroundings')
            # Get the mask and make it compatible with XANES shape
            mask = np.broadcast_to(self.edge_mask(sensitivity=sensitivity),
                                   store.optical_depths.shape[1:])
            # Go through each timestep one at a time (to avoid using all the memory)
            for timestep in range(store.optical_depths.shape[0]):
                bg = store.optical_depths[timestep][mask]
                bg = bg.reshape((*store.optical_depths.shape[1:2], -1))
                bg = bg.mean(axis=-1)
                msg = "Background intensities calculated: {}"
                log.debug(msg.format(bg))
                bg = broadcast_reverse(bg, store.optical_depths.shape[1:])
                # Save the resultant data to disk
                store.optical_depths[timestep] = store.optical_depths[timestep] - bg
    
    @functools.lru_cache(maxsize=64)
    def extent(self, representation='intensities', idx=...):
        """Determine physical dimensions for axes values.
        
        If an index is given, it will first be applied to the frames
        array. For any remaining dimensions besides the last two, the
        median will be taken. For an array of extents for each frame,
        use the ``extent_array`` method.
        
        Arguments
        ---------
        representation : str, optional
          Name for which dataset to use.
        idx : int, optional
          Index for choosing a frame. Any valid numpy index is
          allowed, eg. ``...`` (default) uses all frame.
        
        Returns
        -------
        extent : tuple
          The spatial extent for the frame with order specified by
          ``utilities.Extent``
        
        """
        pixel_size = self.pixel_size(representation=representation, timeidx=idx)
        imshape = self.frame_shape(representation)
        height = imshape[0] * pixel_size
        width = imshape[1] * pixel_size
        # Calculate boundaries from image shapes
        left = -width / 2
        right = +width / 2
        bottom = -height / 2
        top = +height / 2
        extent = Extent(left=left, right=right,
                        bottom=bottom, top=top)
        return extent
    
    def plot_frame(self, idx, ax=None, cmap="bone",
                   representation="optical_depths", component='modulus', *args,
                   **kwargs):
        """Plot the frame with given index as an image.
        
        Parameters
        ==========
        idx : 2-tuple(int)
          Index of the frame to plot in order of (timestep,
          energy).
        ax : mpl.Axes, optional
          Axes to receive the plot.
        cmap : str, optional
          Maptlotlib colormap string for the image.
        component : str, optional
          The complex component (real, imag, modulus, phase) to use
          for plotting. Only applicable to complex-valued data.
        *args, **kwargs :
          Passed to the matplotlib ``imshow`` function.
        
        Returns
        =======
        artist
          The imshow ImageArtist.
        
        """
        if len(idx) != 2:
            raise ValueError("Index must be a 2-tuple in (timestep, energy) order.")
        return self.plot_mean_frame(ax=ax, component=component,
                                    representation=representation, cmap=cmap, timeidx=idx, *args,
                                    **kwargs)
        return artist
    
    @property
    def num_timesteps(self):
        with self.store() as store:
            val = store.optical_depths.shape[0]
        return val
    
    @property
    def timestep_names(self):
        with self.store() as store:
            names = store.timestep_names[()]
            names = names.astype('unicode')
        return names

    @property
    def num_energies(self):
        with self.store() as store:
            val = store.energies.shape[-1]
        return val

    def plot_map(self, ax=None, map_name="whiteline_fit", timeidx=0,
                 vmin=None, vmax=None, v0=0, median_size=0,
                 component="real", edge_filter=False, edge_filter_kw={},
                 *args, **kwargs):
        """Prepare data and plot a map of processed data.

        Parameters
        ==========
        ax : optional
          A matplotlib Axes to receive the plotted map.
        map_name : str, optional
          Which map in the HDF file to plot.
        timeidx : int, optional
          Which timestep to use (default=0).
        vmin : float, optional
          Minimum value used for image plotting.
        vmax : float, optional
          Maximum value used for image plotting.
        median_size : int, optional
          Kernel size for the median rank filter.
        component : str, optional
          If complex-valued data is found, which component to plot.
        edge_filter : bool, optional
          If true, only pixels with a considerable XAS edge will be
          shown.
        edge_filter_kw : dict, optional
          Dictionary of extra parameters to pass to the
          ``XanesFrameset.edge_mask()`` method.
        v0 : float, optional
          The zero point for mapping values. This is subtracted from
          each reported map value. This is done before vmin and vmax
          are applied.

        Returns
        =======
        artists
          A list of matplotlib artists returned when calling
          ``ax.imshow()`` or similar routines.

        """
        with self.store() as store:
            # Get the data from disk
            ds = store.get_dataset(name=map_name)
            data = get_component(ds[timeidx], component)
            if median_size > 0:
                data = median_filter(data, median_size)
            frame_source = store.frame_source(map_name)
        # Get default value ranges
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        # Apply the edge jump filter if necessary
        if edge_filter:
            mask = self.edge_mask(**edge_filter_kw)
            data = np.ma.array(data, mask=mask)
        # Plot the data
        artists = plots.plot_txm_map(
            data=(data - v0),
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            extent=self.extent(representation=frame_source),
            *args, **kwargs)
        return artists

    def plot_map_pixel_spectra(self, pixels, map_ax=None,
                               spectra_ax=None,
                               map_name="whiteline_map", timeidx=0,
                               step_size=0, *args, **kwargs):
        """Plot the frameset's map and highlight some pixels on it then plot
        those pixel's spectra on another set of axes.

        Arguments
        ---------
        pixels : iterable
          An iterable of 2-tuples indicating which (row, column)
          pixels to highlight.
        map_ax : optional
          A matplotlib axes object to put the map onto. If None, a new
          2-wide subplot will be created for both map_ax and
          spectra_ax.
        spectra_ax : optional
          A matplotlib axes to be used for plotting spectra. Will only
          be used if ``map_ax`` is not None.
        map_name : str, optional
          Name of the map to use for plotting. It will be passed to
          the TXM store object and retrieved from the hdf5 file. If
          falsy, no map will be plotted.
        timeidx : int, optional
          Index of which timestep to use (default: 0).
        args, kwargs : optional
          Passed to plots.plot_pixel_spectra()

        """
        # Create some axes if necessary
        if map_ax is None:
            fig, ax = pyplot.subplots(1, 2)
            map_ax, spectra_ax = ax
        # Get the necessary data
        with self.store() as store:
            energies = store.energies[timeidx]
            spectra = store.optical_depths[timeidx]
            # Put energy as the last axis
            spectra = np.moveaxis(spectra, 0, -1)
        if map_name:
            self.plot_map(ax=map_ax, map_name=map_name, timeidx=timeidx)
        plots.plot_pixel_spectra(pixels=pixels,
                                 extent=self.extent('optical_depths'),
                                 spectra=spectra,
                                 energies=energies,
                                 map_ax=map_ax,
                                 spectra_ax=spectra_ax,
                                 step_size=step_size)
    
    def plot_histogram(self, plotter=None, timeidx=None, ax=None,
                       vmin=None, vmax=None, goodness_filter=False,
                       representation="whiteline_fit",
                       component="real",
                       active_pixel=None, bins="energies", *args, **kwargs):
        """Use a default frameset plotter to draw a map of the chemical
        data."""
        with self.store() as store:
            map_ds = store.get_dataset(representation)
            if timeidx is None:
                data = map_ds[:]
            else:
                data = map_ds[timeidx]
        data = get_component(data, component)
        # Add bounds for the colormap if given
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Get bins for the energy steps
        edge = self.edge
        if str(bins) == "energies":
            bins = self.energies()
        artists = plots.plot_txm_histogram(data=data, ax=ax,
                                           norm=norm,
                                           cmap=self.cmap, bins=bins)
        ax = artists[0].axes
        ax.set_xlabel(representation)
        return artists

    def gui_viewer(self):
        """Launch a Qt GUI for inspecting the data."""
        cmd = os.path.join(os.path.split(__file__)[0], 'xanes_viewer.py')
        guiargs = [
            'env', 'python', cmd,
            self.hdf_filename,
            '--groupname', self.parent_name,
        ]
        if self.edge.shell == "K":
            guiargs.append('--k-edge')
        elif self.edge.shell == "L":
            guiargs.append('--l-edge')
        guiargs.append(self.edge.name)
        # Launch the GUI in a subprocess
        subprocess.run(guiargs)
        # finally:
        #     # Restore original values
        #     self.parent_name = init_parent_name
        #     self.data_name = init_data_name

    def apply_internal_reference(self):
        """Use a portion of each frame for internal reference correction.

        This function will extract an $I_0$ from the background by
        thresholding. Then calculate the optical depth by

        .. math:: OD = ln(\\frac{I_0}{I})

        This method is compatible with complex intensity data.

        """
        with self.store() as store:
            Is = store.intensities[()]
            ODs = xm.apply_internal_reference(Is)
        # Save complex image as refractive index (real part is phase change)
        with self.store('r+') as store:
            store.optical_depths = ODs

