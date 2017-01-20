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
from typing import Callable
import os
from time import time
import logging
from collections import namedtuple

import pandas as pd
from matplotlib import pyplot, cm, pyplot as plt
from matplotlib.colors import Normalize
import h5py
import numpy as np
from skimage import morphology, filters, transform,  measure
from sklearn import linear_model, cluster

from utilities import (prog, xycoord, Pixel, Extent, pixel_to_xy,
                       get_component, broadcast_reverse)
from txmstore import TXMStore
import plots
import exceptions
import xanes_math as xm
from edges import Edge


log = logging.getLogger(__name__)


class XanesFrameset():
    """A collection of TXM frames at different energies moving across an
    absorption edge. Iterating over this object gives the individual
    Frame() objects. The class assumes that the data have been
    imported into an HDF file.

    """
    active_group = ''
    cmap = 'plasma'
    _data_name = None
    # Places to store staged image transformations
    _transformations = None

    def __init__(self, filename, edge, groupname=None):
        """
        Parameters
        ----------
        filename : str, optional
          Path to the HDF file that holds these data.
        edge
          An Edge object describing the meterial's X-ray energy response
          characteristics.
       groupname : str, optional
          Top level HDF group corresponding to this frameset. This
          argument is only required if there is more than one top-level
          group.
        
        """
        self.hdf_filename = filename
        self.edge = edge
        self.parent_name = groupname
    
    def __repr__(self):
        s = "<{cls}: '{name}'>"
        return s.format(cls=self.__class__.__name__, name=self.parent_name)
    
    def hdf_path(self, representation=None):
        """Return the hdf path for the active group.
        
        Parameters
        ----------
        representation: str, optional
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
                group = store.get_frames(representation=representation)
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
        
            with self.store() as store:
                # Do stuff with the store
        """
        return TXMStore(hdf_filename=self.hdf_filename,
                        parent_name=self.parent_name,
                        data_name=self.data_name,
                        mode=mode)
    
    def clear_caches(self):
        """Clear cached function values so they will be recomputed with fresh
        data
        
        """
        self.edge_mask.cache_clear()
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
            # Figure out which timestamp is the earliest
            Ts = Ts.astype('datetime64').flatten()
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

    def apply_transformations(self, crop=True, commit=True):
        """Take any transformations staged with `self.stage_transformations()`
        and apply them. If commit is truthy, the staged
        transformations are reset.

        Returns: Transformed array of the absorbances frames.

        Arguments
        ---------
        - crop : If truthy, the images will be cropped after being
        translated, so there are not edges. If falsy, the images will
        be wrapped.

        - commit : If truthy, the changes will be saved to the HDF5
          store for absorbances, intensities and references, and the
          staged transformations will be cleared. Otherwise, only the
          absorbance data will be transformed and returned.

        - frames_name : Name of the frame group to apply this
          transformation too (eg. 'absorbances')

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
                out = store.get_frames('absorbances').value
        else:
            if commit:
                names = ['intensities', 'references', 'absorbances'] # Order matters
            else:
                names = ['absorbances']
            # Apply the transformations
            for frames_name in names:
                with self.store() as store:
                    if not store.has_dataset(frames_name):
                        continue
                    # Prepare an array to hold results
                    out = np.zeros_like(store.get_frames(frames_name))
                    # Apply transformation
                    xm.transform_images(data=store.get_frames(frames_name),
                                        transformations=self._transformations,
                                        out=out)
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
                        store.set_frames(frames_name, out)
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
                     blur=None,
                     method: str="cross_correlation",
                     template=None,
                     passes=1,
                     commit=True,
                     component="modulus",
                     plot_results=True):
        """Use cross correlation algorithm to line up the frames. All frames
        will have their sample position set to (0, 0) since we don't
        know which one is the real position. This operation will
        interpolate between pixels so introduces error. If multiple
        passes are performed, the translations are saved and combined
        at the end so this error is only introduced once. Using the
        `commit=False` argument allows for multiple different types of
        registration to be performed in sequence, since uncommitted
        translations will be applied before the next round of
        registration.
        
        Parameters
        ----------
        reference_frame (int, str or None) : The index of the frame to
          which all other frames should be aligned. If None, the frame
          of highest intensity will be used. If "mean" (default) or
          "median", the average or median of all frames will be
          used. If "max", the frame with highest absorbance is
          used. This attribute has no effect if template matching is
          used.
        blur : A type of filter to apply to each frame of the data
          before attempting registration. Choices are "median" or None
        method : Which technique to use to calculate the translation
          - "cross_correlation" (default)
          - "template_match"
          
          (If "template_match" is used, the `template` argument should
          also be provided.)
        passes : How many times this alignment should be done. Default: 1.
        template : Image data that should be matched if the
          `template_match` method is used.
        commit : If truthy (default), the final translation will be
          applied to the data stored on disk by calling
          `self.apply_translations(crop=True)` after all passes have
          finished.
        component : What component of the data to use: 'modulus',
          'phase', 'imag' or 'real'.
        plot_results : If truthy (default), plot the root-mean-square of the
          translation distance for each pass.

        """
        logstart = time()
        log.info("Aligning frames with %s algorithm over %d passes",
                 method, passes)
        pass_distances = []
        # Check for valid attributes
        valid_filters = ["median", None]
        if blur not in valid_filters:
            msg = 'Invalid blur filter "{}". Choices are {}'
            msg = msg.format(blur, valid_filters)
            raise ValueError(msg)
        # Sanity check on `method` argument
        valid_methods = ['cross_correlation', 'template_match']
        if method not in valid_methods:
            msg = "Unknown method {}. Choices are {}"
            msg = msg.format(method, valid_methods)
            raise ValueError(msg)
        # Guess best reference frame to use
        if reference_frame is "max":
            spectrum = self.xanes_spectrum(representation=component)
            reference_frame = np.argmax(spectrum.values)
        # Keep track of how many passes and where we started
        for pass_ in range(0, passes):
            log.debug("Starting alignment pass %d of %d", pass_, passes)
            # Get data from store
            frames = self.apply_transformations(crop=True, commit=False)
            frames = get_component(frames, component)
            # Calculate axes to use for proper reference image
            if reference_frame == 'mean':
                ref_image = np.mean(frames, axis=(0, 1))
            elif reference_frame == 'median':
                ref_image = np.median(frames, axis=(0, 1))
            else:
                ref_image = frames[reference_frame]
                # Check that the argument results in 2D image_data
                if len(ref_image.shape) != 2:
                    msg = "refrence_frame ({}) does not match"
                    msg += " shape of frameset {}."
                    msg += " Please provide a {}-tuple."
                    msg = msg.format(reference_frame,
                                     frames.shape,
                                     len(frames.shape) - 2)
                    raise IndexError(msg)
            # Prepare blurring if requested
            if blur == "median":
                ref_image = filters.median(ref_image,
                                           morphology.disk(20))
            # Calculate translations for each frame
            desc = "Registering pass {}".format(pass_)
            if method == "cross_correlation":
                translations = xm.register_correlations(frames=frames,
                                                        reference=ref_image,
                                                        desc=desc)
            elif method == "template_match":
                template_ = get_component(template, component)
                translations = xm.register_template(frames=frames,
                                                    reference=ref_image,
                                                    template=template_,
                                                    desc=desc)
            # Add the root-mean-square to the list of distances translated
            rms = np.sqrt((translations**2).sum(axis=-1).mean())
            log.info("RMS of translations for pass %d = %f", pass_, rms)
            pass_distances.append(np.sqrt((translations**2).sum(-1)).flatten())
            # Save translations for deferred calculation
            self.stage_transformations(translations=translations)
            log.debug("Finished alignment pass %d of %d", pass_, passes)
        # Plot the results if requested
        pass_distances = np.array(pass_distances)
        if plot_results:
            x = range(0, passes)
            ax = plots.new_axes()
            # ax.plot(x, pass_distances, marker='o', linestyle=":")
            ax.boxplot(pass_distances.swapaxes(0, 1))
            ax.set_xlabel('Pass')
            ax.set_ylabel("Distance (µm)")
            ax.set_title("$L^2$ Norms of Calculated Translations")
        # Apply result of calculations to disk (if requested)
        if commit:
            log.info("Committing final translations to disk")
            self.apply_transformations(crop=True, commit=True)
        log.info("Aligned %d passes in %d seconds", passes, time() - logstart)

    def label_particles(self, min_distance=20):
        """Use watershed segmentation to identify particles.

        Arguments
        ---------
        - min_distance : Controls how selective the algorithm is at
          grouping areas into particles. Lower numbers means more
          particles, but might split large particles into two.
        """
        with self.store('r+') as store:
            frames = store.absorbances.value
            # Average across all timesteps
            frames = np.median(frames, axis=0)
            Es = np.median(store.energies, axis=0)
            particles = xm.particle_labels(frames=frames, energies=Es,
                                           edge=self.edge,
                                           min_distance=min_distance)
            store.particle_labels = particles
            if hasattr(store.particle_labels, 'attrs'):
                store.particle_labels.attrs['frame_source'] = 'absorbances'
    
    def particle_series(self, map_name="whiteline_map"):
        """Generate an array of values from map_name averaged across each
        particle.
        
        Returns: A 2D array where the first dimension is particles and
        the second is the first dimension of the map dataset (usually time).
        
        """
        steps = []
        with self.store() as store:
            data = store.get_map(map_name).value
            for stepdata in data:
                particles = self.particle_regions(intensity_image=stepdata)
                imgs = [p.intensity_image for p in particles]
                vals = [np.median(im[~np.isnan(im)]) for im in imgs]
                steps.append(vals)
        # Convert from (steps, particles) to (particles, steps)
        steps = np.array(steps)
        steps = np.transpose(steps)
        return steps
    
    def normalize(self, plot_fit=False, new_name='normalized'):
        """Correct for background material not absorbing at this edge. Uses
        method described in DOI 10.1038/ncomms7883: fit line against
        material that fails edge_jump_filter and use this line to
        correct entire frame.

        Arguments
        ---------
        - plot_fit: If True, will plot the background spectrum and the
        best-fit line.
        """
        self.fork_group(new_name)
        # Linear regression on "background" materials
        spectrum = self.xanes_spectrum(edge_jump_filter="inverse")
        regression = linear_model.LinearRegression()
        x = np.array(spectrum.index).reshape(-1, 1)
        regression.fit(x, spectrum.values)
        goodness_of_fit = regression.score(x, spectrum.values)
        # Subtract regression line from each frame

        def remove_offset(payload):
            offset = regression.predict(payload['energy'])
            original_data = payload['data']
            payload['data'] = original_data - offset
            # Remove labels since we don't need to save it
            payload.pop('labels', None)
            return payload

        description = "Normalizing background (R²={:.3f})"
        description = description.format(goodness_of_fit)
        process_with_smp(frameset=self,
                         worker=remove_offset,
                         description=description)
        # Plot the background fit used to normalize
        if plot_fit:
            ax = plots.new_axes()
            ax.plot(x, spectrum.values, marker="o", linestyle="None")
            ax.plot(x, regression.predict(x))

    def particle_regions(self, intensity_image=None, labels=None):
        """Return a list of regions (1 for each particle) sorted by area.
        (largest first). This requires that the `label_particles`
        method be called first.

        Arguments
        ---------
        - intensity_image : 2D array passed on to the skimage
          `regionprops` function to determine what shows up in the
          image for each particle.

        - labels : Dataframe of the same shape as the map, with the
          particles segmented. If None, the `particle_labels`
          attribute of the TXM store will be used.

        """
        with self.store() as store:
            if labels is None:
                labels = store.particle_labels.value
            regions = measure.regionprops(labels,
                                          intensity_image=intensity_image)
        # Put in order of descending area
        regions.sort(key=lambda p: p.area, reverse=True)
        return regions

    def plot_mean_image(self, ax=None, component="modulus"):
        if ax is None:
            ax = plots.new_image_axes()
        with self.store() as store:
            absorbances = np.reshape(store.absorbances,
                                     (-1, *store.absorbances.shape[-2:]))
            data = np.mean(absorbances, axis=0)
            ax_unit = store.pixel_unit
        data = get_component(data, component)
        artist = ax.imshow(data,
                           extent=self.extent(representation='absorbances'),
                           origin="lower", cmap='gray')
        ax.set_xlabel(ax_unit)
        ax.set_ylabel(ax_unit)
        return artist

    def mean_frame(self, representation="absorbances"):
        """Return the mean value with the same shape as an individual
        frame.

        """
        with self.store() as store:
            Is = store.get_frames(representation)
            frame_shape = self.frame_shape(representation)
            Is = np.reshape(Is, (-1, *frame_shape))
            mean = np.mean(Is, axis=0)
        return mean

    def frame_shape(self, representation="absorbances"):
        """Return the shape of the individual energy frames."""
        with self.store() as store:
            imshape = store.get_frames(representation).shape[-2:]
        return imshape

    def pixel_unit(self):
        """Return the unit of measure for the size of a pixel."""
        with self.store() as store:
            unit = store.pixel_unit
        return unit

    def spectra(self, edge_filter=False):
        """Return a two-dimensional array of spectra for all the pixels in
        shape of (pixel, energy).

        Arguments
        ---------
        edge_jump_filter (bool or str): [NOT YET IMPLEMENTED] If
            truthy, only pixels that pass the edge jump filter are
            used to calculate the spectrum. If "inverse" is given,
            then the edge jump filter is logically not-ted and
            calculated with a more conservative threshold.
        """
        with self.store() as store:
            As = store.absorbances
            # Mangle axes to be in (pixel, energy) order
            img_shape = As.shape[-2:]
            spectra = np.reshape(As, (-1, np.prod(img_shape)))
            spectra = np.moveaxis(spectra, 0, -1)
        return spectra

    def spectrum(self, pixel=None, edge_jump_filter=False,
                       representation="absorbances", index=0):
        """Collapse the frameset down to an energy spectrum.

        Any dimensions (besides the energy dimension) that remain
        after applying the arguments below, will be averaged to give
        the final intensity at each energy.

        Returns
        -------
        spectrum : pd.Series
          A pandas Series with the spectrum.

        Parameters
        ----------
        pixel : tuple, optional
          A 2-tuple that causes the returned series to represent
          the spectrum for only 1 pixel in the frameset. If None, a
          larger part of the frame will be used, depending on the
          other arguments.
        edge_jump_filter : bool or str, optional
          If truthy, only pixels that
          pass the edge jump filter are used to calculate the
          spectrum. If "inverse" is given, then the edge jump filter
          is logically not-ted and calculated with a more
          conservative threshold.
        representation : str, optional
          What kind of data to use for creating the spectrum. This
          will be passed to TXMStore.get_map()
        index : int, optional
          Which step in the frameset to use. When used to index
          store().absorbances, this should return a 3D array like
          (energy, rows, columns).

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
                if edge_jump_filter:
                    # Filter out background pixels using edge mask
                    mask = self.edge_mask()
                    mask = np.broadcast_to(array=mask,
                                           shape=(*energies.shape, *mask.shape))
                    frames = np.ma.array(frames, mask=mask)
                # Take average of all pixel frames
                flat = (*frames.shape[:frames.ndim-2], -1) # Collapse image dimension
                spectrum = np.mean(np.reshape(frames, flat), axis=-1)
            # Combine into a series
            series = pd.Series(spectrum, index=energies)
        return series

    def plot_xanes_spectrum(self, ax=None, pixel=None,
                            norm_range=None, normalize=False,
                            representation="modulus",
                            show_fit=False, edge_jump_filter=False,
                            linestyle=":",
                            *args, **kwargs):
        """Calculate and plot the xanes spectrum for this field-of-view.

        Arguments
        ---------
        ax - matplotlib axes object on which to draw

        pixel - Coordinates of a specific pixel on the image to plot.

        normalize - If truthy, will set the pre-edge at zero and the
          post-edge at 1.

        show_fit - If truthy, will use the edge object to fit the data
          and plot the resulting fit line.

        edge_jump_filter - If truthy, will only include those values
          that show a strong absorbtion jump across this edge.
        """
        if norm_range is None:
            norm_range = (self.edge.map_range[0], self.edge.map_range[1])
        norm = Normalize(*norm_range)
        spectrum = self.spectrum(pixel=pixel,
                                 edge_jump_filter=edge_jump_filter,
                                 representation=representation)
        edge = self.edge()
        if ax is None:
            ax = plots.new_axes()
        if normalize:
            # Adjust the limits of the spectrum to be between 0 and 1
            normalized = edge.normalize(spectrum.values, spectrum.index)
            spectrum = pd.Series(normalized, index=spectrum.index)
        scatter = plots.plot_xanes_spectrum(spectrum=spectrum,
                                            ax=ax,
                                            energies=spectrum.index,
                                            norm=Normalize(*self.edge.map_range))
        if pixel is not None:
            xy = pixel_to_xy(pixel, extent=self.extent(), shape=self.map_shape())
            title = 'XANES Spectrum at ({x}, {y}) = {val}'
            masked_map = self.masked_map(goodness_filter=False)
            val = masked_map[pixel.vertical][pixel.horizontal]
            title = title.format(x=round(xy.x, 2),
                                 y=round(xy.y, 2),
                                 val=val)
            ax.set_title(title)
        # Plot lines at edge of normalization range or indicate peak positions
        edge.annotate_spectrum(ax=scatter.axes)
        return scatter
    
    @functools.lru_cache()
    def edge_mask(self, sensitivity: float=1, min_size=0):
        """Calculate a mask for what is likely active material at this
        edge.
        
        Parameters
        ----------
        sensitivity : float
          A multiplier for the otsu value to determine
          the actual threshold.
        min_size : int
          Objects below this size (in pixels) will be removed. Passing
          zero (default) will result in no effect.
        
        """
        with self.store() as store:
            As = store.absorbances.value
            # Check for complex values and convert to absorbances only
            if np.iscomplexobj(As):
                As = np.imag(As)
            mask = self.edge.mask(frames=As,
                                  energies=store.energies.value,
                                  sensitivity=sensitivity,
                                  min_size=min_size)
        return mask
    
    def fit_spectra(self, edge_mask=True):
        """Fit a series of curves to the spectrum at each pixel.
        
        Arguments
        ---------
        edge_mask : bool, optional
          If true, only pixels passing the edge_mask will be fit and
          the remaning pixels will be set to a default value. This can
          help reduce computing time.
        
        """
        logstart = time()
        with self.store() as store:
            frames = store.absorbances
            energies = store.energies.value
            # Get a mask to select only some pixels
            if edge_mask:
                # Active material pixels only
                mask = self.edge_mask()
            else:
                # All pixels
                mask = np.zeros_like(self.edge_mask())
            frames_mask = np.broadcast_to(mask, frames.shape)
            # Convert numpy axes to be in (pixel, energy) form
            frames_mask = np.moveaxis(frames_mask, 1, -1)
            spectra = np.moveaxis(frames, 1, -1)
            map_shape = (*spectra.shape[:-1], len(xm.kedge_params))
            fit_maps = np.empty(map_shape)  # To hold output
            # Make sure spectra and energies have the same shape
            energies = broadcast_reverse(energies, frames.shape)
            spectra = spectra[~frames_mask].reshape((-1, spectra.shape[-1]))
            energies = np.moveaxis(energies, 1, -1)
            energies = energies[~frames_mask].reshape(spectra.shape)
            # Do a preliminary fitting to get good parameters
            I = np.median(spectra, axis=0)[np.newaxis, ...]
            E = np.median(energies, axis=0)[np.newaxis, ...]
            guess = xm.guess_kedge(I[0], E[0], edge=self.edge)
            p0 = xm.fit_kedge(spectra=I, energies=E, p0=guess)
        # Perform full fitting for individual pixels
        all_params = xm.fit_kedge(spectra=spectra,
                                  energies=energies, p0=p0[0])
        # Set actual calculate values
        map_mask = np.broadcast_to(mask, fit_maps.shape[:-1])
        # Set default values
        fit_maps[map_mask] = np.nan
        fit_maps[~map_mask] = all_params
        # Calculate whiteline position
        E0 = fit_maps[..., xm.kedge_params.index('E0')]
        gaus_center = fit_maps[..., xm.kedge_params.index('gb')]
        wl_maps = E0 + gaus_center
        # Save results to disk
        with self.store(mode='r+') as store:
            store.fit_parameters = fit_maps
            store.whiteline_fit = wl_maps
            store.whiteline_fit.attrs['frame_source'] = 'absorbances'
        log.info('fit %d spectra in %d seconds',
                 spectra.shape[0], time() - logstart)

    def calculate_whitelines(self, edge_mask=False):
        """Calculate and save a map of the whiteline position of each pixel by
        calculating the energy of simple maximum absorbance.

        Arguments
        ---------
        - edge_mask : If true, only pixels passing the edge_mask will
          be fit and the remaning pixels will be set to a default
          value. This can help reduce computing time.

        """
        with self.store() as store:
            frames = store.absorbances
            # Insert two axes into energies for image row/cols
            energies = store.energies.value[:, np.newaxis, np.newaxis, :]
            # Convert numpy axes to be in (pixel, energy) form
            spectra = np.moveaxis(frames, 1, -1)
            # Calculate whiteline positions
            whitelines = xm.direct_whitelines(spectra=spectra,
                                              energies=energies,
                                              edge=self.edge)
        # Save results to disk
        with self.store(mode='r+') as store:
            store.whiteline_max = whitelines
            store.whiteline_max.attrs['frame_source'] = 'absorbances'

    def calculate_maps(self, fit_spectra=False):
        """Generate a set of maps based on pixel-wise Xanes spectra: whiteline
        position, particle labels.

        Arguments
        ---------
        -fit_spectra : If truthy, the whiteline will be found by
         fitting curves, instead of the default of taking the direct
         maximum. This is likely to be very slow.
        """
        if fit_spectra:
            self.fit_spectra()
        else:
            self.calculate_whitelines()
        # Calculate particle_labels
        self.label_particles()

    def plot_signals(self, cmap="viridis"):
        """Plot the signals from the previously extracted data. Requires that
        self.store().signals and self.store().signal_weights be
        set.

        """
        with self.store() as store:
            signals = store.signals.value
            n_signals = signals.shape[0]
            weights = store.signal_weights.value
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
        extent = self.extent(representation="absorbances")
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
            ax2.plot(energies, np.real(cmplx_signal), **cmplx_kwargs)
            ax2.plot(energies, np.imag(cmplx_signal), **cmplx_kwargs)
            ax2.plot(energies, np.abs(cmplx_signal), **cmplx_kwargs)
            ax2.legend(["Obs Mod", "Real", "Imag", "Abs"], fontsize=8)
            # ax2.set_ylim(*specrange)
            ax2.set_title("Signal Component {idx}".format(idx=idx))

    def plot_signal_map(self, ax=None, signals_idx=None):
        """Plot the map of signal strength for signals extracted from
        self.calculate_signals().

        Arguments
        ---------
        - ax : A matplotlib Axes object. If "None" (default) a new
          axes object is created.

        - signals_idx : Indices of which signals to plot. This will be
        passed as a numpy array index. Special value None (default)
        means first three signals will be plotted.

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
        extent = self.extent(representation="absorbances")
        artist = plots.plot_composite_map(composite, extent=extent, ax=ax)
        ax = artist.axes
        ax.set_xlabel(px_unit)
        ax.set_ylabel(px_unit)
        ax.set_title("Composite of signals {}".format(signals_idx))

    def calculate_signals(self, n_components=2, method="nmf",
                          edge_mask=True):
        """Extract signals and assign each pixel to a group, then save the
        resulting RGB cluster map.

        Arguments
        ---------

        - n_components : The number of signals and number of clusters
          into which the data will be separated.

        - method : The technique to use for extracting
          signals. Currently only "nmf" is supported.

        - edge_mask : If truthy (default), only those pixels passing
          the edge filter will be considered.
        """
        msg = "Performing {} signal extraction with {} components"
        log.info(msg.format(method, n_components))
        frame_shape = self.frame_shape()
        # Retrieve the absorbances and convert to spectra
        with self.store() as store:
            As = store.absorbances
            # Collapse to (frame, pixel) shape
            n_timesteps = As.shape[0]
            assert n_timesteps == 1  # We can't currently handle timesteps
            num_spectra = np.prod(frame_shape)
            As = np.reshape(As, (-1, num_spectra))
            spectra = np.moveaxis(As, 0, 1)
        # Get the edge mask so only active material is included
        dummy_mask = np.ones(frame_shape, dtype=np.bool)
        if edge_mask:
            try:
                mask = ~self.edge_mask()
            except exceptions.XanesMathError as e:
                log.error(e)
                log.warning("Failed to calculate mask, using all pixels.")
                mask = dummy_mask
            else:
                log.debug("Using edge mask for signal extraction")
        else:
            log.debug("No edge mask for signal extraction")
            mask = dummy_mask

        # Separate the data into signals
        signals, weights = xm.extract_signals_nmf(
            spectra=spectra[mask.flatten()], n_components=n_components)
        # Reshape weights into frame dimensions
        weight_shape = (n_timesteps, *frame_shape, n_components)
        weight_frames = np.zeros(shape=weight_shape)
        weight_frames[:, mask, :] = weights
        # Save the calculated signals and weights
        method_names = {
            "nmf": "Non-Negative Matrix Factorization",
        }
        with self.store(mode="r+") as store:
            store.signals = signals
            store.signal_method = method_names[method]
            store.signal_weights = weight_frames

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
        # ## Perform k-means clustering
        k_results = cluster.k_means(weights, n_clusters=n_components)
        centroid, labels, intertia = k_results
        label_frame = np.zeros(shape=frame_shape)
        label_frame[mask] = labels
        label_frame = label_frame.reshape(1, *frame_shape)
        # Save the resuliting k-means cluster map
        with self.store(mode="r+") as store:
            store.cluster_map = label_frame

    @functools.lru_cache(maxsize=2)
    def map_data(self, timeidx=0, representation="absorbances"):
        """Return map data for the given time index and representation.
        
        If `representation` is really mapping data, then the result
        will have more dimensions than expected.
        
        Parameters
        ----------
        timeidx : int
          Index for the first dimension of the combined data array. If
          the underlying map data has only 2 dimensions, this
          parameter is ignored.
        representation : str
          The group name for these data. Eg "absorbances",
          "whiteline_map", "intensities"
        
        Returns
        -------
        map_data : np.ndarray
          A 2-dimensional array with the form (row, col).

        """
        with self.store() as store:
            map_data = store.get_map(representation)
            if map_data.ndim > 2:
                map_data = map_data[timeidx]
            else:
                map_data = np.array(map_data)
        return map_data
    
    @functools.lru_cache(maxsize=2)
    def frames(self, timeidx=0, representation="absorbances"):
        """Return the frames for the given time index.
        
        If `representation` is really mapping data, then the source
        frames will be returned.
        
        Parameters
        ----------
        timeidx : int
          Index for the first dimension of the combined data array.
        representation : str
          The group name for these data. Eg "absorbances",
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
    
    def subtract_surroundings(self):
        """Use the edge mask to separate "surroundings" from "sample", then
        subtract the average surrounding absorbance from each
        frame. This effective removes effects where the entire frame
        is brighter from one energy to the next.
        """
        with self.store(mode='r+') as store:
            log.debug('Subtracting surroundings')
            mask = np.broadcast_to(self.edge_mask(), store.absorbances.shape)
            bg = store.absorbances[mask].reshape((*store.absorbances.shape[0:2], -1))
            bg = bg.mean(axis=-1)
            msg = "Background intensities calculated: {}"
            log.debug(msg.format(bg))
            bg = broadcast_reverse(bg, store.absorbances.shape)
            # Save the resultant data to disk
            store.absorbances = store.absorbances - bg

    def extent_array(self, representation='intensities'):
        raise NotImplementedError()
        #     # Include all frames
        #     imshape = np.array(frames.shape)[-2:]  # ((rows, cols))
        #     pixel_size = store.pixel_sizes.value
        #     center = store.relative_positions.value
        # left = (center[:, -1] - width) / 2
        # right = (center[:, -1] + width) / 2
        # bottom = (center[:, -2] - height) / 2
        # top = (center[:, -2] + height) / 2
        # ret = np.array((left, right, bottom, top)).T
    
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
        with self.store() as store:
            imshape = self.frame_shape()
            # Filter to only the requested frame
            pixel_size = store.pixel_sizes[idx]
            # Take the median across all pixel sizes (except the xy dim)
            pixel_size = np.median(pixel_size)
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
    
    def plot_frame(self, idx, ax=None, cmap="gray", *args, **kwargs):
        """Plot the frame with given index as an image."""
        if ax is None:
            ax = plots.new_image_axes()
        # Plot image data
        with self.store() as store:
            artist = ax.imshow(store.absorbances[idx],
                               extent=self.extent(idx),
                               cmap=cmap, origin="lower",
                               *args, **kwargs)
            unit = store.pixel_unit
        # Decorate axes
        ax = artist.axes
        ax.set_ylabel(unit)
        ax.set_xlabel(unit)
        return artist
    
    @property
    def num_timesteps(self):
        with self.store() as store:
            val = store.absorbances.shape[0]
        return val
    
    @property
    def num_energies(self):
        with self.store() as store:
            val = store.energies.shape[-1]
        return val

    def plot_map(self, ax=None, map_name="whiteline_map", timeidx=0, vmin=None, vmax=None):
        """Prepare data and plot a map of whiteline positions."""
        # Do the plotting
        with self.store() as store:
            # Add bounds for the colormap if given
            vmin = self.edge.map_range[0] if vmin is None else vmin
            vmax = self.edge.map_range[1] if vmax is None else vmax
            norm = Normalize(vmin=vmin, vmax=vmax)
            # Do the actual plotting
            data = store.get_map(name=map_name)[timeidx]
            plots.plot_txm_map(data=data,
                               ax=ax,
                               norm=norm,
                               edge=self.edge(),
                               extent=self.extent(representation='absorbances'))

    def plot_map_pixel_spectra(self, pixels, map_ax=None,
                               spectra_ax=None,
                               map_name="whiteline_map", timeidx=0,
                               step_size=0):
        """Plot the frameset's map and highlight some pixels on it then plot
        those pixel's spectra on another set of axes.
        
        Arguments
        ---------
        
        - pixels : An iterable of 2-tuples indicating which (row,
          column) pixels to highlight.
        
        - map_ax : A matplotlib axes object to put the map onto. If
          None, a new 2-wide subplot will be created for both map_ax
          and spectra_ax.
        
        - spectra_ax : A matplotlib axes to be used for plotting
          spectra. Will only be used if `map_ax` is not None.
        
        - map_name : Name of the map to use for plotting. It will be
          passed to the TXM store object and retrieved from the hdf5
          file. If falsy, no map will be plotted.
        
        - timeidx : Index of which timestep to use.
        
        """
        # Create some axes if necessary
        if map_ax is None:
            fig, ax = pyplot.subplots(1, 2)
            map_ax, spectra_ax = ax
        # Get the necessary data
        with self.store() as store:
            energies = store.energies[timeidx]
            spectra = store.absorbances[timeidx]
            # Put energy as the last axis
            spectra = np.moveaxis(spectra, 0, -1)
        if map_name:
            self.plot_map(ax=map_ax, map_name=map_name, timeidx=timeidx)
        plots.plot_pixel_spectra(pixels=pixels,
                                 extent=self.extent('absorbances'),
                                 spectra=spectra,
                                 energies=energies,
                                 map_ax=map_ax,
                                 spectra_ax=spectra_ax,
                                 step_size=step_size)
    
    def plot_histogram(self, plotter=None, timeidx=None, ax=None,
                       vmin=None, vmax=None, goodness_filter=False,
                       active_pixel=None, bins="energies", *args, **kwargs):
        """Use a default frameset plotter to draw a map of the chemical
        data."""
        with self.store() as store:
            if timeidx is None:
                data = store.whiteline_map.value
            else:
                data = store.whiteline_map[timeidx]
        # Add bounds for the colormap `if given
        vmin = self.edge.map_range[0] if vmin is None else vmin
        vmax = self.edge.map_range[1] if vmax is None else vmax
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Get bins for the energy steps
        edge = self.edge()
        if str(bins) == "energies":
            bins = edge.energies_in_range(edge.map_range)
        artists = plots.plot_txm_histogram(data=data, ax=ax,
                                           norm=norm,
                                           cmap=self.cmap, bins=bins)
        return artists
   
    def qt_viewer(self):
        from qt_frameset_presenter import QtFramesetPresenter
        from qt_frame_view import QtFrameView
        from qt_map_view import QtMapView
        log.debug("Launching Qt viewer from frameset")
        presenter = QtFramesetPresenter(frameset=self,
                                        frame_view=QtFrameView())
        presenter.add_map_view(QtMapView())
        presenter.prepare_ui()
        presenter.launch()


class PtychoFrameset(XanesFrameset):
    """A set of images ("frames") at different energies moving across an
    absorption edge. The individual frames should be generated by
    ptychographic reconstruction of scanning transmission X-ray
    microscopy (STXM) to produce an array complex intensity
    values. This class does *not* include any code responsible for the
    collection and reconstruction of such data, only for the analysis
    in the context of X-ray absorption near edge spectroscopy."""
    
    def representations(self):
        """Retrieve a list of valid representations for these data, such as
        modulus or phase data for ptychography."""
        reps = super().representations()
        reps += ['modulus', 'phase', 'real', 'imag']
        # Complex image data cannot be properly displayed
        if 'image_data' in reps:
            reps.remove('image_data')
        return reps
    
    def apply_internal_reference(self, plot_background=True, ax=None):
        """Use a portion of each frame for internal reference correction. The
        result is the complex refraction for each pixel: the real
        component describes the phase shift, and the imaginary
        component is exponential decay, ie. absorbance.
        
        Arguments
        ---------
        
        plot_background : If truthy, the values of I_0 are plotted as
          a function of energy.
        
        ax : The axes to use for plotting if `plot_background` is
          truthy.
        
        """
        # Array for holding background correction for plotting
        if plot_background:
            Es = []
            I_0s = []
        with self.store() as store:
            Is = store.intensities
            refraction = xm.apply_internal_reference(Is)
            Es = store.energies.value
        # Save complex image as refractive index (real part is phase change)
        with self.store('r+') as store:
            store.absorbances = refraction
        # Plot background for evaluation
        # if plot_background:
        #     if ax is None:
        #         ax = plots.new_axes()
        #     ax.plot(Es, np.abs(I_0.squeeze()))
        #     ax.set_title("Background Intensity used for Reference Correction")
        #     ax.set_xlabel("Energy (eV)")
        #     ax.set_ylabel("$I_0$")
        return

        # Step through each frame and apply reference correction
        for frame in prog(self, "Reference correction"):
            img = frame.get_data("modulus")
            if mask is None:
                # Calculate background intensity using thresholding
                threshold = filters.threshold_yen(img)
                graymask = img > threshold
                background = img[img > threshold]
                I_0 = np.median(background)
            else:
                data = np.ma.array(img, mask=graymask)
                I_0 = np.ma.median(data)
            # Save values for plotting
            if plot_background:
                Es.append(frame.energy)
                I_0s.append(I_0)
            # Calculate absorbance based on background
            absorbance = np.log(I_0 / frame.image_data)
            # Calculate relative phase shift
            phase = frame.get_data('phase')
            phase - np.median((phase * graymask)[graymask > 0])
            # The phase data has a gradient in the background, so remove it
            x,y = np.meshgrid(np.arange(phase.shape[1]),np.arange(phase.shape[0]))
            A = np.column_stack([y.flatten(), x.flatten(), np.ones_like(x.flatten())])
            p, residuals, rank, s = linalg.lstsq(A, phase.flatten())
            bg = p[0] * y + p[1] * x + p[2]
            phase = phase - bg
            # Save complex image
            frame.image_data = phase + absorbance * complex(0, 1)

        # Plot background for evaluation
        if plot_background:
            if ax is None:
                ax = plots.new_axes()
            ax.plot(Es, I_0s)
            ax.set_title("Background Intensity used for Reference Correction")
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("$I_0$")
