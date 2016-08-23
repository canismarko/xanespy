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
row, column)."""

import functools
from typing import Callable
import os

import pandas as pd
from matplotlib import pyplot
from matplotlib.colors import Normalize
from scipy import linalg
import h5py
import numpy as np
from skimage import morphology, filters, transform, color, measure
from sklearn import linear_model
from units import unit, predefined

from utilities import prog, xycoord, Pixel, Extent, pixel_to_xy
from frame import TXMFrame, PtychoFrame
from txmstore import TXMStore
from plots import new_axes, new_image_axes, plot_txm_map, plot_xanes_spectrum
import exceptions
import xanes_math as xm

predefined.define_units()

# So all modules can use the same HDF indices
# energy_key = "{:.2f}_eV"


def build_series(frames):
    energies = [frame.energy for frame in frames]
    images = [frame.image_data for frame in frames]
    series = pd.Series(images, index=energies)
    return series

def merge_framesets(framesets, group="merged"):
    """Combine two set of frame data into one. No energy should be
    duplicated. A new HDF group will be greated.
    """
    ref_fs = framesets[0]
    dest = "/" + group
    with ref_fs.hdf_file(mode="a") as f:
        # Copy first group as a template
        try:
            del f[dest]
        except KeyError:
            pass
        src_group = f[ref_fs.frameset_group]
        src_group.copy(src_group, dest=dest)
        for key in f[dest]:
            del f[dest][key]
        # Set some attributes
        new_fs_group = f[dest].create_group("merged")
        f[dest].attrs['latest_group'] = new_fs_group.name
        new_fs_group.attrs['level'] = 0
        new_fs_group.attrs['parent'] = ""
        # Now link to the other nodes
        for fs in framesets:
            for frame in fs:
                key = energy_key.format(frame.energy)
                new_fs_group[key] = f[frame.frame_group]
    # return new_frameset
    fs = XanesFrameset(filename=ref_fs.hdf_filename,
                       edge=ref_fs.edge,
                       groupname=dest)
    return fs

def calculate_gaussian_whiteline(data, edge):

    """Calculates the whiteline position of the absorption edge data
    contained in `data`.

    The "whiteline" for an absorption K-edge is the energy at which
    the specimin has its highest absorbance. This function will return
    an 2 arrays with the same shape as each entry in the data
    series. 1st array gives the energy of the highest absorbance and
    2nd array contains goodness of fits.

    Arguments
    ---------
    data - The X-ray absorbance data. Should be similar to a pandas
    Series. Assumes that the index is energy. This can be a Series of
    numpy arrays, which allows calculation of image frames, etc.

    edge - An instantiated edge object that can be used for fitting.

    """
    # Prepare data for manipulation
    energies = data.index.astype(np.float64)
    absorbances = np.array(list(data.values))
    absorbances = absorbances.astype(np.float64)
    orig_shape = absorbances.shape[1:]
    ndim = absorbances.ndim
    # Convert to an array of spectra by moving the 1st axes (energy)
    # to be on bottom
    for dim in range(0, ndim-1):
        absorbances = absorbances.swapaxes(dim, dim+1)
    # Flatten into a 1-D array of spectra
    num_spectra = int(np.prod(orig_shape))
    spectrum_length = absorbances.shape[-1]
    absorbances = absorbances.reshape((num_spectra, spectrum_length))
    # Calculate whitelines and goodnesses(?)
    # whitelines = np.zeros_like(absorbances)
    whitelines = np.zeros(shape=(num_spectra,))
    goodnesses = np.zeros_like(whitelines)
    # Prepare multiprocessing functions
    def result_callback(payload):
        # Unpack and save results to arrays.
        idx = payload['idx']
        whitelines[idx] = payload['center']
        goodnesses[idx] = payload['goodness']
    def worker(payload):
        # Calculate the whiteline position by fitting.
        try:
            peak, goodness = edge.fit(payload['spectrum'])
        except exceptions.RefinementError as e:
            # Fit failed so set as bad cell
            center = edge.E_0
            goodness = 0
        else:
            center = peak.center()
        # Return calculated whiteline position as dictionary
        result = {
            'idx': payload['idx'],
            'center': center,
            'goodness': goodness
        }
        return result
    # Prepare multiprocessing queue
    queue = smp.Queue(worker=worker,
                      totalsize=len(absorbances),
                      result_callback=result_callback,
                      description="Calculating whiteline")
    # Fill queue with tasks
    for idx, spectrum in enumerate(absorbances):
        spectrum = pd.Series(spectrum, index=energies)
        payload = {
            'idx': idx,
            'spectrum': spectrum
        }
        queue.put(payload)
    # Wait for workers to finish
    queue.join()
    # Convert results back to their original shape
    whitelines = np.array(whitelines).reshape(orig_shape)
    goodnesses = np.array(goodnesses).reshape(orig_shape)
    return whitelines, goodnesses

def calculate_direct_whiteline(data, *args, **kwargs):
    """Calculates the whiteline position of the absorption edge data
    contained in `data`. This method uses the energy of maximum
    absorbance and is a faster alternative to `calculate_whiteline`.
    The "whiteline" for an absorption K-edge is the energy at which
    the specimin has its highest absorbance. This function will return
    an 2 arrays with the same shape as each entry in the data
    series. 1st array gives the energy of the highest absorbance and
    2nd array contains a mock array of goodness of fits (all values
    are 1).

    Arguments
    ---------
    data - The X-ray absorbance data. Should be similar to a pandas
    Series. Assumes that the index is energy. This can be a Series of
    numpy arrays, which allows calculation of image frames, etc.

    """
    # First disassemble the data series
    energies = data.index
    imagestack = np.array(list(data.values))
    # Now calculate the indices of the whiteline
    whiteline_indices = np.argmax(imagestack, axis=0)
    # Convert indices to energy
    map_energy = np.vectorize(lambda idx: energies[idx],
                              otypes=[np.float])
    whiteline_energies = map_energy(whiteline_indices)
    goodness = np.ones_like(whiteline_energies)
    return (whiteline_energies, goodness)


def _transform(data, scale=None, rotation=None, translation=None):
    """Apply a similarity transformation to the given (optionally complex)
    data. Arguments are the same as
    http://scikit-image.org/docs/dev/api/skimage.transform.html"""
    transformation = transform.SimilarityTransform(
        scale=scale,
        translation=translation,
        rotation=rotation
    )
    warp_kwargs = {
        'order': 3,
        'inverse_map': transformation,
        'mode': 'wrap',
        'preserve_range': True,
    }
    # Apply the transformation
    new_data = transform.warp(data.real, **warp_kwargs)
    # Apply transformation to imaginary component
    if np.any(data.imag):
        j = complex(0, 1)
        new_data = np.add(new_data, j * transform.warp(data.imag, **warp_kwargs))
    return new_data

class XanesFrameset():
    """A collection of TXM frames at different energies moving across an
    absorption edge. Iterating over this object gives the individual
    Frame() objects. The class assumes that the data have
    been imported into an HDF file.

    Arguments
    ---------
    - filename : path to the HDF file that holds these data.

    - groupname : Top level HDF group corresponding to this
    frameset. This argument is only required if there is more than one
    top-level group.
    """
    FrameClass = TXMFrame
    active_group = ''
    cmap = 'plasma'
    data_name = None
    # Places to store staged image transformations
    _translations = None
    _rotations = None
    _scales = None

    def __init__(self, filename, edge, groupname):
        self.hdf_filename = filename
        self.edge = edge
        self.parent_name = groupname
        # Load cached value for latest data group
        with self.store() as store:
            self.data_name = store.latest_data_name

    def __repr__(self):
        s = "<{cls} '{filename}'>"
        return s.format(cls=self.__class__.__name__, filename=self.hdf_filename)

    def _frame_keys(self):
        """Return a list of valid hdf5 keys in the group that correspond to
        actual energy frames."""
        keys = []
        with self.hdf_file() as f:
            parent_group = f[self.active_group]
            # Filter out groups that do not have an energy associated with them
            for group in parent_group:
                if parent_group[group].attrs.get('energy', False):
                    keys.append(group)
        return keys

    def store(self, mode='r'):
        """Get a TXM Store object that saves and retrieves data from the HDF5
        file. The mode argument is passed to h5py as is. This method
        should be used as a context manager, especially if mode is
        something writeable:
            with self.store() as store:
        """
        return TXMStore(hdf_filename=self.hdf_filename,
                        parent_name=self.parent_name,
                        data_name=self.data_name,
                        mode=mode)

    def save_images(self, directory):
        """Save a series of TIF's of the individual frames in `directory`."""
        if not os.path.exists(directory):
            os.mkdir(directory)
        for frame in self:
            pyplot.imsave(os.path.join(directory, str(frame.energy) + ".tif"), frame.image_data)

    def clear_caches(self):
        """Clear cached function values so they will be recomputed with fresh
        data"""
        self.xanes_spectrum.cache_clear()
        self.image_normalizer.cache_clear()
        self.edge_jump_filter.cache_clear()

    @property
    def active_labels_groupname(self):
        """The group name for the latest frameset of detected particle labels."""
        # Save as an HDF5 attribute
        return None
        group = self.active_group()
        return group.attrs.get('active_labels', None)

    @active_labels_groupname.setter
    def active_labels_groupname(self, value):
        group = self.active_group()
        group.attrs['active_labels'] = value

    @active_labels_groupname.deleter
    def active_labels_groupname(self):
        group = self.active_group()
        del group.attrs['active_labels']

    def starttime(self):
        """Determine the earliest timestamp amongst all of the frames."""
        all_times = [f.starttime for f in self]
        return min(all_times)

    def endtime(self):
        """Determine the latest timestamp amongst all of the frames."""
        all_times = [f.endtime for f in self]
        # Check background frames as well
        old_groupname = self.active_group
        self.switch_group('background_frames')
        all_times += [f.endtime for f in self]
        self.switch_group(old_groupname)
        return max(all_times)

    def particle(self, particle_idx=0):
        """Prepare a particle frameset for the given particle index."""
        fs = ParticleFrameset(parent=self, particle_idx=particle_idx)
        return fs

    def representations(self):
        """Retrieve a list of valid representations for these data."""
        reps = ['modulus']
        return reps

    def switch_group(self, name=""):
        """Set the frameset to retrieve image data from a different hdf
        group. Special value 'background_frames' sets to the reference
        image used during importing.

        Arguments
        ---------
        name (str) - The HDF groupname for this frameset. If omitted
          or "", a list of available options will be listed.
        """
        raise UserWarning("Use switch_data_group() instead")
        # raise UserWarning("This is not correct anymore")
        # with self.hdf_file() as f:
        #     valid_groups = list(f[self.frameset_group].keys())
        if name not in valid_groups:
            msg = "{name} is not a valid group. Choices are {choices}."
            raise exceptions.GroupKeyError(
                msg.format(name=name, choices=valid_groups)
            )
        if name == 'background_frames':
            # Clear cached value
            self.active_group = self.background_groupname
        else:
            self.active_group = os.path.join(self.frameset_group, name)
        self.clear_caches()

    def switch_data_group(self, new_name):
        """Turn on different active data for this frameset's store object."""
        raise UserWarning("Just set self.data_name instead")
        with self.store(mode='r+') as store:
            store.active_data_group = new_name

    def fork_data_group(self, new_name):
        """Turn on different active data for this frameset's store
        object. Similar to `switch_data_group` except that this method
        deletes the existing group and copies symlinks from the current one.
        """
        with self.store(mode='r+') as store:
            store.fork_data_group(new_name=new_name)
        self.data_name = new_name

    # def fork_group(self, name):
    #     """Create a new copy of the current active group inside the HDF parent
    #     with name: `name`. If trying to create a node that already
    #     exists, this will only work if the parent groups are the same,
    #     otherwise we risk breaking the tree. For performance reasons,
    #     datasets are copied as links and will be not stored separately
    #     until set_image_data is called.
    #     """
    #     with self.hdf_file(mode='a') as f:
    #         parent_group = f[self.frameset_group]
    #         old_group = f[self.active_group]
    #         if name in parent_group:
    #             # Existing groups require some validation first
    #             old_parent = parent_group[name].attrs['parent']
    #             if old_parent != self.active_group:
    #                 # Trying to replace an existing group from a different path
    #                 # This can cause an orphaned tree branch
    #                 msg = 'Cannot fork group "{target}".'
    #                 msg += ' Choose a new name or switch to group "{parent}" first.'
    #                 msg = msg.format(
    #                     target=name,
    #                     parent=os.path.basename(old_parent)
    #                 )
    #                 raise exceptions.GroupKeyError(msg)
    #             else:
    #                 del parent_group[name]
    #         # Create a new group to hold the datasets
    #         parent_group.copy(source=old_group, dest=name, shallow=True)
    #         dest = parent_group[name]
    #         new_path = parent_group[name].name
    #         parent_group[name].attrs['parent'] = old_group.name
    #         # Copy links for actual datasets
    #         for E_key in old_group.keys():
    #             # Set a dirty bit for later copy-on-write
    #             dest[E_key].attrs['_copy_on_write'] = True
    #             # Copy sym links for datasets
    #             for src_ds in old_group[E_key]:
    #                 try:
    #                     old_path = old_group[E_key][src_ds].name
    #                 except TypeError:
    #                     pass
    #                 else:
    #                     dest[E_key][str(src_ds)] = h5py.SoftLink(old_path)
    #                     dest[E_key].attrs["_copy_on_write"] = True
    #     self.latest_group = new_path
    #     self.switch_group(name)

    # def fork_labels(self, name):
    #     # Create a new group
    #     if name in self.hdf_group().keys():
    #         del self.hdf_group()[name]
    #     self.hdf_group().copy(self.active_labels_groupname, name)
    #     labels_group = self.hdf_group()[name]
    #     # Update label paths for frame datasets
    #     for frame in self:
    #         key = frame.image_data.name.split('/')[-1]
    #         new_label_name = labels_group[key].name
    #         frame.particle_labels_path = new_label_name
    #     self.latest_labels = name
    #     self.active_labels_groupname = name
    #     return labels_group

    # def apply_references(self, bg_groupname):
    #     """Apply reference corrections for this frameset. Converts raw
    #     intensity frames to absorbance frames."""
    #     self.background_groupname = bg_groupname
    #     self.fork_group('absorbance_frames')
    #     bg_group = self.hdf_file()[bg_groupname]
    #     for frame in prog(self, "Reference correction"):
    #         key = frame.image_data.name.split('/')[-1]
    #         bg_dataset = bg_group[key]
    #         new_data = apply_reference(frame.image_data.value,
    #                                    reference_data=bg_dataset.value)
    #         # Resize the dataset if necessary
    #         if new_data.shape != frame.image_data.shape:
    #             frame.image_data.resize(new_data.shape)
    #         frame.image_data.write_direct(new_data)

    # def correct_magnification(self):
    #     """Correct for changes in magnification at different energies.

    #     As the X-ray energy increases, the focal length of the zone
    #     plate changes and so the image is zoomed-out at higher
    #     energies. This method applies a correction to each frame to
    #     make the magnification similar to that of the first
    #     frame. Some beamlines correct for this automatically during
    #     acquisition: APS 8-BM-B
    #     """
    #     raise NotImplementedError("Mag correction done during import")

    # def apply_translation(self, shift_func, new_name,
    #                       description="Applying translation"):
    #     """Apply a translation to every frame using
    #     multiprocessing. `shift_func` should be a function that
    #     accepts a dictionary and returns an offset tuple (x, y) of
    #     corrections to be applied to the frame and the labels. The
    #     dictionary will contain the key, energy, data and labels for a
    #     frame. All frames have their sample position set set to (0, 0)
    #     since we don't know which one is the real position.
    #     """
    #     raise NotImplementedError()
    #     # Create new data groups to hold shifted image data
    #     self.fork_group(new_name)
    #     self.fork_labels(new_name + "_labels")
    #     # Multiprocessing setup

    #     def worker(payload):
    #         # key, data, labels = payload
    #         key = payload['key']
    #         data = payload['data']
    #         labels = payload['labels']
    #         # Apply net transformation with bicubic interpolation
    #         shift = shift_func(payload)
    #         transformation = transform.SimilarityTransform(translation=shift)
    #         new_data = transform.warp(data, transformation,
    #                                   order=3, mode="wrap", preserve_range=True)
    #         # Transform labels
    #         original_dtype = labels.dtype
    #         labels = labels.astype(np.float64)
    #         new_labels = transform.warp(labels, transformation,
    #                                     order=0, mode="constant", preserve_range=True)
    #         new_labels = new_labels.astype(original_dtype)
    #         ret = {
    #             'key': key,
    #             'energy': payload['energy'],
    #             'data': new_data,
    #             'labels': new_labels
    #         }
    #         return ret

        # # Launch multiprocessing queue
        # process_with_smp(frameset=self,
        #                  worker=worker,
        #                  description=description)
        # # Update new positions
        # for frame in self:
        #     frame.sample_position = position(0, 0, frame.sample_position.z)

    # def correct_drift(self, new_name="aligned_frames", method="ransac",
    #                   loc=xycoord(x=20, y=20), reference_frame=0,
    #                   plot_fit=False):
    #     """Apply a linear correction for a misalignment of zoneplate in APS
    #     8BM-B beamline as of Nov 2015. Deprecated in favor of
    #     align_frames method.

    #     Arguments
    #     ---

    #     method (default: "ransac"): Which type of regression to use:
    #         "ransac", "linear"

    #     loc (default 20, 20): Which particle to use to track drift

    #     reference_frame: index of the frame that stands still

    #     plot_fit (default False): If truthy, plot the resulting fit
    #         line of frame drift.

    #     """
    #     # Create new data groups to hold shifted image data
    #     # self.fork_group(new_name)
    #     # self.fork_labels(new_name + "_labels")
    #     # Regression values determined from cell 1 charge on 2015-11-11
    #     slope_v = 0.34693115
    #     slope_h = -0.28493559
    #     # slope_v = 0.32418403
    #     # slope_h = -0.25658992
    #     E_0 = self[reference_frame].energy
    #     # Prepare particle positions for regression
    #     centroids = self.particle_centroid_spectrum(loc=loc)
    #     x = np.array(centroids.index).reshape(-1, 1)
    #     # Perform linear regression (RANSAC ignores outliers)
    #     if method == "ransac":
    #         regression_v = linear_model.RANSACRegressor(linear_model.LinearRegression())
    #         regression_h = linear_model.RANSACRegressor(linear_model.LinearRegression())
    #     elif method == "linear":
    #         regression_v = linear_model.LinearRegression()
    #         regression_h = linear_model.LinearRegression()
    #     regression_v.fit(x, centroids.vertical)
    #     regression_h.fit(x, centroids.horizontal)
    #     if method == "ransac":
    #         inliers_v = np.count_nonzero(regression_v.inlier_mask_)
    #         regression_v = regression_v.estimator_
    #         inliers_h = np.count_nonzero(regression_h.inlier_mask_)
    #         regression_h = regression_h.estimator_
    #     slope_v = regression_v.coef_[0]
    #     slope_h = regression_h.coef_[0]
    #     # icpt_v = regression_v.intercept_
    #     # icpt_h = regression_h.intercept_
    #     error_v = regression_v.score(x, centroids.vertical)
    #     error_h = regression_h.score(x, centroids.horizontal)

    #     # Plot results of regression
    #     if plot_fit:
    #         pyplot.plot(x, centroids.vertical, marker="o", linestyle="None")
    #         pyplot.plot(x, centroids.horizontal, marker="o", linestyle="None")
    #         pyplot.plot(x, regression_v.predict(x))
    #         pyplot.plot(x, regression_h.predict(x))
    #         pyplot.legend(["Vertical", "Horizontal"])

    #     # Display status
    #     if method == "ransac":
    #         description = "Correcting drift (R²: {}v, {}h, #inliers: {}v, {}h)".format(
    #             round(error_v, 3), round(error_h, 3),
    #             inliers_v, inliers_h
    #         )
    #     else:
    #         description = "Correcting drift (R²: {}v, {}h)".format(
    #             round(error_v, 3), round(error_h, 3)
    #         )

    #     # Move frames
    #     def shift_func(payload):
    #         delta_E = payload['energy'] - E_0
    #         correction = xycoord(x=(slope_h * delta_E), y=(slope_v * delta_E))
    #         return correction
    #     self.apply_translation(shift_func, new_name=new_name, description=description)
    #     # Set active particles
    #     for frame in self:
    #         frame.activate_closest_particle(loc=loc)

    def apply_transformations(self, crop=True, commit=True):
        """Take any transformations staged with `self.stage_transformations()`
        and apply them. If commit is truthy, the staged
        transformations are reset.

        Returns: A numpy array with the transformed images.

        Arguments
        ---------
        - crop : If truthy, the images will be cropped after being
        translated, so there are not edges. If falsy, the images will
        be wrapped.

        - commit : If truthy, the changes will be saved to the HDF5
          store and the staged transformations will be cleared.

        """
        not_actionable = (self._translations is None and
                          self._rotations is None and
                          self._scales is None)
        if not_actionable:
            # Nothing to apply, so no-op
            with self.store() as store:
                out = store.absorbances.value
        else:
            # Apply the transformations
            with self.store() as store:
                # Prepare an array to hold results
                out = np.zeros_like(store.absorbances)
                # Apply transformation
                xm.transform_images(data=store.absorbances,
                                    translations=self._translations,
                                    rotations=self._rotations,
                                    scales=self._scales,
                                    mode='wrap', out=out)
            # Calculate and apply cropping bounds for the image stack
            if crop:
                tx = self._translations[...,0]
                ty = self._translations[...,1]
                new_rows = out.shape[-2] - (np.max(ty) - np.min(ty))
                new_cols = out.shape[-1] - (np.max(tx) - np.min(tx))
                rlower = int(np.ceil(-np.min(ty)))
                rupper = int(np.floor(new_rows + rlower))
                clower = int(np.ceil(-np.min(tx)))
                cupper = int(np.floor(clower + new_cols))
                out = out[...,rlower:rupper,clower:cupper]
            # Save result and clear saved transformations if appropriate
            if commit:
                with self.store('r+') as store:
                    store.absorbances = out
                self._translations = None
                self._scales = None
                self._rotations = None
        # Return calculated result
        return out

    def stage_transformations(self, translations=None, rotations=None, scales=None):

        """Allows for deferred transformation of the frame data. Since each
        transformation introduces interpolation error, the best
        results occur when the translations are saved up and then
        applied all in one shot. Takes a combination of arrays of
        translations (x, y), rotations and/or scales and saves them for later
        application. This method should be used in conjunction
        apply_transformations().
        """
        # Save translations for later
        if translations is not None:
            if self._translations is None:
                self._translations = np.copy(translations)
            else:
                self._translations += translations
        # Save scale factors for later
        if scales is not None:
            if self._scales is None:
                self._scales = np.copy(scales)
            else:
                self._scales *= scales
        # Save rotations for later
        if rotations is not None:
            if self._rotations is None:
                self._rotations = np.copy(rotations)
            else:
                self._rotations += rotations

    def align_frames(self,
                     reference_frame="mean",
                     blur=None,
                     method: str="cross_correlation",
                     template=None,
                     passes=1,
                     commit=True,
                     representation="modulus",
                     plot_results=True):
        """Use cross correlation algorithm to line up the frames. All frames
        will have their sample position set to (0, 0) since we don't
        know which one is the real position. This operation will
        interpolate between pixels so introduces error. If multiple
        passes are performed, the translations are saved and combined
        at the end so this error is only introduced once. Using the
        `commit=False` argument allows for multiple different types of
        registration to be performed in sequence, since uncommitted
        translations will be applied before the next round of registration.

        Arguments
        ---------

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

        representation : What component of the data to use: 'modulus',
          'phase', 'imag' or 'real'.

        plot_results : If truthy (default), plot the root-mean-square of the
          translation distance for each pass.
        """
        pass_distances = []
        # Check for valid attributes
        valid_filters = ["median", None]
        if blur not in valid_filters:
            msg = "Invalid blur filter {}. Choices are {}".format(blur,
                                                                  valid_filters)
            raise AttributeError(msg) from None
        # Sanity check on `method` argument
        valid_methods = ['cross_correlation', 'template_match']
        if method not in valid_methods:
            msg = "Unknown method {}. Choices are {}".format(method, valid_methods)
            raise ValueError(msg)
        # Guess best reference frame to use
        if reference_frame is "max":
            spectrum = self.xanes_spectrum(representation=representation)
            reference_frame = np.argmax(spectrum.values)
        # Keep track of how many passes and where we started
        for pass_ in range(0, passes):
            # Get data from store
            frames = self.apply_transformations(crop=True, commit=False)
            # Calculate axes to use for proper reference image
            if reference_frame == 'mean':
                ref_image = np.mean(frames, axis=(0, 1))
            elif reference_frame == 'median':
                ref_image = np.median(frames, axis=(0, 1))
            else:
                ref_image = frames[reference_frame]
                # Check that the argument results in 2D image_data
                if len(ref_image.shape) != 2:
                    msg = "refrence_frame ({}) does not match shape of frameset {}."
                    msg += "Please provide a {}-tuple."
                    msg = msg.format(reference_frame,
                                     frames.shape,
                                     len(frames.shape) - 2)
                    raise IndexError(msg)
            # Prepare blurring if requested
            if blur == "median":
                ref_image = filters.median(ref_image,
                                           morphology.disk(20))
            # original_shape = shape(*reference_image.shape)
            # Calculate translations for each frame
            if method == "cross_correlation":
                translations = xm.register_correlations(frames=frames,
                                                     reference=ref_image)
            elif method == "template_match":
                translations = xm.register_template(frame=frames, template=template)
            # Add the root-mean-square to the list of distances translated
            rms = np.sqrt((translations**2).sum(axis=-1).mean())
            pass_distances.append(rms)
            # Save translations for deferred calculation
            self.stage_transformations(translations=translations)
        # Plot the results if requested
        if plot_results:
            x = range(0, passes)
            ax = new_axes()
            ax.plot(x, pass_distances, marker='o', linestyle=":")
            ax.set_xlabel('Pass')
            ax.set_ylabel("RMS Translation")
        # Apply result of calculations to disk (if requested)
        if commit:
            self.apply_transformations(crop=True, commit=True)
        # with self.store(mode='r+') as store:
        #     transform_images(data=frames, translations=translations,
        #                      out=store.absorbances, mode='wrap')
        # reference_match = feature.match_template(component(reference_image, "imag"),
        #                                                  component(reference_target, "imag"),
        #                                                  pad_input=True)
        #         reference_center = np.unravel_index(reference_match.argmax(),
        #                                             reference_match.shape)
        #         reference_center = Pixel(vertical=reference_center[0],
        #                                  horizontal=reference_center[1])
        #     # Multiprocessing setup
        #     def worker(payload):
        #         key = payload['key']
        #         data = payload['data']
        #         # Temporarily rescale the data to be between -1 and 1
        #         scaled_data = data
        #         if blur == "median":
        #             blurred_data = filters.median(scaled_data, morphology.disk(20))
        #         elif blur is None:
        #             blurred_data = np.copy(scaled_data)
        #         labels = payload.get('labels', None)
        #         if current_method == "cross_correlation":
        #             # Determine what the new translation should be
        #             results = feature.register_translation(reference_image,
        #                                                    blurred_data,
        #                                                    upsample_factor=upsampling)
        #             shift, error, diffphase = results
        #             shift = xycoord(-shift[1], -shift[0])
        #         elif current_method == "template_match":
        #             # Determine what the new translation should be
        #             match = feature.match_template(component(scaled_data, "imag"),
        #                                            component(reference_target, "imag"),
        #                                            pad_input=True)
        #             center = np.unravel_index(match.argmax(), match.shape)
        #             center = Pixel(vertical=center[0], horizontal=center[1])
        #             # Determine the net translation necessary to align to reference frame
        #             shift = xycoord(
        #                 x=center.horizontal - reference_center.horizontal,
        #                 y=center.vertical - reference_center.vertical,
        #             )
        #         # Apply net transformation with bicubic interpolation
        #         # transformation = transform.SimilarityTransform(translation=shift)
        #         # new_data = transform.warp(data, transformation,
        #         #                           order=3, mode="wrap", preserve_range=True)
        #         new_data = _transform(data, translation=shift)
        #         # # Reset intensities of original values
        #         # new_data = exposure.rescale_intensity(new_data,
        #         #                                       in_range=out_range,
        #         #                                       out_range=in_range)
        #         result = {
        #             'key': key,
        #             'energy': payload['energy'],
        #             'data': new_data,
        #             'shift': shift,
        #         }
        #         # Transform labels
        #         if labels:
        #             original_dtype = labels.dtype
        #             labels = labels.astype(np.float64)
        #             new_labels = transform.warp(labels, transformation, order=0, mode="constant", preserve_range=True)
        #             new_labels = new_labels.astype(original_dtype)
        #             result['labels'] = new_labels
        #         return result

        #     # Save coordinates for determining cropping later on
        #     limits = {
        #         'left': 0,
        #         'right': 0,
        #         'top': 0,
        #         'bottom': 0,
        #     }

        #     def process_result(payload):
        #         key = payload['key']
        #         shift = payload.pop('shift')
        #         # Check if these shifts set new cropping limits
        #         if shift.y > limits['bottom']:
        #             limits['bottom'] = shift.y
        #         elif shift.y < limits['top']:
        #             limits['top'] = shift.y
        #         if shift.x > limits['right']:
        #             limits['right'] = shift.x
        #         elif shift.x < limits['left']:
        #             limits['left'] = shift.x
        #         # Save shift for final transformation
        #         past_shifts = shifts.get(key, [])
        #         past_shifts.append(shift)
        #         shifts[key] = past_shifts
        #         return payload

        #     # Launch the multiprocessing queue
        #     description = "Aligning pass {curr}/{total}"
        #     description = description.format(curr=current_pass+1,
        #                                      total=passes,
        #                                      frame=reference_frame)
        #     process_with_smp(frameset=self,
        #                      worker=worker,
        #                      process_result=process_result,
        #                      description=description)
        #     # Update new positions
        #     for frame in self:
        #         frame.sample_position = position(0, 0, frame.sample_position.z)
        #     # Crop frames and save for later
        #     if crop:
        #         bottom = math.ceil(abs(limits['top']))
        #         top = math.floor(original_shape.rows - abs(limits['bottom']))
        #         left = math.ceil(abs(limits['left']))
        #         right = math.floor(original_shape.columns - abs(limits['right']))
        #         for frame in prog(self, "Cropping frames"):
        #             frame.crop(bottom=bottom, left=left, top=top, right=right)
        #         # Save cropping dimensions for final crop after last pass
        #         all_crops.append(
        #             Crop(top=top, left=left, bottom=bottom, right=right)
        #         )
        #     # Increment counter to keep track of current position
        #     current_pass += 1

        # # Perform a final, complete translation and cropping if necessary
        # if passes > 1:
        #     # Revert back to original frameset
        #     self.switch_group(os.path.basename(original_group))
        #     self.fork_group(new_name)
        #     if self.active_labels_groupname:
        #         self.fork_labels(new_name + "_labels")
        #     # Multiprocessing setup
        #     def worker(payload):
        #         key = payload['key']
        #         data = payload['data']
        #         # Temporarily rescale the data to be between -1 and 1
        #         labels = payload.get('labels', None)
        #         # Compute the net translation needed for this frame
        #         curr_shifts = shifts[key]
        #         shift = xycoord(
        #             sum([n[0] for n in curr_shifts]),
        #             sum([n[1] for n in curr_shifts])
        #         )
        #         new_data = _transform(data, translation=shift)
        #         result = {
        #             'key': key,
        #             'energy': payload['energy'],
        #             'data': new_data,
        #             'shift': shift,
        #         }
        #         # Transform labels
        #         if labels:
        #             original_dtype = labels.dtype
        #             labels = labels.astype(np.float64)
        #             new_labels = transform.warp(labels,
        #                                         transformation,
        #                                         order=0,
        #                                         mode="constant", preserve_range=True)
        #             new_labels = new_labels.astype(original_dtype)
        #             result['labels'] = new_labels
        #         return result
        #     process_with_smp(frameset=self,
        #                      worker=worker,
        #                      description="Final alignment")
        #     # Calculate smallest cropping size
        #     if crop:
        #         top, bottom, left, right = (0, 0, 0, 0)
        #         for crop in all_crops:
        #             bottom += crop.bottom
        #             left += crop.left
        #         last_crop = all_crops[-1]
        #         right = left + last_crop.right - last_crop.left
        #         top = bottom + last_crop.top - last_crop.bottom
        #         crop = Crop(top=top, left=left, bottom=bottom, right=right)
        #         # Crop frames down to size
        #         for frame in prog(self, "Final crop"):
        #             frame.crop(left=crop.left, bottom=crop.bottom,
        #                        right=crop.right, top=crop.top)

    # def align_to_particle(self, loc, new_name, reference_frame=None):
    #     """Use template matching algorithm to line up the frames. Similar to
    #     `align_frames` but matches only to the particle closest to the
    #     argument `loc`.
    #     """
    #     # Autoguess best reference frame
    #     if reference_frame is None:
    #         spectrum = self.xanes_spectrum()
    #         reference_frame = np.argmax(spectrum.values)
    #     # Create new data groups to hold shifted image data
    #     self.fork_group(new_name)
    #     self.fork_labels(new_name + "_labels")
    #     # Determine which particle to use
    #     particle = self[reference_frame].activate_closest_particle(loc=loc)
    #     particle_img = np.copy(particle.image())
    #     # Set all values outside the particle itself to 0
    #     particle_img[np.logical_not(particle.mask())] = 0
    #     reference_key = self[reference_frame].key()
    #     reference_img = self[reference_frame].image_data.value
    #     reference_match = feature.match_template(reference_img, particle_img, pad_input=True)
    #     reference_center = np.unravel_index(reference_match.argmax(),
    #                                         reference_match.shape)
    #     reference_center = Pixel(vertical=reference_center[0],
    #                              horizontal=reference_center[1])

    #     # Multiprocessing setup
    #     def worker(payload):
    #         key = payload['key']
    #         energy = payload['energy']
    #         data = payload['data']
    #         labels = payload['labels']
    #         # Determine where the reference particle is in this frame's image
    #         match = feature.match_template(data, particle_img, pad_input=True)
    #         center = np.unravel_index(match.argmax(), match.shape)
    #         center = Pixel(vertical=center[0], horizontal=center[1])
    #         # Determine the net translation necessary to align to reference frame
    #         shift = [
    #             center.horizontal - reference_center.horizontal,
    #             center.vertical - reference_center.vertical,
    #         ]
    #         if key == reference_key:
    #             # Sanity check to ensure that reference frame does not shift
    #             assert shift == [0, 0], "Reference frame is shifted by " + shift
    #             ret = {
    #                 'key': key,
    #                 'energy': energy,
    #                 'data': data,
    #                 'labels': labels,
    #             }
    #         else:
    #             # Apply the translation with bicubic interpolation
    #             transformation = transform.SimilarityTransform(translation=shift)
    #             new_data = transform.warp(data, transformation,
    #                                       order=3, mode="wrap", preserve_range=True)
    #             # Transform labels
    #             original_dtype = labels.dtype
    #             labels = labels.astype(np.float64)
    #             new_labels = transform.warp(labels, transformation, order=0, mode="constant", preserve_range=True)
    #             new_labels = new_labels.astype(original_dtype)
    #             ret = {
    #                 'key': key,
    #                 'energy': energy,
    #                 'data': new_data,
    #                 'labels': new_labels
    #             }
    #         return ret

    #     def process_result(payload):
    #         frame = self[payload['key']]
    #         frame.activate_closest_particle(loc=loc)
    #         return payload

    #     # Launch the multiprocessing queue
    #     description = "Aligning to frame [{}]".format(reference_frame)
    #     process_with_smp(frameset=self,
    #                      worker=worker,
    #                      process_result=process_result,
    #                      description=description)

    #     # Update new positions
    #     for frame in self:
    #         frame.sample_position = position(0, 0, frame.sample_position.z)
    #     return reference_match

    # def crop_to_particle(self, loc=None, new_name='cropped_particle'):
    #     """Reduce the image size to just show the particle in
    #     question. Requires that particles be already labeled using the
    #     `label_particles()` method. Can either find the right particle
    #     using the `loc` argument, or using each frame's
    #     `active_particle_idx` attribute, allowing for more
    #     fine-grained control.  particles based on location.

    #     Arguments
    #     ---------
    #     - loc : 2-tuple of relative (x, y) position indicated the
    #       point to search from. If omitted or None, each frame's
    #       `active_particle_idx` attribute will be used.
    #     - new_name : Name to give the new group.
    #     """
    #     # Create a copy of the data group
    #     self.fork_group(new_name)
    #     # Activate particle if necessary
    #     if loc is not None:
    #         for frame in prog(self, 'Identifying closest particle'):
    #             frame.activate_closest_particle(loc=loc)
    #     # Make sure an active particle is assigned to all frames
    #     for frame in self:
    #         if frame.active_particle_idx is None:
    #             msg = "Frame {idx} has no particle assigned. Try {cls}.label_particles()"
    #             raise exceptions.NoParticleError(msg.format(idx=frame, cls=frame))
    #     # Determine largest bounding box based on all energies
    #     boxes = [frame.particles()[frame.active_particle_idx].bbox()
    #              for frame in self]
    #     left = min([box.left for box in boxes])
    #     bottom = min([box.bottom for box in boxes])
    #     top = max([box.top for box in boxes])
    #     right = max([box.right for box in boxes])

    #     # Make sure the expanded box is square
    #     def expand_dims(lower, upper, target):
    #         center = (lower + upper) / 2
    #         new_lower = center - target / 2
    #         new_upper = center + target / 2
    #         return (new_lower, new_upper)
    #     vertical = top - bottom
    #     horizontal = right - left
    #     if horizontal > vertical:
    #         bottom, top = expand_dims(bottom, top, target=horizontal)
    #     elif vertical > horizontal:
    #         left, right = expand_dims(left, right, target=vertical)
    #     # Sanity checks to make sure the new window is square
    #     vertical = top - bottom
    #     horizontal = right - left
    #     assert abs(horizontal) == abs(vertical), "{}h ≠ {}v".format(horizontal, vertical)
    #     assert bottom < top
    #     assert left < right
    #     # Roll each image to have the particle top left
    #     for frame in prog(self, 'Cropping frames'):
    #         frame.crop(top=top, left=left, bottom=bottom, right=right)
    #         # Determine new main particle index
    #         new_idx = np.argmax([p.convex_area() for p in frame.particles()])
    #         frame.active_particle_idx = new_idx
    #         # Set the new relative position for this frames position in the image
    #         frame.relative_position = position(*loc, z=frame.sample_position.z)

    # def align_frame_positions(self):
    #     """Correct for inaccurate motion in the sample motors."""
    #     self.fork_group('aligned_frames')
    #     self.fork_labels('aligned_labels')
    #     # Determine average positions
    #     total_x = 0
    #     total_y = 0
    #     n = 0
    #     for frame in prog(self, 'Computing true center'):
    #         n += 1
    #         total_x += frame.sample_position.x
    #         total_y += frame.sample_position.y
    #     global_x = total_x / n
    #     global_y = total_y / n
    #     for frame in prog(self, 'Aligning frames'):
    #         um_per_pixel_x = 40 / frame.image_data.shape[1]
    #         um_per_pixel_y = 40 / frame.image_data.shape[0]
    #         offset_x = int(round(
    #             (global_x - frame.sample_position.x) / um_per_pixel_x
    #         ))
    #         offset_y = int(round(
    #             (global_y - frame.sample_position.y) / um_per_pixel_y
    #         ))
    #         frame.shift_data(x_offset=offset_x, y_offset=offset_y)
    #         # Store updated position info
    #         new_position = (
    #             frame.sample_position.x + offset_x * um_per_pixel_x,
    #             frame.sample_position.y + offset_y * um_per_pixel_y,
    #             frame.sample_position.z
    #         )
    #         frame.sample_position = new_position

    def label_particles(self, min_distance=20):
        """Use watershed segmentation to identify particles.

        Arguments
        ---------
        - min_distance : Controls how selective the algorithm is at
          grouping areas into particles. Lower numbers means more
          particles, but might split large particles into two.
        """
        with self.store('r+') as store:
            frames = store.absorbances
            Es = store.energies
            particles = xm.particle_labels(frames=frames, energies=Es, edge=self.edge())
            store.particle_labels = particles

    def rebin(self, new_shape=None, factor=None):
        """Resample all images into new shape. Arguments `shape` and `factor`
        passed to txm.frame.TXMFrame.rebin().
        """
        self.fork_group('rebinned')
        if self.active_labels_groupname:
            self.fork_labels('rebinned_labels')
        for frame in prog(self, "Rebinning"):
            frame.rebin(new_shape=new_shape, factor=factor)

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
            ax = new_axes()
            ax.plot(x, spectrum.values, marker="o", linestyle="None")
            ax.plot(x, regression.predict(x))

    def particle_regions(self, map_name="whiteline_map", labels=None):
        """Return a list of regions (1 for each particle) sorted by area.
        (largest first). This requires that the `label_particles`
        method be called first.

        Arguments
        ---------
        - map_name : string with the attribute name to get for intensity data.

        - labels : Dataframe of the same shape as the map, with the
          particles segmented. If None, the `particle_labels`
          attribute of the TXM store will be used.
        """
        with self.store() as store:
            if labels is None:
                labels = store.particle_labels.value
            map_ = store.get_map(map_name).value
            regions = measure.regionprops(labels,
                                          intensity_image=map_)
        # Put in order of descending area
        regions.sort(key=lambda p: p.area)
        return regions

    def particle_area_spectrum(self, loc=xycoord(20, 20)):
        """Calculate a spectrum based on the area of the particle closest to
        the given location in the frame. This may be useful for assessing
        magnification across multiple frames.
        """
        energies = [f.energy for f in self]
        areas = []
        for frame in self:
            particle_idx = frame.closest_particle_idx(loc)
            particle = frame.particles()[particle_idx]
            areas.append(particle.area())
        return pd.Series(areas, index=energies)

    def particle_centroid_spectrum(self, loc=xycoord(20, 20)):
        """Calculate a spectrum based on the image centroid of the particle
        closest to the given location in the frame. This may be useful
        for assessing systematic drift across multiple frames.
        """
        energies = [f.energy for f in self]
        vs = []
        hs = []
        for frame in self:
            particle_idx = frame.closest_particle_idx(loc)
            particle = frame.particles()[particle_idx]
            centroid = particle.centroid()
            vs.append(centroid.vertical)
            hs.append(centroid.horizontal)
        return pd.DataFrame({'vertical': vs, 'horizontal': hs}, index=energies)

    def plot_mean_image(self, ax=None):
        if ax is None:
            ax = new_image_axes()
        data = self.mean_image()
        artist = ax.imshow(data, extent=self.extent(), origin="lower",
                           cmap='gray')
        return artist

    # def mean_image(self):
    #     """Determine an overall image by taking the mean intensity of each
    #     pixel across all frames."""
    #     frames = np.array([f.image_data for f in self])
    #     avg_frame = np.mean(frames, axis=0)
    #     return avg_frame

    def median_image(self):
        """Determine an overall image by taking the median intensity of each
        pixel across all frames."""
        frames = np.array([f.image_data for f in self])
        median_frame = np.median(frames, axis=0)
        return median_frame

    def xanes_spectrum(self, *args, **kwargs):
        raise (UserWarning('use spectrum()'))
        return self.spectrum(*args, **kwargs)

    @functools.lru_cache()
    def spectrum(self, pixel=None, edge_jump_filter=False,
                       representation="modulus"):
        """Collapse the dataset down to a two-dimensional spectrum. Returns a
        pandas series containing the resulting spectrum.

        Arguments
        ---------
        pixel: A 2-tuple that causes the returned series to represent
            the spectrum for only 1 pixel in the frameset. If None, a
            larger part of the frame will be used, depending on the
            other arguments.

        edge_jump_filter (bool or str): If truthy, only pixels that
            pass the edge jump filter are used to calculate the
            spectrum. If "inverse" is given, then the edge jump filter
            is logically not-ted and calculated with a more
            conservative threshold.

        """
        # energies = []
        # intensities = []
        # # Calculate masks if necessary
        # if edge_jump_filter == "inverse":
        #     mask = ~self.edge_jump_mask(sensitivity=0.4)
        # elif edge_jump_filter:
        #     mask = self.edge_jump_mask()
        # # Determine the contribution from each energy frame
        # for frame in self:
        #     data = frame.get_data(name=representation)
        #     # Determine which subset of pixels to use
        #     if pixel is not None:
        #          # Specific pixel is requested
        #         intensity = data[pixel.vertical][pixel.horizontal]
        #     elif edge_jump_filter:
        #         masked_data = np.ma.array(data, mask=mask)
        #         # Average absorbances for datasets
        #         intensity = np.sum(masked_data) / np.sum(masked_data.mask)
        #     else:
        #         masked_data = data
        #         # Sum absorbances for datasets
        #         intensity = np.sum(data) / np.prod(data.shape)
        #     # Add to cumulative arrays
        #     intensities.append(intensity)
        #     energies.append(frame.energy)
        # Retrieve data
        with self.store() as store:
            energies = store.energies.value
            frames = store.absorbances
            if pixel is not None:
                pixel = Pixel(*pixel)
                # Get a spectrum for a single pixel
                spectrum = frames[..., pixel.vertical, pixel.horizontal]
            else:
                if edge_jump_filter:
                    # Filter out background pixels using edge mask
                    mask = self.edge_mask()
                    mask = np.broadcast_to(array=mask,
                                           shape=(*energies.shape, *mask.shape))
                    frames = np.ma.array(frames, mask=mask)
                # Take average of all pixel frames
                spectrum = np.mean(np.reshape(frames, (frames.shape[0], -1)), axis=(1))
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
            ax = new_axes()
        # if normalize or show_fit:
        #     # Prepare an edge for fitting
        #     edge.post_edge_order = 1
        #     try:
        #         edge.fit(spectrum)
        #     except (exceptions.RefinementError,
        #             validation.NotFittedError,):
        #         # Fit failed, so we can't normalize
        #         normalize = False
        #         show_fit = False
        if normalize:
            # Adjust the limits of the spectrum to be between 0 and 1
            normalized = edge.normalize(spectrum.values, spectrum.index)
            spectrum = pd.Series(normalized, index=spectrum.index)
        scatter = plot_xanes_spectrum(spectrum=spectrum,
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
        # ax.axvline(x=norm_range[0], linestyle='-', color="0.55", alpha=0.4)
        # ax.axvline(x=norm_range[1], linestyle='-', color="0.55", alpha=0.4)
        return scatter

    def plot_xanes_edge(self, *args, **kwargs):
        """Call self.plot_xanes_spectrum() but zoomed in on the edge."""
        scatter = self.plot_xanes_spectrum(*args, **kwargs)
        ax = scatter.axes
        # Determine plotting limits
        start = self.edge.map_range[0] - 5
        stop = self.edge.map_range[1] + 5
        ax.set_xlim(start, stop)
        return scatter

    def plot_edge_jump(self, ax=None, alpha=1):
        """Plot the results of the edge jump filter."""
        if ax is None:
            ax = new_image_axes()
        ej = self.edge_jump_filter()
        artist = ax.imshow(ej, extent=self.extent(), cmap=self.cmap,
                           origin="lower", alpha=alpha)
        ax.set_xlabel('µm')
        ax.set_ylabel('µm')
        return artist

    @functools.lru_cache()
    def edge_jump(self):
        """Calculate a image showing the difference in
        signal across the X-ray edge."""
        with self.store() as store:
            ej = xm.edge_jump(frames=store.absorbances,
                              energies=store.energies.value,
                              edge=self.edge)
        return ej


    @functools.lru_cache()
    def edge_mask(self, sensitivity: float=1, min_size=0):
        """Calculate a mask for what is likely active material at this
        edge.

        Arguments
        ---------
        - sensitivity : A multiplier for the otsu value to determine
          the actual threshold.

        - min_size : Objects below this size (in pixels) will be
          removed. Passing zero (default) will result in no effect.
        """
        with self.store() as store:
            mask = xm.edge_mask(frames=store.absorbances,
                                energies=store.energies.value, edge=self.edge(),
                                sensitivity=sensitivity, min_size=min_size)
        return mask

    def goodness_filter(self):
        """Calculate an image based on the goodness of fit. `calculate_map`
        will be called if not already calculated."""
        raise DeprecationWarning()
        if not self.map_goodness_name:
            map_data, goodness = self.calculate_map()
        else:
            with self.hdf_file() as f:
                goodness = f[self.map_goodness_name].value
        return goodness

    def goodness_mask(self, sensitivity: float=0.5):
        """Calculate an image based on the goodness of fit. `calculate_map`
        must have been called previously. Goodness filter is converted
        to a binary map using (Otsu) threshold filtering.

        Arguments
        ---------
        - sensitivity: A multiplier for the otsu value to determine
          the actual threshold.
        """
        goodness = self.goodness_filter()
        try:
            threshold = filters.threshold_otsu(goodness)
        except TypeError:
            mask = np.zeros_like(goodness)
            # If thresholding failed, just show everything
        else:
            # If thresholding succeeded make a mask
            mask = goodness > (sensitivity * threshold)
            mask = morphology.dilation(mask)
            mask = np.logical_not(mask)
        return mask

    def calculate_whitelines(self):
        """Calculate and save a map of the whiteline position of each pixel."""
        with self.store() as store:
            energies = store.energies.value
            # Convert numpy axes to be in (pixel, energy) form
            frames = store.absorbances
            spectra = np.moveaxis(frames, 1, -1)
            # Calculate whiteline positions
            whitelines = xm.direct_whitelines(spectra=spectra,
                                              energies=energies, edge=self.edge)
        # Save results to disk
        with self.store(mode='r+') as store:
            store.whiteline_map = whitelines

    def calculate_maps(self):
        """Generate a set of maps based on pixel-wise Xanes spectra: whiteline
        position, particle labels."""
        # Calculate particle_labels
        self.calculate_whitelines()
        # self.label_particles()

    def masked_map(self, goodness_filter=True):
        """Generate a map based on pixel-wise Xanes spectra and apply an
        edge-jump filter mask. Default is to compute X-ray whiteline
        position.
        """
        raise DeprecationWarning()
        # Check for cached map of the whiteline position for each pixel
        if not self.map_name:
            map_data, goodness = self.calculate_map()
        else:
            with self.hdf_file() as f:
                map_data = f[self.map_name].value
        masked_map = np.ma.array(map_data, mask=self.goodness_mask())
        return masked_map

    def extent(self, representation, idx=0):
        """Determine physical dimensions for axes values.

        Returns: If idx was given, a single tuple of (left, right,
        bottom, up).

        Arguments
        ---------

        -idx : Index for a given frame. This allows for faster
         calculation if only a single frame is required.

        """
        with self.store() as store:
            frames = store.get_frames(representation)
            if idx is not None:
                # Filter to only the requested frame
                imshape = np.array(frames[idx].shape) # (rows, cols)
                pixel_size = store.pixel_sizes[idx]
                center = store.relative_positions[idx]
            else:
                # Include all frames
                imshape = np.array(frames.shape)[-2:] # ((rows, cols))
                pixel_size = store.pixel_sizes.value
                center = store.relative_positions.value
        width = imshape[-1] * pixel_size
        height = imshape[-2] * pixel_size
        # Calculate boundaries from image shapes
        if idx is not None:
            left = (center[-1] - width)/ 2
            right = (center[-1] + width) / 2
            bottom = (center[-2] - height) / 2
            top = (center[-2] + height) / 2
            ret = Extent(left=left, right=right,
                         bottom=bottom, top=top)
        else:
            left = (center[:,-1] - width)/ 2
            right = (center[:,-1] + width) / 2
            bottom = (center[:,-2] - height) / 2
            top = (center[:,-2] + height) / 2
            ret = np.array((left, right, bottom, top)).T
        return ret

    def plot_frame(self, idx, ax=None, cmap="gray", *args, **kwargs):
        """Plot the frame with given index as an image."""
        if ax is None:
            ax = new_image_axes()
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

    def plot_map(self, ax=None, map_name="whiteline_map"):
        """Prepare data and plot a map of whiteline positions."""
        # Do the plotting
        with self.store() as store:
            data = store.get_map(name=map_name)
            plot_txm_map(data=data,
                         ax=ax,
                         norm=None,
                         edge=self.edge(),
                         extent=self.extent(representation='absorbances'))

    def plot_goodness(self, plotter=None, ax=None, norm_range=None,
                      *args, **kwargs):
        """Use a default frameset plotter to draw a map of the goodness of fit
        as determined by the Edge object."""
        if plotter is None:
            plotter = FramesetPlotter(frameset=self, goodness_ax=ax)
        plotter.draw_goodness(norm_range=norm_range, *args, **kwargs)
        return plotter

    def plot_histogram(self, plotter=None, ax=None, norm_range=None,
                       goodness_filter=False,
                       active_pixel=None,
                       *args, **kwargs):
        """Use a default frameset plotter to draw a map of the chemical data."""
        if plotter is None:
            plotter = FramesetPlotter(frameset=self, map_ax=ax)
        artists = plotter.plot_histogram(norm_range=norm_range,
                                         goodness_filter=goodness_filter,
                                         *args, **kwargs)
        return artists

    def movie_plotter(self):
        """Creates an animation of all the frames in ascending energy, but
        does not display it anywhere, that's up to you."""
        pl = FramesetMoviePlotter(frameset=self)
        pl.create_axes(figsize=(10, 6))
        pl.connect_animation(repeat=False)
        return pl

    def save_movie(self, filename, *args, **kwargs):
        """Save an animation of all the frames and XANES to the specified
        filename."""
        pl = FramesetMoviePlotter(frameset=self)
        pl.create_axes()
        pl.connect_animation()
        pl.save_movie(filename=filename, *args, **kwargs)

    # def whiteline_map(self, method="direct"):
    #     """Calculate a map where each pixel is the energy of the whiteline.

    #     Arguments
    #     ---------
    #     method: A string declaring which method to use
    #       - "gaussian": fit the whiteline with a gaussian peak (accurate)
    #       - "direct": Find the energy with maximum absorbance (fast)
    #     """
    #     imagestack = build_series(self)
    #     # Call the appropriate calculation function
    #     if method == "gaussian":
    #         whiteline, goodness = calculate_gaussian_whiteline(
    #             imagestack, edge=self.edge()
    #         )
    #     elif method == "direct":
    #         if not prog.quiet:
    #             print("Calculating whiteline map...", end="")
    #         whiteline, goodness = calculate_direct_whiteline(
    #             imagestack, edge=self.edge()
    #         )
    #         if not prog.quiet:
    #             print("done")
    #     else:
    #         # Unknown value for method
    #         msg = 'Unknown method "{}".'.format(method)
    #         raise ValueError(msg)
    #     self.map_method = "whiteline_" + method
    #     return whiteline, goodness

    # def whiteline_energy(self):
    #     """Calculate the energy corresponding to the whiteline (maximum
    #     absorbance) for the whole frame. This first applies an
    #     edge-jump filter.
    #     """
    #     spectrum = self.xanes_spectrum(edge_jump_filter=True)
    #     whiteline = calculate_whiteline(spectrum, edge=self.edge())
    #     return whiteline

    # def fit_whiteline(self, width=4):
    #     """Calculate the energy corresponding to the whiteline (maximum
    #     absorbance) for the whole frame using gaussian peak fitting.
    #     """
    #     spectrum = self.xanes_spectrum(edge_jump_filter=True)
    #     peak, goodness = fit_whiteline(spectrum, width=width)
    #     return peak

    def hdf_file(self, mode='r'):
        """Return an open h5py.File object for this (any maybe other) frameset.

        Arguments
        ---------
        mode : A mode string, see h5py documentation for options. To
        avoid file corruption, calling this method as a context
        manager is recommended, especially with a mode other than 'r'.
        """
        if self.hdf_filename is not None:
            file = h5py.File(self.hdf_filename, mode)
        else:
            file = None
        return file

    def background_group(self):
        return self.hdf_file()[self.background_groupname]

    def hdf_node(self):
        """For use with HDFAttribute descriptor."""
        return self.hdf_group()

    def is_background(self):
        return self.active_group == self.background_group

    def add_frame(self, frame):
        setname_template = "{energy}_eV"
        frames_group = self.active_group()
        # Find a unique frame dataset name
        setname = setname_template.format(
            energy=frame.approximate_energy,
        )
        # Name found, so create the actual dataset
        frame.create_dataset(setname=setname,
                             hdf_group=frames_group,
                             compression="gzip")
        return setname

    def drop_frame(self, index):
        """Delete frame with the given index (int) or energy (float). This
        destructively removes the data from the HDF5 file, so use with
        caution.
        """
        frame = self[index]
        with self.hdf_file(mode="a") as f:
            # Delete group
            del f[frame.frame_group]

    def background_normalizer(self):
        # Find global limits
        global_min = 0
        global_max = 99999999999
        bg_group = self.background_group()
        for key in bg_group.keys():
            data = bg_group[key].value
            local_min = np.min(data)
            if local_min < global_min:
                global_min = local_min
            local_max = np.max(data)
            if local_max < global_max:
                global_max = local_max
        return Normalize(global_min, global_max)

    @functools.lru_cache()
    def image_normalizer(self, representation):
        # Find global limits
        # global_min = 99999999999
        # global_max = 0
        # for frame in self:
        #     data = frame.get_data(name=representation)
        #     # Remove outliers temporarily
        #     sigma = 9
        #     median = np.median(data)
        #     sdev = np.std(data)
        #     d = np.abs(data - median)
        #     s = d / sdev if sdev else 0.
        #     data[s >= sigma] = median
        #     # Check if this frame has the minimum intensity
        #     local_min = np.min(data)
        #     if local_min < global_min:
        #         global_min = local_min
        #     # Check if this has the maximum intensity
        #     local_max = np.max(data)
        #     if local_max > global_max:
        #         global_max = local_max
        with self.store() as store:
            global_min = np.min(store.absorbances)
            global_max = np.max(store.absorbances)
        return Normalize(global_min, global_max)

    def gtk_viewer(self):
        """Launch a GTK gui to view the data in the frameset."""
        from gtk_viewer import GtkTxmViewer
        viewer = GtkTxmViewer(edge=self.edge,
                              hdf_filename=self.hdf_filename, parent_name=self.parent_name)
        viewer.show()
        # Close the current blank plot
        pyplot.close()


class PtychoFrameset(XanesFrameset):
    """A set of images ("frames") at different energies moving across an
    absorption edge. The individual frames should be generated by
    ptychographic reconstruction of scanning transmission X-ray
    microscopy (STXM) to produce an array complex intensity
    values. This class does *not* include any code responsible for the
    collection and reconstruction of such data, only for the analysis
    in the context of X-ray absorption near edge spectroscopy."""

    FrameClass = PtychoFrame

    def representations(self):
        """Retrieve a list of valid representations for these data, such as
        modulus or phase data for ptychography."""
        reps = super().representations()
        reps += ['modulus', 'phase', 'real', 'imag']
        # Complex image data cannot be properly displayed
        if 'image_data' in reps:
            reps.remove('image_data')
        return reps

    def apply_internal_reference(self, mask=None, plot_background=True, ax=None):
        """Use a portion of each frame for internal reference correction. The
        result is the complex refraction for each pixel: the real
        component describes the phase shift, and the imaginary
        component is exponential decay, ie. absorbance.

        Arguments
        ---------

        mask : An array the same size as each frame that contains a
          mask to be provided to numpy's ma module.

        plot_background : If truthy, the values of I_0 are plotted as
          a function of energy.

        ax : The axes to use for plotting if `plot_background` is
          truthy.

        """
        self.fork_group("reference_corrected")
        # Convert the supplied mask to grayscale
        if mask is not None:
            graymask = color.rgb2gray(mask)
        # Array for holding background correction for plotting
        if plot_background:
            Es = []
            I_0s = []
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
                ax = new_axes()
            ax.plot(Es, I_0s)
            ax.set_title("Background Intensity used for Reference Correction")
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("$I_0$")


def process_with_smp(frameset: XanesFrameset,
                     worker: Callable[[dict], dict],
                     process_result: Callable[[dict], dict]=None,
                     description: str="Processing frames"):
    """Runs a computation on all frames in a frameset using parrallel
    processing and saves the result.

    Arguments
    ---------
    - frameset: Set of frames to be process.

    - worker: function to do the actual calculation. Should accept a
      dictionary payload which contains the frame's `key`, `energy`,
      `data`, and `labels`. Should return a similar dictionary with
      modified `data` and/or `labels`. If worker does not modify
      `data` or `labels`, it should remove them from the return
      dictionary to avoid unnecessary disk I/O.

    - process_result: A callback to perform additional
      post-processing. Should accept return a similar dictionary as
      worker. If the returned dicionary does not contain `data` or
      `labels` then the corresponding data will not be saved
      automatically.

    - description: A string describing the operation. Used in a status
      bar.

    """
    # Prepare callbacks and queue
    def result_callback(payload):
        frame = frameset.FrameClass(frameset=frameset, groupname=payload['key'])
        # Call user-provided result callback
        if process_result is not None:
            payload = process_result(payload)
        # Save data and/or labels if necessary
        if 'data' in payload.keys():
            frame.image_data = payload['data']
        if 'labels' in payload.keys():
            frame.particle_labels = payload['labels']
        if 'pixel_size_value' in payload.keys():
            px_unit = unit(payload['pixel_size_unit'])
            px_size = px_unit(payload['pixel_size_value'])
            frame.pixel_size = px_size
    queue = smp.Queue(worker=worker,
                      totalsize=len(frameset),
                      result_callback=result_callback,
                      description=description)

    # Populate the queue
    for frame in frameset:
        payload = {
            'data': frame.image_data,
            'key': frame.frame_group,
            'energy': frame.energy,
            'pixel_size_value': frame.pixel_size.num,
            'pixel_size_unit': str(frame.pixel_size.unit),
        }
        labels = frame.particle_labels
        if labels is not None:
            payload['labels'] = labels
        queue.put(payload)
    # Join the queue and wait for all processes to complete
    queue.join()
