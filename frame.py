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

from collections import namedtuple, OrderedDict
import os
import math
import warnings

import dateutil.parser
import numpy as np
from scipy import ndimage
from skimage.morphology import (disk, closing, remove_small_objects,
                                watershed, dilation)
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops, label
from skimage.filters import threshold_adaptive, rank
from skimage.feature import peak_local_max
from skimage.restoration import unwrap_phase
from units import unit, predefined

import exceptions
import plots
from utilities import xycoord, Pixel, shape, component
from particle import Particle

predefined.define_units()
position = namedtuple('position', ('x', 'y', 'z'))
Extent = namedtuple('extent', ('left', 'right', 'bottom', 'top'))


def remove_outliers(data, sigma):
    """Mark as invalid any pixels more that `sigma` standard deviations
    away from the median."""
    median = np.median(data)
    d = np.abs(data - median)
    sdev = np.std(data)
    s = d / sdev if sdev else 0.
    data[s >= sigma] = median
    return data


def rebin_image(data, new_shape):
    """Resample image into new shape, but only if the new dimensions are
    smaller than the old. This is not meant to apply zoom corrections,
    only correct sizes in powers of two. Eg, a 2048x2048 images can be
    down-sampled to 1024x1024. If there are extra pixels in either
    direction, they are discarded from the bottom and right.

    Kwargs:
    -------
    new_shape (tuple): The target shape for the new array.

    """
    # Return original data is shapes are the same
    if data.shape == new_shape:
        return data
    # Check that the new shape is not larger than the old shape
    for idx, dim in enumerate(new_shape):
        if dim > data.shape[idx]:
            msg = 'New shape {new} is larger than original shape {original}.'
            msg = msg.format(new=new_shape, original=data.shape)
            raise ValueError(msg)
    ratios = shape(data.shape[0] / new_shape[0],
                   data.shape[1] / new_shape[1])
    # Check for and discard extra pixels
    if (ratios[0] % 1) or (ratios[1] % 1):
        ratios = shape(int(ratios[0]), int(ratios[1]))
        data = data[:ratios[0] * new_shape[0],:ratios[1] * new_shape[1]]
    # Determine new dimensions
    sh = (new_shape[0],
          int(ratios[0]),
          new_shape[1],
          int(ratios[1]))
    # new_data = data.reshape(sh).mean(-1).mean(1)
    new_data = data.reshape(sh).sum(-1).sum(1)
    return new_data


def apply_reference(data, reference_data):
    """Apply a reference correction to a raw image. This turns intensity
    data into absorbance data. If data and reference data have
    different shapes, they will be down-samples to the lower of the
    two.

    Arguments:
    ----------
    data (numpy array): The sample image to be corrected.

    reference_data (numpy array): The reference data that will be
        corrected against.
    """
    # Rebin datasets in case they don't match
    min_shape = [
        min([ds.shape[0] for ds in [data, reference_data]]),
        min([ds.shape[1] for ds in [data, reference_data]]),
    ]
    reference_data = rebin_image(reference_data, new_shape=min_shape)
    data = rebin_image(data, new_shape=min_shape)
    new_data = np.log10(reference_data / data)
    return new_data


def xy_to_pixel(xy, extent, shape):
    """Take an xy location on an image and convert it to a pixel location
    suitable for numpy indexing."""
    ratio_x = (xy.x - extent.left) / (extent.right - extent.left)
    pixel_h = int(round(ratio_x * shape[1]))
    ratio_y = (xy.y - extent.bottom) / (extent.top - extent.bottom)
    # (1 - ratio) for y because images are top indexed
    pixel_v = int(round(ratio_y * shape[0]))
    return Pixel(vertical=pixel_v, horizontal=pixel_h)


def pixel_to_xy(pixel, extent, shape):
    """Take an xy location on an image and convert it to a pixel location
    suitable for numpy indexing."""
    # ratio_x = (xy.x-extent.left)/(extent.right-extent.left)
    # pixel_h = int(round(ratio_x * shape[1]))
    # ratio_y = (xy.y-extent.bottom)/(extent.top-extent.bottom)
    # # (1 - ratio) for y because images are top indexed
    # pixel_v = int(round((1 - ratio_y) * shape[0]))
    ratio_h = (pixel.horizontal / shape[1])
    x = extent.left + ratio_h * (extent.right - extent.left)
    ratio_v = (pixel.vertical / shape[0])
    y = extent.bottom + ratio_v * (extent.top - extent.bottom)
    return xycoord(x=x, y=y)


class TXMFrame():
    """A single microscopy image at a certain energy."""

    # HDF Attributes
    hdf_default_scope = "frame"

    # TODO: These should be reimplement as a TXMStore object
    # energy = hdf.Attr(key='energy')
    # approximate_energy = hdf.Attr('approximate_energy', default=0.0)
    # _starttime = hdf.Attr('starttime')
    # _endtime = hdf.Attr('endtime')
    # _pixel_size_value = hdf.Attr(key="pixel_size_value", default=1)
    # _pixel_size_unit = hdf.Attr(key="pixel_size_unit",  default="px")
    # position_unit = hdf.Attr(key="position_unit", default="m", wrapper=unit)
    # relative_position = hdf.Attr(key="relative_position",
    #                               default=position(0, 0, 0),
    #                               wrapper=lambda coords: position(*coords))
    # sample_position = hdf.Attr(key="sample_position",
    #                              default=(0, 0, 0),
    #                              wrapper=lambda coords: position(*coords))
    # original_filename = hdf.Attr('original_filename')
    # particle_labels_path = hdf.Attr(key="particle_labels_path")
    # active_particle_idx = hdf.Attr(key="active_particle_idx")

    def __init__(self, frameset=None, groupname=None):
        self.frameset = frameset
        self.frame_group = groupname

    def __repr__(self):
        name = '<TXMFrame: "{}">'
        return name.format(
            self.frame_group
        )

    def get_data(self, name):
        """Retrieve image data with the given name. "Modulus" in this case
        just returns the original data, since it is only real."""
        if name == "modulus":
            data = self._get_data(name="image_data")
        else:
            data = self._get_data(name=name)
        return data

    def _get_data(self, name):
        """Retrieve image data with the given name."""
        with self.hdf_file(mode="a") as f:
            # Get current representation
            try:
                data = f[self.frame_group][name].value
            except (KeyError, TypeError):
                msg = 'Could not load group "{}"'.format(name)
                raise exceptions.GroupKeyError(msg)
        return data

    def set_data(self, name, data):
        return self._set_data(name=name, data=data)

    def _set_data(self, name, data):
        """Save image data as a dataset whose name is the given
        representation."""
        with self.hdf_file(mode="a") as f:
            try:
                del f[self.frame_group][name]
            except KeyError:
                pass
            f[self.frame_group].create_dataset(name, data=data, compression="gzip")

    @property
    def image_data(self):
        img = self.get_data(name="image_data")
        return img

    @image_data.setter
    def image_data(self, new_data):
        self.set_data(name="image_data", data=new_data)

    @property
    def particle_labels(self):
        try:
            img = self.get_data(name="particle_labels")
        except exceptions.GroupKeyError:
            img = None
        return img

    @particle_labels.setter
    def particle_labels(self, new_data):
        self.set_data(name="particle_labels", data=new_data)

    def hdf_file(self, *args, **kwargs):
        return self.frameset.hdf_file(*args, **kwargs)

    @property
    def approximate_position(self):
        actual_position = self._sample_position
        return position(x=round(actual_position.x, -1),
                        y=round(actual_position.y, -1),
                        z=round(actual_position.z, -1))

    @property
    def starttime(self):
        return dateutil.parser.parse(self._starttime)

    @starttime.setter
    def starttime(self, newdt):
        self._starttime = newdt.isoformat()

    @property
    def endtime(self):
        return dateutil.parser.parse(self._endtime)

    @endtime.setter
    def endtime(self, newdt):
        self._endtime = newdt.isoformat()

    def transmission_data(self, background_group):
        bg_data = self.background_dataset(group=background_group)
        return bg_data / np.exp(self.image_data)

    def background_dataset(self, group):
        return group[self.key()]

    def key(self):
        return self.image_data.name.split('/')[-1]

    def hdf_node(self):
        return self.image_data

    def extent(self, img_shape=None):
        """Determine physical dimensions for axes values."""
        if img_shape is None:
            img_shape = shape(*self.image_data.shape)
        else:
            img_shape = shape(*img_shape)
        pixel_size = self.pixel_size
        center = self.relative_position
        width = img_shape.columns * pixel_size
        height = img_shape.rows * pixel_size
        # Calculate boundaries from image shapes
        left = width.unit(center.x) - width / 2
        right = width.unit(center.x) + width / 2
        bottom = height.unit(center.y) - height / 2
        top = height.unit(center.y) + height / 2
        return Extent(left=left.num, right=right.num,
                      bottom=bottom.num, top=top.num)

    def plot_histogram(self, ax=None, bins=100, representation=None, *args, **kwargs):
        if ax is None:
            ax = plots.new_axes()
        ax.set_xlabel('Absorbance (AU)')
        ax.set_ylabel('Occurences')
        data = np.nan_to_num(self.get_data(name=representation))
        artist = ax.hist(data.flatten(), bins=bins, *args, **kwargs)
        return artist

    def plot_image(self, data=None, ax=None, cmap="gray",
                   show_particles=True, representation=None,
                   *args, **kwargs):
        """Plot a frame's data image. Use frame.image_data if no data are
        given."""
        if ax is None:
            ax = plots.new_image_axes()
        if data is None:
            data = self.get_data(name=representation)
        extent = self.extent(img_shape=shape(*data.shape))
        im_ax = ax.imshow(data, *args, cmap=cmap, extent=extent,
                          origin="lower", **kwargs)
        # Plot particles
        if show_particles:
            self.plot_particle_labels(ax=im_ax.axes, extent=extent)
        # Set labels, etc
        unit = self.pixel_size.unit
        ax.set_xlabel(unit)
        ax.set_ylabel(unit)
        im_ax.set_extent(extent)
        return im_ax

    def plot_particle_labels(self, ax=None, *args, **kwargs):
        """Plot the identified particles (as an overlay if ax is given)."""
        artists = []
        # Get axes extent (if passed) or use default
        extent = kwargs.pop('extent', self.extent())
        if ax is None:
            opacity = 1
            ax = plots.new_image_axes()
        else:
            opacity = 0.3
        if self.particle_labels_path:
            data = self.particle_labels
            # Mask out anything not labeled (ie. set to zero)
            mask = np.logical_not(data.value.astype(np.bool))
            masked_data = np.ma.array(data, mask=mask)
            artists.append(ax.imshow(masked_data, *args,
                                     alpha=opacity, cmap="Dark2",
                                     origin="lower", extent=extent,
                                     **kwargs))
            # Plot text for label index
            particles = self.particles()
            xs = [particle.relative_position().x for particle in particles]
            ys = [particle.relative_position().y for particle in particles]
            for idx, x in enumerate(xs):
                y = ys[idx]
                txt = ax.text(x, y, str(idx),
                              horizontalalignment='center',
                              verticalalignment='center')
                artists.append(txt)
        return artists

    def crop(self, bottom, left, top, right):
        """Reduce the image size to given box (in pixels)."""
        # Move image data to top left
        xoffset = -left
        yoffset = -bottom
        new_img = self.shift_data(x_offset=xoffset,
                                  y_offset=yoffset,
                                  data=self.image_data)
        # Shrink image to bounding box size
        new_shape = shape(rows=int((top - bottom)),
                          columns=int((right - left)))
        data = np.array(new_img[0:new_shape[0],
                                0:new_shape[1]])
        self.image_data = data
        # Repeat for particle labels
        labels = self.particle_labels
        if labels is not None:
            new_labels = self.shift_data(x_offset=xoffset,
                                         y_offset=yoffset,
                                         data=labels)
            new_labels = np.array(new_labels[0:new_shape[0],
                                             0:new_shape[1]])
            self.particle_labels = new_labels
        # Update stored position information
        pixel_size = self.pixel_size
        px_unit = pixel_size.unit
        pos_unit = self.position_unit
        sample_position = position(*[px_unit(pos_unit(c))
                                     for c in self.sample_position])
        new_position = position(
            x=sample_position.x + xoffset * pixel_size,
            y=sample_position.y + yoffset * pixel_size,
            z=sample_position.z
        )
        self.sample_position = [c.num for c in new_position]

    @staticmethod
    def shift_data(x_offset, y_offset, data):
        """Move the image data within the view field by the given offsets in
        pixels.  New values are rolled around to the other side.

        Arguments
        ---------
        x_offset : int
            Distance to move in pixels in x-diraction
        y_offset : int
            Distance to move in pixels in y-diraction
        dataset : Dataset
            Optional dataset to manipulate . If None, self.image_data
            will be used (default None)
        """
        # Roll along x axis
        new_data = np.roll(data, int(x_offset), axis=1)
        # Roll along y axis
        new_data = np.roll(new_data, int(y_offset), axis=0)
        return new_data

    def rebin(self, new_shape=None, factor=None):
        """Resample image into new shape. One of the kwargs `shape` or
        `factor` is required. Process is most effective when factors
        are powers of 2 (2, 4, 8, 16, etc). New shape is calculated
        and passed to the rebin_image() function.

        Kwargs:
        -------
        new_shape (tuple): The target shape for the new array. Will
            override `factor` if both are provided.
        factor (int): Factor by which to decrease the frame size. factor=2 would
            take a (1024, 1024) to (512, 512)

        """
        original_shape = shape(*self.image_data.shape)
        if new_shape is None and factor is None:
            # Raise an error if not arguments are passed.
            raise ValueError("Must pass one of `shape` or `factor`")
        elif new_shape is None:
            # Determine new shape from factor if not provided.
            new_shape = tuple(int(dim / factor) for dim in original_shape)
        new_shape = shape(*new_shape)
        # Calculate new, rebinned image data
        new_data = rebin_image(self.image_data.value, new_shape=new_shape)
        # Resize existing dataset
        self.image_data.resize(new_shape)
        self.image_data.write_direct(new_data)
        # Calculate new, rebinned particle labels
        try:
            labels = self.particle_labels()
        except exceptions.NoParticleError:
            pass
        else:
            new_labels = rebin_image(labels.value, new_shape=new_shape)
            # Resize existing particle labels
            labels.resize(new_shape)
            labels.write_direct(new_labels)
        # Adjust pixel size appropriately
        old_px_size = self.um_per_pixel
        new_px_size = Pixel(
            vertical=old_px_size.vertical * original_shape.rows / new_shape.rows,
            horizontal=old_px_size.horizontal * original_shape.columns / new_shape.columns
        )
        self.um_per_pixel = new_px_size
        return new_data

    @property
    def pixel_size(self):
        px_unit = unit(self._pixel_size_unit)
        px_size = px_unit(self._pixel_size_value)
        return px_size

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size_value = value.num
        self._pixel_size_unit = str(value.unit)
        # print("Saved", value.num, self._pixel_size_value)

    def create_dataset(self, setname, hdf_group):
        """Save data and metadata to an HDF dataset."""
        if setname in hdf_group.keys():
            msg = "{name} already exists in group {group}"
            msg = msg.format(name=setname, group=hdf_group.name)
            raise exceptions.DatasetExistsError(msg)
        else:
            attrs = getattr(self.image_data, 'attrs', self._attrs)
            self.image_data = hdf_group.create_dataset(name=setname,
                                                       data=self.image_data,
                                                       maxshape=self.image_data.shape,
                                                       compression="gzip")
            # Set metadata attributes
            for attr_name in attrs.keys():
                self.image_data.attrs[attr_name] = attrs[attr_name]

    @classmethod
    def load_from_dataset(Cls, dataset):
        """Accept an HDF5 frame dataset and return a new frame."""
        new_frame = Cls()
        new_frame.image_data = dataset
        return new_frame

    def activate_closest_particle(self, loc):
        """Get a particle that's closest to frame location and save for future
        reference."""
        if loc is not None:  # Some calling routines may pass `None`
            self.active_particle_idx = self.closest_particle_idx(loc)
            return self.particles()[self.active_particle_idx]
        else:
            return None

    def closest_particle_idx(self, loc):
        """Get the particle that's closest to frame location. Returns None if no
        particles are found.
        """
        particles = self.particles()
        current_min = 999999
        current_idx = None # (default if no particle found)
        # Include units on location tuple
        target = xycoord(*(self.pixel_size.unit(l) for l in loc))
        for idx, particle in enumerate(particles):
            center = particle.relative_position()
            distance = math.sqrt(
                (target.x - center.x)**2 + (target.y - center.y)**2
            )
            if distance < current_min:
                # New closest match
                current_min = distance
                current_idx = idx
        return current_idx

    def particles(self):
        labels = self.particle_labels
        props = regionprops(labels, intensity_image=self.image_data)
        particles = []
        for prop in props:
            particles.append(Particle(regionprops=prop, frame=self))
        return particles


class PtychoFrame(TXMFrame):
    def get_data(self, name):
        """
        Retrieves the data as representation `name`. Since the image data
        are complex, some special names are defined (anything else is assumed
        to be an HDF5 dataset name):
        - "modulus" or None: Magnitude of complex number
        - "phase": Phase of complex number
        - "real": Just real component of each value
        - "imag": Just the imaginary component of each value
        """
        if name is None:
            realname = "modulus"
        else:
            realname = name
        # Get complex data if necessary
        if realname in ['modulus', 'phase', 'real', 'imag']:
            data = component(self._get_data(name="image_data"), name=name)
        else:
            # No magic name was given, just get the requested dataset
            data = self._get_data(name=realname)
        return data

def calculate_particle_labels(data, return_intermediates=False,
                              min_distance=0.016):
    """Identify and label material particles in the image data.

    Generate and save a scikit-image style labels frame identifying
    the different particles. Returns an array with the same shape as
    `data` with values ap`oximating the particles in the frame. If
    kwarg `return_all` is truthy, returns an ordered dictionary of the
    intermediate arrays calculated during the algorithm (useful for
    debugging).

    Parameters
    ----------
    return_intermediates : bool
        Return intermediate images as a dict (default False)
    min_distance : float
        How far away (as a portion of image size) particle centers need to be in
        order to register as different particles (default 0.016)

    """
    with warnings.catch_warnings():
        # Supress precision loss warnings when we convert to binary mask
        warnings.simplefilter("ignore")
        # Shift image into range -1 to 1 by contrast stretching
        original = data
        in_range = (data.min(), data.max())
        equalized = rescale_intensity(original,
                                      in_range=in_range,
                                      out_range=(0, 1))
        # Identify foreground vs background with adaptive Otsu filter
        average_shape = sum(equalized.shape) / len(equalized.shape)
        # threshold = filters.threshold_otsu(equalized)
        # multiplier = 1
        block_size = average_shape / 2.72  # Determined emperically
        mask = threshold_adaptive(equalized, block_size=block_size, offset=0)
        mask_copy = np.copy(mask)
        # Determine minimum size for discarding small objects
        min_size = 4. * average_shape
        large_only = remove_small_objects(mask, min_size=min_size)
        # Fill in the shapes a lot to round them out
        reclosed = closing(large_only, disk(average_shape * 20 / 1024))
        # Expand the particles to make sure we capture the edges
        dilated = dilation(reclosed, disk(average_shape * 10 / 1024))
        # Compute each pixel's distance from the edge of a blob
        distances = ndimage.distance_transform_edt(dilated)
        in_range = (distances.min(), distances.max())
        distances = rescale_intensity(distances,
                                      in_range=in_range,
                                      out_range=(0, 1))
        # Blur the distances to help avoid split particles
        mean_distances = rank.mean(distances, disk(average_shape / 64))
        # Use the local distance maxima as peak centers and compute labels
        local_maxima = peak_local_max(
            mean_distances,
            indices=False,
            min_distance=min_distance * average_shape,
            labels=dilated
        )
        markers = label(local_maxima)
        labels = watershed(-mean_distances, markers, mask=dilated)
    if return_intermediates:
        result = OrderedDict()
        result['original'] = original
        result['equalized'] = equalized
        result['mask'] = mask_copy
        result['large_only'] = large_only
        result['reclosed'] = reclosed
        result['dilated'] = dilated
        result['mean_distances'] = mean_distances
        result['distances'] = distances
        result['local_maxima'] = local_maxima
        result['markers'] = markers
        result['labels'] = labels
    else:
        result = labels
    return result
