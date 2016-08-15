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

"""A collection of classes and functions that arent' specific to any
one type of measurement. Also, it defines some namedtuples for
describing coordinates.
"""

from collections import namedtuple
import sys

from tqdm import tqdm
import numpy as np
import h5py

position = namedtuple('position', ('x', 'y', 'z'))
xycoord = namedtuple('xycoord', ('x', 'y'))
Pixel = namedtuple('Pixel', ('vertical', 'horizontal'))
shape = namedtuple('shape', ('rows', 'columns'))
Extent = namedtuple('extent', ('left', 'right', 'bottom', 'top'))


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


def component(data, name):
    """If complex, turn to given component, otherwise return original data."""
    if np.any(data.imag):
        # Sort out complex components
        if name == "modulus":
            data = np.abs(data)
        elif name == "phase":
            data = np.angle(data)
        elif name == "real":
            data = data.real
        elif name == "imag":
            data = data.imag
    return data


class Prog:
    """A progress bar for displaying how many iterations have been
    completed. This is mostly just a wrapper around the tqdm
    library. Additionally it makes use of the borg pattern, so setting
    Prog.quiet to True once silences all progress bars. This is useful
    for testing.
    """
    __global_state = {
        'quiet': False
    }

    def __init__(self):
        self.__dict__ = self.__global_state

    def __call__(self, iterable, *args, **kwargs):
        """Progress meter. Wraps around tqdm with some custom defaults."""
        if self.quiet:
            # Just return the iterable with no progress meter
            ret = iterable
        else:
            kwargs['file'] = kwargs.get('file', sys.stdout)
            kwargs['leave'] = kwargs.get('leave', True)
            ret = tqdm(iterable, *args, **kwargs)
        return ret


prog = Prog()


def prepare_hdf_group(*args, **kwargs):
    """Check the filenames and create an hdf file as needed. Will
    overwrite the group if it already exists.

    Returns: The opened HDF5 group.

    Arguments
    ---------

    - filename : name of the requested hdf file, may be None if not
      provided, in which case the filename will be generated
      automatically based on `dirname`.

    - groupname : Requested groupname for these data.

    - dirname : Used to derive a default filename if None is passed
      for `filename` attribute.
    """
    raise UserWarning("Use txmstore.prepare_txm_store() instead")
    # return prepare_txm_store(*args, **kwargs)
