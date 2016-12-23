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
import multiprocessing
import threading
import sys
import logging
import math

from tqdm import tqdm, tqdm_notebook
import numpy as np
import h5py

position = namedtuple('position', ('x', 'y', 'z'))
xycoord = namedtuple('xycoord', ('x', 'y'))
Pixel = namedtuple('Pixel', ('vertical', 'horizontal'))
shape = namedtuple('shape', ('rows', 'columns'))
Extent = namedtuple('extent', ('left', 'right', 'bottom', 'top'))


CPU_COUNT = multiprocessing.cpu_count()

def foreach(f, l, threads=CPU_COUNT, return_=False):
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
    ratio_h = ((pixel.horizontal+0.5) / shape[1])
    x = extent.left + ratio_h * (extent.right - extent.left)
    ratio_v = ((pixel.vertical+0.5) / shape[0])
    y = extent.bottom + ratio_v * (extent.top - extent.bottom)
    return xycoord(x=x, y=y)


def broadcast_reverse(array, shape, *args, **kwargs):
    """Take the array and extends it as much as possible to match
    `shape`. Similar to numpy's broadcast_to function, but starts with
    the most significant axis. For example, if `array` has shape (7,
    29), it can be broadcast to (7, 29, 1024, 1024).

    """
    def reverse_axes(array):
        ndims = array.ndim
        for n in range(int(ndims/2)):
            array = np.swapaxes(array, n, ndims - 1 - n)
        return array
    # Convert the array to its reverse representation
    rev_array = reverse_axes(array)
    rev_shape = shape[::-1]
    # Broadcast the array
    new_array = np.broadcast_to(rev_array, rev_shape, *args, **kwargs)
    # Convert the array back to its original representation
    new_array = reverse_axes(new_array)
    return new_array


def get_component(data, name):
    """If complex, turn to given component, otherwise return original data."""
    if np.iscomplexobj(data):
        # Sort out complex components
        if name == "modulus":
            data = np.abs(data)
        elif name == "phase":
            data = np.angle(data)
        elif name == "real":
            data = np.real(data)
        elif name == "imag":
            data = np.imag(data)
    else:
        # Just return the real part
        data = np.real(data)
    return data


def is_kernel():
    """Detect whether or not we're running inside an IPython kernel. NB:
    This does not distinguish between eg IPython notebook and IPython
    QtConsole.
    """
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported, definitely not
        return False
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), 'kernel', None) is not None


def  prog(*args, leave=False, dynamic_ncols=True, **kwargs):
    """A progress bar for displaying how many iterations have been
    completed. This is mostly just a wrapper around the tqdm
    library. *args and **kwargs are passed directly to either
    tqdm.tqdm or tqdm.tqdm_notebook.
    """
    if is_kernel():
        # Use the IPython widgets version of tqdm
        wrapper = tqdm_notebook
    else:
        # Use the text-version of tqdm
        wrapper = tqdm
    # Create and return the progress bar iterable
    prog_iter = wrapper(*args, leave=leave,
                        dynamic_ncols=dynamic_ncols, **kwargs)
    return prog_iter


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
