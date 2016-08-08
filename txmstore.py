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

"""Tools for accessing TXM data stored in an HDF5 file."""

import h5py
import numpy as np

import exceptions


class TXMStore():
    """Wrapper around HDF5 file that stores TXM data. It has a series of
    properties that return the corresponding HDF5 dataset object; the
    TXMStore().attribute.value pattern can be used to get pure numpy
    arrays. These objects should be used as a context manager to ensure
    that the file is closed, especially if using a writing mode:
        with TXMStore() as store:
            # Do stuff with store here

    Arguments
    ---------
    - hdf_filename : String with the filename to be used.

    - groupname : String with the top-level HDF5 groupname. If None
    and only 1 group is present, it will be used, otherwise this
    argument is required.

    - mode : String, passed directly to h5py.File constructor.

    """
    VERSION = 1
    def __init__(self, hdf_filename: str, groupname: str, mode='r'):
        self.hdf_filename = hdf_filename
        self._file = h5py.File(self.hdf_filename, mode=mode)
        keys = self._file.keys()
        if groupname is None:
            # Try and automatically determine the group name (if there's only 1)
            if len(keys) == 1:
                self.groupname = list(keys)[0]
            else:
                # Multiple groupnames, so we can't decide
                msg = "Cannot determine best groupname please choose from {}"
                raise exceptions.GroupKeyError(msg.format(list(keys)))
        else:
            self.groupname = groupname
        self.mode = mode

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._file.close()

    def group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        return self._file[self.groupname]

    def replace_dataset(self, name, data, attrs={}, *args, **kwargs):
        """Wrapper for h5py.create_dataset that removes the existing dataset
        if it exists.

        Arguments
        ---------

        - name : String with the HDF5 name to give this dataset.

        - data : Numpy array of data to be saved.

        - attrs : Dictionary containing HDF5 metadata attributes to be
          set on the resulting dataset.

        - *args, **kwargs : Will get passed to h5py's `create_dataset`
           method.
        """
        # Remove the existing dataset if possible
        try:
            del self.group()[name]
        except KeyError:
            pass
        # Perform the actual group creation
        ds = self.group().create_dataset(name=name, data=data, *args, **kwargs)
        # Set metadata attributes
        for key, val in attrs.items():
            ds.attrs[key] = val

    @property
    def pixel_sizes(self):
        raise NotImplementedError

    @pixel_sizes.setter
    def pixel_sizes(self, val):
        self.replace_dataset('pixel_sizes', val)

    @property
    def positions(self):
        raise NotImplementedError

    @positions.setter
    def positions(self, val):
        self.replace_dataset('positions', val)

    @property
    def intensities(self):
        return self.group()['intensities']

    @intensities.setter
    def intensities(self, val):
        self.replace_dataset('intensities', val)

    @property
    def references(self):
        return self.group()['references']

    @references.setter
    def references(self, val):
        self.replace_dataset('references', val)

    @property
    def absorbances(self):
        return self.group()['absorbances']

    @absorbances.setter
    def absorbances(self, val):
        self.replace_dataset('absorbances', val)

    @property
    def pixel_sizes(self):
        return self.group()['pixel_sizes']

    @pixel_sizes.setter
    def pixel_sizes(self, val):
        self.replace_dataset('pixel_sizes', val)

    @property
    def energies(self):
        return self.group()['energies']

    @energies.setter
    def energies(self, val):
        self.replace_dataset('energies', val)

    @property
    def timestamps(self):
        return self.group()['timestamps']

    @timestamps.setter
    def timestamps(self, val):
        # S32 is the 32-character ACSII string type for numpy
        val = np.array(val, dtype="S32")
        self.replace_dataset('timestamps', val, dtype="S32")

    @property
    def positions(self):
        return self.group()['positions']

    @positions.setter
    def positions(self, val):
        self.replace_dataset('positions', val)

    @property
    def filenames(self):
        return self.group()['filenames']

    @filenames.setter
    def filenames(self, val):
        # S100 is the 100-character ACSII string type for numpy
        val = np.array(val, dtype="S100")
        self.replace_dataset('filenames', val, dtype="S100")
