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

import warnings

import h5py
import numpy as np

import exceptions
from utilities import prog


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

    def fork_data_group(self, new_name):
        """Turn on different active data group for this store. This method
        deletes the existing group and copies symlinks from the
        current one.
        """
        # Check that the current and target groups are not the same
        if new_name == self.active_data_group:
            msg = "Refusing to fork myself to myself"
            raise exceptions.CreateGroupError(msg)
        else:
            if new_name in self.parent_group().keys():
                # Delete the old group and overwrite it
                del self.parent_group()[new_name]
            # Create the new group
            self.active_data_group = new_name

    @property
    def active_data_group(self):
        return self.parent_group().attrs['active_data_group']

    @active_data_group.setter
    def active_data_group(self, val):
        parent_group = self.parent_group()
        if val in parent_group.keys():
            # Group already exists, so just save the new reference
            target_group = parent_group[val]
        else:
            # Group does not exist, so copy the old one with symlinks
            target_group = parent_group.create_group(val)
            old_group = parent_group[self.active_data_group]
            for key in old_group.keys():
                target_group[key] = old_group[key]
        # Save the new reference to the active group
        parent_group.attrs['active_data_group'] = target_group.name

    def parent_group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        return self._file[self.groupname]

    def data_group(self):
        """Retrieve the currently active second-level HDF5 group object for
        this file and groupname. Ex. "imported" or "aligned_frames".
        """
        return self._file[self.groupname][self.active_data_group]

    def group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        raise UserWarning('Using data_group() instead')
        warnings.warn()
        return self.data_group()

    def replace_dataset(self, name, data, attrs={}, compression=None, *args, **kwargs):
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
            del self.data_group()[name]
        except KeyError:
            pass
        # Perform the actual group creation
        ds = self.data_group().create_dataset(name=name, data=data,
                                              compression=compression,
                                              *args, **kwargs)
        # Set metadata attributes
        for key, val in attrs.items():
            ds.attrs[key] = val

    @property
    def pixel_sizes(self):
        return self.data_group()['pixel_sizes']

    @pixel_sizes.setter
    def pixel_sizes(self, val):
        self.replace_dataset('pixel_sizes', val)

    @property
    def relative_positions(self):
        return self.data_group()['relative_positions']

    @relative_positions.setter
    def relative_positions(self, val):
        self.replace_dataset('relative_positions', val)

    @property
    def original_positions(self):
        return self.data_group()['original_positions']

    @original_positions.setter
    def original_positions(self, val):
        self.replace_dataset('original_positions', val)

    @property
    def pixel_unit(self):
        return self.data_group()['pixel_sizes'].attrs['unit']

    @pixel_unit.setter
    def pixel_unit(self, val):
        self.data_group()['pixel_sizes'].attrs['unit'] = val

    @property
    def intensities(self):
        return self.data_group()['intensities']

    @intensities.setter
    def intensities(self, val):
        self.replace_dataset('intensities', val)

    @property
    def references(self):
        return self.data_group()['references']

    @references.setter
    def references(self, val):
        self.replace_dataset('references', val)

    @property
    def absorbances(self):
        return self.data_group()['absorbances']

    @absorbances.setter
    def absorbances(self, val):
        self.replace_dataset('absorbances', val)

    @property
    def pixel_sizes(self):
        return self.data_group()['pixel_sizes']

    @pixel_sizes.setter
    def pixel_sizes(self, val):
        self.replace_dataset('pixel_sizes', val)

    @property
    def energies(self):
        return self.data_group()['energies']

    @energies.setter
    def energies(self, val):
        self.replace_dataset('energies', val)

    @property
    def timestamps(self):
        return self.data_group()['timestamps']

    @timestamps.setter
    def timestamps(self, val):
        # S32 is the 32-character ACSII string type for numpy
        val = np.array(val, dtype="S32")
        self.replace_dataset('timestamps', val, dtype="S32")

    @property
    def positions(self):
        return self.data_group()['positions']

    @positions.setter
    def positions(self, val):
        self.replace_dataset('positions', val)

    @property
    def filenames(self):
        return self.data_group()['filenames']

    @filenames.setter
    def filenames(self, val):
        # S100 is the 100-character ACSII string type for numpy
        val = np.array(val, dtype="S100")
        self.replace_dataset('filenames', val, dtype="S100")

    @property
    def whiteline_map(self):
        return self.data_group()['whiteline_map']

    @whiteline_map.setter
    def whiteline_map(self, val):
        self.replace_dataset('whiteline_map', val)


def prepare_txm_store(filename: str, groupname: str, dirname: str=None):
    """Check the filenames and create an hdf file as needed. Will
    overwrite the group if it already exists.

    Returns: An opened TXMStore object ready to accept data (r+ mode).

    Arguments
    ---------

    - filename : name of the requested hdf file, may be None if not
      provided, in which case the filename will be generated
      automatically based on `dirname`.

    - groupname : Requested groupname for these data.

    - dirname : Used to derive a default filename if None is passed
      for `filename` attribute.
    """
    # Get default filename and groupname if necessary
    if filename is None:
        real_name = os.path.abspath(dirname)
        new_filename = os.path.split(real_name)[1]
        hdf_filename = "{basename}-results.h5".format(basename=new_filename)
    else:
        hdf_filename = filename
    if groupname is None:
        groupname = os.path.split(os.path.abspath(dirname))[1]
    # Open actual file
    hdf_file = h5py.File(hdf_filename, mode='a')
    # Delete the group if it already exists
    if groupname in hdf_file.keys():
        del hdf_file[groupname]
    new_group = hdf_file.create_group("{}/imported".format(groupname))
    # Prepare a new TXMStore object to accept data
    store = TXMStore(hdf_filename=hdf_filename,groupname=groupname, mode="r+")
    store.active_data_group = 'imported'
    return store
