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


class TXMStore():
    """Wrapper around HDF5 file that stores TXM data."""
    VERSION = 1
    def __init__(self, hdf_filename: str, groupname: str, mode='r'):
        self.hdf_filename = hdf_filename
        self.groupname = groupname
        self.mode = mode
        self._file = h5py.File(self.hdf_filename, mode=mode)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._file.close()

    def replace_dataset(self, name, data, *args, **kwargs):
        """Wrapper for h5py.create_dataset that removes the existing dataset
        if it exists."""
        # Remove the existing dataset if possible
        try:
            del self.group()[name]
        except KeyError:
            pass
        # Perform the actual group creation
        self.group().create_dataset(name=name, data=data, *args, **kwargs)
