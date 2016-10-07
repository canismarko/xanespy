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
from skimage.measure import regionprops

import exceptions
from utilities import prog

import xanes_math as xm


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

    - parent_name : String with the top-level HDF5 groupname.

    - data_name : String wit the second level HDF5 groupname, used for
      specific data iterations (eg. imported, aligned)

    - mode : String, passed directly to h5py.File constructor.

    """
    VERSION = 1
    def __init__(self, hdf_filename: str,
                 parent_name: str, data_name: str,
                 mode='r'):
        self.hdf_filename = hdf_filename
        self._file = h5py.File(self.hdf_filename, mode=mode)
        self.parent_name = parent_name
        self._data_name = data_name
        self.mode = mode

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._file.close()

    @property
    def data_name(self):
        return self._data_name

    @data_name.setter
    def data_name(self, val):
        if val not in self.parent_group().keys():
            msg = "Group {} does not exists. Run TXMStore.fork_data_group('{}') first"
            raise exceptions.CreateGroupError(msg.format(val, val))
        self._data_name = val

    def data_tree(self):
        """Create a tree of the possible groups this store could access. The
        first level is samples, then data_groups (ie. same sample but
        different analysis status), then representations. Maps are not
        included in this tree.
        """
        # Define a recursive function to walk the groups in the file
        def walk_groups(parent, level):
            # Recurse into children
            keys = getattr(parent, 'keys', lambda: [])
            datas = []
            for key in keys():
                # Check for whether this object is a frameset or a map
                node = parent[key]
                grpdata = {
                    'name': key,
                    'path': node.name,
                    'context': node.attrs.get('context', None),
                    'ndim': getattr(node, 'ndim', 0),
                    'level': level,
                    # Resurse
                    'children': walk_groups(parent=node, level=level+1),
                }
                datas.append(grpdata)
            return datas
        # Start the recursion at the top
        tree = walk_groups(self._file, level=0)
        return tree

    def fork_data_group(self, new_name):
        """Turn on different active data group for this store. This method
        deletes the existing group and copies symlinks from the
        current one.
        """
        # Check that the current and target groups are not the same
        if new_name == self.data_name:
            msg = "Refusing to fork myself to myself"
            raise exceptions.CreateGroupError(msg)
        # Delete the old group and overwrite it
        parent = self.parent_group()
        if new_name in parent.keys():
            del parent[new_name]
        # Copy the old one with symlinks
        new_group = parent.copy(self.data_group(), new_name, shallow=True)
        # target_group = parent[new_name]
            #         old_gruop = parent_group[self.data_group]
            #         for key in old_group.keys():
            #             target_group[key] = old_group[key]
            #     # Save the new reference to the active group
            #     self.parent_group().attrs['latest_data_group'] = target_group.name
        self.latest_data_name = new_name
        self.data_name = new_name
        return new_group

    @property
    def latest_data_name(self):
        name = self.parent_group().attrs['latest_data_name']
        return name

    @latest_data_name.setter
    def latest_data_name(self, val):
        self.parent_group().attrs['latest_data_name'] = val

    # @latest_data_name.setter
    # def latest_data_name(self, val):
    #     parent_group = self.parent_group()
    #     if val in parent_group.keys():
    #         # Group already exists, so just save the new reference
    #         target_group = parent_group[val]
    #     else:
    #         # Group does not exist, so copy the old one with symlinks
    #         target_group = parent_group.create_group(val)
    #         old_group = parent_group[self.data_group]
    #         for key in old_group.keys():
    #             target_group[key] = old_group[key]
    #     # Save the new reference to the active group
    #     self.parent_group().attrs['latest_data_group'] = target_group.name

    def parent_group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        return self._file[self.parent_name]

    def data_group(self):
        """Retrieve the currently active second-level HDF5 group object for
        this file and groupname. Ex. "imported" or "aligned_frames".
        """
        return self.parent_group()[self.data_name]

    def group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        raise UserWarning('Using data_group() or parent_group() instead')

    def replace_dataset(self, name, data, context=None, attrs={},
                        compression=None, *args, **kwargs):
        """Wrapper for h5py.create_dataset that removes the existing dataset
        if it exists.

        Arguments
        ---------

        - name : String with the HDF5 name to give this dataset.

        - data : Numpy array of data to be saved.

        - context : A string specifying what kind of data is
          stored. Eg. "frameset", "metadata", "map".

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
        if context is not None:
            ds.attrs['context'] = context
        for key, val in attrs.items():
            ds.attrs[key] = val

    def get_frames(self, name):
        """Get a set of frames, specified by the value of `name`."""
        return self.data_group()[name]

    def set_frames(self, name, val):
        """Set data for a set of frames, specificied by the value of `name`."""
        return self.replace_dataset(name, val, context='frameset')

    def get_map(self, name):
        """Get a map of the frames, specified by the value of `name`."""
        if name in self.data_group().keys():
            data = self.data_group()[name]
        else:
            msg = "dataset '{}' not found in file '{}'"
            msg = msg.format(name, self.hdf_filename)
            raise exceptions.GroupKeyError(msg)
        return data

    def has_dataset(self, name):
        """Return a boolean indicated whether this dataset exists in the HDF
        file."""
        return (name in self.data_group().keys())

    @property
    def timestep_names(self):
        return self.data_group()['timestep_names']

    @timestep_names.setter
    def timestep_names(self, val):
        self.replace_dataset('timestep_names', val, context='metadata')

    @property
    def pixel_sizes(self):
        return self.data_group()['pixel_sizes']

    @pixel_sizes.setter
    def pixel_sizes(self, val):
        self.replace_dataset('pixel_sizes', val, context='metadata')

    @property
    def relative_positions(self):
        """(x, y, z) position values for each frame."""
        return self.data_group()['relative_positions']

    @relative_positions.setter
    def relative_positions(self, val):
        self.replace_dataset('relative_positions', val, context='metadata')
        self.data_group()['relative_positions'].attrs['order'] = "(x, y, z)"

    @property
    def original_positions(self):
        return self.get_map('original_positions')

    @original_positions.setter
    def original_positions(self, val):
        self.replace_dataset('original_positions', val, context='metadata')

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
        self.replace_dataset('intensities', val, context='frameset')

    @property
    def references(self):
        return self.data_group()['references']

    @references.setter
    def references(self, val):
        self.replace_dataset('references', val, context='frameset')

    @property
    def absorbances(self):
        return self.get_frames('absorbances')

    @absorbances.setter
    def absorbances(self, val):
        self.replace_dataset('absorbances', val, context="frameset")

    @property
    def energies(self):
        return self.data_group()['energies']

    @energies.setter
    def energies(self, val):
        self.replace_dataset('energies', val, context='metadata')

    @property
    def timestamps(self):
        return self.data_group()['timestamps']

    @timestamps.setter
    def timestamps(self, val):
        # S32 is the 32-character ACSII string type for numpy
        val = np.array(val, dtype="S32")
        self.replace_dataset('timestamps', val, dtype="S32", context='metadata')

    @property
    def filenames(self):
        return self.data_group()['filenames']

    @filenames.setter
    def filenames(self, val):
        # S100 is the 100-character ACSII string type for numpy
        val = np.array(val, dtype="S100")
        self.replace_dataset('filenames', val, dtype="S100", context='metadata')

    @property
    def fit_parameters(self):
        return self.get_map('fit_parameters')

    @fit_parameters.setter
    def fit_parameters(self, val):
        attrs = {
            'parameters': str(xm.kedge_params),
        }
        return self.replace_dataset('fit_parameters', val,
                                    attrs=attrs, context="metadata",
                                    dtype=np.float64)

    @property
    def whiteline_map(self):
        return self.get_map('whiteline_map')

    @whiteline_map.setter
    def whiteline_map(self, val):
        self.replace_dataset('whiteline_map', val, context='map')

    @property
    def particle_labels(self):
        return self.get_map('particle_labels')

    @particle_labels.setter
    def particle_labels(self, val):
        self.replace_dataset('particle_labels', val, context='map')


def prepare_txm_store(filename: str, parent_name: str, data_name='imported', dirname: str=None):
    """Check the filenames and create an hdf file as needed. Will
    overwrite the group if it already exists.

    Returns: An opened TXMStore object ready to accept data (r+ mode).

    Arguments
    ---------

    - filename : name of the requested hdf file, may be None if not
      provided, in which case the filename will be generated
      automatically based on `dirname`.

    - parent_name : Requested groupname for this sample.

    - data_name : Requested name for this data iteration.

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
    if parent_name is None:
        groupname = os.path.split(os.path.abspath(dirname))[1]
    # Open actual file
    hdf_file = h5py.File(hdf_filename, mode='a')
    # Delete the group if it already exists
    if parent_name in hdf_file.keys():
        del hdf_file[parent_name]
    new_group = hdf_file.create_group("{}/{}".format(parent_name, data_name))
    # Prepare a new TXMStore object to accept data
    store = TXMStore(hdf_filename=hdf_filename,
                     parent_name=parent_name,
                     data_name=data_name,
                     mode="r+")
    store.latest_data_name = 'imported'
    return store
