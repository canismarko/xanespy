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
import logging

import exceptions

import xanes_math as xm


log = logging.getLogger(__name__)


class TXMStore():
    """Wrapper around HDF5 file that stores TXM data. It has a series of
    properties that return the corresponding HDF5 dataset object; the
    TXMStore().attribute.value pattern can be used to get pure numpy
    arrays. These objects should be used as a context manager to ensure
    that the file is closed, especially if using a writing mode:

        with TXMStore() as store:
            # Do stuff with store here

    Parameters
    ----------
    hdf_filename : str
      Path to the HDF file to be used.
    parent_name : str
      Name of the top-level HDF5 group.
    data_name : str
      Name of the second level HDF5 group, used for specific data
      iterations (eg. imported, aligned)
    mode : str
      Eg. 'r' for read-only, 'r+' for read-write. Passed directly to
      h5py.File constructor.

    """
    VERSION = 1
    _data_name = None

    def __init__(self, hdf_filename: str,
                 parent_name: str, data_name=None,
                 mode='r'):
        self.hdf_filename = hdf_filename
        self._file = h5py.File(self.hdf_filename, mode=mode)
        self.parent_name = parent_name
        self.mode = mode
        # Use the latest_data_name if one isn't provided
        if data_name is None:
            data_name = self.latest_data_name
        self._data_name = data_name

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

    def fork_data_group(self, dest, src=None):
        """Turn on different active data group for this store. This method
        deletes the existing group and copies symlinks from the
        current one.

        """
        # Switch to the group given by `src`
        if src is not None:
            self.data_name = src
        # Check that the current and target groups are not the same
        if dest == self.data_name:
            log.critical('Refusing to fork group "%s" to itself', dest)
            msg = 'Refusing to fork myself to myself ({})'.format(dest)
            raise exceptions.CreateGroupError(msg)
        log.info('Forking data group "%s" to "%s"', src, dest)
        # Delete the old group and overwrite it
        parent = self.parent_group()
        if dest in parent.keys():
            del parent[dest]
        # Copy the old one with symlinks
        new_group = parent.copy(self.data_group(), dest, shallow=True)
        self.latest_data_name = dest
        self.data_name = dest
        return new_group

    @property
    def latest_data_name(self):
        name = self.parent_group().attrs['latest_data_name']
        return name

    @latest_data_name.setter
    def latest_data_name(self, val):
        self.parent_group().attrs['latest_data_name'] = val

    def parent_group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        return self._file[self.parent_name]

    def data_group(self):
        """Retrieve the currently active second-level HDF5 group object for
        this file and groupname. Ex. "imported" or "aligned_frames".
        """
        return self.parent_group()[self.data_name]

    def replace_dataset(self, name, data, context=None, attrs={},
                        compression=None, *args, **kwargs):
        """Wrapper for h5py.create_dataset that removes the existing dataset
        if it exists.

        Parameters
        ----------
        name : str
          HDF5 groupname name to give this dataset.
        data : np.ndarray
          Numpy array of data to be saved.
        context : str, optional
          Specifies what kind of data is stored. Eg. "frameset",
          "metadata", "map".
        attrs : dict, optional
          Dictionary containing HDF5 metadata attributes to be set on
          the resulting dataset.
        *args
          Arguments to pass to h5py's ``create_dataset`` method.
        **kwargs
          Keyword arguments to pass to h5py's ``create_dataset`` method.
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
        frames = self.get_dataset(name)
        # If it's a map, then return the source frames
        if frames.attrs['context'] == 'map':
            frames = self.get_dataset(frames.attrs['frame_source'])
        return frames

    def set_frames(self, name, val):
        """Set data for a set of frames, specificied by the value of `name`."""
        return self.replace_dataset(name, val, context='frameset')

    def get_map(self, name):
        """Get a map of the frames, specified by the value of `name`."""
        map_ds = self.get_dataset(name)
        # If it's not a map, throw an exception
        context = map_ds.attrs['context']
        if context != "map":
            msg = "Expected {name} to be a map, but was a {context}."
            msg = msg.format(name=name, context=context)
            raise exceptions.GroupKeyError(msg)
        # Return the data
        return map_ds

    def get_dataset(self, name):
        """Attempt to open the requested dataset.

        Returns
        -------
        data : hyp5.Dataset
          An open HDF5 dataset

        Raises
        ------
        exceptions.GroupKeyError
          If the dataset does not exist in the file.
        """
        # Check for some bad dataset names
        if name is None:
            msg = "dataset `None` not found in file '{}'"
            msg = msg.format(self.hdf_filename)
            raise exceptions.GroupKeyError(msg)
        elif name not in self.data_group().keys():
            msg = "dataset '{}' not found in file '{}'"
            msg = msg.format(name, self.hdf_filename)
            raise exceptions.GroupKeyError(msg)
        else:
            data = self.data_group()[name]
        return data

    def has_dataset(self, name):
        """Return a boolean indicated whether this dataset exists in the HDF
        file."""
        try:
            result = name in self.data_group().keys()
        except TypeError:
            result = False
        return result

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
        return self.get_dataset('original_positions')

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
    def signals(self):
        """Get the previously extracted signals using any one of a variety of
        decomposition methods, saved as `signal_method`.
        """
        return self.data_group()['extracted_signals']

    @signals.setter
    def signals(self, val):
        self.replace_dataset('extracted_signals', val, context="metadata")

    @property
    def signal_method(self):
        """String describing how the previously extracted signals were
        calculated.
        """
        return self.data_group()['extracted_signals'].attrs['method']

    @signal_method.setter
    def signal_method(self, val):
        self.data_group()['extracted_signals'].attrs['method'] = val

    @property
    def signal_weights(self):
        """Get the pixel weights of the previously extracted signals using any
        one of a variety of decomposition methods, saved as
        `signal_method`.
        """
        return self.data_group()['extracted_signal_weights']

    @signal_weights.setter
    def signal_weights(self, val):
        self.replace_dataset('extracted_signal_weights', val,
                             context="frameset")

    @property
    def signal_map(self):
        return self.get_map('signal_map')

    @signal_map.setter
    def signal_map(self, val):
        self.replace_dataset('signal_map', val, context='map')

    @property
    def timestamps(self):
        return self.data_group()['timestamps']

    @timestamps.setter
    def timestamps(self, val):
        # S32 is the 32-character ACSII string type for numpy
        val = np.array(val, dtype="S32")
        self.replace_dataset('timestamps', val, dtype="S32", context='metadata',
                             attrs={'timezone', "UTC"})

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
    def whiteline_max(self):
        return self.get_map('whiteline_max')

    @whiteline_max.setter
    def whiteline_max(self, val):
        self.replace_dataset('whiteline_max', val, context='map')

    @property
    def whiteline_fit(self):
        return self.get_map('whiteline_fit')

    @whiteline_fit.setter
    def whiteline_fit(self, val):
        self.replace_dataset('whiteline_fit', val, context='map')

    @property
    def cluster_map(self):
        return self.get_map('cluster_map')

    @cluster_map.setter
    def cluster_map(self, val):
        self.replace_dataset('cluster_map', val, context='map')

    @property
    def particle_labels(self):
        return self.get_map('particle_labels')

    @particle_labels.setter
    def particle_labels(self, val):
        self.replace_dataset('particle_labels', val, context='map')
