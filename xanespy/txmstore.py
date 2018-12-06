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

import logging
from collections import namedtuple

import h5py
import numpy as np
from tqdm import tqdm

from exceptions import (GroupKeyError, CreateGroupError, FrameSourceError,)
import xanes_math as xm
from utilities import get_component


log = logging.getLogger(__name__)


def merge_stores(base_store, new_store, destination, energy_difference=0.25, upsample=True):
    """Merge two open txm stores into a third store.
    
    Framesets will be combined from both ``base_store`` and
    ``new_store``. If frames in both sets are within
    ``energy_difference`` or each other, then the one from
    ``new_store`` will be used. The resulting frames will be cropped
    and up-sampled. Maps will not be copied, since they are unlikely
    to be reliable with the merged framesets. The metadata will
    reflect the merging as best as possible.
    
    """
    num_timesteps = len(base_store.timestep_names)
    M = namedtuple('M', ('energy', 'idx', 'store'))
    # Create some arrays to hold the results
    energies = []
    intensities = []
    optical_depths = []
    filenames = []
    pixel_sizes = []
    for t_idx in range(num_timesteps):
        # Prepare a "master list" of frames to used
        num_energies_A = len(base_store.energies[t_idx])
        energies_A = base_store.energies[t_idx]
        master_list = [M(energies_A[E_idx], E_idx, base_store)
                       for E_idx in range(num_energies_A)]
        # Update the master list for each frame in the new store
        energies_B = new_store.energies[t_idx]
        for E_idx, E in enumerate(energies_B):
            # Check if this entry exists in the master list already
            matches = [m for m in master_list
                       if abs(m.energy - E) < energy_difference]
            if matches:
                # It already exists, so replace it
                for m in matches:
                    idx = master_list.index(m)
                    master_list[idx] = M(E, E_idx, new_store)
            else:
                # It doesn't exist, so add it
                master_list.append(M(E, E_idx, new_store))
        # Sort the master list to be in energy ascending order
        master_list.sort(key=lambda m: m.energy)
        # Empty array for catching processed data
        Es = []
        Is = []
        ODs = []
        fnames = []
        # Prepare the arguments for resizing each image
        shapes = [m.store.intensities.shape[2:] for m in master_list]
        max_shape = (max(s[0] for s in shapes),
                     max(s[1] for s in shapes))
        dims = [np.array(m.store.intensities.shape[2:]) * m.store.pixel_sizes[t_idx,E_idx]
                for m in master_list]
        min_dims = (min(d[0] for d in dims), min(d[1] for d in dims))
        target_px_size = min(m.store.pixel_sizes[t_idx,E_idx] for m in master_list)
        pixel_sizes.append((target_px_size,) * len(master_list))
        # Retrieve and resize each image
        for m, dim in tqdm(zip(master_list, dims), total=len(master_list)):
            Es.append(m.energy)
            px_size = m.store.pixel_sizes[t_idx]
            I = m.store.intensities[t_idx,m.idx]
            if np.iscomplexobj(I):
                comp = 'imag'
            else:
                comp = 'real'
            I = get_component(I, comp)
            I = xm.resample_image(I, new_shape=max_shape, src_dims=dim, new_dims=min_dims)
            Is.append(I)
            OD = m.store.optical_depths[t_idx,m.idx]
            OD = get_component(OD, comp)
            OD = xm.resample_image(OD, new_shape=max_shape, src_dims=dim, new_dims=min_dims)
            ODs.append(OD)
            # Save the necessary metadata
            fnames.append(m.store.filenames[t_idx][m.idx])
        # Save the combined framesets
        energies.append(Es)
        intensities.append(Is)
        optical_depths.append(ODs)
        filenames.append(fnames)
    # Set the newly merged frames
    destination.energies = energies
    destination.intensities = np.array(intensities)
    destination.optical_depths = np.array(optical_depths)
    destination.filenames = filenames
    destination.timestep_names = base_store.timestep_names
    destination.pixel_sizes = pixel_sizes
    destination.pixel_unit = base_store.pixel_unit


class TXMDataset():
    """Data descriptor for accessing HDF datasets.
    
    Parameters
    ----------
    name : str
      The dataset name in the HDF file.
    context : str, optional
      Type of dataset this is: frameset, map, metadata, etc.
    dtype : np.dtype, optional
      The data-type to use when saving new data to disk. Using lower
      precision datatypes can save significant disk space.
    
    """
    def __init__(self, name, context=None, dtype=None):
        self.name = name
        self.context = context
        self.dtype = dtype
    
    def __get__(self, store, type=None):
        dataset = store.get_dataset(self.name)
        return dataset
    
    def __set__(self, store, value):
        store.replace_dataset(name=self.name, data=value,
                              context=self.context, dtype=self.dtype)
    
    def __delete__(self, store):
        del store.data_group()[self.name]


class TXMStore():
    """Wrapper around HDF5 file that stores TXM data.
    
    It has a series of descriptors and properties that return the
    corresponding HDF5 dataset object; the TXMStore().attribute.value
    pattern can be used to get pure numpy arrays directly. These
    objects should be used as a context manager to ensure that the
    file is closed, especially if using a writing mode:
    
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
    
    # HDF5 Descriptors
    # ----------------
    energies = TXMDataset('energies', context='metadata')
    signals = TXMDataset('extracted_signals', context='metadata')
    original_positions = TXMDataset('original_positions', context='metadata')
    timestep_names = TXMDataset('timestep_names', context='metadata')
    pixel_sizes = TXMDataset('pixel_sizes', context='metadata')
    linear_combination_sources = TXMDataset('linear_combination_sources', context='metadata')
    
    intensities = TXMDataset('intensities', context='frameset')
    optical_depths = TXMDataset('optical_depths', context='frameset')
    references = TXMDataset('references', context='frameset')
    signal_weights = TXMDataset('extracted_signal_weights', context='frameset')
    linear_combination_parameters = TXMDataset('linear_combination_parameters', context='frameset')
    
    optical_depth_mean = TXMDataset('optical_depth_mean', context='map')
    intensity_mean = TXMDataset('intensity_mean', context='map')
    signal_map = TXMDataset('signal_map', context='map')
    edge_mask = TXMDataset('edge_mask', context='map')
    whiteline_max = TXMDataset('whiteline_max', context='map')
    whiteline_fit = TXMDataset('whiteline_fit', context='map')
    cluster_fit = TXMDataset('cluster_fit', context='map')
    particle_labels = TXMDataset('particle_labels', context='map')
    segments = TXMDataset('segments', context='map')
    linear_combination_residuals = TXMDataset('linear_combination_residuals', context='map')
    
    def __init__(self, hdf_filename: str,
                 parent_name: str, data_name=None,
                 mode='r'):
        self.hdf_filename = hdf_filename
        self._file = self.open_file(self.hdf_filename, mode=mode)
        self.parent_name = parent_name
        self.mode = mode
        # Use the latest_data_name if one isn't provided
        if data_name is None:
            self.data_name = self.latest_data_name
        else:
            self.data_name = data_name
    
    def __str__(self):
        return self.parent_name + '-' + self.data_name

    def __repr__(self):
        fmt = '<TXMStore: {}/{}/{}>'
        fmt = fmt.format(self.hdf_filename, self.parent_name, self.data_name)
        return fmt
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()
    
    def frameset_names(self):
        """Returns a list of all the valid frameset representations."""
        names = []
        for key in self.data_group().keys():
            # Check if it's a "frameset" or not
            grp = self.data_group()
            context = grp[key].attrs.get('context', '')
            if context == 'frameset':
                names.append(key)
        return names        
    
    def map_names(self):
        """Returns a list of all the valid map representations."""
        names = []
        for key in self.data_group().keys():
            # Check if it's a "map" or not
            if self.data_group()[key].attrs.get('context','') == 'map':
                names.append(key)
        return names
    
    def open_file(self, filename, mode):
        return h5py.File(filename, mode=mode)
    
    def close(self):
        self._file.close()
    
    def data_tree(self):
        """Create a tree of the possible groups this store could access. The
        first level is samples, then data_groups (ie. same sample but
        different analysis status), then representations.
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
            raise CreateGroupError(msg)
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
    
    def validate_parent_group(self, name):
        """Retrieve the real parent group name for a possible parent_group.
        
        If ``name`` is None and only one group exists in the file,
        then that group name will be returned. If ``name`` is in the
        file, then ``name`` will be returned. If ``name`` is not in
        the file, a GroupKeyError will be raised.
        """
        if name is None and len(self._file.keys()) == 1:
            new_name = list(self._file.keys())[0]
        elif name not in self._file.keys():
            raise GroupKeyError("Cannot load parent group '{group}'. "
                                "Valid choices are {choices}."
                                "".format(group=name, choices=list(self._file.keys())))
        else:
            new_name = name
        return new_name
    
    def parent_group(self):
        """Retrieve the top-level HDF5 group object for this file and
        groupname."""
        try:
            parent_group = self.validate_parent_group(self.parent_name)
            group = self._file[parent_group]
        except (TypeError, KeyError):
            # Invalid group name, throw an exception
            msg = 'Cannot load parent group "{group}". Valid choices are {choices}'
            try:
                choices = list(self._file.keys())
            except:
                choices = 'unavailable'
            msg = msg.format(group=self.parent_name, choices=choices)
            raise GroupKeyError(msg) from None
        return group
    
    def data_group(self):
        """Retrieve the currently active second-level HDF5 group object for
        this file and groupname. Ex. "imported" or "aligned_frames".
        """
        if self.data_name not in self.parent_group().keys():
            msg = "Group {} does not exists. Run TXMStore.fork_data_group('{}') first"
            raise CreateGroupError(msg.format(self.data_name, self.data_name))
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
        compression : str, optional
          What type of compression to use. See HDF5 documentation for
          options.
        *args
          Arguments to pass to h5py's ``create_dataset`` method.
        **kwargs
          Keyword arguments to pass to h5py's ``create_dataset`` method.
        
        """
        # Remove the existing dataset if possible
        try:
            attrs = self.data_group()[name].attrs
            del self.data_group()[name]
        except KeyError as e:
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
        return ds
    
    def get_dataset(self, name):
        """Attempt to open the requested dataset.
        
        Parameters
        ==========
        name : str
          The name of the dataset to open in the data group.
        
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
            raise GroupKeyError(msg)
        elif name not in self.data_group().keys():
            msg = "dataset '{}' not found in group '{}' file '{}'"
            msg = msg.format(name, self.data_group().name, self.hdf_filename)
            raise GroupKeyError(msg)
        else:
            data = self.data_group()[name]
        return data
    
    def has_dataset(self, name):
        """Return a boolean indicated whether this dataset exists in the HDF
        file.
        
        """
        try:
            result = name in self.data_group().keys()
        except TypeError:
            result = False
        return result
    
    def frame_source(self, name):
        """Get the name of the frames that went into creating a map."""
        attrs = getattr(self.get_dataset(name), 'attrs', {})
        source = attrs.get('frame_source', name)
        return source
    
    def get_frames(self, name):
        """Return the source frame data for the given data name.
        
        This is similar to ``get_dataset`` except that if the
        data are a map, then get the frames that went into making it.
        
        Parameters
        ----------
        name : str
          The dataset name for which to retrieve frames.
        
        Returns
        -------
        dataset : h5py.Dataset
          The requested frameset. If the dataset is actually a map, as
          determined by the "context" attribute, then the related
          frame source attribute will be retrieved.
        
        """
        dataset = self.get_dataset(name)
        # If it's a map, then return the source frames instead
        if dataset.attrs.get('context', None) == 'map':
            try:
                dataset = self.get_dataset(dataset.attrs['frame_source'])
            except KeyError:
                source_desc = dataset.attrs.get('frame_source', 'None')
                # raise FrameSourceError(
                log.warning(
                    "Invalid frame source {} specified for group {}"
                    "".format(source_desc, self.data_name))
                dataset = self.get_dataset('optical_depths')
        return dataset
    
    @property
    def relative_positions(self):
    
        """(x, y, z) position values for each frame."""
        return self.data_group()['relative_positions']
    
    @relative_positions.setter
    def relative_positions(self, val):
        self.replace_dataset('relative_positions', val, context='metadata')
        self.data_group()['relative_positions'].attrs['order'] = "(x, y, z)"
   
    @property
    def pixel_unit(self):
        return self.data_group()['pixel_sizes'].attrs['unit']
    
    @pixel_unit.setter
    def pixel_unit(self, val):
        self.data_group()['pixel_sizes'].attrs['unit'] = val
    
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
        return self.get_dataset('fit_parameters')
    
    @fit_parameters.setter
    def fit_parameters(self, val):
        attrs = {
            'parameters': str(xm.kedge_params),
        }
        return self.replace_dataset('fit_parameters', val,
                                    attrs=attrs, context="metadata",
                                    dtype=np.float64)
