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

import os
import h5py
import re
from time import time
from collections import namedtuple
import warnings
import logging
import datetime as dt
import contextlib
from functools import partial

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.constants import physical_constants
from scipy.ndimage.filters import median_filter
from scipy.ndimage import center_of_mass
import pytz

from xradia import XRMFile, TXRMFile
from nanosurveyor import CXIFile, HDRFile
from sxstm import SxstmDataFile
from utilities import prog, get_component
import exceptions
from xanes_math import (transform_images, apply_references,
                        transformation_matrices, downsample_array,
                        apply_internal_reference, crop_image)
from txmstore import TXMStore, merge_stores


format_classes = {
    '.xrm': XRMFile
}


CURRENT_VERSION = "0.3" # Lets file loaders deal with changes to storage

log = logging.getLogger(__name__)


def import_aps4idc_sxstm_files(filenames, hdf_filename, hdf_groupname,
                               shape, energies, flux_correction=True):
    """Import scanning X-ray tunneling microscopy absorbance frames.
    
    These frames are STM images from APS 4-ID-C with incident X-rays
    at increasing energies, providing pixel-resolved spectral data. If
    the files are all in one directory with no other data, then the
    ``filenames`` argument can be the directory name, otherwise it
    should be a list of the filenames to import. It is assumed that
    the fast axis is energy, then the two spatial axes.
    
    Parameters
    ----------
    filenames : list, str
      List of filenames to import relative to working
      directory. Alternately, it can be a string with a directory path
      and all files will be imported.
    hdf_filename : str
      Path to a filename to use for saving imported data. If it
      doesn't exist, it will be created.
    hdf_groupname : str
      HDF groupname to use for saving data. If it exists, it will be
      overwritten.
    shape : 2-tuple
      Shape for the resulting maps.
    energies : iterable
      Incident beam energies for each frame, in electron-volts.
    flux_correction : bool, optional
      If true, certain channels will be corrected to account for
      changing beam flux.
    
    """
    # Convert to a list of filenames if it's a directory
    try:
        is_directory = os.path.isdir(filenames)
    except TypeError:
        is_directory = False
    if is_directory:
        directory = filenames
        filenames = os.listdir(filenames)
    else:
        directory = os.getcwd()
    # Process the metadata from filenames
    metadata = []
    file_re = re.compile('([a-zA-Z0-9_]+)_(\d+)_(\d+).3ds')
    for f in filenames:
        # Extract metadata from filenames
        file_match = file_re.match(os.path.basename(f))
        if file_match is None:
            warnings.warn('Cannot process filename {}.'.format(os.path.basename(f)))
            continue
        # Filename matches so continue the importing
        scan_name, pos_idx, E_idx = file_match.groups()
        E_idx = int(E_idx) - 1 # Convert from 1-index to 0-index
        pos_idx = int(pos_idx) - 1
        # Convert to actual 2D indices
        pos_idx = np.unravel_index(int(pos_idx), shape)
        # Add to the dataframe
        metadata.append([os.path.join(directory, f), scan_name, pos_idx, E_idx])
    metadata = pd.DataFrame(data=metadata,
                            columns=['filename', 'scan_name', 'pos_idx', 'E_idx'])
    # Create a new HDF file
    with h5py.File(hdf_filename, mode='a') as f:
        # Create the new parent HDF group (delete if it exists)
        if hdf_groupname in f.keys():
            del f[hdf_groupname]
            log.warn('Overwriting existing group "%s."' % hdf_groupname)
        parent_group = f.create_group(hdf_groupname)
        data_group = parent_group.create_group('imported')
        parent_group.attrs['latest_data_name'] = 'imported'
        # Set some metadata for the experiment
        new_attrs = {
            'technique': 'Synchrotron X-ray Scanning Tunneling Microscopy',
            'xanespy_version': CURRENT_VERSION,
            'beamline': 'APS 4-ID-C',
            'original_directory': os.path.abspath(directory),
        }
        parent_group.attrs.update(new_attrs)
        # Calculate the desired shape
        num_Es = len(metadata.E_idx.unique())
        full_shape = (1, num_Es, *shape)
        # Create empty datasets to hold the data
        ds_names = {
            'Bias calc (V)': 'bias_calc',
            'Current (A)': 'current',
            'LIA Tip Ch1 (V)': 'LIA_tip_ch1',
            'LIA Tip Ch2 (V)': 'LIA_tip_ch2',
            'LIA Sample Ch1 (V)': 'LIA_sample',
            'LIA Gold Shielding Ch1 (V)': 'LIA_shielding',
            'LIA topo Ch1 (V)': 'LIA_topo',
            'Gold Shielding (V)': 'shielding',
            'Flux (V)': 'flux',
            'Bias (V)': 'bias',
            'Z (m)': 'height',
        }
        for ds_name in ds_names.values():
            ds = data_group.create_dataset(ds_name, shape=full_shape,
                                           dtype='float32', compression='gzip')
            ds.attrs['context'] = 'frameset'
        # Load each source file one at a time
        Xs, Ys = [], []
        for row in prog(metadata.itertuples(), desc='Importing', total=len(filenames)):
            with SxstmDataFile(row.filename) as sxstm_file:
                df = sxstm_file.dataframe()
            medians = df.median()
            Xs.append(medians['X (m)'] / 1e-6)
            Ys.append(medians['Y (m)'] / 1e-6)
            full_idx = (0,row.E_idx,*row.pos_idx)
            # Import each data column to the HDF5 file
            for old_name, new_name in ds_names.items():
                if new_name == 'height':
                    data = medians[old_name] / 1e-9
                else:
                    data = medians[old_name]
                data_group[new_name][full_idx] = data
        # Determine the pixel sizes
        px_size_X = (np.max(Xs) - np.min(Xs)) / (shape[1] - 1)
        px_size_Y = (np.max(Ys) - np.min(Ys)) / (shape[0] - 1)
        # 1D line-scans produce NaN values
        if np.isnan(px_size_X) or np.isinf(px_size_X):
            px_size_X = px_size_Y
        elif np.isnan(px_size_Y) or np.isinf(px_size_Y):
            px_size_Y = px_size_X
        elif px_size_X != px_size_Y:
            warnings.warn("X and Y pixel sizes do not match (%f vs %f). "
                          "X axis values will be unreliable."
                          "" % (px_size_X, px_size_Y))
        px_ds = data_group.create_dataset('pixel_sizes', shape=(1, num_Es),
                                       dtype='float32')
        px_ds[:,:] = px_size_Y
        px_ds.attrs['unit'] = 'µm'
        px_ds.attrs['context'] = 'metadata'
        # Save the beam energy list
        Es = np.reshape(energies, (1, len(energies)))
        E_ds = data_group.create_dataset('energies', data=Es)
        E_ds.attrs['context'] = 'metadata'
        # Save an array for filenames
        filenames = np.array(filenames, dtype="S100")
        filenames = np.reshape(filenames, (*shape, num_Es))
        filenames = np.moveaxis(filenames, -1, 0)
        filename_ds = data_group.create_dataset('filenames', data=filenames)
        filename_ds.attrs['context'] = 'metadata'
        # Save a time step name
        ts_names = np.array(['ex-situ'], dtype="S30")
        ts_names = data_group.create_dataset('timestep_names', data=ts_names)
        ts_names.attrs['context'] = 'metadata'
        # Correct some channels for flux changes
        if flux_correction:
            log.debug("Fixing flux")
            flux = data_group['flux'].value
            def fix_flux(ds):
                ds.write_direct(ds.value / flux)
            fix_flux(data_group['LIA_topo'])
            fix_flux(data_group['LIA_tip_ch1'])
            fix_flux(data_group['LIA_tip_ch2'])
            fix_flux(data_group['LIA_sample'])


def _average_frames(*frames):
    """Accept several Xradia frames and return the first frame with new
    image data. Assumes metadata from first frame in list.
    
    """
    raise UserWarning("Just use numpy.mean instead.")
    new_image = np.zeros_like(frames[0].image_data(), dtype=np.float)
    # Sum all images
    for frame in frames:
        new_image += frame.image_data()
    # Divide to get average
    new_image = new_image / len(frames)
    # Return average data as a txm frame
    new_frame = frames[0]
    new_frame.image_data = new_image
    return new_frame


def import_aps32idc_xanes_files(filenames, hdf_filename, hdf_groupname, *args, **kwargs):
    """Import XANES data from a HDF5 file produced at APS beamline 32-ID-C.
    
    This is used for importing a full operando experiment at once.
    
    Parameters
    ----------
    filenames : str
      List of paths to the HDF5 files containing the source data.
    hdf_filename : str
      The path to the HDF5 file that will receive the imported
      data. Will be created if it doesn't exist.
    hdf_groupname : str, optional
      A description of the dataset that will be used to form the HDF
      data group.
    args, kwargs
      Passed to ``import_aps32idc_xanes_file``
    
    """
    append = False
    for idx, filename in enumerate(filenames):
        import_aps32idc_xanes_file(filename, hdf_filename, hdf_groupname,
                                   timestep=idx, total_timesteps=len(filenames),
                                   append=append, *args, **kwargs)
        append = True


def import_aps32idc_xanes_file(filename, hdf_filename, hdf_groupname,
                               timestep=0, total_timesteps=1,
                               append=False, downsample=1,
                               square=True, exclude=[],
                               median_filter_size=(1, 3, 3),
                               dark_idx=slice(None)):
    """Import XANES data from a HDF5 file produced at APS beamline 32-ID-C.
    
    This is used for importing a single XANES dataset.
    
    Parameters
    ----------
    filename : str
      The path to the HDF5 file containing the source data.
    hdf_filename : str
      The path to the HDF5 file that will receive the imported
      data. Will be created if it doesn't exist.
    hdf_groupname : str, optional
      A description of the dataset that will be used to form the HDF
      data group.
    timestep : int, optional
      Which timestep index to use for saving data.
    total_timesteps : int, optional
      How many timesteps to use for creating new datasteps. Only
      meaningful if ``append`` is truthy.
    append : bool, optional
      If true, existing datasets will be saved and only the timestep
      will be overwritten.
    downsample : int, optional
      Improves signal-to-noise at the expense of spatial
      resolution. Applied to intensities, flat-field, etc before
      converting to optical_depth.
    square : bool, optional
      If true (default), the edges will be cut to make a square
      array. Eg (2048, 2448) becomes (2048, 2048).
    exclude : iterable, optional
      Indices of frames to exclude from importing if, for example, the
      frame contains artifacts or is otherwise problematic.
    median_filter_size : float or tuple
      If not None, apply a median rank filter to each flat and data
      frame. The value of this parameters matches the ``size``
      parameter to :py:func:`scipy.ndimage.filters.median_filter`, for
      example using ``(1, 3, 3)`` will filter only along the *x* and
      *y* axes, and not the energy axis. Median filtering takes places
      after downsampling.
    dark_idx : slice, optional
      A slice object (or an index) for which dark-field images to
      use. Must be the same for all timesteps in an experiment. Useful
      if some dark field images are not usable.
    
    """
    # Open the source HDF file
    src_file = h5py.File(filename, mode='r')
    # Prepare the destination HDF file and create datasets if needed
    log.info("Importing APS 32-ID-C file %s", filename)
    with h5py.File(hdf_filename, mode='a') as h5file:
        # Prepare an HDF5 group with metadata for this experiment
        if append:
            parent_group = h5file[hdf_groupname]
            data_group = parent_group['imported']
        else:
            # Check for existing HDF5 group
            if hdf_groupname in h5file.keys():
                log.warning("Overwriting old HDF group {}".format(hdf_groupname))
                del h5file[hdf_groupname]
            # Create new group
            parent_group = h5file.create_group(hdf_groupname)
            metadata = {
                'technique': 'Full-field TXM',
                'xanespy_version': CURRENT_VERSION,
                'beamline': 'APS 32-ID-C',
                'original_directory': os.path.dirname(filename),
            }
            parent_group.attrs.update(metadata)
            # Prepare an HDF5 sub-group for this dataset
            data_group = parent_group.create_group('imported')
            parent_group.attrs['latest_data_name'] = 'imported'
        # Check if there's any value to exclude
        frm_idx = [i for i in range(src_file['/exchange/data'].shape[0]) if i not in exclude]
        kw = dict(factor=downsample, axis=(1, 2))
        src_data = downsample_array(src_file['/exchange/data'][frm_idx], **kw)
        src_flat = downsample_array(src_file['/exchange/data_white'][frm_idx], **kw)
        src_dark = downsample_array(src_file['/exchange/data_dark'][dark_idx], **kw)
        # Cut off extra edges if requested
        if square:
            im_shape = src_data.shape[1:]
            delta = int(min(im_shape) / 2)
            slices = tuple(slice(int(s/2-delta), int(s/2+delta)) for s in im_shape)
            slices = (slice(None),) + slices
            src_data = src_data[slices]
            src_flat = src_flat[slices]
            src_dark = src_dark[slices]
        shape = (total_timesteps, *src_data.shape)
        time_idx = timestep
        data_group.require_dataset('intensities', shape=shape, dtype=src_data.dtype)
        data_group['intensities'].attrs['context'] = 'frameset'
        data_group.require_dataset('flat_fields', shape=shape, dtype=src_flat.dtype)
        data_group['flat_fields'].attrs['context'] = 'frameset'
        dark_shape = (total_timesteps, *src_dark.shape)
        data_group.require_dataset('dark_fields', shape=dark_shape, dtype=src_dark.dtype)
        data_group['dark_fields'].attrs['context'] = 'frameset'
        data_group.require_dataset('optical_depths', shape=shape, dtype='float32')
        data_group['optical_depths'].attrs['context'] = 'frameset'
        # Create datasets for metadata
        pixels_shape = (total_timesteps, src_data.shape[0])
        data_group.require_dataset('pixel_sizes', shape=pixels_shape, dtype='float32')
        data_group['pixel_sizes'][time_idx] = 0.02999 * (2**downsample)
        data_group['pixel_sizes'].attrs['unit'] = 'µm'
        data_group.require_dataset('energies', shape=shape[0:2], dtype='float32')
        timestamp_shape = (*shape[0:2], 2)
        data_group.require_dataset('timestamps', shape=timestamp_shape, dtype="S32")
        data_group.require_dataset('filenames', shape=shape[0:2], dtype="S100")
        data_group.require_dataset('timestep_names', shape=(shape[0],), dtype="S100")
        # Import start and end dates
        fmt = "%Y-%m-%dT%H:%M:%S%z"
        end_str = bytes(src_file['/process/acquisition/end_date'][0][:24])
        end_dt = dt.datetime.strptime(end_str.decode('ascii'), fmt)
        end_dt = end_dt.astimezone(pytz.utc)
        start_str = bytes(src_file['/process/acquisition/start_date'][0][:24])
        start_dt = dt.datetime.strptime(start_str.decode('ascii'), fmt)
        start_dt = start_dt.astimezone(pytz.utc)
        out_fmt = "%Y-%m-%d %H:%M:%S"
        start_bytes = bytes(start_dt.strftime(out_fmt), encoding='ascii')
        data_group['timestamps'][time_idx,:,0] = start_bytes
        end_bytes = bytes(end_dt.strftime(out_fmt), encoding='ascii')
        data_group['timestamps'][time_idx,:,1] = end_bytes
        soc_bytes = bytes("soc{:03d}".format(time_idx), encoding='ascii')
        data_group['timestep_names'][time_idx] = soc_bytes
        # Import actual datasets
        data_group['intensities'][time_idx] = src_data
        data_group['flat_fields'][time_idx] = src_flat
        data_group['dark_fields'][time_idx] = src_dark
        energies = 1000 * src_file['/exchange/energy'][frm_idx]
        data_group['energies'][time_idx] = energies
        data_group['filenames'][time_idx,:] = filename.encode('ascii')
        # Apply median filter if requested
        if median_filter_size is not None:
            src_data = median_filter(src_data, size=median_filter_size)
            src_flat = median_filter(src_flat, size=median_filter_size)
        # Convert the intensity data to optical depth
        # keys_ = ('intensities', 'flat_fields', 'dark_fields')
        # Is, flat, dark = [data_group[key][time_idx] for key in keys_]
        dark = np.median(src_dark, axis=0)
        Is = (src_data) / (src_flat)
        Is[Is<=0] = 1e-6
        ODs = -np.log(Is)
        # Check for nan values
        if np.any(np.isnan(ODs)):
            log.warn('nan values found after OD conversion')
        # Save optical depth data to disk
        data_group['optical_depths'][time_idx] = ODs


def read_metadata(filenames, flavor, quiet=False):
    """Take a list of filenames and return a pandas dataframe with all the
    metadata.
    
    Arguments
    ---------
    filenames : iterable
      Iterable of filenames to use for extracting metadata.
    flavor : str
      Same as in ``import_frameset``.
    quiet : bool, optional
      Whether to suppress the progress bar, etc.
    """
    log.info("Importing metadata with flavor %s", flavor)
    logstart = time()
    columns = ('timestep_name', 'position_name', 'is_background',
               'energy', 'shape', 'starttime')
    df = pd.DataFrame(columns=columns)
    for filename in prog(filenames, desc="Reading", disable=quiet):
        log.debug("Reading metadata for {}".format(filename))
        ext = os.path.splitext(filename)[-1]
        if ext not in format_classes.keys():
            continue # Skip non-TXM files
        # Extract metadata from the filename
        if flavor == 'aps':
            metadata = decode_aps_params(filename)
        elif flavor == 'ssrl':
            metadata = decode_ssrl_params(filename)
        # Fetch metadata from the file itself
        with format_classes[ext](filename, flavor=flavor) as f:
            if f.is_valid():
                metadata['shape'] = f.image_shape()
                # Get time in utc
                metadata['starttime'] = f.starttime()
                df.loc[filename] = metadata
            else:
                # Invalid file data, so warn the user and keep going
                msg = "Ignoring invalid file {}".format(filename)
                warnings.warn(RuntimeWarning(msg))
                continue
        # try:
        #     with format_classes[ext](filename, flavor=flavor) as f:
        #         metadata['shape'] = f.image_shape()
        #         # Get time in utc
        #         metadata['starttime'] = f.starttime()
        # except exceptions.DataFormatError as e:
        #     # Invalid file data, so warn the user and keep going
        #     msg = "Ignoring invalid file {}".format(filename)
        #     warnings.warn(RuntimeWarning(msg))
        #     continue
    # Remove any incomplete framesets
    if len(df) > 0:
        timestep_names, timesteps = zip(*df.groupby('timestep_name'))
        lengths = np.array([len(df_.groupby('energy')) for df_ in timesteps])
        max_length = max(lengths)
        bad_samples = []
        for name, length in zip(timestep_names, lengths):
            if length < max_length:
                df = df[df.timestep_name != name]
                bad_samples.append(name)
        # Warn the user about dropped frames
        if bad_samples:
            msg = "Dropping incomplete framesets {}".format(bad_samples)
            log.info(msg)
        # Log summary of files read results
        msg = "Read metadata for %d files in %f sec"
        log.info(msg, len(df), time() - logstart)
    # Return file metadata in order of collection time
    return df.sort_values(by=['starttime'])


@contextlib.contextmanager
def open_files(paths, opener=open):
    """Context manager that opens files and closes them again.
    
    Example usage::
    
        file_paths = ('file_a.txt', 'file_b.txt')
        with open_files(file_paths) as files:
            for f in files:
                f.read()
    
    Parameters
    ==========
    paths :
      Iterable of file paths to open.
    opener : callable
      A class or function such that one would normally run::
        with opener(path) as f:
          ...do stuff with f...
    
    """
    files = tuple(opener(path) for path in paths)
    yield files
    # Close files after exiting context manager
    [f.close() for f in files]


def load_cosmic_files(files, store, median_filter_size=None):
    """Take a collection of STXM or ptycho files and load their data.
    
    Parameters
    ==========
    files : iterable
      A collection of open files (either CXIFile or HDRFile) that will
      be loaded and saved.
    store : TXMStore
      The TXMStore object that will receive the loaded data.
    median_filter_size : int or 3-tuple, optional
      Size of median filter to apply to image data. If tuple, should
      be in (energy, row, column) order. See
    ``scipy.ndimage.filters.median_filter`` for more details.
    
    """
    # Compile the filenames metadata
    filenames = []
    for f in files:
        filenames.extend(f.filenames())
    store.filenames = [filenames]
    store.energies = [np.array([f.energies() for f in files]).ravel()]
    store.timestep_names = np.array(['ex-situ'], dtype="S20")
    # Get pixel sizes (if possible)
    px_sizes = []
    for f in files:
        try:
            this_px_size = f.pixel_size()
        except exceptions.DataFormatError:
            warnings.warn('Cannot load pixel sizes from %s' % f)
            this_px_size = 6
        px_sizes.extend([this_px_size] * f.num_images())
    store.pixel_sizes = [px_sizes]
    store.pixel_unit = 'nm'
    # Load intensity data and calculate optical depths
    Is = [I for f in files for I in f.image_frames()]
    # Crop any over-sized frames and make sure they're all the same
    row_min = min(I.shape[0] for I in Is)
    row_max = max(I.shape[0] for I in Is)
    col_min = min(I.shape[1] for I in Is)
    col_max = max(I.shape[1] for I in Is)
    def do_crop(img, new_shape):
        # Convert to optical density
        OD = apply_internal_reference(img)
        if np.iscomplexobj(OD):
            OD = get_component(OD, 'imag')
        # Crop the image
        center = center_of_mass(OD)
        new_img = crop_image(img, center=center, shape=new_shape)
        return new_img
    Is = np.array(tuple(map(partial(do_crop, new_shape=(row_min, col_min)), Is)))
    # Apply median filter if requested
    if median_filter_size is not None:
        Is = median_filter(Is, size=median_filter_size)
    store.intensities = [Is]
    # Convert to optical depths
    ODs = apply_internal_reference(store.intensities)
    store.optical_depths = ODs


def import_cosmic_frameset(hdf_filename, stxm_hdr=(), ptycho_cxi=(),
                           hdf_groupname=None, energy_difference=0.25):
    """Import a combination of STXM and ptychography frames.
    
    Order is preserved, so later entries in ``stxm_hdr`` over-ride
    previous ones. Additionally, ptychography frames will be given
    precidence over stxm frames of similar energy (within +/-
    ``energy_difference`` eV). If both types of files are provided,
    both sets may potentially be scaled and/or interpolated for
    matching resolution.
    
    Parameters
    ==========
    hdf_filename : str
      Path to HDF5 file that will receive the data.
    stxm_hdr : iterable
      A list of hdr file paths to import from.
    ptycho_cxi : iterable
      A list of ptychography .cxi file paths to import from.
    hdf_groupname : str, optional
      Name of the HDF group to use. If omitted, this value will be
      guessed from the first file provided.
    energy_difference : float, optional
      When merging ptycho and stxm files, how close in energy (eV) two
      frames should be before they are considered at the same energy.

    """
    all_paths = ptycho_cxi + stxm_hdr
    has_ptycho = len(ptycho_cxi) > 0
    has_stxm = len(stxm_hdr) > 0
    # Check that at least some data are given
    if len(all_paths) == 0:
        raise ValueError("`stxm_hdr` and `ptycho_cxi` cannot both be empty")
    # Decide how to arrange the data groups
    if has_ptycho and has_stxm:
        ptycho_data_name = 'imported_ptychography'
        stxm_data_name = 'imported_stxm'
    else:
        ptycho_data_name = 'imported'
        stxm_data_name = 'imported'
    # Prepare the HDF5 file and sample group
    if hdf_groupname is None:
        hdf_groupname = os.path.splitext(os.path.basename(all_paths[0]))[0]
    log.info("Importing to file %s", hdf_filename)
    with h5py.File(hdf_filename) as h5file:
        # Delete old HDF5 group
        if hdf_groupname in h5file:
            log.warning('Overwriting existing group: "%s"', hdf_groupname)
            del h5file[hdf_groupname]
        parent_group = h5file.create_group(hdf_groupname)
        if has_stxm:
            new_grp = parent_group.create_group(stxm_data_name)
            log.debug("Created new STXM group: %s", new_grp.name)
        if has_ptycho:
            new_grp = parent_group.create_group(ptycho_data_name)
            log.debug("Created new ptycho group: %s", new_grp.name)
        if has_stxm and has_ptycho:
            new_grp = parent_group.create_group('imported')
            log.debug("Created new merged group: imported")
    # Load the STXM frames    
    store_kw = dict(hdf_filename=hdf_filename, parent_name=hdf_groupname,
                    mode='a')
    if has_stxm:
        log.debug("Loading STXM frames for %s", stxm_hdr)
        stxm_store = TXMStore(**store_kw, data_name=stxm_data_name)
        with stxm_store, open_files(stxm_hdr, HDRFile) as stxm_files:
            # ((1, 3, 1) median filter gets rid of row artifacts)
            load_cosmic_files(files=stxm_files, store=stxm_store,
                              median_filter_size=(1, 3, 1))
            stxm_store.latest_data_name = stxm_data_name
            # 
    # Load the ptychography frames
    if has_ptycho:
        log.debug("Loading ptycho frames for %s", ptycho_cxi)
        ptycho_store = TXMStore(**store_kw, data_name=ptycho_data_name)
        with ptycho_store, open_files(ptycho_cxi, CXIFile) as ptycho_files:
            load_cosmic_files(files=ptycho_files, store=ptycho_store)
            ptycho_store.latest_data_name = ptycho_data_name
    # Merge the framesets together into a new frameset
    if has_ptycho and has_stxm:
        # Prepare the relevant data stores
        merged_store = TXMStore(**store_kw, data_name='imported')
        ro_store_kw = dict(**store_kw)
        ro_store_kw['mode'] = 'r'
        stxm_store = TXMStore(**ro_store_kw, data_name=stxm_data_name)
        ptycho_store = TXMStore(**ro_store_kw, data_name=ptycho_data_name)
        # Do the merging
        merge_stores(base_store=stxm_store, new_store=ptycho_store,
                     destination=merged_store,
                     energy_difference=energy_difference)


def import_nanosurveyor_frameset(directory: str, quiet=False,
                                 hdf_filename=None, hdf_groupname=None,
                                 energy_range=None, exclude_re=None, append=False,
                                 frame_shape=None):
    """Import a set of images from reconstructed ptychography scanning microscope data.
    
    This generates ptychography chemical maps based on data collected
    at ALS beamline 5.3.2.1. The arguments ``energy_range`` and
    ``exclude_re`` can be used to fine-tune the set of imported
    file. For example: passing ``exclude_re='(019|017)'`` will import
    everything except scans 019 and 017.
    
    Parameters
    ----------
    directory : str
      Directory where to look for results. It should contain .cxi
      files that are the output of the ptychography reconstruction."
    quiet : Bool, optional
      If truthy, progress bars will not be shown.
    hdf_filename : str, optional
      HDF File used to store computed results. If omitted or None, the
      `directory` basename is used
    hdf_groupname : str, optional
      Name to use for the hdf group of this dataset. If omitted or
      None, the `directory` basename is used. Raises an exception if
      the group already exists in the HDF file.
    energy_range : 2-tuple, optional
      A 2-tuple with the (min, max) energy to be imported. This is
      useful if only a subset of the available data is usable. Values
      are assumed to be in electron-volts.
    exclude_re : str, optional
      Any filenames matching this regular expression will not be
      imported. A string or compiled re object can be given.
    append : bool, optional
      If True, any existing dataset will be added to, rather
      than replaced (default False)
    frame_shape : 2-tuple, optional
      If given, images will be trimmed to this shape. Must be smaller
      than the smallest frame. This may be useful if the frames are
      slightly different shapes. Does not apply to STXM images.
    
    """
    # Prepare logging info
    logstart = time()
    # Check if exclude_re is a string or regex object.
    if exclude_re is not None and not hasattr(exclude_re, 'search'):
        exclude_re = re.compile(exclude_re)
    # Prepare the HDF5 file and sample group
    log.info("Importing ptychography directory %s", directory)
    h5file = h5py.File(hdf_filename)
    # Get a default groupname if none is given
    path, sam_name = os.path.split(os.path.abspath(directory))
    if hdf_groupname is None:
        hdf_groupname = sam_name
    # Create the group if necessary
    if hdf_groupname in h5file.keys() and not append:
        msg = 'Overwriting existing group "{}"'.format(hdf_groupname)
        log.warning(msg)
        del h5file[hdf_groupname]
    sam_group = h5file.require_group(hdf_groupname)
    log.info("Created HDF group %s", sam_group.name)
    # Set some metadata
    sam_group.attrs["xanespy_version"] = CURRENT_VERSION
    sam_group.attrs["technique"] = "ptychography STXM"
    sam_group.attrs["beamline"] = "ALS 5.3.2.1"
    sam_group.attrs["original_directory"] = os.path.abspath(directory)
    # Prepare groups for data
    imported = sam_group.require_group('imported')
    sam_group.attrs['latest_data_name'] = 'imported'
    # Check that the directory exists
    if not os.path.exists(directory):
        msg = "Could not find directory {}".format(directory)
        raise exceptions.DataNotFoundError(msg)
    # Look in each directory for cxi files
    cxifiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cxi"):
                cxifiles.append(os.path.join(root, file))
    # Check that we actually found some data
    if len(cxifiles) == 0:
        msg = "{} contained no cxi files to import."
        msg = msg.format(directory)
        raise exceptions.DataNotFoundError(msg)
    # Import any cxi files that were found
    intensities = []
    energies = []
    filenames = []
    stxm_frames = []
    pixel_sizes = []
    log.info("Importing %d .cxi files", len(cxifiles))
    for filename in cxifiles:
        # Skip this energy is its exclude by the regular expression
        if exclude_re is not None and exclude_re.search(filename):
            msg = 'Skipping {filename} (matches exclude_re: "{re}")'
            msg = msg.format(filename=filename, re=exclude_re.pattern)
            log.info(msg)
            continue
        # Open the hdf5 file and get the data
        with h5py.File(filename, mode='r') as f:
            # Extract energy in Joules and convert to eV
            energy = f['/entry_1/instrument_1/source_1/energy'].value
            energy = energy / physical_constants['electron volt'][0]
            # Skip this energy if it's outside the desired range
            is_in_range = (energy_range is None or
                           min(energy_range) <= energy <= max(energy_range))
            if not is_in_range:
                log.info('Skipping %s (%f eV is outside of range)',
                         filename, energy)
                continue
            log.debug("Importing %s -> %f eV", filename, energy)
            filenames.append(os.path.relpath(filename))
            energies.append(energy)
            # Import complex reconstructed image
            if frame_shape is not None:
                # User requested the frames be cropped
                data = f['/entry_1/image_1/data'][:frame_shape[0], :frame_shape[1]]
            else:
                # No cropping, import full frames
                data = f['/entry_1/image_1/data'].value
            intensities.append(data)
            # Import STXM interpretation
            stxm = f['entry_1/instrument_1/detector_1/STXM'].value
            stxm_frames.append(stxm)
            # Save pixel size
            px_size = float(f['/entry_1/process_1/Param/pixnm'].value)
            log.debug("Scan %s has pixel size %f", filename, px_size)
            pixel_sizes.append(px_size)
    # Check that we have actual data to import
    if len(intensities) == 0:
        msg = "No files in directory {} pass import filters. "
        msg += "Consider changing `exclude_re` or `energy_range` parameters."
        raise exceptions.DataNotFoundError(msg.format(directory))
    # Check that all the frames are the same shape
    shapes = [d.shape for d in intensities]
    if len(set(shapes)) > 1:
        msg = "Frames are different shapes: {}".format(shapes)
        raise exceptions.DataFormatError(msg)
    # Helper function to save image data to the HDF file
    def replace_ds(name, parent, *args, **kwargs):
        if name in parent.keys():
            # Delete previous dataset
            del parent[name]
        else:
            log.info("Creating new dataset %s from %s", name, directory)
        # Save new combined dataset
        new_ds = parent.create_dataset(name, *args, **kwargs)
        return new_ds
    # Load previous datasets
    if 'intensities' in imported.keys():
        log.info("Appending data from %s", directory)
        # Check for redundant energies
        old_energies = imported['energies'][0]
        energies = np.array(energies)
        overlap = np.in1d(energies.astype(old_energies.dtype), old_energies)
        if np.any(overlap):
            msg = "Imported redundant energies from {directory}: {energies}"
            msg = msg.format(directory=directory, energies=energies[overlap])
            log.warning(msg)
        # Combine new data with previously imported data
        intensities = np.concatenate([intensities, imported['intensities'][0]])
        stxm_frames = np.concatenate([stxm_frames, imported['stxm'][0]])
        energies = np.concatenate([energies, old_energies])
        pixel_sizes = np.concatenate([pixel_sizes, imported['pixel_sizes'][0]])
        filenames = np.concatenate([filenames, imported['filenames'][0]])
        del imported['relative_positions']
        del imported['original_positions']
    else:
        log.info("Creating datasets from %s", directory)
        imported.create_dataset('timestep_names',
                                data=np.array([sam_name], dtype="S50"))
    # Sort all the datasets by energy
    sort_idx = np.argsort(energies)
    energies = np.array(energies)[sort_idx]
    intensities = np.array(intensities)[sort_idx]
    stxm_frames = np.array(stxm_frames)[sort_idx]
    pixel_sizes = np.array(pixel_sizes)[sort_idx]
    filenames = np.array(filenames)[sort_idx]
    # Save updated data to HDF file
    replace_ds('intensities', parent=imported, data=[intensities],
               dtype=np.complex64)
    replace_ds('stxm', parent=imported, data=[stxm_frames],
               dtype=np.float32)
    replace_ds('energies', parent=imported, data=[energies], dtype=np.float32)
    log.debug("Found energies %s", energies)
    # Save pixel size information
    px_grp = replace_ds('pixel_sizes', parent=imported, data=[pixel_sizes])
    px_unit = 'nm'
    px_grp.attrs['unit'] = px_unit
    # Save metadata
    filenames = np.array(filenames, dtype="S100")
    replace_ds('filenames', parent=imported, data=[filenames], dtype="S100")
    zero_positions = [np.zeros(shape=(*filenames.shape, 3), dtype=np.float32)]
    imported.create_dataset('relative_positions', data=zero_positions)
    imported['intensities'].attrs['context'] = 'frameset'
    nan_pos = np.empty(shape=(1, *filenames.shape, 3), dtype=np.float32)
    nan_pos.fill(np.nan)
    imported.create_dataset('original_positions', data=nan_pos)
    # Clean up any open files, etc
    h5file.close()
    log.info("Importing finished in %f seconds", time() - logstart)


def import_stxm_frameset(directory: str, quiet=False,
                         hdf_filename=None, hdf_groupname=None,
                         energy_range=None, exclude_re=None, append=False):
    """Import a set of images from scanning microscope data.

    This generates Scanning Tranmission X-ray Microscopy chemical maps
    based on data collected at ALS beamline 5.3.2.1

    Parameters
    ----------
    directory : str
      Directory where to look for results. It should contain .hdr and
      xim files that are the output of the ptychography
      reconstruction."
    quiet : Bool, optional
      If truthy, progress bars will not be shown.
    hdf_filename : str, optional
      HDF File used to store computed results. If omitted or None, the
      `directory` basename is used
    hdf_groupname : str, optional
      Name to use for the hdf group of this dataset. If omitted or
      None, the `directory` basename is used. Raises an exception if
      the group already exists in the HDF file.
    energy_range : 2-tuple, optional
      A 2-tuple with the (min, max) energy to be imported. This is
      useful if only a subset of the available data is usable. Values
      are assumed to be in electron-volts.
    exclude_re : str, optional
      Any filenames matching this regular expression will not be
      imported. A string or compiled re object can be given.
    append : bool, optional
      If True, any existing dataset will be added to, rather
      than replaced (default False)
    
    """
    # Prepare logging info
    logstart = time()
    # Check if exclude_re is a string or regex object.
    if exclude_re is not None and not hasattr(exclude_re, 'search'):
        exclude_re = re.compile(exclude_re)
    # Get a default groupname if none is given
    path, sam_name = os.path.split(os.path.abspath(directory))
    # Prepare the HDF5 file and sample group
    log.info("Importing STXM directory %s", directory)
    if hdf_filename is None:
        hdf_filename = sam_name + '.h5'
    h5file = h5py.File(hdf_filename)
    if hdf_groupname is None:
        hdf_groupname = sam_name
    # Create the group if necessary
    if hdf_groupname in h5file.keys() and not append:
        msg = 'Overwriting existing group "{}"'.format(hdf_groupname)
        log.warning(msg)
        del h5file[hdf_groupname]
    sam_group = h5file.require_group(hdf_groupname)
    log.info("Created HDF group %s", sam_group.name)
    # Set some metadata
    sam_group.attrs["xanespy_version"] = CURRENT_VERSION
    sam_group.attrs["technique"] = "STXM"
    sam_group.attrs["beamline"] = "ALS 5.3.2.1"
    sam_group.attrs["original_directory"] = os.path.abspath(directory)
    # Prepare groups for data
    imported = sam_group.require_group('imported')
    sam_group.attrs['latest_data_name'] = 'imported'
    # Check that the directory exists
    if not os.path.exists(directory):
        msg = "Could not find directory {}".format(directory)
        raise exceptions.DataNotFoundError(msg)
    # Look in each directory for cxi files
    ximfiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xim"):
                ximfiles.append(os.path.join(root, file))
    # Check that we actually found some data
    if len(ximfiles) == 0:
        msg = "{} contained no .xim files to import."
        msg = msg.format(directory)
        raise exceptions.DataNotFoundError(msg)
    # Import any xim files that were found
    intensities = []
    filenames = ximfiles
    pixel_sizes = []
    log.info("Importing %d .xim files", len(ximfiles))
    # Load data from header file
    hdrfile = os.path.join(directory, '{}.hdr'.format(sam_name))
    with open(hdrfile) as f:
        # Get rid of extra tabs and newlines
        lines = f.readlines()
        lines = [l.replace('\t', '').replace('\n', '') for l in lines]
        # Extract energies
        energy_re = "Points = \(\d+, ([0-9., ]+)\);"
        e_string = re.search(energy_re, lines[14]).groups()[0]
        energies = [float(s) for s in e_string.split(', ')]
        # Fake pixel sizes for now
        log.warning('Using unity pixel sizes')
        pixel_sizes = [1 for e in energies]
    # Open XIM files and load data
    for idx, filename in enumerate(ximfiles):
        # Skip this energy is its exclude by the regular expression
        if exclude_re is not None and exclude_re.search(filename):
            msg = 'Skipping {filename} (matches exclude_re: "{re}")'
            msg = msg.format(filename=filename, re=exclude_re.pattern)
            log.info(msg)
            continue
        # Load the XIM file
        data = np.loadtxt(filename, dtype='int')
        intensities.append(data)
        # Get the energy
        # Open the hdf5 file and get the data
        # with h5py.File(filename, mode='r') as f:
        #     # Extract energy in Joules and convert to eV
        #     energy = f['/entry_1/instrument_1/source_1/energy'].value
        #     energy = energy / physical_constants['electron volt'][0]
        #     # Skip this energy if it's outside the desired range
        #     is_in_range = (energy_range is None or
        #                    min(energy_range) <= energy <= max(energy_range))
        #     if not is_in_range:
        #         log.info('Skipping %s (%f eV is outside of range)',
        #                  filename, energy)
        #         continue
        #     log.debug("Importing %s -> %f eV", filename, energy)
        #     filenames.append(os.path.relpath(filename))
        #     energies.append(energy)
        #     # Import complex reconstructed image
        #     data = f['/entry_1/image_1/data'].value
        #     intensities.append(data)
        #     # Import STXM interpretation
        #     stxm = f['entry_1/instrument_1/detector_1/STXM'].value
        #     stxm_frames.append(stxm)
        #     # Save pixel size
        #     px_size = float(f['/entry_1/process_1/Param/pixnm'].value)
        #     log.debug("Scan %s has pixel size %f", filename, px_size)
        #     pixel_sizes.append(px_size)

    # Check that we have actual data to import
    if len(intensities) == 0:
        msg = "No files in directory {} pass import filters. "
        msg += "Consider changing `exclude_re` or `energy_range` parameters."
        raise exceptions.DataNotFoundError(msg.format(directory))
    # Helper function to save image data to the HDF file
    def replace_ds(name, parent, *args, **kwargs):
        if name in parent.keys():
            # Delete previous dataset
            del parent[name]
        else:
            log.info("Creating new dataset %s from %s", name, directory)
        # Save new combined dataset
        new_ds = parent.create_dataset(name, *args, **kwargs)
        return new_ds
    # Load previous datasets
    if 'intensities' in imported.keys():
        log.info("Appending data from %s", directory)
        # Check for redundant energies
        old_energies = imported['energies'][0]
        energies = np.array(energies)
        overlap = np.in1d(energies.astype(old_energies.dtype), old_energies)
        if np.any(overlap):
            msg = "Imported redundant energies from {directory}: {energies}"
            msg = msg.format(directory=directory, energies=energies[overlap])
            log.warning(msg)
        # Combine new data with previously imported data
        intensities = np.concatenate([intensities, imported['intensities'][0]])
        energies = np.concatenate([energies, old_energies])
        pixel_sizes = np.concatenate([pixel_sizes, imported['pixel_sizes'][0]])
        filenames = np.concatenate([filenames, imported['filenames'][0]])
        del imported['relative_positions']
        del imported['original_positions']
    else:
        log.info("Creating datasets from %s", directory)
        imported.create_dataset('timestep_names',
                                data=np.array([sam_name], dtype="S50"))
    # Sort all the datasets by energy
    sort_idx = np.argsort(energies)
    energies = np.array(energies)[sort_idx]
    intensities = np.array(intensities)[sort_idx]
    pixel_sizes = np.array(pixel_sizes)[sort_idx]
    filenames = np.array(filenames)[sort_idx]
    # Save updated data to HDF file
    replace_ds('intensities', parent=imported, data=[intensities],
               dtype=np.int)
    replace_ds('energies', parent=imported, data=[energies], dtype=np.float32)
    log.debug("Found energies %s", energies)
    # Save pixel size information
    px_grp = replace_ds('pixel_sizes', parent=imported, data=[pixel_sizes])
    px_unit = 'nm'
    px_grp.attrs['unit'] = px_unit
    # Save metadata
    filenames = np.array(filenames, dtype="S100")
    replace_ds('filenames', parent=imported, data=[filenames], dtype="S100")
    zero_positions = [np.zeros(shape=(*filenames.shape, 3), dtype=np.float32)]
    imported.create_dataset('relative_positions', data=zero_positions)
    imported['intensities'].attrs['context'] = 'frameset'
    nan_pos = np.empty(shape=(1, *filenames.shape, 3), dtype=np.float32)
    nan_pos.fill(np.nan)
    imported.create_dataset('original_positions', data=nan_pos)
    # Clean up any open files, etc
    h5file.close()
    log.info("Importing finished in %f seconds", time() - logstart)
    

_SsrlResponse = namedtuple("_SsrlResponse", ("data", "starttime", "endtime"))


def _average_ssrl_files(files):
    starttimes = []
    endtimes = []
    arrays = []
    # Average sample frames
    for filename in files:
        name, extension = os.path.splitext(filename)
        Importer = format_classes[extension]
        with Importer(filename=filename, flavor="ssrl") as txmfile:
            starttimes.append(txmfile.starttime())
            endtimes.append(txmfile.endtime())
            # Append to the running total array (create if first frame)
            arrays.append(txmfile.image_data())
    # Prepare the result tuple
    response = _SsrlResponse(
        data=np.mean(arrays, axis=0),
        starttime=min(starttimes),
        endtime=max(endtimes)
    )
    return response


def magnification_correction(frames, pixel_sizes):
    """Correct for changes in magnification at different energies.
    
    As the X-ray energy increases, the focal length of the zone plate
    changes and so the image is zoomed-out at higher energies. This
    method applies a correction to each frame to make the
    magnification similar to that of the first frame. Some beamlines
    correct for this automatically during acquisition and don't need
    this function: APS 8-BM-B, 32-ID-C.
    
    Arguments
    ---------
    frames : np.ndarray
      Numpy array of image frames that need to be corrected.
    pixel_sizes : np.ndarray
      Numpy array of pixel sizes corresponding to entries in `frames`.
    
    Returns
    -------
    (scales2D, translations) : (np.ndarray, np.ndarray)
      An array of scale factors to use for applying a correction to
      each frame. Translations show how much to move each frame array by to re-center it.
    
    """
    scales = np.min(pixel_sizes) / pixel_sizes
    datashape = frames.shape[:-2]
    imshape = np.array(frames.shape[-2:])
    scales2D = scales.reshape(*datashape, 1).repeat(2, axis=-1)
    translations = (imshape-1) * (1-scales2D) / 2
    return (scales2D, translations)


def import_ssrl_xanes_dir(directory, hdf_filename, groupname=None, *args, **kwargs):
    """Import all files in the given directory collected at SSRL beamline
    6-2c and process into framesets. Images are assumed to full-field
    transmission X-ray micrographs and repetitions will be
    averaged. Passed on to ``xanespy.importers.import_frameset``
    
    Arguments
    ---------
    directory : str
      Where to look for files to import.
    hdf_filename : str
      Path to an HDF5 to receive the data.
    groupname : str, optional
      HDF group name to use for saving these data. If omitted, try to
      guess from directory path.
    args, kwargs
      Arguments and keyword arguments passed on to
    ``import_frameset``.
    
    """
    imp_group = import_frameset(directory, hdf_filename=hdf_filename,
                                flavor='ssrl', return_val="group",
                                groupname=groupname, *args, **kwargs)
    # Set some beamline specific metadata
    imp_group.parent.attrs['technique'] = 'Full-field TXM'
    imp_group.parent.attrs['beamline'] = 'SSRL 6-2c'
    # Cleanup and exit
    imp_group.file.close()
    return imp_group


def decode_ssrl_params(filename):
    """Accept the filename of an XRM file and return sample parameters as
    a dictionary."""
    # Beamline 6-2c at SSRL
    ssrl_regex_bg = re.compile(
        'rep(\d{2})_(\d{6})_ref_[0-9]+_([-a-zA-Z0-9_]+)_([0-9.]+)_eV_(\d{3})of(\d{3})\.xrm'
    )
    ssrl_regex_sample = re.compile(
        'rep(\d{2})_[0-9]+_([-a-zA-Z0-9_]+)_([0-9.]+)_eV_(\d{3})of(\d{3}).xrm'
    )
    # Check for background frames
    bg_result = ssrl_regex_bg.search(filename)
    sample_result = ssrl_regex_sample.search(filename)
    if bg_result:
        params = {
            'timestep_name': "rep{}".format(bg_result.group(1)),
            'position_name': bg_result.group(3).strip("_"),
            'is_background': True,
            'energy': float(bg_result.group(4)),
        }
    elif sample_result:
        params = {
            'timestep_name': "rep{}".format(sample_result.group(1)),
            'position_name': sample_result.group(2).strip("_"),
            'is_background': False,
            'energy': float(sample_result.group(3)),
        }
    else:
        msg = "Could not parse filename {filename} using flavor {flavor}"
        msg = msg.format(filename=filename, flavor='ssrl')
        raise exceptions.FilenameParseError(msg)
    return params


def decode_aps_params(filename):
    """Accept the filename of an XRM file and return sample parameters as
    a dictionary."""
    regex = re.compile(
        '(?P<pos>[a-zA-Z0-9_]+)_xanes(?P<sam>[a-zA-Z0-9_]+)_(?P<E_int>[0-9]+)_(?P<E_dec>[0-9]+)eV.xrm'
    )
    match = regex.search(filename)
    if not match:
        msg = "{filename} does not match {regex}"
        raise RuntimeError(msg.format(regex=regex, filename=filename))
    match = match.groupdict()
    energy = float("{}.{}".format(match['E_int'], match['E_dec']))
    result = {
        'timestep_name': match['sam'],
        'position_name': match['pos'],
        'is_background': match['pos'] == 'ref',
        'energy': energy,
    }
    return result


def import_aps8bm_xanes_file(filename, ref_filename, hdf_filename,
                              groupname=None, quiet=False):
    """Extract an entire xanes framestack from one xradia file.
    
    A single TXRM file can contain multiple frames at different
    energies. This function will import such a file along with the
    corresponding reference frames into an HDF file. If the given
    ``groupname`` exists in ``hdf_filename``, it will be overwritten
    and a ``RuntimeWarning`` will be issued.
    
    Parameters
    ----------
    filename : str
      File path to the txrm file to import.
    ref_filename : str
      File path to the txrm file that contains the white-field
      reference images. This will be used to calculate optical depth
      from transmitted intensity.
    hdf_filename : str
      File path to the destination HDF5 file that will receive the
      imported data.
    groupname : str, optional
      The name for the top-level HDF group. If omitted, a group name
      will be generated from the ``filename`` parameter. '{}' can be
      included and will receive the position name using the
      ``format()`` method. The '{}' is required if more than one field
      of view exists and a groupname is given.
    quiet : bool, optional
      Whether to suppress the progress bar, etc.
    
    """
    log.debug("Starting import of APS 8-BM frameset.")
    if groupname is None:
        groupname = os.path.splitext(os.path.basename(filename))[0]
    with TXRMFile(filename, flavor='aps') as f:
        int_data = f.image_stack()
        energies = f.energies()
        sample_pos = f.sample_position()
        timestamps = f.starttimes()
    with TXRMFile(ref_filename, flavor='aps') as f:
        ref_data = f.image_stack()
    # Delete old data groups for this file
    h5file = h5py.File(hdf_filename, mode='a')
    if groupname in h5file.keys():
        log.warning("Overwriting old HDF group {}".format(groupname))
        del h5file[groupname]
    # Save data the HDF file
    sam_group = h5file.create_group(groupname)
    imp_group = sam_group.create_group('imported')
    ds_shape = (1, *int_data.shape)
    int_ds = imp_group.create_dataset('intensities', data=[int_data])
    int_ds.attrs['context'] = 'frameset'
    ref_ds = imp_group.create_dataset('references', data=[ref_data])
    ref_ds.attrs['context'] = 'frameset'
    E_ds = imp_group.create_dataset('energies', data=[energies])
    E_ds.attrs['context'] = 'metadata'
    # Prepare and save timestamps
    timestamps = np.array([timestamps], dtype='S32')
    timestamp_ds = imp_group.create_dataset('timestamps', data=timestamps)
    # Apply reference correction
    abs_ds = imp_group.create_dataset('optical_depths', shape=ds_shape,
                              dtype=np.float32, maxshape=ds_shape)
    abs_ds.attrs['context'] = 'frameset'
    apply_references(int_ds, ref_ds, out=abs_ds, quiet=quiet)
    # Set some other metadata
    h5file[groupname].attrs["original_file"] = filename
    h5file[groupname].attrs["xanespy_version"] = CURRENT_VERSION
    h5file[groupname].attrs['latest_data_name'] = 'imported'
    h5file[groupname].attrs['technique'] = 'Full-field TXM'
    h5file[groupname].attrs['beamline'] = 'APS 8-BM-B'
    imp_group.create_dataset('timestep_names', data=[0])
    pixel_size = 40/int_data.shape[0]
    pixel_sizes = np.full(ds_shape[0: 2], pixel_size)
    pixel_ds = imp_group.create_dataset('pixel_sizes', data=pixel_sizes)
    pixel_ds.attrs['unit'] = 'µm'
    # Save original filename
    filenames = np.array([[filename]], dtype="S100")
    imp_group.create_dataset('filenames', data=filenames)
    # Save original sample positions
    pos_shape = (*int_ds.shape[0:2], 3)
    original_pos = np.broadcast_to(sample_pos, pos_shape)
    imp_group.create_dataset('original_positions', data=original_pos)
    # Clean up and exit
    h5file.close()


def import_aps8bm_xanes_dir(directory, hdf_filename, groupname=None,
                             *args, **kwargs):
    imp_group = import_frameset(directory=directory, flavor="aps",
                                hdf_filename=hdf_filename, return_val="group",
                                groupname=groupname, *args, **kwargs)
    # Set some beamline specific metadata
    imp_group.parent.attrs['technique'] = 'Full-field TXM'
    imp_group.parent.attrs['beamline'] = 'APS 8-BM-B'
    # Cleanup and exit
    imp_group.file.close()
    return imp_group


def import_frameset(directory, flavor, hdf_filename, groupname=None, return_val=None,
                    quiet=False):
    """Import all files in the given directory collected at an X-ray
    microscope beamline.
    
    Images are assumed to full-field transmission X-ray micrographs.
    
    If ``return_val`` is "group", the return value for this function
    will be a data group in an **open** HDF5 file. The underlying file
    should be explicitly closed to avoid corruption. This is done
    automatically if ``return_val`` is None (default).
    
    Parameters
    ----------
    directory : str
      A valid path to a directory containing the frame data to import.
    flavor : str
      Indicates what type of naming conventions and data structure to
      assume. See documentation for ``xanespy.xradia.XRMFile`` for
      possible choice.
    hdf_filename : str
      Where to save the output to. Will overwrite previous data-sets
      with the same name.
    groupname : str, optional
      What to use as the name for HDF group storing the data. If
      omitted, guess the name from the directory path.  '{}' can be
      included and will receive the position name using the
      ``format()`` method. The '{}' is required if more than one field
      of view exists and a groupname is given.
    return_val : str, optional
      Request a specific return value.
      - None: No return value (default)
      - "group": The open HDF5 group for this experiment.
    quiet : bool, optional
      Whether to suppress the progress bar
    
    """
    # Check arguments for sanity
    valid_return_vals = ["group", None]
    if return_val not in valid_return_vals:
        msg = "Invalid `return_val`: {}. Choices are {}"
        raise ValueError(msg.format(return_val, valid_return_vals))
    # Get list of all possible files in the directory tree
    files = [os.path.join(dp, f) for dp, dn, filenames in
             os.walk(directory) for f in filenames]
    # Process filename metadata into separate dataframes
    metadata = read_metadata(files, flavor=flavor, quiet=quiet)
    reference_files = metadata[metadata['is_background'] == True]
    sample_files = metadata[metadata['is_background'] == False]
    total_files = sample_files.count()['is_background']
    # Make sure there are at least some sample data files to import
    if len(sample_files) == 0:
        msg = 'No data files found in directory "{}"'
        msg = msg.format(os.path.abspath(directory))
        raise exceptions.DataNotFoundError(msg)
    # Import each sample-position combination
    log.debug("Opening hdf file %s.", hdf_filename)
    h5file = h5py.File(hdf_filename)
    # Get some shape information for all the datasets
    shapes = metadata['shape'].unique()
    assert len(shapes) == 1
    num_samples = len(reference_files['timestep_name'].unique())
    num_positions = len(sample_files['position_name'].unique())
    pos_name = sample_files['position_name'].unique()[0]
    num_energies = len(reference_files['energy'].unique())
    ds_shape = (num_samples, num_energies, *shapes[0])
    chunk_shape = (1, 1, *shapes[0])
    # Sanity check that we have a usable groupname that is format()-able
    groupname_is_valid = (num_positions == 1) or (groupname is None) or ("{}" in groupname)
    if not groupname_is_valid:
        msg = 'Invalid groupname "{}" for {} position names.'
        msg = msg.format(groupname, num_positions)
        msg += ' Please include a "{}" to receive the position name.'
        raise exceptions.CreateGroupError(msg)
    # Import reference frames
    assert len(reference_files.groupby('position_name')) == 1
    ref_groups = reference_files.groupby('timestep_name')
    imp_name = pos_name if groupname is None else groupname
    imp_group = h5file.require_group("{}/imported".format(imp_name))
    ref_ds = imp_group.require_dataset('references', shape=ds_shape,
                                       maxshape=ds_shape,
                                       chunks=chunk_shape,
                                       dtype=np.uint16)
    progbar = prog(desc="Importing", total=len(metadata),
                   unit="files", disable=quiet)
    ref_ds.attrs['context'] = 'frameset'
    median_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for ts_idx, (ts_name, ts_df) in enumerate(ref_groups):
        Is = []
        for E_idx, (E_name, E_df) in enumerate(ts_df.groupby('energy')):
            Importer = format_classes[os.path.splitext(E_df.index[0])[-1]]
            images = []
            for f in E_df.index:
                with Importer(f, flavor=flavor) as E_file:
                    images.append(E_file.image_data())
                progbar.update(1)
            # Take the average of all the files and apply median filter
            images = np.mean(np.array(images), axis=0)
            Is.append(median_filter(images, footprint=median_kernel))
        # Save to disk
        ref_ds[ts_idx] = np.array(Is)
    pos_groups = enumerate(sample_files.groupby('position_name'))
    for pos_idx, (pos_name, pos_df) in pos_groups:
        imp_name = pos_name if groupname is None else groupname.format(pos_name)
        log.info('Saving imported data to group "%s"', imp_name)
        logstart = time()
        # Create HDF5 datasets to hold the data
        Importer = format_classes[os.path.splitext(pos_df.index[0])[-1]]
        with Importer(pos_df.index[0], flavor=flavor) as f:
            im_shape = f.image_data().shape
            num_samples = len(pos_df.groupby('timestep_name'))
            num_energies = len(pos_df.groupby('energy'))
        ds_shape = (num_samples, num_energies, *im_shape)
        h5group = h5file.require_group("{}/imported".format(imp_name))
        h5file["{}".format(imp_name)].attrs['latest_data_name'] = 'imported'
        int_ds = h5group.require_dataset('intensities',
                                         shape=ds_shape,
                                         maxshape=ds_shape,
                                         chunks=True,
                                         dtype=np.uint16)
        int_ds.attrs['context'] = 'frameset'
        abs_ds = h5group.require_dataset('optical_depths',
                                         shape=ds_shape,
                                         maxshape=ds_shape,
                                         dtype=np.float32)
        abs_ds.attrs['context'] = 'frameset'
        px_ds = h5group.require_dataset('pixel_sizes',
                                        shape=(num_samples, num_energies),
                                        dtype=np.float16)
        px_ds.attrs['unit'] = 'µm'
        px_ds.attrs['context'] = 'metadata'
        E_ds = h5group.require_dataset('energies',
                                       shape=(num_samples, num_energies),
                                       dtype=np.float32)
        E_ds.attrs['context'] = 'metadata'
        timestamp_ds = h5group.require_dataset('timestamps',
                                               shape=(num_samples, num_energies, 2),
                                               dtype="S32")
        timestamp_ds.attrs['context'] = 'metadata'
        timestamp_ds.attrs['timezone'] = 'UTC'
        filename_ds = h5group.require_dataset('filenames',
                                              shape=(num_samples, num_energies),
                                              dtype="S100",
                                              compression="gzip")
        filename_ds.attrs['context'] = 'metadata'
        orig_pos_ds = h5group.require_dataset('original_positions',
                                              shape=(num_samples, num_energies, 3),
                                              dtype=np.float32,
                                              compression="gzip")
        orig_pos_ds.attrs['context'] = 'metadata'
        rel_pos_ds = h5group.require_dataset('relative_positions',
                                              shape=(num_samples, num_energies, 3),
                                              dtype=np.float32)
        rel_pos_ds.attrs['context'] = 'metadata'
        dt = h5py.special_dtype(vlen=str)
        timestep_ds = h5group.require_dataset('timestep_names',
                                              shape=(num_samples,),
                                              dtype=dt)
        timestep_ds.attrs['context'] = 'metadata'
        timestep_groups = enumerate(pos_df.groupby('timestep_name'))
        for ts_idx, (ts_name, ts_df) in timestep_groups:
            # Save name to HDF file
            timestep_ds[ts_idx] = ts_name
            # Import the data from each energy frame...
            E_groups = ts_df.sort_values('energy').groupby('energy')
            for E_idx, (energy, group) in enumerate(E_groups):
                energies = []
                images = []
                pos = []
                starts = []
                ends = []
                px_sizes = []
                Importer = format_classes[os.path.splitext(group.index[0])[-1]]
                for f in group.index:
                    with Importer(f, flavor=flavor) as E_file:
                        energies.append(E_file.energy())
                        images.append(E_file.image_data())
                        pos.append(E_file.sample_position())
                        starts.append(E_file.starttime())
                        ends.append(E_file.endtime())
                        px_sizes.append(E_file.um_per_pixel())
                        progbar.update(1)
                # ...x-ray energies...
                E_ds[ts_idx, E_idx] = np.mean(np.array(energies))
                # ...intensity data...
                #   (Combine and store the image arrays with a median filter)
                images = np.mean(np.array(images), axis=0)
                image = median_filter(images, footprint=median_kernel)
                int_ds[ts_idx, E_idx] = image
                # ...position data...
                orig_pos_ds[ts_idx, E_idx] = np.mean(np.array(pos), axis=0)
                # ...filenames...
                filename_ds[ts_idx, E_idx] = np.array([group.index[0]],
                                                      dtype="S")
                # ...timestamp data...
                timestamp_ds[ts_idx, E_idx] = np.array([min(starts), max(ends)],
                                                       dtype='S32')
                # ...pixel size data.
                px_ds[ts_idx, E_idx] = np.mean(np.array(px_sizes))
        # Create a references dataset if one doesn't exist yet
        if 'references' not in h5group.keys():
            h5group['references'] = ref_ds
        progbar.close()
        # Convert to absorbance values
        abs_start = time()
        apply_references(int_ds, ref_ds, out=abs_ds, quiet=quiet)
        log.info("Applied reference correction in %f seconds.", time() - abs_start)
        # Correct magnification from different energy focusing
        if flavor == 'ssrl':
            # Correct magnification changes due to zone-plate movement
            scales, translations = magnification_correction(frames=abs_ds,
                                                            pixel_sizes=px_ds)
            tmatrices = transformation_matrices(scales=scales,
                                                translations=translations)
            transform_images(abs_ds, transformations=tmatrices,
                             out=abs_ds, quiet=quiet)
        # Remove dead or hot pixels
        progbar = prog(desc="Median filter", total=1, disable=quiet)
        progbar.update(0)
        median_filter(abs_ds, size=1, output=abs_ds)
        progbar.update(1)
        progbar.close()
        # Set some metadata
        h5group.parent.attrs['xanespy_version'] = CURRENT_VERSION
        h5group.parent.attrs['original_directory'] = directory
        # Write logging output
        log.info("Imported %s (%d files) in %f sec",
                 imp_name, len(pos_df), time() - logstart)
    # Clean up and sort out the return value
    if return_val == "group":
        ret = imp_group
    else:
        h5file.close()
        ret = None
    return ret
