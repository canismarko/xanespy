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

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.constants import physical_constants
from scipy.ndimage.filters import median_filter

from xradia import XRMFile
from utilities import prog
import exceptions
from xanes_math import transform_images, apply_references, transformation_matrices


format_classes = {
    '.xrm': XRMFile
}


CURRENT_VERSION = "0.3" # Let's file loaders deal with changes to storage

log = logging.getLogger(__name__)


def _average_frames(*frames):
    """Accept several frames and return the first frame with new image
    data. Assumes metadata from first frame in list."""
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


def read_metadata(filenames, flavor):
    """Take a list of filenames and return a pandas dataframe with all the
    metadata.
    
    Arguments
    ---------
    filenames : iterable
      Iterable of filenames to use for extracting metadata.
    flavor : str
      Same as in ``import_frameset``.
    
    """
    log.info("Importing metadata with flavor %s", flavor)
    logstart = time()
    columns = ('timestep_name', 'position_name', 'is_background',
               'energy', 'shape', 'starttime')
    df = pd.DataFrame(columns=columns)
    for filename in prog(filenames, desc="Reading"):
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
            metadata['shape'] = f.image_shape()
            # Get time in utc
            metadata['starttime'] = f.starttime()
        df.loc[filename] = metadata
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


def import_nanosurveyor_frameset(directory: str, quiet=False,
                                 hdf_filename=None, hdf_groupname=None,
                                 energy_range=None, exclude_re=None, append=False):
    """Import a set of images from reconstructed ptychography scanning microscope data.

    This generates ptychography chemical maps based on data collected at ALS beamline
    5.3.2.1

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
            data = f['/entry_1/image_1/data'].value
            intensities.append(data)
            # Import STXM interpretation
            stxm = f['entry_1/instrument_1/detector_1/STXM'].value
            stxm_frames.append(stxm)
            # Save pixel size
            px_size = float(f['/entry_1/process_1/Param/pixnm'].value)
            log.debug("Scan %s has pixel size %f", filename, px_size)
            pixel_sizes.append(px_size)
    
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


def import_ssrl_frameset(directory, hdf_filename):
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
    """
    imp_group = import_frameset(directory, hdf_filename=hdf_filename,
                                flavor='ssrl', return_val="group")
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
        '(?P<pos>[a-zA-Z0-9_]+)_xanes(?P<sam>[a-zA-Z0-9_]+)_(?P<E_int>[0-9]+)_(?P<E_dec>[0-9])eV.xrm'
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


def import_aps_8BM_frameset(directory, hdf_filename, quiet=False):
    imp_group = import_frameset(directory=directory, flavor="aps",
                                hdf_filename=hdf_filename, return_val="group")
    # Set some beamline specific metadata
    imp_group.parent.attrs['technique'] = 'Full-field TXM'
    imp_group.parent.attrs['beamline'] = 'APS 8-BM-B'
    # Cleanup and exit
    imp_group.file.close()
    return imp_group


def import_frameset(directory, flavor, hdf_filename, return_val=None):
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
      Where to save the output to. An exception is throw if this file
      already exists.
    return_val : str
      Request a specific return value.
      - None: No return value
      - "group": The open HDF5 group for this experiment.
    
    """
    # Check arguments for sanity
    valid_return_vals = ["group", None]
    if return_val not in valid_return_vals:
        raise ValueError("Invalid `return_val`: {}. Choices are {}".format(return_val, valid_return_vals))
    # Check that file does not exist
    if os.path.exists(hdf_filename):
        raise OSError("File {} exists".format(hdf_filename))
    # Get list of all possible files in the directory tree
    files = [os.path.join(dp, f) for dp, dn, filenames in
             os.walk(directory) for f in filenames]
    # Process filename metadata into separate dataframes
    metadata = read_metadata(files, flavor=flavor)
    reference_files = metadata[metadata['is_background'] == True]
    sample_files = metadata[metadata['is_background'] == False]
    total_files = sample_files.count()['is_background']
    # Make sure there are at least some sample data files to import
    if len(sample_files) == 0:
        msg = 'No data files found in directory "{}"'
        msg = msg.format(os.path.abspath(directory))
        raise exceptions.DataNotFoundError(msg)
    # Import each sample-position combination
    h5file = h5py.File(hdf_filename)
    # Get some shape information for all the datasets
    shapes = metadata['shape'].unique()
    assert len(shapes) == 1
    num_samples = len(reference_files['timestep_name'].unique())
    pos_name = sample_files['position_name'].unique()[0]
    num_energies = len(reference_files['energy'].unique())
    ds_shape = (num_samples, num_energies, *shapes[0])
    chunk_shape = (1, 1, *shapes[0])
    # Import reference frames
    assert len(reference_files.groupby('position_name')) == 1
    ref_groups = reference_files.groupby('timestep_name')
    imp_group = h5file.require_group("{}/imported".format(pos_name))
    ref_ds = imp_group.require_dataset('references', shape=ds_shape,
                                       maxshape=ds_shape,
                                       chunks=chunk_shape,
                                       dtype=np.uint16)
    progbar = prog(desc="Importing", total=len(metadata), unit="files")
    ref_ds.attrs['context'] = 'frameset'
    median_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for ts_idx, (ts_name, ts_df) in enumerate(ref_groups):
        Is = []
        for E_idx, (E_name, E_df) in enumerate(ts_df.groupby('energy')):
            Importer = format_classes[os.path.splitext(E_df.index[0])[-1]]
            # E_files = [Importer(f, flavor=flavor) for f in E_df.index]
            # images = np.array([f.image_data() for f in E_files])
            # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            # image = median_filter(np.mean(images, axis=0), footprint=kernel)
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
        logstart = time()
        # Create HDF5 datasets to hold the data
        Importer = format_classes[os.path.splitext(pos_df.index[0])[-1]]
        with Importer(pos_df.index[0], flavor=flavor) as f:
            im_shape = f.image_data().shape
            num_samples = len(pos_df.groupby('timestep_name'))
            num_energies = len(pos_df.groupby('energy'))
        ds_shape = (num_samples, num_energies, *im_shape)
        h5group = h5file.require_group("{}/imported".format(pos_name))
        h5file["{}".format(pos_name)].attrs['latest_data_name'] = 'imported'
        int_ds = h5group.require_dataset('intensities',
                                         shape=ds_shape,
                                         maxshape=ds_shape,
                                         chunks=True,
                                         dtype=np.uint16)
        int_ds.attrs['context'] = 'frameset'
        abs_ds = h5group.require_dataset('absorbances',
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
        apply_references(int_ds, ref_ds, out=abs_ds)
        # Correct magnification from different energy focusing
        if flavor == 'ssrl':
            # Correct magnification changes due to zone-plate movement
            scales, translations = magnification_correction(frames=abs_ds,
                                                            pixel_sizes=px_ds)
            tmatrices = transformation_matrices(scales=scales,
                                                translations=translations)
            transform_images(abs_ds, transformations=tmatrices,
                             out=abs_ds)
        # Remove dead or hot pixels
        progbar = prog(desc="Median filter", total=1)
        progbar.update(0)
        median_filter(abs_ds, size=1, output=abs_ds)
        progbar.update(1)
        progbar.close()
        # Set some metadata
        h5group.parent.attrs['xanespy_version'] = CURRENT_VERSION
        h5group.parent.attrs['original_directory'] = directory
        # Write logging output
        log.info("Imported %s (%d files) in %f sec",
                 pos_name, len(pos_df), time() - logstart)
    # Clean up and sort out the return value
    if return_val == "group":
        ret = imp_group
    else:
        h5file.close()
        ret = None
    return ret
