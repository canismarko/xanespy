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
from PIL import Image
from scipy.constants import physical_constants
from scipy.ndimage.filters import median_filter

from xradia import XRMFile
from xanes_frameset import XanesFrameset
from frame import remove_outliers
from txmstore import TXMStore, prepare_txm_store
from utilities import prog, prepare_hdf_group, parallel_map, foreach
import exceptions
from xanes_math import transform_images, apply_references


format_classes = {
    '.xrm': XRMFile
}


CURRENT_VERSION = "0.3" # Let's file loaders deal with changes to storage

logger = logging.getLogger(__name__)


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


def import_txm_framesets(*args, **kwargs):
    msg = "This function is ambiguous."
    msg += " Choose from the more specific importers."
    raise NotImplementedError(msg)


def read_metadata(filenames, flavor):
    """Take a list of filenames and return a pandas dataframe with all the
    metadata."""
    columns = ('timestep_name', 'position_name', 'is_background',
               'energy', 'shape', 'starttime')
    df = pd.DataFrame(columns=columns)
    for filename in prog(filenames, 'Preparing metadata'):
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
            # Get total seconds since unix epoch
            metadata['starttime'] = f.starttime().timestamp()
        df.loc[filename] = metadata
    # Remove any incomplete framesets
    timestep_names, timesteps = zip(*df.groupby('timestep_name'))
    lengths = np.array([len(df_.groupby('energy')) for df_ in timesteps])
    max_length = max(lengths)
    bad_samples = []
    for name, length in zip(timestep_names, lengths):
        if length < max_length:
            df = df[df.timestep_name != name]
            bad_samples.append(name)
    # Warn the user about dropped frames
    if bad_samples and not prog.quiet:
        msg = "Dropping incomplete framesets {}".format(bad_samples)
        warnings.warn(UserWarning(msg))
    return df.sort_values(by=['starttime'])


def import_ptychography_frameset(directory: str, quiet=False,
                                 hdf_filename=None, hdf_groupname=None):
    """Import a set of images as a new frameset for generating
    ptychography chemical maps based on data collected at ALS beamline
    5.3.2.1

    Arguments
    ---------

    - directory : Directory where to look for results. It should
    contain .cxi files that are the output of the ptychography reconstruction."

    - quiet : If truthy, progress bars will not be shown.

    - hdf_filename : HDF File used to store computed results. If
      omitted or None, the `directory` basename is used

    - hdf_groupname : String to use for the hdf group of this
    dataset. If omitted or None, the `directory` basename is
    used. Raises an exception if the group already exists in the HDF file.
    """
    # Prepare logging info
    logstart = time()
    # Prepare the HDF5 file and sample group
    logger.info("Importing ptychography directory %s", directory)
    h5file = h5py.File(hdf_filename)
    path, sam_name = os.path.split(os.path.abspath(directory))
    try:
        sam_group = h5file.create_group(sam_name)
    except ValueError:
        raise exceptions.DatasetExistsError(sam_name)
    logger.info("Created HDF group %s", sam_group.name)
    # Set some metadata
    sam_group.attrs["xanespy_version"] = CURRENT_VERSION
    sam_group.attrs["technique"] = "ptychography STXM"
    sam_group.attrs["beamline"] = "ALS 5.3.2.1"
    sam_group.attrs["original_directory"] = os.path.abspath(directory)
    # Prepare groups for data
    imported = sam_group.create_group('imported')
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
    timestamps = []
    filenames = []
    stxm_frames = []
    logger.info("Importing %d .cxi files", len(cxifiles))
    for filename in cxifiles:
        filenames.append(filename)
        with h5py.File(filename, mode='r') as f:
            # Extract energy in Joules and convert to eV
            energy = f['/entry_1/instrument_1/source_1/energy'].value
            energy = energy / physical_constants['electron volt'][0]
            energies.append(energy)
            # Import complex reconstructed image
            data = f['/entry_1/image_1/data'].value
            intensities.append(data)
            # Import STXM interpretation
            stxm = f['entry_1/instrument_1/detector_1/STXM'].value
            stxm_frames.append(stxm)
    # Save image data to the HDF file
    intensities = np.array([intensities])
    intensity_ds = imported.create_dataset('intensities',
                                           data=intensities,
                                           dtype=np.complex64)
    logger.info("Saving intensity data: %s", intensity_ds.name)
    stxm_frames = np.array([stxm_frames])
    imported.create_dataset('stxm', data=stxm_frames)
    # Save X-ray energies to the HDF File
    energies = np.array([energies], dtype=np.float32)
    imported.create_dataset('energies', data=energies)
    logger.debug("Found energies %s", energies)
    # Save pixel size information
    px_sizes = np.empty(shape=intensities.shape[0:-2])
    px_size = 4.17
    px_sizes[:] = 4.17
    px_grp = imported.create_dataset('pixel_sizes', data=px_sizes)
    px_unit = 'nm'
    px_grp.attrs['unit'] = px_unit
    logger.info("Using pixel size of %f %s", px_size, px_unit)
    # Save metadata
    imported.create_dataset('timestep_names',
                            data=np.array([sam_name], dtype="S50"))
    filenames = np.array(filenames, dtype="S100")
    imported.create_dataset('filenames', data=[filenames], dtype="S100")
    imported.create_dataset('relative_positions',
                            data=[np.zeros(shape=(*filenames.shape, 3), dtype=np.float32)])
    nan_pos = np.empty(shape=(1, *filenames.shape, 3), dtype=np.float32)
    nan_pos.fill(np.nan)
    imported.create_dataset('original_positions', data=nan_pos)
    # Clean up any open files, etc
    h5file.close()
    logger.info("Importing finished in %f seconds", time() - logstart)


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
    correct for this automatically during acquisition: APS 8-BM-B, 32-ID-C.

    Returns a 2-tuple of (scale, translation) arrays. Each array has
    the same length as `frames`.

    Arguments
    ---------
    - frames : Numpy array of image frames that need to be corrected.

    - pixel_sizes : Numpy array of pixel sizes corresponding to
    entries in `frames`.
    """
    scales = np.min(pixel_sizes) / pixel_sizes
    datashape = frames.shape[:-2]
    imshape = np.array(frames.shape[-2:])
    scales2D = scales.reshape(*datashape, 1).repeat(2, axis=-1)
    translations = (imshape-1) * (1-scales2D) / 2
    return (scales, translations)


def import_ssrl_frameset(directory, hdf_filename=None, quiet=False):
    """Import all files in the given directory collected at SSRL beamline
    6-2c and process into framesets. Images are assumed to full-field
    transmission X-ray micrographs and repetitions will be averaged.
    """
    imp_group = import_frameset(directory, hdf_filename=hdf_filename,
                                quiet=quiet, flavor='ssrl')
    # Set some beamline specific metadata
    imp_group.parent.attrs['technique'] = 'Full-field TXM'
    imp_group.parent.attrs['beamline'] = 'SSRL 6-2c'
    # Cleanup and exit
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
                          hdf_filename=hdf_filename, quiet=quiet)
    # Set some beamline specific metadata
    imp_group.parent.attrs['technique'] = 'Full-field TXM'
    imp_group.parent.attrs['beamline'] = 'APS 8-BM-B'
    # Cleanup and exit
    imp_group.file.close()
    return imp_group


def import_frameset(directory, flavor, hdf_filename, quiet=False):
    """Import all files in the given directory collected at APS beamline
    8-BM-B and process into framesets. Images are assumed to
    full-field transmission X-ray micrographs. This beamline does not
    produce the flux to warrant averaging.
    """
    prog.quiet = quiet
    # Check that file does not exist
    if os.path.exists(hdf_filename):
        raise OSError("File {} exists".format(hdf_filename))
    files = [os.path.join(dp, f) for dp, dn, filenames in
             os.walk(directory) for f in filenames]
    # Process filename metadata into the dataframe
    metadata = read_metadata(files, flavor=flavor)
    reference_files = metadata[metadata['is_background'] == True]
    sample_files = metadata[metadata['is_background'] == False]
    # Prepare counters and functions for progress bar
    curr_file = 0
    total_files = len(metadata)
    init_time = time()

    def set_progbar(val, total, init_time):
        if not prog.quiet:
            # Update the progress bar
            status = tqdm.format_meter(n=val,
                                       total=total,
                                       elapsed=time() - init_time,
                                       unit='fm',
                                       prefix="Importing frames: ")
            print("\r", status, end='')  # Avoid new line every time
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
    ref_ds.attrs['context'] = 'frameset'
    for ts_idx, (ts_name, ts_df) in enumerate(ref_groups):
        Is = []
        for E_idx, (E_name, E_df) in enumerate(ts_df.groupby('energy')):
            Importer = format_classes[os.path.splitext(E_df.index[0])[-1]]
            E_files = [Importer(f, flavor=flavor) for f in E_df.index]
            images = np.array([f.image_data() for f in E_files])
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            image = median_filter(np.mean(images, axis=0), footprint=kernel)
            Is.append(image)
            [f.close() for f in E_files]
            # Update progress bar
            curr_file += len(E_files)
            set_progbar(curr_file, total=total_files, init_time=init_time)
        # Save to disk
        ref_ds[ts_idx] = np.array(Is)
    pos_groups = enumerate(sample_files.groupby('position_name'))
    for pos_idx, (pos_name, pos_df) in pos_groups:
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
                Importer = format_classes[os.path.splitext(group.index[0])[-1]]
                E_files = [Importer(f, flavor=flavor) for f in group.index]
                # ...x-ray energies...
                energies = np.array([f.energy() for f in E_files])
                E_ds[ts_idx, E_idx] = np.mean(energies)
                # ...intensity data...
                images = np.array([f.image_data() for f in E_files])
                kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                image = median_filter(np.mean(images, axis=0),
                                      footprint=kernel)
                int_ds[ts_idx, E_idx] = image
                # ...position data...
                pos = np.array([f.sample_position() for f in E_files])
                orig_pos_ds[ts_idx, E_idx] = np.mean(pos, axis=0)
                # ...filenames...
                filename_ds[ts_idx, E_idx] = np.array([E_files[0].filename],
                                                      dtype="S")
                # ...timestamp data...
                starts = [f.starttime() for f in E_files]
                ends = [f.endtime() for f in E_files]
                timestamp_ds[ts_idx, E_idx] = np.array([min(starts), max(ends)],
                                                       dtype='S32')
                # ...pixel size data.
                px_sizes = np.array([f.um_per_pixel() for f in E_files])
                px_ds[ts_idx, E_idx] = np.mean(px_sizes)
                # Increment progress bars
                curr_file += len(E_files)
                set_progbar(curr_file, total=total_files, init_time=init_time)
                # Close all the files
                for f in E_files:
                    f.close()
        # Create a references dataset if one doesn't exist yet
        if 'references' not in h5group.keys():
            h5group['references'] = ref_ds
        # Convert to absorbance values
        apply_references(int_ds, ref_ds, out=abs_ds)
        print("")  # To avoid over-writing the status bar
        # Correct magnification from different energy focusing
        if flavor == 'ssrl':
            # Correct magnification changes due to zone-plate movement
            scales, translations = magnification_correction(frames=abs_ds,
                                                            pixel_sizes=px_ds)
            transform_images(abs_ds, translations=translations,
                             scales=scales, out=abs_ds)
        # Remove dead or hot pixels
        median_filter(abs_ds, size=1, output=abs_ds)
        # Set some metadata
        h5group.parent.attrs['xanespy_version'] = CURRENT_VERSION
        h5group.parent.attrs['original_directory'] = directory
        # Clean-up and return data
        return h5group
