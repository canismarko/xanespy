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
import datetime as dt
import warnings

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
from utilities import prog, prepare_hdf_group
import exceptions
from xanes_math import transform_images, apply_references, parallel_map, foreach

format_classes = {
    '.xrm': XRMFile
}


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
    msg = "This function is ambiguous. Choose from the more specific importers."
    raise NotImplementedError(msg)


def read_metadata(filenames, flavor):
    """Take a list of filenames and return a pandas dataframe with all the
    metadata."""
    columns = ('sample_name', 'position_name', 'is_background',
               'energy', 'shape', 'starttime')
    df = pd.DataFrame(columns=columns)
    for filename in prog(filenames, 'Sorting'):
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
            metadata['starttime'] = f.starttime()
        df.loc[filename] = metadata
    # Remove any incomplete framesets
    sam_names, samples = zip(*df.groupby('sample_name'))
    lengths = np.array([len(df_.groupby('energy')) for df_ in samples])
    max_length = max(lengths)
    bad_samples = []
    for name, length in zip(sam_names, lengths):
        if length < max_length:
            df = df[df.sample_name != name]
            bad_samples.append(name)
    # Warn the user about dropped frames
    if bad_samples and not prog.quiet:
        msg = "Dropping incomplete framesets {}".format(bad_samples)
        warnings.warn(UserWarning(msg))
    return df.sort_values(by=['starttime'])


def import_ptychography_frameset(directory: str,
                                 hdf_filename=None, hdf_groupname=None):
    """Import a set of images as a new frameset for generating
    ptychography chemical maps based on data collected at ALS beamline
    5.3.2.1

    Arguments
    ---------

    - directory : Directory where to look for results. It should
    contain .cxi files that are the output of the ptychography reconstruction."

    - hdf_filename : HDF File used to store computed results. If
      omitted or None, the `directory` basename is used

    - hdf_groupname : String to use for the hdf group of this
    dataset. If omitted or None, the `directory` basename is
    used. Raises an exception if the group already exists in the HDF file.
    """
    CURRENT_VERSION = "0.2" # Let's file loaders deal with changes to storage
    # Prepare some filesystem information
    # tiff_dir = os.path.join(directory, "tiffs")
    # modulus_dir = os.path.join(tiff_dir, "modulus")
    # stxm_dir = os.path.join(tiff_dir, "modulus")
    # Prepare the HDF5 file and metadata
    hdf_group = prepare_hdf_group(filename=hdf_filename,
                                   groupname=hdf_groupname,
                                   dirname=directory)
    hdf_group.attrs["scimap_version"] = CURRENT_VERSION
    hdf_group.attrs["technique"] = "ptychography STXM"
    hdf_group.attrs["beamline"] = "ALS 5.3.2.1"
    hdf_group.attrs["original_directory"] = os.path.abspath(directory)
    # Prepare groups for data
    imported = hdf_group.create_group("imported")
    hdf_group.attrs["active_group"] = "imported"
    imported_group = imported.name
    hdf_group["imported"].attrs["level"] = 0
    hdf_group["imported"].attrs["parent"] = ""
    hdf_group["imported"].attrs["default_representation"] = "modulus"
    file_re = re.compile("projection_modulus_(?P<energy>\d+\.\d+)\.tif")
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
    for filename in cxifiles:
        with h5py.File(filename, mode='r') as f:
            # Extract energy in Joules and convert to eV
            energy = f['/entry_1/instrument_1/source_1/energy'].value
            energy = energy / physical_constants['electron volt'][0]
            # All dataset names will be the energy with two decimal places
            energy_set = imported.create_group(energy_key.format(energy))
            energy_set.attrs['energy'] = energy
            energy_set.attrs['approximate_energy'] = round(energy, 2)
            energy_set.attrs['pixel_size_value'] = 4.17
            energy_set.attrs['pixel_size_unit'] = "nm"
            # Save dataset
            data = f['/entry_1/image_1/data'].value
            energy_set.create_dataset('image_data',
                                      data=data,
                                      chunks=True,
                                      compression="gzip")
            # Import STXM interpretation
            data = f['entry_1/instrument_1/detector_1/STXM'].value
            energy_set.create_dataset('stxm',
                                      data=data,
                                      chunks=True,
                                      compression="gzip")
    for filename in []:#os.listdir(modulus_dir):
        # (assumes each image type has the same set of energies)
        # Extract energy from filename
        match = file_re.match(filename)
        if match is None:
            msg = "Could not read energy from filename {}".format(filename)
            raise exceptions.FilenameParseError(msg)
        energy_str = match.groupdict()['energy']
        # All dataset names will be the energy with two decimal places
        energy_set = imported.create_group(energy_key.format(float(energy_str)))
        energy_set.attrs['energy'] = float(energy_str)
        energy_set.attrs['approximate_energy'] = round(float(energy_str), 2)
        energy_set.attrs['pixel_size_value'] = 4.17
        energy_set.attrs['pixel_size_unit'] = "nm"
        def add_files(name, template="projection_{name}_{energy}.tif"):
            # Import modulus (total value)
            filename = template.format(name=name, energy=energy_str)
            filepath = os.path.join(tiff_dir, name, filename)
            data = Image.open(filepath)
            energy_set.create_dataset(name,
                                      data=data,
                                      chunks=True,
                                      compression="gzip")
        representations = ['modulus', 'phase', 'complex', 'intensity']
        [add_files(name) for name in representations]
        add_files("stxm", template="stxm_{energy}.tif")
    # Create the frameset object
    hdf_filename = hdf_group.file.filename
    hdf_groupname = hdf_group.name
    hdf_group.file.close()
    frameset = XanesFrameset(filename=hdf_filename,
                             groupname=hdf_groupname,
                             edge=None)
    frameset.latest_group = imported_group


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
    correct for this automatically during acquisition: APS 8-BM-B

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
            'sample_name': "rep{}".format(bg_result.group(1)),
            'position_name': bg_result.group(3).strip("_"),
            'is_background': True,
            'energy': float(bg_result.group(4)),
        }
    elif sample_result:
        params = {
            'sample_name': "rep{}".format(sample_result.group(1)),
            'position_name': sample_result.group(2).strip("_"),
            'is_background': False,
            'energy': float(sample_result.group(3)),
        }
    else:
        msg = "Could not parse filename {filename} using flavor {flavor}"
        raise exceptions.FilenameParseError(msg.format(filename=filename, flavor='ssrl'))
    return params


def import_ssrl_frameset(directory, hdf_filename=None, quiet=False):
    """Import all files in the given directory collected at SSRL beamline
    6-2c and process into framesets. Images are assumed to full-field
    transmission X-ray micrographs and repetitions will be averaged.
    """
    return import_aps_8BM_frameset(directory, hdf_filename=hdf_filename, quiet=quiet, flavor='ssrl')
    # Everything else below here is deprecated
    prog.quiet = quiet
    # Prepare list of dataframes to be imported
    samples = {}
    reference_files = {}
    start_time = time()
    total_files = 0 # Counter for progress meter
    curr_file = 0
    # Prepare a dictionary of samples, each sample is a dictionary of
    # energies, which contains a list of filenames to be imported
    for filename in os.listdir(directory):
        # Make sure it's a file
        fullpath = os.path.join(directory, filename)
        if os.path.isfile(fullpath):
            # Queue the file for import if the extension is known
            name, extension = os.path.splitext(filename)
            if extension in format_classes.keys():
                metadata = decode_ssrl_params(filename)
                framesetname = "{name}_rep{rep}"
                framesetname = framesetname.format(name=metadata['sample_name'],
                                                   rep=str(metadata['repetition']))
                if metadata['position_name']:
                    framesetname += "_" + metadata['position_name']
                if metadata['is_background']:
                    root = reference_files
                else:
                    root = samples
                energies = root.get(framesetname, {})
                replicates = energies.get(metadata['energy'], [])
                # Update the stored tree
                root[framesetname] = energies
                replicates.append(fullpath)
                energies[metadata['energy']] = replicates
                total_files += 1
    # Check that in the ssrl flavor, each sample has a reference set
    if not samples.keys() == reference_files.keys():
        msg = "SSRL data should have 1-to-1 sample to reference: {} and {}"
        raise exceptions.DataFormatError(msg.format(list(samples.keys()),
                                                    list(reference_files.keys())))
    # Go through each sample and import
    for sample_name, sample in samples.items():
        # Empty arrays for holding results of importing
        intensities, references, absorbances = [], [], []
        energies, positions, filenames = [], [], []
        starttimes, endtimes = [], []
        pixel_sizes = []
        # Average data for each energy
        for energy in sample:
            averaged_I = _average_ssrl_files(sample[energy])
            intensities.append(averaged_I.data)
            starttimes.append(averaged_I.starttime.isoformat())
            endtimes.append(averaged_I.endtime.isoformat())
            file1 = sample[energy][0]
            name, extension = os.path.splitext(file1)
            Importer = format_classes[extension]
            with Importer(file1, flavor='ssrl') as first_file:
                pixel_sizes.append(first_file.um_per_pixel())
                energies.append(first_file.energy())
                positions.append(first_file.sample_position())
                filenames.append(file1)
            # Increment counter
            curr_file += len(sample[energy])
            # Display progress meter
            if not prog.quiet:
                status = tqdm.format_meter(n=curr_file,
                                           total=total_files,
                                           elapsed=time() - start_time,
                                           prefix="Importing frames: ")
                print("\r", status, end='')
            # Average reference frames
            averaged_ref = _average_ssrl_files(reference_files[sample_name][energy])
            references.append(averaged_ref.data)
            # Apply reference correction to get absorbance data
            abs_data = np.log(averaged_ref.data / averaged_I.data)
            absorbances.append(abs_data)
            # Increment counter
            curr_file += len(reference_files[sample_name][energy])
            # Display progress meter
            if not prog.quiet:
                status = tqdm.format_meter(n=curr_file,
                                           total=total_files,
                                           elapsed=time() - start_time,
                                           prefix="Importing frames: ")
                print("\r", status, end='')
        # Convert data to numpy arrays
        absorbances = np.array(absorbances)
        pixel_sizes = np.array(pixel_sizes)
        # Correct magnification changes due to zone-plate movement
        scales, translations = magnification_correction(frames=absorbances,
                                                        pixel_sizes=pixel_sizes)
        transform_images(absorbances, translations=translations,
                         scales=scales, out=absorbances)
        pixel_sizes[:] = np.min(pixel_sizes)
        # Save data to HDF5 file
        store = prepare_txm_store(filename=hdf_filename,
                                  parent_name=sample_name,
                                  dirname=directory)
        def save_data(name, data):
            # Sort by energy
            data = [d for (E, d) in sorted(zip(energies, data), key=lambda x: x[0])]
            # Save as new HDF5 dataset
            setattr(store, name, data)
        save_data('intensities', data=intensities)
        save_data('references', data=references)
        save_data('absorbances', data=absorbances)
        save_data('pixel_sizes', data=pixel_sizes)
        store.data_group()['pixel_sizes'].attrs['unit'] = 'µm'
        save_data('energies', data=energies)
        save_data('timestamps', data=zip(starttimes, endtimes))
        save_data('filenames', data=filenames)
        save_data('original_positions', data=positions)
        store.data_group()['original_positions'].attrs['order'] = "(energy, (x, y, z))"
        # Convert to relative positions
        save_data('relative_positions', data=np.zeros_like(positions))
        store.data_group()['relative_positions'].attrs['order'] = "(energy, (x, y, z))"
        # All done, clean up
        store.close()


def decode_aps_params(filename):
    """Accept the filename of an XRM file and return sample parameters as
    a dictionary."""
    regex = re.compile(
        '(?P<pos>[a-zA-Z0-9_]+)_xanes(?P<sam>[a-zA-Z0-9_]+)_(?P<E_int>[0-9]+)_(?P<E_dec>[0-9])eV.xrm'
    )
    match = regex.search(filename).groupdict()
    energy = float("{}.{}".format(match['E_int'], match['E_dec']))
    result = {
        'sample_name': match['sam'],
        'position_name': match['pos'],
        'is_background': match['pos'] == 'ref',
        'energy': energy,
    }
    return result

# @profile
def import_aps_8BM_frameset(directory, hdf_filename=None, quiet=False, flavor="aps"):
    """Import all files in the given directory collected at APS beamline
    8-BM-B and process into framesets. Images are assumed to
    full-field transmission X-ray micrographs. This beamline does not
    produce the flux to warrant averaging.
    """
    prog.quiet = quiet
    # files = [os.path.join(dp, f) for dp, dn, os.listdir(directory)
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory) for f in filenames]
    # dirs = [os.path.join(dp, f) for f in fldp, dn, os.listdir(directory)]
    # print(list(os.walk(directory)))
    # metadata = decode_aps_params(files[0])
    # Prepare a dataframe to hold all the file metadata
    # Process filename metadata into the dataframe
    metadata = read_metadata(files, flavor=flavor)
    reference_files = metadata[metadata['is_background']==True]
    sample_files = metadata[metadata['is_background']==False]
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
            print("\r", status, end='') # Avoid new line every time
    # Import each sample-position combination
    h5file = h5py.File(hdf_filename)
    # Get some shape information for all the datasets
    shapes = metadata['shape'].unique()
    assert len(shapes) == 1
    num_samples = len(reference_files['sample_name'].unique())
    pos_name = sample_files['position_name'].unique()[0]
    num_energies = len(reference_files['energy'].unique())
    ds_shape = (num_samples, num_energies, *shapes[0])
    chunk_shape = (1, 1, *shapes[0])
    ## Import reference frames
    assert len(reference_files.groupby('position_name')) == 1
    ref_groups = reference_files.groupby('sample_name')
    imp_group = h5file.require_group("{}/imported".format(pos_name))
    ref_ds = imp_group.require_dataset('references', shape=ds_shape,
                                       maxshape=ds_shape,
                                       chunks=chunk_shape,
                                       dtype=np.uint16)
    for sam_idx, (sam_name, sam_df) in enumerate(ref_groups):
        Is = []
        for E_idx, (E_name, E_df) in enumerate(sam_df.groupby('energy')):
            Importer = format_classes[os.path.splitext(E_df.index[0])[-1]]
            E_files = [Importer(f, flavor="aps") for f in E_df.index]
            images = np.array([f.image_data() for f in E_files])
            Is.append(np.mean(images, axis=0))
            [f.close() for f in E_files]
            # Update progress bar
            curr_file += len(E_files)
            set_progbar(curr_file, total=total_files, init_time=init_time)
        # import pdb; pdb.set_trace()
        # Save to disk
        ref_ds[sam_idx] = np.array(Is)
    # Import the actual sample frames
    pos_groups = enumerate(sample_files.groupby('position_name'))
    for pos_idx, (pos_name, pos_df) in pos_groups:
        # Create HDF5 datasets to hold the data
        Importer = format_classes[os.path.splitext(pos_df.index[0])[-1]]
        with Importer(pos_df.index[0], flavor="aps") as f:
            im_shape = f.image_data().shape
            num_samples = len(pos_df.groupby('sample_name'))
            num_energies = len(pos_df.groupby('energy'))
        ds_shape = (num_samples, num_energies, *im_shape)
        h5group = h5file.require_group("{}/imported".format(pos_name))
        h5file["{}".format(pos_name)].attrs['latest_data_name'] = 'imported'
        int_ds = h5group.require_dataset('intensities',
                                         shape=ds_shape,
                                         maxshape=ds_shape,
                                         chunks=True,
                                         dtype=np.uint16)
        abs_ds = h5group.require_dataset('absorbances',
                                         shape=ds_shape,
                                         maxshape=ds_shape,
                                         dtype=np.float32)
        px_ds = h5group.require_dataset('pixel_sizes',
                                        shape=(num_samples, num_energies),
                                        dtype=np.float16)
        px_ds.attrs['unit'] = 'µm'
        E_ds = h5group.require_dataset('energies',
                                       shape=(num_samples, num_energies),
                                       dtype=np.float32)
        timestamp_ds = h5group.require_dataset('timestamps',
                                               shape=(num_samples, num_energies, 2),
                                               dtype="S32")
        filename_ds = h5group.require_dataset('filenames',
                                              shape=(num_samples, num_energies),
                                              dtype="S100",
                                              compression="gzip")
        orig_pos_ds = h5group.require_dataset('original_positions',
                                              shape=(num_samples, num_energies, 3),
                                              dtype=np.float32,
                                              compression="gzip")
        rel_pos_ds = h5group.require_dataset('relative_positions',
                                              shape=(num_samples, num_energies, 3),
                                              dtype=np.float32)
        sam_groups = enumerate(pos_df.groupby('sample_name'))
        for sam_idx, (sam_name, sam_df) in sam_groups:
            E_groups = sam_df.sort_values('energy').groupby('energy')
            for E_idx, (energy, group) in enumerate(E_groups):
                # Import the data from each energy frame...
                Importer = format_classes[os.path.splitext(group.index[0])[-1]]
                E_files = [Importer(f, flavor="aps") for f in group.index]
                # ...x-ray energies...
                energies = np.array([f.energy() for f in E_files])
                E_ds[sam_idx,E_idx] = np.mean(energies)
                # ...intensity data...
                images = np.array([f.image_data() for f in E_files])
                int_ds[sam_idx,E_idx] = np.mean(images, axis=0)
                # ...position data...
                pos = np.array([f.sample_position() for f in E_files])
                orig_pos_ds[sam_idx,E_idx] = np.mean(pos, axis=0)
                # ...filenames...
                filename_ds[sam_idx,E_idx] = np.array([E_files[0].filename],
                                                      dtype="S")
                # ...timestamp data...
                starts = [f.starttime() for f in E_files]
                ends = [f.endtime() for f in E_files]
                timestamp_ds[sam_idx,E_idx] = np.array([min(starts), max(ends)],
                                                       dtype='S32')
                # ...pixel size data.
                px_sizes = np.array([f.um_per_pixel() for f in E_files])
                px_ds[sam_idx,E_idx] = np.mean(px_sizes)
                # pixel_sizes.append(np.mean(px_sizes))
                # Increment progress bars
                curr_file += len(E_files)
                set_progbar(curr_file, total=total_files, init_time=init_time)
                # Get reference if it isn't saved
                # ref = references.get(energy, None)
                # if ref is None:
                #     ref_idx = ((reference_files['sample_name'] == sample)
                #            & (reference_files['energy'] == energy))
                #     ref_fnames = reference_files[ref_idx].index
                #     ref_files = np.array([Importer(f, flavor='aps') for f in ref_fnames])
                #     ref = (np.mean(np.array([f.image_data() for f in ref_files]), axis=0))
                #     references[energy] = ref
                #     # Increment progress bars
                #     curr_file += len(ref_files)
                #     set_progbar(curr_file, total=total_files, init_time=init_time)
                # Close all the files
                for f in E_files: f.close()
            # Save data to disk
            # groupname = "{sample}_{fov}".format(sample=sample, fov=position)
            # store = prepare_txm_store(filename=hdf_filename,
            #                           parent_name=groupname)
            # all_groupnames.append(groupname)
            # def save_data(name, data, energies):
            #     """Sort data by energy then save to the store."""
            #     # Sort by energy
            #     data = [d for (E, d) in sorted(zip(energies, data), key=lambda x: x[0])]
            #     # Save as new HDF5 dataset
            #     setattr(store, name, data)
            # save_data('intensities', data=Is, energies=Es)
            # save_data('energies', data=Es, energies=Es)
            # save_data('pixel_sizes', data=pixel_sizes, energies=Es)
            # store.data_group()['pixel_sizes'].attrs['unit'] = 'µm'
            # save_data('timestamps', data=zip(starttimes, endtimes), energies=Es)
            # save_data('filenames', data=filenames, energies=Es)
            # save_data('original_positions', data=positions, energies=Es)
            # store.data_group()['original_positions'].attrs['order'] = "(energy, (x, y, z))"
            # positions = np.array(positions)
            # save_data('relative_positions', data=np.zeros_like(positions), energies=Es)
            # store.data_group()['relative_positions'].attrs['order'] = "(energy, (x, y, z))"
            # Look for reference frames
            # if ref_ds is None:
            #     # Save the references to disk
            #     save_data('references', data=references.values(),
            #               energies=references.keys())
            #     # Save for later so we can link instead of re-writing
            #     ref_ds = store.data_group()['references'].name
            # else:
            #     # Reference frames are already saved, so just create a hard-link
            #     store.data_group()['references'] = store.data_group().file[ref_ds]
            # Save the reference data if not set
            # Save datasets
        if not 'references' in h5group.keys():
            h5group['references'] = ref_ds
        # Convert to absorbance values
        apply_references(int_ds, ref_ds, out=abs_ds)
            # Remove dead or hot pixels
            # median_filter(abs_ds, size=1, output=abs_ds)
            # sigma = 9
            # absorbances = remove_outliers(data=absorbances, sigma=sigma)
            # Save absorbances to disk
    # Print a summary of data saved to disk
    # if not quiet:
    #     print() # To put in a newline
    #     msg = "Saved to HDF5 file {h5file} in {num} groups:"
    #     print(msg.format(h5file=hdf_filename, num=len(all_groupnames)))
    #     for gname in all_groupnames:
    #         print('• {}'.format(gname))
    # for (sample, pos), group in sample_files.groupby(['sample_name', 'position_name']):
    #     groupname = "{sample}_{fov}".format(sample=sample, fov=pos)
    #     sample_group = prepare_hdf_group(filename=hdf_filename,
    #                                      groupname=groupname)
    #     energies, intensities = 
    #     print(group)
    # ref_name = "reference"
    # reference = sample_group.create_group(ref_name)
    # reference.attrs['default_representation'] = 'image_data'
    # reference.attrs['parent'] = ""
    # absorbance = sample_group.create_group("reference_corrected")
    # absorbance.attrs['default_representation'] = 'image_data'
    # absorbance.attrs['parent'] = imported.name
    # Go through each file and import it to the correct position group
    # position_groups = {}

    # Empty arrays to hold the imported images and metadata
    # intensities, references = [], []
    # I_energies, ref_energies = [], []
    # starttimes, endtimes = [], []
    # pixel_sizes = []
    # for filename in prog(files, 'Importing frames'):
    #     # Make sure it's a file
    #     fullpath = os.path.join(directory, filename)
        # Determine what group to put it in
        # if metadata['is_background']:
        #     fs_group = reference
        # elif pos_name not in sample_group.keys():
        #     fs_group = sample_group.create_group(pos_name)
        #     fs_group.attrs['default_representation'] = 'image_data'
        #     fs_group.attrs['parent'] = ""
        #     sample_group.attrs['latest_group'] = fs_group.name
        #     # Save for later manipulation (reference correction, etc)
        #     position_groups[metadata['position_name']] = fs_group
        # else:
        #     fs_group = sample_group[pos_name]
        # approximate_energy = round(float(actual_energy), 1)
        # Retrieve data
    #     basename, extension = os.path.splitext(filename)
    #     Importer = format_classes[extension]
    #     with Importer(fullpath, flavor='aps') as f:
    #         actual_energy = f.energy()
    #         data = f.image_data()
    #     # Remove dead or hot pixels
    #     sigma = 9
    #     data = remove_outliers(data=data, sigma=sigma)
    #     # Sort data into categories
    #     if metadata['is_background']:
    #         ref_energies.append(actual_energy)
    #         references.append(data)
    #     else:
    #         # Actual sample frame
    #         I_energies.append(actual_energy)
    #         intensities.append(data)
    #         # Save image data
    #         # intensity_group = fs_group.create_group(key)
    #         # intensity_group.create_dataset(name="image_data",
    #         #                                data=data,
    #         #                                compression="gzip")
    #         # Set metadata attributes
    #         # intensity_group.attrs['starttime'] = f.starttime().isoformat()
    #         # intensity_group.attrs['endtime'] = f.endtime().isoformat()
    #         # intensity_group.attrs['pixel_size_value'] = f.um_per_pixel()
    #         # intensity_group.attrs['pixel_size_unit'] = 'um'
    #         # intensity_group.attrs['energy'] = actual_energy
    #         # intensity_group.attrs['approximate_energy'] = approximate_energy
    #         # intensity_group.attrs['sample_position'] = f.sample_position()
    #         # intensity_group.attrs['original_filename'] = filename
    # # Apply reference correction
    # for pos_name, imported_group in position_groups.items():
    #     # Create a new HDF5 group
    #     abs_name = "{}_refcorr".format(pos_name)
    #     new_fs_group = sample_group.create_group(abs_name)
    #     # Copy attributes from intensity group
    #     for attr_key in imported_group.attrs.keys():
    #         new_fs_group.attrs[attr_key] = imported_group.attrs[attr_key]
    #     # Calculate absorbance frames for each energy
    #     for key in prog(imported_group.keys(), 'Applying reference'):
    #         new_fs_group.attrs['default_representation'] = 'image_data'
    #         new_fs_group.attrs['parent'] = imported_group.name
    #         # Apply reference correction
    #         ref_data = reference[key]['image_data'].value
    #         sam_data = imported_group[key]['image_data'].value
    #         abs_data = np.log(ref_data / sam_data)
    #         # Save new data to hdf5 file
    #         abs_group = new_fs_group.create_group(key)
    #         abs_group.create_dataset(name='image_data',
    #                                  data=abs_data,
    #                                  compression="gzip")
    #         # Copy attrs from old group
    #         for attr_key in imported_group[key].attrs.keys():
    #             abs_group.attrs[attr_key] = imported_group[key].attrs[attr_key]
    # # Close HDF file to prevent corruption
    # sample_group.file.close()
