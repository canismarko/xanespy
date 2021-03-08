# -*- Coding: utf-8 -*-
#
# Copyright © 2021 Mark Wolf
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


from typing import Sequence, Union
from pathlib import Path
import pandas as pd
import re

import h5py

# Type aliases
HDFPath = Union[Path, str]


def tomopy_metadata(source_files, regex):
    """Extract metadata from filenames.
    
    *regex* is a regular expression that will be given the
    filename. Filenames that don't match will be ignored. Named groups
    will be used to extract metadata. Valid named groups are:
    
      - scan_num
      - energy
    
    Returns
    =======
    metadata
      A pandas dataframe with the extracted metadata.
    
    """
    # Convert strings to regular expressions
    if isinstance(regex, str):
        regex = re.compile(regex)
    # Parse all the files for their metadata
    metadata = []        
    for fname in source_files:
        match = regex.match(fname)
        if match:
            # Found a valid match to the regular expression, so parse it
            match_dict = match.groupdict()
            row = {
                'tomogram_path': fname,
                'scan_num': match_dict.get('scan_num'),
                'energy': match_dict.get('energy'),
            }
            metadata.append(row)
    # Convert all the matched rows to a dataframe
    metadata = pd.DataFrame(metadata)
    return metadata


def import_tomopy_hdf(source_files: Sequence[HDFPath],
                      target_file: HDFPath,
                      experiment_name: str):
    """Import reconstructed tomograms used by the *tomopy_cli* package.
    
    Parameters
    ==========
    source_files
      The paths to use for loading data. Each entry should point to an
      HDF5 file created using *tomopy_cli* with
      ``--output-format=hdf5``.
    target_file
      The destination HDF5 file that will hold the processed data.
    experiment_name
      The name of the experiment. This will be used to create a new
      group inside the *target_file* file.
    
    """
    # Parse the file metadata
    metadata = tomopy_metadata(source_files)
    with h5py.File(target_file, mode="a") as dest_hdf:
        # Create groups to hold the data
        experiment_group = dest_hdf.create_group(experiment_name)
        imported_group = experiment_group.create_group("imported")

        # Import the files
