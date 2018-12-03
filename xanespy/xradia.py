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

"""Tools for importing X-ray microscopy frames in formats produced by
Xradia instruments."""

import datetime as dt
from collections import namedtuple
import struct
import os
import re
import warnings

import olefile
import numpy as np
import pytz

import exceptions
from utilities import shape, Pixel


# Some of the byte decoding was taken from
# https://github.com/data-exchange/data-exchange/blob/master/xtomo/src/xtomo_reader.py


class XRMFile():
    """Single X-ray micrscopy frame created using XRadia XRM format.
    
    Formats from different beamlines have subtly different storage
    patterns. The ``flavor`` argument controls this parameter. The
    following metadata are affected by this coice:
    
    - Energy: SSRL 6-2c does not store the X-ray beam energy in the
      file so it must be extracted from the filename.
    - starttime, endtime : The XRMFile does not store timezone info,
      so we have to guess based on beamline.
    - pixel_size : The APS microscope automatically corrects
      magnification when changing energy, the SSRL microscope does
      not.
    
    Arguments
    ---------
    filename : str
      The path to the .xrm file
    flavor : str
      The variety of data represented in the xrm file. Valid
      choices are ['ssrl', 'aps', 'aps-old1']. These choices should
      line up with whatever is generated using the scripts in
      beamlines moudles.
    
    """
    aps_old1_regex = re.compile(r"(\d{8})_([a-zA-Z0-9_]+)_([a-zA-Z0-9]+)_(\d{4}).xrm")
    
    def __init__(self, filename, flavor: str):
        self.filename = filename
        self.flavor = flavor
        self.ole_file = olefile.OleFileIO(self.filename)
        # Filename parameters
        # params = self.parameters_from_filename()
        # self.sample_name = params['sample_name']
        # self.position_name = params['position_name']
        # self.is_background = params['is_background']
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return "<XRMFile: '{}'>".format(os.path.basename(self.filename))
    
    def close(self):
        """Close original XRM (ole) file on disk."""
        self.ole_file.close()
    
    def um_per_pixel(self):
        """Describe the size of a pixel in microns. If this is an SSRL frame,
        the pixel size is dependent on energy. For APS frames, the pixel size
        is uniform and assumes a 40µm field-of-view.
        """
        # Based on calibration data from Yijin:
        #   For 9 keV, at binning 2, the pixel size is 35.54 nm.  In our
        #   system, the pixel size is proportional to the X-ray energy.
        #   Say you have 8.5 keV, your pixel size at binning 2 will be
        #   36.39296/9*8.5
        if self.flavor == "ssrl":
            energy = self.energy()
            field_size = 36.39296 * energy / 9000
        else:
            field_size = 40
        num_pixels = max(self.image_data().shape)
        pixel_size = field_size / num_pixels
        return pixel_size
    
    def ole_value(self, stream, fmt=None, as_array=False):
        """Get arbitrary data from the ole file and convert from bytes."""
        try:
            stream_bytes = self.ole_file.openstream(stream).read()
        except OSError:
            msg = "Cannot find stream {} in file {}"
            msg = msg.format(stream, self.filename)
            raise exceptions.DataFormatError(msg)
        # Modify the format string to read all values if an array
        if fmt is not None and len(fmt) == 2:
            endian, dtype = fmt
            num_vals = int(len(stream_bytes) / struct.calcsize(fmt))
            fmt = "{endian}{num_vals}{dtype}"
            fmt = fmt.format(endian=endian, num_vals=num_vals, dtype=dtype)
        # Decode the bytes value
        if fmt is not None:
            stream_value = struct.unpack(fmt, stream_bytes)
            # Return just the first value if requested
            if not as_array:
                stream_value = stream_value[0]
        else:
            stream_value = stream_bytes
        return stream_value
    
    def print_ole(self):  # pragma: no cover
        for l in self.ole_file.listdir():
            if l[0] == 'ImageInfo':
                try:
                    val = self.ole_value(l, '<f')
                except:
                    pass
                else:
                    print(l, ':', val)
    
    def endtime(self):
        """Retrieve a datetime object representing when this frame was
        finished collecting. Duration is decided by exposure time of the frame
        and the start time."""
        exptime = self.ole_value('ImageInfo/ExpTimes', '<f')
        startime = self.starttime()
        duration = dt.timedelta(seconds=exptime)
        return startime + duration
    
    def starttimes(self):
        """Retrieve all datetime objects representing when these frames were
        collected. Timezone is inferred from flavor (eg. ssrl ->
        california time).
        
        """
        # Determine most likely timezone
        if self.flavor == 'ssrl':
            timezone = pytz.timezone('US/Pacific')
        elif self.flavor in ['aps', 'aps-old1']:
            # timezone = pytz.timezone('US/Central')
            timezone = pytz.timezone("America/Chicago")
        else:  # pragma: no cover
            # Unknown flavor. Raising here instead of constructor
            # means you forgot to put in the proper timezone.
            msg = "Unknown timezone for flavor '{flavor}'. Assuming UTC"
            warnings.warn(msg.format(flavor=self.flavor))
            timezone = pytz.utc
        # Retrieve the raw bytes from disk
        key = 'ImageInfo/Date'
        bytes_ = self.ole_value(key)
        # Split the date into chunks
        n_images = self.num_images()
        chunk_size = int(len(bytes_) / n_images)
        chunk_starts = range(0, len(bytes_), chunk_size)
        date_chunks = [bytes_[i:i+chunk_size] for i in chunk_starts]
        # Convert each chunk into a date
        fmt = "%m/%d/%y %H:%M:%S"
        dt_list = []
        for dt_bytes in date_chunks:
            dt_str = dt_bytes[0:17].decode()
            # Convert string into datetime object
            timestamp = dt.datetime.strptime(dt_str, fmt)
            timestamp = timezone.localize(timestamp)
            # Now convert to datetime naive UTC format
            timestamp = timestamp.astimezone(pytz.utc).replace(tzinfo=None)
            dt_list.append(timestamp)
        return dt_list
    
    def num_images(self):
        return self.ole_value('ImageInfo/NoOfImages', '<I')
    
    def starttime(self):
        """Return the earliest timestamp for the collected frames."""
        timestamp = min(self.starttimes())
        return timestamp
    
    def energy(self):
        """Beam energy in electronvoltes."""
        # Try reading from file first
        energy = self.ole_value('ImageInfo/Energy', '<f')
        if not energy > 0:
            # if not, read from filename
            re_result = re.search(r"(\d+\.?\d?)_?eV", self.filename)
            if re_result:
                energy = float(re_result.group(1))
            else:
                msg = "Could not read energy for file {}"
                raise exceptions.FileFormatError(msg.format(self.filename))
        return energy
    
    def is_valid(self):
        """Check that the XRM file has valid data."""
        keys = ['ImageData1/Image1', 'ImageInfo/Date']
        has_keys = [self.ole_file.exists(k) for k in keys]
        if all(has_keys):
            is_valid = True
        else:
            is_valid = False
        # The following code is more robust but much slower
        # try:
        #     self.image_data()
        # except OSError:
        #     is_valid = False
        return is_valid
    
    def image_data(self, idx=0):
        """TXM Image frame."""
        # Figure out byte size
        dimensions = self.image_shape()
        num_bytes = dimensions.columns * dimensions.rows
        # Determine format string
        image_dtype = self.image_dtype()
        if image_dtype == 'uint16':
            fmt_str = "<{}h".format(num_bytes)
        elif image_dtype == 'float32':
            fmt_str = "<{}f".format(num_bytes)
        # Return decoded image data
        try:
            stream = self.ole_file.openstream('ImageData1/Image{}'.format(idx+1))
        except OSError:
            raise
            msg = "Cannot open image data {} in file {}"
            msg = msg.format('ImageData1/Image1', self.filename)
            raise exceptions.DataFormatError(msg)
        img_data = struct.unpack(fmt_str, stream.read())
        img_data = np.reshape(img_data, dimensions)
        return img_data
    
    # def is_background(self):
    #     """Look at the file name for clues to whether this is a background
    #     frame."""
    #     result = re.search(r'bkg|_ref_', self.filename)
    #     return bool(result)
    
    def sample_position(self):
        position = namedtuple('position', ('x', 'y', 'z'))
        x = self.ole_value('ImageInfo/XPosition', '<f')
        y = self.ole_value('ImageInfo/YPosition', '<f')
        z = self.ole_value('ImageInfo/ZPosition', '<f')
        return position(x, y, z)
    
    def binning(self):
        """Binning mode of the image.
        
        Binning reduces pixel density but improves signal/noise ratio
        by combining adjacent pixels. For example, a 2048 x 2048 CCD
        with binning 4 would produce a 512 x 512 image.
        
        """
        vertical = self.ole_value('ImageInfo/VerticalalBin', '<L')
        horizontal = self.ole_value('ImageInfo/HorizontalBin', '<L')
        binning = namedtuple('binning', ('horizontal', 'vertical'))
        return binning(horizontal, vertical)
    
    def image_dtype(self):
        dtypes = {
            5: 'uint16',
            10: 'float32',
        }
        dtype_number = self.ole_value('ImageInfo/DataType', '<1I')
        return dtypes[dtype_number]
    
    def image_shape(self):
        horizontal = self.ole_value('ImageInfo/ImageWidth', '<I')
        vertical = self.ole_value('ImageInfo/ImageHeight', '<I')
        return shape(columns=vertical, rows=horizontal)


class TXRMFile(XRMFile):
    """Similar to :py:class:`XRMFile` but contains multiple images in a data-set."""
    def image_stack(self):
        entries = [s for s in self.ole_file.listdir() if s[0] == 'ImageData1']
        stack = []
        for idx in range(len(entries)):
            stack.append(self.image_data(idx=idx))
        return np.array(stack)
    
    def energies(self):
        bytesize = 4
        energy_bytes = self.ole_value('ImageInfo/Energy')
        # Unpack the array
        length = int(len(energy_bytes) / bytesize)
        fmt = "<{}f".format(length)
        energies = struct.unpack(fmt, energy_bytes)
        # Filter out any zero values
        energies = np.array([e for e in energies if e > 0.])
        return energies
    
