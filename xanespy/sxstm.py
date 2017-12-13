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

"""Tools for importing Soft X-ray Scanning Tunneling Microscopy data
as output from the sector 4 ID-C beamline."""

import struct
import re

import numpy as np
import pandas as pd

class SxstmDataFile():
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, mode="r+b")
        
    
    def header_lines(self):
        self.file.seek(0)
        lines = self.file.readlines()[0:34]
        # Convert from bytestring to unicode
        lines = [l.decode('utf-8') for l in lines]
        # Remove excess whitespace
        lines = [l.strip() for l in lines]
        assert lines[-1] == ':HEADER_END:', lines[-1]
        return lines[:-1]
    
    def dataframe(self):
        # Find the start of the data section
        self.file.seek(0)
        last_line = ""
        while last_line != ":HEADER_END:":
            last_line = self.file.readline().decode('ascii').strip()
        # Load list of columns
        self.file.read(44) # Garbage??
        channels = self.file.read(259)
        channels = channels.split(b'\x00\x00\x00')
        # Clean out unhelpful bytes from the column names
        bad_chars = [0x0b, 0x0f, 0x12, 0x10, 0x08, 0x0c, 0x05, 0x1a]
        clean_first = lambda b: b[1:] if b[0] in bad_chars else b
        channels = list(map(clean_first, channels))
        # Convert column names to unicode
        channels = [c.decode('latin1') for c in channels]
        # Read experimental data and convert to float/int/etc.
        lines = self.file.read()
        word_len = 4
        fmt = ">%df" % (len(lines) / word_len)
        numbers = struct.unpack_from(fmt, lines)
        numbers = np.array(numbers)
        # Reshape the values to be in the correct order for pandas
        numbers = numbers.reshape((len(channels), -1))
        numbers = numbers.swapaxes(0, 1)
        # Create the pandas dataframe
        df = pd.DataFrame(data=numbers, columns=channels)
        return df
    
    def channels(self):
        hdr = self.header_lines()
        reg = re.compile('^Channels="([a-zA-Z0-9 ();]+)"')
        for l in hdr:
            match = reg.match(l)
            if match:
                channels = match.group(1).split(';')
                return channels
        
    def num_points(self):
        hdr = self.header_lines()
        reg = re.compile('^Points=(\d+)')
        for l in hdr:
            match = reg.match(l)
            if match:
                return int(match.group(1))
    
    def close(self, *args, **kwargs):
        self.file.close(*args, **kwargs)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()
