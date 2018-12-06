import os
import json
import re
import warnings
import datetime as dt

import h5py
import numpy as np
from scipy.constants import physical_constants

import exceptions

class CXIFile():
    """A ptychography CXI file.
    """
    def __init__(self, path, mode='r'):
        self.path = path
        self.filename = os.path.basename(path)
        self.hdf_file = h5py.File(path, mode=mode)
    
    def __repr__(self):
        return '<CXI: {}>'.format(self.filename)
    
    def __str__(self):
        return self.filename
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def close(self):
        self.hdf_file.close()
    
    def filenames(self):
        filenames = [os.path.basename(self.path)]
        return filenames
    
    def image_frames(self):
        data = self.hdf_file['/entry_1/image_1/data'].value[::-1]
        return np.array([data])
    
    def image_shape(self):
        shape = self.hdf_file['/entry_1/image_1/data'].shape
        return shape
    
    def num_images(self):
        return 1
    
    def energies(self):
        energy = self.hdf_file['/entry_1/instrument_1/source_1/energy'].value
        energy = energy / physical_constants['electron volt'][0]
        return [energy]
    
    def pixel_size(self):
        try:
            px_size = float(self.hdf_file['/entry_1/process_1/Param/pixnm'].value)
        except KeyError:
            raise exceptions.DataFormatError(
                'File {} does not contain pixel size'
                ''.format(self.filename))
        return px_size


class HDRFile():
    """A STXM HDR file."""
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.xim_fmt = '{base}_a{idx}.xim'.format(
            base=os.path.splitext(self.path)[0], idx='{idx:03d}')
        with open(self.path, mode='r') as f:
            self._contents = f.read()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def close(self):
        pass
    
    def filenames(self):
        # Extract the file IDs from the image metadata
        ids = (int(s['id'].split('_')[0]) for s in self._image_metadata())
        # Convert the image IDs to filenames
        fnames = [self.xim_fmt.format(idx=i) for i in ids]
        return fnames
    
    def image_shape(self):
        shape_re = re.compile(r'XPoints = (\d+); YPoints = (\d+)')
        match = shape_re.search(self._contents)
        if not match:
            raise exceptions.DataFormatError('Could not read image_shape in'
                                             '{}'.format(self.path))
        else:
            shape = tuple(int(match.group(i)) for i in (1, 2))
        return shape
    
    def energies(self):
        regex = (
            r'\s*StackAxis = { Name = "Energy"; Unit = "eV";'
            r' Min = (\d+); Max = (\d+); Dir = (\d+);\n'
            r'\s*Points = \((\d+), ([0-9., ]+)\);\n'
            r'};'
        )
        E_re = re.compile(regex)
        match = E_re.search(self._contents)
        # Check if a valid match was found
        if not match:
            raise exceptions.DataFormatError('Could not read energies in'
                                             '{}'.format(self.path))
        else:
            # Decode the groups from the regular expression
            num_energies = int(match.group(4))
            energies = [float(e) for e in match.group(5).split(',')]
            if num_energies != len(energies):
                log.warning('Wrong number of energies in "{}": expected {} got {}'
                            ''.format(self.path, num_energies, len(energies)))
        # Check that the number of energies matches the number of images
        num_images = self.num_images()
        if num_energies != num_images:
            log.warning('Number of energies ({}) does not match '
                        'number of images ({}) in {}'
                        ''.format(num_energies, num_images, self.filename))
        return energies
    
    def num_images(self):
        return len(self._image_metadata())
    
    def image_frames(self):
        frames = np.array([np.loadtxt(xim, dtype='int')
                           for xim in self.filenames()])
        return frames
    
    def _image_metadata(self):
        """Get a list of dictionaries with metadata for each image frame."""
        regex = re.compile(
            r'Image(?P<id>[0-9_]+) = {'
            r'StorageRingCurrent = (?P<current>[-0-9.]+); '
            r'Energy = (?P<energy>[-0-9.]+); '
            r'Time = "(?P<time>[0-9a-zA-Z :]+)"; '
            r'ZP_dest= (?P<zp_dest>[-0-9.]+); '
            r'ZP_error = (?P<zp_error>[-0-9.]+);};'
        )
        matches = tuple(m.groupdict() for m in regex.finditer(self._contents))
        # Convert strings to numbers, dates, etc.
        types = {
            'energy': float,
            'time': lambda t: dt.datetime.strptime(t, '%Y %b %d %H:%M:%S'),
            'current': float,
            'zp_dest': float,
            'zp_error': float,
        }
        for scan in matches:
            for key, val in scan.items():
                type_ = types.get(key, lambda x: x)
                scan[key] = type_(val)
        return matches
    
    def scan_metadata(self):
        regex = (
            r'\{\s*CentreXPos = (?P<centerx>\d+\.?\d*);'
            r'\s*CentreYPos = (?P<centery>\d+\.?\d*);'
            r'\s*XRange = (?P<xrange>\d+\.?\d*);'
            r'\s*YRange = (?P<yrange>\d+\.?\d*);'
            r'\s*XStep = (?P<xstep>\d+\.?\d*);'
            r'\s*YStep = (?P<ystep>\d+\.?\d*);'
            r'\s*XPoints = (?P<xpoints>\d+\.?\d*);'
            r'\s*YPoints = (?P<ypoints>\d+\.?\d*);'
            r'\s*SquareRegion = (?P<squareregion>[a-z]+);'
            r'\s*SquarePixels = (?P<squarepixels>[a-z]+);'
        )
        scan_re = re.compile(regex)
        match = scan_re.search(self._contents)
        metadata = match.groupdict()
        # Convert value to proper datatypes
        to_bool = lambda x: True if x == 'true' else False
        types = {
            'centerx': float,
            'centery': float,
            'xrange': float,
            'yrange': float,
            'xstep': float,
            'ystep': float,
            'xpoints': int,
            'ypoints': int,
            'squareregion': to_bool,
            'squarepixels': to_bool,
            
        }
        for key, val in metadata.items():
            type_ = types.get(key, lambda x: x)
            metadata[key] = type_(val)
        return metadata
    
    def pixel_size(self):
        """Retrieve the nominal pixel size in nm."""
        metadata = self.scan_metadata()
        xstep, ystep = metadata['xstep'], metadata['ystep']
        # Convert values from µm to nm
        xstep, ystep = xstep * 1000, ystep * 1000
        # Check for non-square scans
        if xstep != ystep:
            warnings.warn('Mismatched x ({}) and y ({}) steps in scan {}. '
                          'Ignoring y step.'.format(xstep, ystep, self.filename))
        # Just use the xstep as the step size
        return xstep
