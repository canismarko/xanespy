#!/usr/bin/env python

"""A script to launch spectrum fitting for all the pixels of a XANES
Frameset. This script should be launched with `mpiexec
mpi_spectrum_fit.py` to make proper use of parallel processing."""


import logging
logging.basicConfig(level=logging.DEBUG)
import os

import xanespy as xp

HDF_FILE = '/tmp/txm_data_from_youngsang.h5'
# HDF_FILE = os.path.join(os.path.dirname(__file__), 'imported-ssrl-data.h5')

log = logging.getLogger()

def get_frameset():
    """Retrieve some imported data for testing."""
    fs = xp.XanesFrameset(HDF_FILE, edge=xp.k_edges['Ni_NCA'],
                          groupname="NAT1050_Insitu03_p01_OCV")
    return fs

if __name__ == '__main__':
    fs = get_frameset()

    # The ``fit_spectra()`` method contains all the MPI magic
    fs.fit_spectra()
