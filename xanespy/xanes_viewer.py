#!/bin/python

import argparse
import sys

from xanes_frameset import XanesFrameset
from edges import l_edges

HDF_FILENAME = '/run/media/mwolf/WOLF_METAPOD/ptycho-2017-05-24-als/LNA_Ptychography_Data.h5'


def get_frameset(name):
    return fs

def main():
    # Set up CLI argument parser
    parser = argparse.ArgumentParser(description='A GUI viewer for inspecting xanespy results.')
    parser.add_argument('hdf_filename', metavar='filename', type=str,
                        help='Path to HDF5 file with processed data.')
    parser.add_argument('-g', '--groupname', type=str,
                        help="HDF Groupname to use for frameset")
    args = parser.parse_args()
    # Create the Frameset
    fs = XanesFrameset(filename=args.hdf_filename,
                       edge=l_edges['Ni_NCA'],
                       groupname=args.groupname)
    return_status = fs.qt_viewer()
    return return_status

if __name__ == "__main__":
    sys.exit(main())
