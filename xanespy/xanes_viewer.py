#!/bin/python

import argparse
import sys

import h5py

from xanes_frameset import XanesFrameset
from edges import k_edges, l_edges
from qt_frameset_presenter import QtFramesetPresenter
from qt_frame_view import QtFrameView
from qt_map_view import QtMapView

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
    parser.add_argument('-k', '--k-edge', type=str,
                        help='Name of xanespy K absorption edge')
    parser.add_argument('-l', '--l-edge', type=str,
                        help='Name of xanespy L absorption edge')
    args = parser.parse_args()
    # Get a default argument if necessary
    if args.groupname is None:
        with h5py.File(args.hdf_filename, mode='r') as f:
            groupname = list(f.keys())[0]
    else:
        groupname = args.groupname
    # Get the K or L-edge object
    if args.k_edge is not None:
        edge = k_edges[args.k_edge]
    elif args.l_edge is not None:
        edge = l_edges[args.l_edge]
    else:
        edge = None
    # Get the xanes frameset object
    fs = XanesFrameset(filename=args.hdf_filename,
                       edge=edge,
                       groupname=groupname)
    # Lauch the Qt viewer
    presenter = QtFramesetPresenter(frameset=fs,
                                    frame_view=QtFrameView())
    presenter.add_map_view(QtMapView())
    expand_tree = args.groupname is not None
    presenter.prepare_ui(expand_tree=expand_tree)
    ret = presenter.launch()


if __name__ == "__main__":
    sys.exit(main())
