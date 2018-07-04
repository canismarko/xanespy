#!/bin/python

import argparse
import sys
import logging
import os

import h5py

from xanes_frameset import XanesFrameset
from edges import k_edges, l_edges
from qt_frameset_presenter import QtFramesetPresenter
from qt_frame_view import QtFrameView
from qt_map_view import QtMapView

log = logging.getLogger(__name__)

def parse_args(argv):
    # Set up CLI argument parser
    parser = argparse.ArgumentParser(description='A GUI viewer for inspecting xanespy results.')
    parser.add_argument('hdf_filename', metavar='filename', type=str,
                        help='Path to HDF5 file with processed data.')
    parser.add_argument('-g', '--groupname', type=str,
                        help="HDF Groupname to use for frameset")
    # Add arguments for K and L edges
    k_edge_list = str(tuple(k_edges.keys()))
    parser.add_argument('-k', '--k-edge', type=str,
                        help='Name of xanespy K absorption edge'
                        ' %s' % k_edge_list)
    l_edge_list = str(tuple(l_edges.keys()))
    parser.add_argument('-l', '--l-edge', type=str,
                        help='Name of xanespy L absorption edge'
                        ' %s' % l_edge_list)
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Show detailed logging and disable threading.")
    args = parser.parse_args(argv)
    return args


def launch_viewer(argv, Presenter):
    # Validate arguments
    args = parse_args(argv)
    if not os.path.exists(args.hdf_filename):
        sys.exit("File not found: {}".format(os.path.abspath(args.hdf_filename)))
    # Prepare logging
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.DEBUG
        loglevel = logging.WARNING
    logging.basicConfig(level=loglevel)
    # Get a default groupname if necessary
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
    is_threaded = not args.debug
    is_threaded = None
    presenter = Presenter(frameset=fs)
    presenter.create_app()
    presenter.add_map_view(QtMapView(), threaded=is_threaded)
    # This is a temporary workaround until frame_view signals are fixed
    presenter.add_frame_view(QtFrameView(), threaded=is_threaded)
    expand_tree = args.groupname is not None
    presenter.prepare_ui(expand_tree=expand_tree)
    ret = presenter.launch()
    return ret


def main():
    return launch_viewer(argv=None, Presenter=QtFramesetPresenter)


if __name__ == "__main__":
    sys.exit(main())
