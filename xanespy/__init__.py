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
# along with Xanespy.  If not, see <http://www.gnu.org/licenses/>.

__version__ = "0.1.1"

import sys
import os

# Make sure this directory is in python path for imports
sys.path.append(os.path.dirname(__file__))

import exceptions
import xradia
from xradia import XRMFile, TXRMFile
import beamlines
import sxstm
import fitting
from fitting import L3Curve, LinearCombination, prepare_p0, Gaussian
from beamlines import (sector8_xanes_script, ZoneplatePoint,
                       Zoneplate, Detector, DetectorPoint)
from importers import (import_frameset,
                       import_ssrl_xanes_dir,
                       import_aps4idc_sxstm_files,
                       import_aps8bm_xanes_dir,
                       import_aps8bm_xanes_file,
                       import_nanosurveyor_frameset,
                       import_cosmic_frameset,
                       import_stxm_frameset,
                       import_aps32idc_xanes_files,
                       import_aps32idc_xanes_file)
from xanes_frameset import XanesFrameset
import edges
from edges import k_edges, l_edges, Edge, KEdge, LEdge
from plots import (dual_axes, new_axes, new_image_axes, plot_txm_map,
                   set_axes_color, plot_pixel_spectra,
                   plot_txm_histogram, plot_spectrum, latexify,
                   remove_extra_spines, plot_kedge_fit)
import plots
import xanes_math as xanes_math
import utilities
from utilities import get_component, xy_to_pixel, pixel_to_xy, xycoord, Pixel
# from qt_frameset_presenter import QtFramesetPresenter
# from qt_frame_view import QtFrameView
# from qt_map_view import QtMapView
