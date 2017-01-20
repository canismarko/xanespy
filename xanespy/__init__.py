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

__version__ = "0.1.0"

import sys
import os

# Make sure this directory is in python path for imports
sys.path.append(os.path.dirname(__file__))

import exceptions
import xradia
from importers import (import_frameset, import_ssrl_frameset,
                       import_aps_8BM_frameset,
                       import_nanosurveyor_frameset)
from xanes_frameset import XanesFrameset, PtychoFrameset
from edges import k_edges, l_edges
from plots import (dual_axes, new_axes, new_image_axes, plot_txm_map,
                   set_axes_color, plot_pixel_spectra, plot_txm_histogram,
                   plot_xanes_spectrum)
import plots
import xanes_math as xanes_math
import utilities
# from qt_frameset_presenter import QtFramesetPresenter
# from qt_frame_view import QtFrameView
# from qt_map_view import QtMapView
