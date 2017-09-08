# -*- coding: utf-8 -*-
#
# Copyright © 2017 Mark Wolf
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

"""A collection of callables that can be used for fitting spectra
against."""


from collections import namedtuple as nt

import numpy as np

class L3Edge():
    """An L_3 absorption edge.
    
    This function is a combination of two Gaussian peaks and a step
    function. The first 3 parameters give the height, position and
    width of one peak, and parameters 3:6 give the same for a second
    peak. Parameters 6:9 are height, position and width of an arctan
    step function. Parameter 9 is a global offset.
    
    """
    param_names = (
        'a0', 'b0', 'c0', # First gaussian
        'a1', 'b1', 'c1', # Second gaussian
        'sigh', 'sigb', 'sigw', # Arc-tan step function
        'offset' # Global offset
    )
    
    def __init__(self, energies):
        self.params = nt('L3Params', self.param_names)
        self.energies = energies
    
    @staticmethod
    def _gauss(x, a, b, c):
        return a * np.exp(-(x-b)**2 / 2 / c**2)
    
    def __call__(self, *params):
        p = self.params(*params)
        Es = self.energies
        # Add two gaussian fields
        ODs = self._gauss(Es, p.a0, p.b0, p.c0)
        ODs += self._gauss(Es, p.a1, p.b1, p.c1)
        # Add arctan step function
        ODs += p.sigh * np.arctan((Es-p.sigb)*p.sigw) / np.pi + 1/2
        # Add vertical offset
        ODs += p.offset
        return ODs
