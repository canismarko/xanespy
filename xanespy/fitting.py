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


from collections import namedtuple

import numpy as np


def prepare_p0(p0, frame_shape, num_timesteps=1):
    """Create an initial parameter guess for fitting.
    
    Takes a starting guess (p0) and returns a numpy array with this
    inital guess that matches the frameset.
    
    For example, if a frameset has 12 timesteps and (1024, 2048) frames,
    then a 5-tuple input for ``p0`` will result in a return value with
    shape (12, 5, 1024, 2048)
    
    """
    # Prepare an empty array for the results
    out_shape = (num_timesteps, *frame_shape, len(p0))
    out = np.empty(shape=out_shape)
    # Now populate the fields and put the param axis in the energy spot
    out[:] = p0
    out = np.moveaxis(out, -1, 1)
    return out


class LinearCombination():
    """Combines other curves into one callable.
    
    The constructor accepts the keyword argument ``sources``, which
    should be a list of numpy arrays. The resulting object can then be
    called with parameters for the weight of each function plus an
    offset. For example, with two sources, the object is called as
    
    .. code:: python
    
        # Prepare the separate sources
        x = np.linspace(0, 2*np.pi, num=361)
        sources = [np.sin(x), np.sin(2*x)]
        
        # Produce a combo with 0.5*sin(x) + 0.25*sin(2x) + 2
        lc = LinearCombination(sources=sources)
        out = lc(0.5, 0.25, 2)
    
    The final output will have the same shape as the sources, which
    should all be the same shape as each other.
    
    """
    name = "linear_combination"
    
    def __init__(self, sources):
        self.sources = sources
    
    def __call__(self, *params):
        # Prepare data and parameters
        out = 0
        p_sources = params[0:-1]
        # Add each source weighted by input parameters
        for coeff, source in zip(p_sources, self.sources):
            out += coeff * source
        # Add global offset
        out += params[-1]
        return out
    
    @property
    def param_names(self):
        names = ['weight_%d' % idx for idx in range(len(self.sources))]
        names.append('offset')
        names = tuple(names)
        return names


class L3Curve():
    """An L_3 absorption edge.
    
    This function is a combination of two Gaussian peaks and a step
    function. The first 3 parameters give the height, position and
    width of one peak, and parameters 3:6 give the same for a second
    peak. Parameters 6:9 are height, position and width of an arctan
    step function. Parameter 9 is a global offset.
    
    Parameters
    ----------
    peaks : int, optional
      How many peaks to fit across the edge.
    
    """
    name = "L3-gaussian"
    
    def __init__(self, energies, num_peaks=2):
        self.num_peaks = num_peaks
        self.params = namedtuple('L3Params', self.param_names)
        self.energies = energies
        self.dtype = energies.dtype
    
    @staticmethod
    def _gauss(x, a, b, c):
        return a * np.exp(-(x-b)**2 / 2 / c**2)
    
    def __call__(self, *params):
        p = self.params(*params)
        Es = self.energies
        # Add two gaussian fields
        out = np.zeros_like(Es)
        for idx in range(self.num_peaks):
            i = 3*idx
            p_i = p[i:i+3]
            out += self._gauss(Es, *p_i)
        # Add arctan step function
        out += p.sig_height * (np.arctan((Es-p.sig_center)*p.sig_sigma) / np.pi + 0.5)
        # Add vertical offset
        out += p.offset
        return out
    
    @property
    def param_names(self):
        pnames = []
        # Add Gaussian parameters
        for idx in range(self.num_peaks):
            pnames.append('height_%d' % idx)
            pnames.append('center_%d' % idx)
            pnames.append('sigma_%d' % idx)
        # Add sigmoid parameters
        pnames.append('sig_height')
        pnames.append('sig_center')
        pnames.append('sig_sigma')
        # Add global y-offset parameter
        pnames.append('offset')
        return tuple(pnames)
