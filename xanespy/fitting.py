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
import warnings

import numpy as np
from scipy.optimize import leastsq

from xanes_math import k_edge_jump, iter_indices, foreach


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


def fit_spectra(observations, func, p0, nonnegative=False, quiet=False):
    """Fit a function to a series observations.
    
    The shapes of ``observations`` and ``p0`` parameters must match in
    the first dimension, and the callable ``func`` should take a
    series of parameters (the exact number is determined by the last
    dimension of ``p0``) and return a set of observations (the length
    of which is determined by the last dimension of ``observations``).
    
    Parameters
    ----------
    observations : np.ndarray
      A 1- or 2-dimensional array of observations against which to fit
      the function ``func``.
    func : callable, str
      The function that will be used for fitting. It should match
      ``func(p0, p1, ...)`` where p0, p1, etc are the fitting
      parameters. Some useful functions can be found in the
      ``xanespy.fitting`` module.
    p0 : np.ndarray
      Initial guess for parameters, with similar dimensions to a
      frameset. Example, fitting 3 sources (plus offset) for a (1,
      40, 256, 256) 40-energy frameset requires p0 to be (1, 4,
      256, 256).
    nonnegative : bool, optional
      If true (default), negative parameters will be avoided. This
      can also be a tuple to allow for fine-grained control. Eg:
      (True, False) will only punish negative values in the first
      of the two parameters.
    quiet : bool, optional
      Whether to suppress the progress bar, etc.
    
    Returns
    -------
    params : numpy.ndarray
      The fit parameters (as frames) for each source.
    residuals : numpy.ndarray
      Residual error after fitting, as maps.

    """
    # Massage the datas
    observations = np.array(observations)
    p0 = np.array(p0)
    if observations.ndim == 1:
        residuals = np.zeros(shape=(1,))
    else:
        residuals = np.zeros(shape=(observations.shape[0],))
    params = np.empty_like(p0)
    # Prepare error function
    def errfun(guess, obs):
        if np.any(np.logical_and(guess < 0, nonnegative)):
            # Punish negative values
            diff = np.empty_like(obs)
            diff[:] = 1e6
        else:
            # Calculate error for this guess
            predicted = func(*guess)
            # Compare predicted with observed values
            diff = np.abs(obs - predicted)
        assert not np.any(np.isnan(diff))
        return diff
    # Execute fitting for each spectrum
    indices = iter_indices(observations, desc="Fitting spectra",
                           leftover_dims=1, quiet=quiet)
    def fit_sources(idx):
        spectrum = observations[idx]
        # Don't bother fitting if there's NaN values
        if np.any(np.isnan(spectrum)):
            params[idx] = np.nan
            residuals[idx] = np.nan
            return
        # Valid data, so fit the spectrum
        guess = tuple(p0[idx])
        results = leastsq(func=errfun, x0=guess, args=(spectrum,), full_output=True)
        x, cov_x, infodict, mesg, status = results
        # Status 4 is often a sign of mismatched datatypes.
        if status == 4:
            msg = "Precision errors encountered during fitting. Check dtypes."
            warnings.warn(msg, RuntimeWarning)
        params[idx] = x
        # Calculate residual errors
        res_ = (spectrum - func(*x))
        res_ = np.sqrt(np.mean(np.power(res_, 2)))
        residuals[idx] = res_
    foreach(fit_sources, indices, threads=1)
    return (params, residuals)


class Curve():
    """Base class for a callabled Curve."""
    name = "curve"
    param_names = ()


class LinearCombination(Curve):
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


class Gaussian(Curve):
    """A Gaussian curve.
    
    Mathematically:
    
    .. math::
        a e^{\frac{-(x-b)**2}{2c^2)}
    
    Parameters
    ----------
    x : np.ndarray
      Array of x-values to input into the Gaussian function.
    
    """
    name = "gaussian"
    param_names = ('height', 'center', 'width')
    
    def __init__(self, x):
        self.x = x
    
    def __call__(self, height, center, width):
        x = self.x
        a, b, c = (height, center, width)
        return a * np.exp(-(x-b)**2 / 2 / c**2)


class L3Curve(Curve):

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
    
    def __call__(self, *params):
        p = self.params(*params)
        Es = self.energies
        # Add two gaussian fields
        out = np.zeros_like(Es)
        gaussian = Gaussian(x=Es)
        for idx in range(self.num_peaks):
            i = 3*idx
            p_i = p[i:i+3]
            out += gaussian(*p_i)
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


class KCurve(Curve):
    param_names = (
        'scale', 'voffset', 'E0',  # Global parameters
        'sigw',  # Sharpness of the edge sigmoid
        'bg_slope', # Linear reduction in background optical_depth
        'ga', 'gb', 'gc',  # Gaussian height, center and width
    )
    
    def __init__(self, energies):
        self.energies = energies
    
    def guess_params(self, intensities, edge):
        """Guess initial starting parameters for a k-edge curve. This will
        give a rough estimate, appropriate for giving to the fit_kedge
        function as the starting parameters, p0.
        
        Arguments
        ---------
        intensities : np.ndarray
          An array containing optical_depth data that represents a
          K-edge spectrum. Only 1-dimensional data are currently
          accepted.
        edge : xanespy.edges.KEdge
          An X-ray Edge object, will be used for estimating the actual
          edge energy itself.
        
        Returns
        -------
        p0 : tuple
          A named tuple with the estimated parameters (see KEdgeParams
          for definition)
        
        """
        Is = intensities
        assert Is.shape == self.energies.shape
        # Guess the overall scale and offset parameters
        scale = k_edge_jump(frames=Is, energies=self.energies, edge=edge)
        voffset = np.min(Is)
        # Estimate the edge position
        E0 = edge.E_0
        # Estimate the whiteline Gaussian parameters
        ga = 5 * (np.max(Is) - scale - voffset)
        gb = self.energies[np.argmax(Is)] - E0
        gc = 2  # Arbitrary choice, should improve this in the future
        # Construct the parameters tuple
        KParams = namedtuple('KParams', self.param_names)
        p0 = KParams(scale=scale, voffset=voffset, E0=E0,
                             sigw=0.5, bg_slope=0,
                             ga=ga, gb=gb, gc=gc)
        return p0
    
    def __call__(self, *params):
        # Named tuple to help keep track of parameters
        Params = namedtuple('Params', self.param_names)
        p = Params(*params)
        x = self.energies
        # Adjust the x's to be relative to E_0
        x = x - p.E0
        # Sigmoid
        sig = np.arctan(x*p.sigw) / np.pi + 1/2
        # Gaussian
        gaus = p.ga*np.exp(-(x-p.gb)**2/2/p.gc**2)
        # Background
        bg = x * p.bg_slope
        curve = sig + gaus + bg
        curve = p.scale * curve + p.voffset
        return curve
