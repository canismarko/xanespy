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
# along with Xanespy. If not, see <http://www.gnu.org/licenses/>.

"""Descriptions of X-ray energy absorption edge."""

import math
import warnings

import numpy as np
from pandas import Series
from sklearn import linear_model, svm, utils
from matplotlib.colors import Normalize

import exceptions
from peakfitting import Peak
import plots

import pyximport; pyximport.install()
from xanes_math import normalize_Kedge


class Edge():
    """An X-ray absorption edge. It is defined by a series of energy
    ranges. All energies are assumed to be in units of
    electron-volts. This class is intended to be extended into K-edge,
    L-edge, etc.
    """
    pass


class LEdge(Edge):
    pass


class KEdge(Edge):
    """An X-ray absorption K-edge corresponding to a 2p→3d transition.

    Attributes
    ---------
    E_0: number - The energy of the absorption edge itself.

    *regions: 3-tuples - All the energy regions. Each tuple is of the
      form (start, end, step) and is inclusive at both ends.

    name: string - A human-readable name for this edge (eg "Ni K-edge")

    pre_edge: 2-tuple (start, stop) - Energy range that defines points
      below the edge region, inclusive.

    post_edge: 2-tuple (start, stop) - Energy range that defines points
      above the edge region, inclusive.

    post_edge_order - What degree polynomial to use for fitting
      the post_edge region.

    map_range: 2-tuple (start, stop) - Energy range used for
      normalizing maps. If not supplied, will be determine from pre-
      and post-edge arguments.

    """
    regions = []
    E_0 = None
    pre_edge = None
    post_edge = None
    map_range = None
    post_edge_order = 2
    pre_edge_fit = None

    def annotate_spectrum(self, ax):
        ax.axvline(x=self.edge_range[0], linestyle='-', color="0.55", alpha=0.4)
        ax.axvline(x=self.edge_range[1], linestyle='-', color="0.55", alpha=0.4)

    def normalize(self, spectra, energies):
        """Takes a set of spectra and energies and passes them on to the
        computation module for normalizing by the `normalize_Kedge`
        function."""
        ret = normalize_Kedge(spectra=spectra, energies=energies,
                              pre_edge=self.pre_edge,
                              post_edge=self.post_edge, E_0=self.E_0)
        return ret

    def all_energies(self):
        energies = []
        for region in self.regions:
            start = region[0]
            stop = region[1]
            step = region[2]
            num = int(stop - start) / step + 1
            energies.append(np.linspace(region[0], region[1], num))
        energies = np.concatenate(energies)
        return sorted(list(set(energies)))

    def energies_in_range(self, norm_range=None):
        if norm_range is None:
            norm_range = (self.map_range[0],
                          self.map_range[1])
        energies = [e for e in self.all_energies()
                    if norm_range[0] <= e <= norm_range[1]]
        return energies

    def _post_edge_xs(self, x):
        """Convert a set of x values to a power series up to an order
        determined by self.post_edge_order."""
        X = []
        for power in range(1, self.post_edge_order+1):
            X.append(x**power)
        # Reshape data for make sklearn regression happy
        X = np.array(X)
        if X.shape == (1,):
            # Single value for x
            X = X.reshape(-1, 1)
        elif X.ndim == 1:
            # Single feature (eg [x, x^2])
            X = X.reshape(1, -1)
        elif X.ndim > 1:
            # Multiple values for x
            X = X.swapaxes(0, 1)
        return X

    def fit(self, data: Series):
        """Regression fitting. First the pre-edge is linearlized and the
        extended (post) edge normalized using a polynomial. Pending: a
        step function is fit to the edge itself and any gaussian peaks
        are then added as necessary. This method is taken mostly from
        the Athena manual chapter 4.

        Arguments
        ---------
        data - The X-ray absorbance data. Should be similar to a pandas
          Series. Assumes that the index is energy. This can be a Series of
          numpy arrays, which allows calculation of image frames, etc.
          Returns a tuple of (peak, goodness) where peak is a fitted peak
          object and goodness is a measure of the goodness of fit.

        """
        warnings.warn(UserWarning("KEdge.fit()  not implemented"))
        return data
        # Determine linear background region in pre-edge
        pre_edge = data.ix[self.pre_edge[0]:self.pre_edge[1]]
        self._pre_edge_fit = linear_model.LinearRegression()
        try:
            self._pre_edge_fit.fit(
                X=np.array(pre_edge.index).reshape(-1, 1),
                y=pre_edge.values
            )
        except ValueError:
            raise exceptions.RefinementError
        # Correct the post edge region with polynomial fit
        self._post_edge_fit = linear_model.LinearRegression()
        post_edge = data.ix[self.post_edge[0]:self.post_edge[1]]
        if len(post_edge) > 0:
            x = np.array(post_edge.index)
            self._post_edge_fit.fit(
                X=self._post_edge_xs(x),
                y=post_edge.values
            )

        # Fit the whiteline peak to those values above E_0 in the map range
        normalized = self.normalize(spectrum=data)
        subset = normalized[self.map_range[0]:self.map_range[1]]
        subset = subset[subset > 1]
        # Correct for background
        vertical_offset = 1
        normed_subset = subset - vertical_offset
        # Perform fitting
        peak = Peak(method="gaussian")
        peak.vertical_offset = vertical_offset
        peak.fit(x=normed_subset.index, y=normed_subset.values)
        # Save results
        self.whiteline_peak = peak
        goodness = peak.goodness(subset)
        return (peak, goodness)

    # def calculate_direct_whiteline(self, imagestack, energies, *args, **kwargs):
    #     """Calculates the whiteline position of the absorption edge data
    #     contained in `data`. This method uses the energy of maximum
    #     absorbance and is a faster alternative to `calculate_whiteline`.
    #     The "whiteline" for an absorption K-edge is the energy at which
    #     the specimin has its highest absorbance. This function will return
    #     an 2 arrays with the same shape as each entry in the data
    #     series. 1st array gives the energy of the highest absorbance and
    #     2nd array contains a mock array of goodness of fits (all values
    #     are 1).

    #     Arguments
    #     ---------
    #     data - The X-ray absorbance data. Should be similar to a pandas
    #     Series. Assumes that the index is energy. This can be a Series of
    #     numpy arrays, which allows calculation of image frames, etc.

    #     """
    #     # Calculate the indices of the whiteline
    #     whiteline_indices = np.argmax(imagestack, axis=0)
    #     # Convert indices to energy
    #     map_energy = np.vectorize(lambda idx: energies[idx],
    #                               otypes=[np.float])
    #     whiteline_energies = map_energy(whiteline_indices)
    #     goodness = np.ones_like(whiteline_energies)
    #     return (whiteline_energies, goodness)

    # def normalize(self, spectrum: Series) -> Series:
    #     """Adjust the given spectrum so that the pre-edge is around 0 and the
    #     post-edge is around 1. The `fit()` method should have been
    #     previously called, ideally (though not required) on the same data.
    #     """
    #     # Calculate predicted pre-edge
    #     energies = np.array(spectrum.index)
    #     preedge = self._pre_edge_fit.predict(energies.reshape(-1, 1))
    #     # Calculate predicted absorbance at whiteline
    #     E_0 = self._post_edge_xs(self.E_0)
    #     try:
    #         abs_0 = self._post_edge_fit.predict(E_0)
    #     except utils.validation.NotFittedError:
    #         raise exceptions.RefinementError() from None
    #     pre_E_0 = np.array(self.E_0).reshape(-1, 1)
    #     abs_0 = abs_0 - self._pre_edge_fit.predict(pre_E_0)
    #     # Perform normalization
    #     new_spectrum = (spectrum - preedge) / abs_0
    #     return new_spectrum

    # def plot(self, ax=None):
    #     """Plot this edge on an axes. If the edge has been fit to data, then
    #     this fit will be plotted. Otherwise, just the ranges of the
    #     edge will be shown.
    #     """
    #     if ax is None:
    #         ax = plots.new_axes()
    #     # Find range of values to plot based on edge energies
    #     all_energies = self.all_energies()
    #     xmin = min(all_energies)
    #     xmax = max(all_energies)
    #     x = np.linspace(xmin, xmax, num=50)
    #     # Plot pre-edge line
    #     y = self._pre_edge_fit.predict(x.reshape(-1, 1))
    #     ax.plot(x, y)
    #     # Plot post-edge curve
    #     y = self._post_edge_fit.predict(self._post_edge_xs(x))
    #     ax.plot(x, y)
    #     # Plot whiteline fit if performed
    #     # if self.whiteline_peak is not None:
    #     #     self.whiteline_peak.plot_fit(ax=ax)

class NCANickelLEdge(KEdge):
    E_0 = 853
    regions = [
        (844, 848, 1),
        (849, 856, 0.25),
        (857, 862, 1),
    ]
    pre_edge = (844, 848)
    post_edge = (857, 862)
    map_range = (0, 1)
    _peak1 = 850.91
    _peak2 = 853.16

    # def calculate_direct_map(self, imagestack, energies):
    #     """Return a map with the ratios of intensities at 851 and 853 eV."""
    #     idx1 = energies.index(self._peak1)
    #     idx2 = energies.index(self._peak2)
    #     # Now calculate the peak ratio
    #     peak1 = imagestack[idx1]
    #     peak2 = imagestack[idx2]
    #     ratio = peak2 / (peak1 + peak2)
    #     goodness = (peak1 + peak2)
    #     return (ratio, goodness)

    # def map_normalizer(self, method="direct"):
    #     return Normalize(0, 1)

    # def annotate_spectrum(self, ax):
    #     ax.axvline(x=self._peak1, linestyle='-', color="0.55", alpha=0.4)
    #     ax.axvline(x=self._peak2, linestyle='-', color="0.55", alpha=0.4)

# class NickelKEdge(KEdge):
#     E_0 = 8333
#     regions = [
#         (8250, 8310, 20),
#         (8324, 8344, 2),
#         (8344, 8356, 1),
#         (8356, 8360, 2),
#         (8360, 8400, 4),
#         (8400, 8440, 8),
#         (8440, 8640, 50),
#     ]
#     # pre_edge = (8250, 8325)
#     pre_edge = (8250, 8290)
#     # post_edge = (8352, 8640)
#     post_edge = (8440, 8640)
#     map_range = (8341, 8358)

class LMOMnKEdge(KEdge):
    regions = [
        (6450, 6510, 20),
        (6524, 6542, 2),
        (6544, 6564, 1),
        (6566, 6568, 2),
        (6572, 6600, 4),
        (6610, 6650, 10),
        (6700, 6850, 50),
    ]

class NCANickelKEdge(KEdge):
    E_0 = 8333
    shell = 'K'
    regions = [
        (8250, 8310, 20),
        (8324, 8344, 2),
        (8344, 8356, 1),
        (8356, 8360, 2),
        (8360, 8400, 4),
        (8400, 8440, 8),
        (8440, 8640, 50),
    ]
    pre_edge = (8250, 8290)
    post_edge = (8440, 8640)
    map_range = (8341, 8358)
    edge_range = (8341, 8358)

    # def calculate_direct_map(self, imagestack, energies):
    #     return self.calculate_direct_whiteline(imagestack, energies)


class NCANickelKEdge61(NCANickelKEdge):
    regions = [
        (8250, 8310, 15),
        (8324, 8360, 1),
        (8360, 8400, 4),
        (8400, 8440, 8),
        (8440, 8640, 50),
    ]

class NCANickelKEdge62(NCANickelKEdge):
    regions = [
        (8250, 8310, 15),
        (8324, 8360, 1),
        (8360, 8400, 4),
        (8400, 8440, 8),
        (8440, 8690, 50),
    ]

# Dictionaries make it more intuitive to access these edges by element
k_edges = {
    'Ni_NCA': NCANickelKEdge,
    'Ni_NMC': NCANickelKEdge, # They're pretty much the same
    'Mn_LMO': LMOMnKEdge,
}

l_edges = {
    'Ni_NCA': NCANickelLEdge,
}
