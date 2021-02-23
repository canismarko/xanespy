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

from typing import List, Tuple

import numpy as np

from .xanes_math import k_edge_mask, l_edge_mask, normalize_k_edge


class Edge():
    """A definition for  an element's X-ray absorption edge.
    
    It is defined by a series of energy ranges. All energies are
    assumed to be in units of electron-volts. This class is intended
    to be extended into K-edge, L-edge, etc. ``pre_edge`` and
    ``post_edge`` are used for fitting and applying edge jump filters,
    etc.
    
    Attributes
    ----------
    E_0 : float
      The energy of the absorption edge itself.
    regions: list of 3-tuples
      All the energy regions. Each tuple is of the form (start, end,
      step) and is inclusive at both ends.
    name : string
      A human-readable name for this edge (eg "Ni K-edge")
    pre_edge: 2-tuple
      Energy range (start, stop) that defines points below the edge
      region, inclusive.
    post_edge: 2-tuple
      Energy range (start, stop) that defines points above the edge
      region, inclusive.
    edge_range : 2-tuple
      Energy range (start, stop) used to determine the official
      beginning and edge of the edge itself.
    
    """
    regions = [] # type: List[Tuple[float, float, float]]
    E_0 = None # type: float
    pre_edge = None # type: Tuple[float, float]
    post_edge = None # type: Tuple[float, float]
    edge_range = None # type: Tuple[float, float]
    
    def all_energies(self):
        """Combine all the regions into one array.
        
        Returns
        -------
        energies : list
          Flat array with the energies for this edge.
        """
        energies = []
        if self.regions == []:
            energies = np.linspace(self.edge_range[0], self.edge_range[1], num=20)
        else:
            for region in self.regions:
                start = region[0]
                stop = region[1]
                step = region[2]
                rng = int(stop - start)
                num = int(rng / step + 1)
                energies.append(np.linspace(region[0], region[1], num))
            energies = np.concatenate(energies)
            energies = sorted(list(set(energies)))
        return energies
    
    def mask(self, frames, *args, **kwargs):
        return np.zeros(shape=frames.shape[-2:])
    
    def normalize(self, spectrum, energies):
        raise NotImplementedError()
    
    def annotate_spectrum(self, ax):
        pass


class LEdge(Edge):
    """An X-ray absorption K-edge corresponding to a 2s or 2p
    transition.
    
    """
    shell = 'L'
    def annotate_spectrum(self, ax):
        """Draw lines on the axes to indicate the position of the edge."""
        ax.axvline(x=np.max(self.pre_edge), linestyle='-', color="0.55",
                   alpha=0.4)
        ax.axvline(x=np.min(self.post_edge), linestyle='-', color="0.55",
                   alpha=0.4)
        return ax
    
    def mask(self, *args, **kwargs):
        """Return a numpy array mask for material that's active at this
        edge. Calculations are done in `xanes_math.l_edge_mask()`.
        
        """
        return l_edge_mask(*args, edge=self, **kwargs)


class KEdge(Edge):
    shell = 'K'
    """An X-ray absorption K-edge corresponding to a 1s transition."""
    
    def annotate_spectrum(self, ax):
        """Draw lines on the axes to indicate the position of the edge."""
        ax.axvline(x=self.edge_range[0], linestyle='-', color="0.55",
                   alpha=0.4)
        ax.axvline(x=self.edge_range[1], linestyle='-', color="0.55",
                   alpha=0.4)
        return ax
    
    def mask(self, *args, **kwargs):
        """Return a numpy array mask for material that's active at this
        edge. Calculations are done in ``xanes_math.l_edge_mask()``.
        
        """
        return k_edge_mask(*args, edge=self, **kwargs)
    
    def normalize(self, spectrum, energies):
        """Normalize so that pre- and post-edges scale to 0 and 1."""
        return normalize_k_edge(spectrum, energies, edge=self)


class NCACobaltLEdge(LEdge):
    name = "Co_NCA"
    E_0 = 793.2
    regions = [
        (770, 775, 1.0),
        (775, 785, 0.5),
        (785, 790, 1),
    ]
    pre_edge = (770, 775)
    post_edge = (785, 790)
    edge_range = (775, 785)
    _peak1 = 780.5


class NCANickelLEdge(LEdge):
    name = "Ni_NCA"
    E_0 = 853
    regions = [
        (844, 848, 1),
        (849, 856, 0.25),
        (857, 862, 1),
    ]
    pre_edge = (844, 848)
    post_edge = (857, 862)
    edge_range = (848, 857)
    _peak1 = 850.91
    _peak2 = 853.16


class LMOMnKEdge(KEdge):
    name = "Mn_LMO"
    regions = [
        (6450, 6510, 20),
        (6524, 6542, 2),
        (6544, 6564, 1),
        (6566, 6568, 2),
        (6572, 6600, 4),
        (6610, 6650, 10),
        (6700, 6850, 50),
    ]
class FeKEdge(KEdge):
    name = "Fe"
    E_0 = 7100.0
    shell = "K"
    regions = [
        (7100, 7110, 3),
        (7110, 7117, 1),
        (7117, 7130, 5),
        (7130, 7200, 5),
    ]
    pre_edge = (7100, 7108)
    post_edge = (7150, 7250)
    edge_range = (7115, 7140)

class GeKEdge(KEdge):
    name = "Ge"
    E_0 = 11100.0
    shell = "K"
    regions = [
        (11050, 11075, 5),
        (11075, 11150, 1.5),
        (11150, 11300, 4),
    ]
    pre_edge = (11050, 11075)
    post_edge = (11075, 11150)
    edge_range = (11150, 11300)
    map_range = (11050, 11300)


class NCACobaltKEdge(KEdge):
    name = "Co_NCA"
    E_0 = 7712
    shell = 'K'
    pre_edge = (7600, 7715)
    post_edge = (7780, 7900)
    edge_range = (7715, 7740)


class CuKEdge(KEdge):
    name = 'Cu'
    E_0 = 8978.9
    shell = "K"
    pre_edge = (8940, 8970)
    post_edge = (9010, 9200)
    edge_range = (8970, 9010)


class NCANickelKEdge(KEdge):
    name = "Ni_NCA"
    E_0 = 8345
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
    pre_edge = (8249, 8320)
    post_edge = (8380, 8500)
    edge_range = (8341, 8360)


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


class NMCNickelKEdge29(NCANickelKEdge):
    regions = [
        (8250, 8310, 20),
        (8324, 8346, 6),
        (8346, 8358, 1),
        (8360, 8400, 10),
        (8400, 8440, 15),
        (8440, 8640, 100),
    ]


class OKEdge(KEdge):
    name = 'O'
    E_0 = 530
    shell = "K"
    pre_edge = (525, 528)
    post_edge = (545, 550)
    edge_range = (537, 545)
    map_range = (528, 537)

# Dictionaries make it more intuitive to access these edges by element
k_edges = {
    'Fe': FeKEdge(),
    'Ni_NCA': NCANickelKEdge(),
    'Ni_NMC': NCANickelKEdge(), # They're pretty much the same
    'Co_NCA': NCACobaltKEdge(),
    'Mn_LMO': LMOMnKEdge(),
    'Ni': NCANickelKEdge(),
    'Cu': CuKEdge(),
    'Ge': GeKEdge()
}


l_edges = {
    'Co_NCA': NCACobaltLEdge(),
    'Ni_NCA': NCANickelLEdge(),
}
