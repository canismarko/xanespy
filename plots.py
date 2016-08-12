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

"""Helper functions for setting up and displaying plots using matplotlib."""

import numpy as np
from matplotlib import pyplot, cm
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter


class ElectronVoltFormatter(ScalarFormatter):
    """Matplotlib formatter for showing energies as electon-volts."""
    def __call__(self, *args, **kwargs):
        formatted_value = super().__call__(*args, **kwargs)
        formatted_value = "{value} eV".format(value=formatted_value)
        return formatted_value


def remove_extra_spines(ax):
    """Removes the right and top borders from the axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax


def set_outside_ticks(ax):
    """Convert all the axes so that the ticks are on the outside and don't
    obscure data."""
    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    return ax


def new_axes(height=5, width=None):
    """Create a new set of matplotlib axes for plotting. Height in inches."""
    # Adjust width to accomodate colorbar
    if width is None:
        width = height / 0.8
    fig = pyplot.figure(figsize=(width, height))
    # Set background to be transparent
    fig.patch.set_alpha(0)
    # Create axes
    ax = pyplot.gca()
    # Remove borders
    remove_extra_spines(ax)
    return ax


def new_image_axes(height=5, width=5):
    """Square axes with ticks on the outside."""
    ax = new_axes(height, width)
    return set_outside_ticks(ax)


def big_axes():
    """Return a new Axes object, but larger than the default."""
    return new_axes(height=9, width=16)


def xrd_axes():
    """Return a new Axes object, with a size appropriate for display x-ray
    diffraction data."""
    return new_axes(width=8)


def dual_axes(fig=None, orientation='horizontal'):
    """Two new axes for mapping, side-by-side."""
    if fig is None:
        fig = pyplot.figure()
    if orientation == 'vertical':
        fig, (ax1, ax2) = fig.subplots(2, 1)
        fig.set_figwidth(6.9)
        fig.set_figheight(13.8)
    else:
        fig, (ax1, ax2) = pyplot.subplots(1, 2)
        fig.set_figwidth(13.8)
        fig.set_figheight(5)
    # Remove redundant borders
    remove_extra_spines(ax1)
    remove_extra_spines(ax2)
    # Set background to be transparent
    fig.patch.set_alpha(0)
    return (ax1, ax2)


def draw_colorbar(ax, cmap, norm, energies):
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.arange(0, 3))
    # Add the colorbar to the axes
    cbar = pyplot.colorbar(mappable,
                           ax=ax,
                           ticks=energies,
                           spacing="proportional")
    # Annotate the colorbar
    cbar.ax.xaxis.get_major_formatter().set_useOffset(False)
    cbar.ax.set_title('eV')
    return cbar


def plot_xanes_spectrum(spectrum, energies, norm=Normalize(),
                        show_fit=False, ax=None, linestyle=':',
                        cmap="plasma"):
    """Plot a XANES spectrum on an axes. Applies some color formatting if
    `edge` is a valid XANES Edge object."""
    if ax is None:
        ax = new_axes()
    norm.autoscale_None(spectrum)
    print('out here')
    # Color code the markers by energy
    colors = cm.get_cmap(cmap)(norm(energies))
    ax.plot(energies, spectrum, linestyle=linestyle)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if show_fit:
        # Plot the predicted values from edge fitting
        edge.plot(ax=ax)
    scatter = ax.scatter(energies, spectrum, c=colors, s=25)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Energy /eV')
    ax.set_ylabel('Absorbance')
    return scatter


def plot_txm_map(data, edge, norm=None, ax=None, cmap='plasma',
                 origin="lower", *args, **kwargs):
    # Get a default normalizer
    if norm is None:
        norm = Normalize()
        norm.autoscale_None(data)
    # Create axes if necessary
    if ax is None:
        ax = new_image_axes()
        energies = edge.energies_in_range(norm_range=(norm.vmin, norm.vmax))
        draw_colorbar(ax=ax, cmap=cmap, norm=norm, energies=energies)
    # Do the plotting
    artist = ax.imshow(data,
                       cmap=cmap,
                       origin=origin,
                       norm=norm,
                       *args, **kwargs)
    # Add the colorbar
    # Add annotations and formatting stuff
    return artist


def plot_txm_histogram(data, ax=None, norm=None, bins=100, cmap='plasma'):
    """Take an array of data values and show a histogram with some
    color-coding related to normalization value.
    """
    if ax is None:
        ax = plots.new_axes()
    # Flatten the data so it can be nicely plotted
    data = data.flatten()
    # Set normalizer
    if norm is None:
        norm = Normalize()
        norm.autoscale_None(data)
    # norm = Normalize(norm_range[0], norm_range[1])
    # masked_map = self.frameset.masked_map(goodness_filter=goodness_filter)
    # mask = masked_map.mask
    # # Add a bin for any above and below the range
    # edge = self.frameset.edge()
    # # energies = self.frameset.edge().energies_in_range(norm_range=norm_range)
    # # energies = [
    # #     2 * energies[0] - energies[1],
    # #     *energies,
    # #     2 * energies[-1] - energies[-2]
    # # ]
    # clipped_map =  np.clip(masked_map, edge.map_range[0], edge.map_range[1])
    n, bins, patches = ax.hist(data, bins=bins)
    # Set colors on histogram
    for patch in patches:
        x_position = patch.get_x()
        real_cmap = cm.get_cmap(cmap)
        color = real_cmap(norm(x_position))
        patch.set_color(color)
    # Set axes decorations
    ax.set_xlabel("Whiteline position /eV")
    ax.set_ylabel("Pixels")
    ax.set_xlim(norm.vmin, norm.vmax)
    # ax.xaxis.set_ticks(energies)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    return ax


def plot_txm_intermediates(images):
    """Accept a dictionary of images and plots them each on its own axes
    using matplotlib's `imshow`. This is a complement to routines that
    operate on a microscopy frame and optionally return all the
    intermediate calculated frames.

    """
    for key in images.keys():
        ax1, ax2 = dual_axes()
        ax1.imshow(images[key], cmap='gray')
        ax1.set_title(key)
        ax2.hist(images[key].flat, bins=100)
