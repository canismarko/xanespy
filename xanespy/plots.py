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

from contextlib import contextmanager
from typing import List, NoReturn

import numpy as np
from matplotlib import pyplot, cm, rcParams, rc_context, style
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

from utilities import pixel_to_xy, Extent, Pixel

@contextmanager
def latexify(styles: List[str]=[], preamble: List[str]=[]):
    """Set some custom options for saving matplotlib graphics in PGF
    format.
    
    Use this as a context manager, along with additional matplotlib styles:
    
    .. code:: python
        
        with xp.latexify(['beamer']):
            plt.plot(...)
            
    
    This will let you add in LaTeX tools and mpl styles together. By
    default, ``siunitx`` and ``mhchem`` packages are
    included. Additional ``\\usepackage`` statements can be included
    using the ``preamble`` parameter.
    
    Parameters
    ==========
    styles : optional
      Additional matplotlib styles in load in the context.
    preamble : optional
      Additional lines to add to the LaTeX preamble.
    
    """
    # Set default LaTeX PGF style
    pgf_with_latex = {                      # setup matplotlib to use latex for output# {{{
        "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
        # "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots 
        "font.sans-serif": [],              # to inherit fonts from the document
        "font.monospace": [],
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts 
            r"\usepackage[T1]{fontenc}",        # plots will be generated
            r"\usepackage{fontspec}",
            r"\usepackage[detect-all,locale=DE,per-mode=reciprocal]{siunitx}",
            r"\usepackage[version=4]{mhchem}",
        ] + preamble,
    }
    # Enter the context library
    with rc_context(rc=pgf_with_latex):
        style.use(styles)
        yield


def remove_extra_spines(ax):
    """Removes the right and top borders from the axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax


def set_axes_color(ax, color):  # pragma: no cover
    """Set the axes, tick marks, etc of `ax` to mpl color `color`. Also,
    "doegreen" has special significance as the color associated with the
    US department of energy."""
    if color == "doegreen":
        color = (33/255, 99/255, 50/255)
    # Set the spine colors
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)
    # Change ticks colors
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)
    # Change label color
    ax.yaxis.label.set_color(color)
    ax.xaxis.label.set_color(color)


def set_outside_ticks(ax):  # pragma: no cover
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


def new_image_axes(height=5, width=5):  # pragma: no cover
    """Square axes with ticks on the outside."""
    ax = new_axes(height, width)
    return set_outside_ticks(ax)


def big_axes():  # pragma: no cover
    """Return a new Axes object, but larger than the default."""
    return new_axes(height=9, width=16)


def dual_axes(fig=None, longdim=13.8, shortdim=6.9, orientation='horizontal'):  # pragma: no cover
    """Two new axes for mapping, side-by-side.

    Parameters
    ----------
    longdim : float
      Size in inches for the long dimension. If orientation is
      "vertical", this will be the height.
    shortdim : float
      Size in inches for the short dimension. If orientation is
      "vertical", this will be the width.
    """
    if fig is None:
        fig = pyplot.figure()
    if orientation == 'vertical':
        fig, (ax1, ax2) = fig.subplots(2, 1)
        fig.set_figwidth(shortdim)
        fig.set_figheight(longdim)
    else:
        fig, (ax1, ax2) = pyplot.subplots(1, 2)
        fig.set_figwidth(longdim)
        fig.set_figheight(shortdim)
    # Remove redundant borders
    remove_extra_spines(ax1)
    remove_extra_spines(ax2)
    # Set background to be transparent
    fig.patch.set_alpha(0)
    return (ax1, ax2)


def draw_histogram_colorbar(ax, *args, **kwargs):  # pragma: no cover
    """Similar to `draw_colorbar()` with some special formatting options
    to put it along the X-axis of the axes."""
    cbar = draw_colorbar(ax=ax, pad=0, orientation="horizontal", energies=None, *args, **kwargs)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off',
        labeltop='off')
    ax.spines['bottom'].set_visible(False)
    cbar.ax.set_xlabel(ax.get_xlabel())
    ax.xaxis.set_visible(False)
    cbar.ax.set_title("")
    cbar.outline.set_visible(False)
    gray = (0.1, 0.1, 0.1)
    cbar.ax.axhline(cbar.ax.get_ylim()[1], linewidth=2, linestyle=":", color=gray)
    cbar.ax.tick_params(
        axis='x',
        which='both',
        bottom='on',
        top='on',
        labelbottom="on",
    )
    return cbar


def draw_colorbar(ax, cmap, norm, energies=None, orientation="vertical",
                  *args, **kwargs):  # pragma: no cover
    """Draw a colorbar on the side of a mapping axes to show the range of
    colors used. Returns the newly created colorbar object.

    Parameters
    ---------
    ax
      Matplotlib axes object against which to plot.
    cmap : str
      String or mpl Colormap instance indicating which colormap to
      use.
    norm : matplotlib.Normalize
      Describes the range of values to use.
    energies : iterable, optional
      Values to put as the tick marks on the colorbar. If not given, 3
      points across ``norm`` will be used.
    orientation : str, optional
      "horizontal" or "vertical" (default)
    
    """
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.arange(0, 3))
    # Prepare default energies if necessary
    if energies is None:
        energies = np.linspace(norm.vmin, norm.vmax, num=3)
    # Add the colorbar to the axes
    cbar = pyplot.colorbar(mappable,
                           ax=ax,
                           ticks=energies,
                           spacing="proportional",
                           orientation=orientation,
                           *args, **kwargs)
    # Annotate the colorbar
    cbar.ax.set_title('eV')
    # Make sure the ticks don't use scientific notation
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()
    return cbar


def plot_pixel_spectra(pixels, extent, spectra, energies, map_ax,
                       spectra_ax=None, step_size=0):  # pragma: no cover
    """Highlight certain pixels in an already-plotted map and plot their
    spectra. The map should already have been plotted.

    Arguments
    ---------

    - pixels : List of 2-tuples with (row, column) positions of pixels
      to highlight and plot.

    - extent : Matplotlib extent specification for converting (y, x)
      positions to (row, column) positions.

    - spectra : Numpy array with spectra that will be plotted. Entries
      in `pixels` will use this array to get spectra. Shape is assumed
      to be (row, column, energy).

    - energies : Numpy array with energies to use as the x-axis for
      plotting spectra.

    - map_ax : The matplotlib Axes that will be used for highlighting
      pixels.

    - spectra_ax : A matplotlib Axes on which to plot the spectra. If
      None (default) a new axes will be created.

    - step_size : An offset to add to each spectrum so they are not
      directly on top of each other.
    """
    # Prepare some metadata
    map_shape = spectra.shape[0:2]
    extent = Extent(*extent)
    # Prepare axes if necessary (and remove spines for arbitrary units)
    if spectra_ax is None:
        spectra_ax = new_axes()
    remove_extra_spines(spectra_ax)
    # Remove ticks on the y axes
    tick_params = {
        'axis': 'y',
        'which': 'both',
        'left': 'off',
        'right': 'off',
        'labelright': 'off',
        'labelleft': 'off',
    }
    # Prepare a secondary y axes if necessary for complex data
    if np.iscomplexobj(spectra):
        spectra_ax2 = spectra_ax.twinx()
        remove_extra_spines(spectra_ax2)
        spectra_ax2.tick_params(**tick_params)
    spectra_ax.tick_params(**tick_params)
    # Do the plotting and annotating for each pixel
    min_val = 0
    max_val = 0
    for idx, px in enumerate(pixels):
        # Put an annotation on the map
        px = Pixel(*px)
        xy = pixel_to_xy(px, extent, shape=map_shape)
        map_ax.text(xy.x, xy.y, str(idx), color="white",
                    horizontalalignment="center", verticalalignment="center",)
        # Normalize the spectrum
        spectrum = spectra[px]
        min_ = np.min(np.abs(spectrum))
        max_ = np.max(np.abs(spectrum))
        spectrum = (spectrum - min_) / (max_ - min_)
        # Plot the spectrum on the spectrum axes
        spectrum += step_size * idx
        plot_spectrum(spectrum, energies, ax=spectra_ax,
                      ax2=spectra_ax2)
        # Put labels on the axes to indicate which specturm is which
        spectra_ax.text(
            x=np.max(energies) + 2,
            y=np.mean((np.real(spectrum), np.imag(spectrum))),
            s=idx,
        )
        # Save values for proper scaling of axes
        min_val = complex(
            min(np.real(min_val), np.real(spectrum).min()),
            min(np.imag(min_val), np.imag(spectrum).min())
        )
        max_val = complex(
            max(np.real(max_val), np.real(spectrum).max()),
            max(np.imag(max_val), np.imag(spectrum).max())
        )
    # Set axes limits based on min and max values
    min_val -= min_val * 0.1
    max_val += max_val * 0.1
    spectra_ax.set_ylim(bottom=np.real(min_val), top=np.real(max_val))
    if spectra_ax2 is not None:
        spectra_ax2.set_ylim(bottom=np.imag(min_val), top=np.imag(max_val))
    spectra_ax.set_xlim(right=np.max(energies)+4)

def plot_kedge_fit(energies, params):
    """Plot the fit based on the given k-edge params. The params will be
    given to ``xanespy.xanes_math.KEdgeParams`` for conversion.

    energies : np.ndarray
      A 1D array with the x values to plot.

    params : array-like
      Fitted K-edge parameters.
    """
    p = KEdgeParams(*params)
    # Plot the sigmoid by making everything else zero
    sig_p = dict(zip(kedge_params, p))
    sig_p['ga'] = 0
    sig_p = KEdgeParams(**sig_p)
    sig = predict_edge(energies, *sig_p)
    pyplot.plot(energies, sig, linestyle=":")
    # Plot the whole predicted line
    fit = predict_edge(energies, *p)
    pyplot.plot(energies, fit, linestyle=":")

def plot_spectrum(spectrum, energies, norm=Normalize(),
                  show_fit=False, ax=None, ax2=None,
                  linestyle=':', color="blue", cmap="plasma",
                  polar_coords=False,
                  *args, **kwargs):  # pragma: no cover
    """Plot an energy spectrum on an axes.
    
    Applies some color formatting if `edge` is a valid XANES Edge
    object.
    
    Arguments
    ---------
    spectrum : np.ndarray
      Array of intensity values.
    energies : np.ndarray
      Array of energy values.
    norm : optional
      Matplotlib Normalize() object that shows the map
      range. This will be used to annotate the plot if it is give.
    show_fit : bool, optional
      Whether to plot lines showing the best fit.
    ax : mpl.Axes, optional
      Matplotlib Axes on which to plot. If not given, a new axes
      will be generated.
    ax2 : mpl.Axes, optional
      A second y-axes for plotting the imaginary component if
      the data are complex.
    linestyle : optional
      Passed on to matplotlib.
    cmap : str, optional
      Colormap, passed on to matplotlib
    color : optional
      Specifies the color for the circles plotted. Either "x"
      or "y" will decide based on the numerical value, `norm` and
      `cmap` arguments. Anything else will be passed as a color spec
      to the matplotlib commands.
    polar_coords : bool, optional
      If truthy, the spectrum will be plotted as modulus-phase instead
      of real-imag. For purely real data this is equivalent to taking
      the absolute value.
    
    """
    if ax is None:
        ax = new_axes()
    if norm is not None:
        norm.autoscale_None(np.real(spectrum))
    # Retrieve `values` in case it's a pandas series
    spectrum = getattr(spectrum, 'values', spectrum)
    # Color code the markers by energy
    is_complex = np.iscomplexobj(spectrum)
    if is_complex:
        colors = None
    elif color == "x":
        colors = cm.get_cmap(cmap)(norm(energies))
    elif color == "y":
        colors = cm.get_cmap(cmap)(norm(spectrum))
    else:
        colors = [color] * len(energies)
    # Remove secondary axes and re-add them for complex values
    if is_complex:
        # Check if modulus-phase or real-imag components
        if polar_coords:
            comp0, comp1 = (np.abs, np.angle)
            label0, label1 = ('Modulus', 'Phase')
        else:
            comp0, comp1 = (np.real, np.imag)
            label0, label1 = ('Real', 'Imag')
        # Create a secondary axis
        if ax2 is None:
            ax2 = ax.twinx()
        # Convert complex values to two lines
        ys = spectrum
        artist = ax.plot(energies, comp0(ys), linestyle=linestyle, color="C0")
        artist.extend(ax2.plot(energies, comp1(ys),
                               linestyle=linestyle, color="C1", *args, **kwargs))
    else:
        # Just plot the real numbers
        ys = np.real(spectrum)
        artist = ax.plot(energies, ys, linestyle=linestyle, *args, **kwargs)
    # save limits, since they get messed up by scatter plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Draw scatter plot of data points
    markersize = kwargs.get('markersize', rcParams['lines.markersize'])
    if is_complex:
        ax.scatter(energies, comp0(ys), c="C0", s=25, alpha=0.5)
        ax.set_ylabel(label0, color="C0")
        for t1 in ax.get_yticklabels():
            t1.set_color("C0")
        ax2.scatter(energies, comp1(ys), c="C1", s=markersize ** 2, alpha=0.5)
        ax2.set_ylabel(label1, color="C1")
        for t1 in ax2.get_yticklabels():
            t1.set_color("C1")
    else:
        ax.scatter(energies, ys, c=colors, s=markersize ** 2)
        ax.set_ylabel('Optical Depth')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Energy /eV')
    # Plot the edges of the map range
    if norm is not None:
        vlineargs = dict(linestyle='--', alpha=0.7,
                         color="lightgrey", zorder=0,)
        ax.axvline(norm.vmin, **vlineargs)
        ax.axvline(norm.vmax, **vlineargs)
    return artist


def plot_composite_map(data, ax=None, origin="upper", *args, **kwargs):  # pragma: no cover
    """Plot an RGB composite map on the given axes."""
    while data.shape[-1] < 3:
        data = np.concatenate((data, np.zeros_like(data)),
                              axis=-1)
    data = data[..., 0:3]
    if ax is None:
        ax = new_image_axes()
    artist = ax.imshow(data, origin=origin, *args, **kwargs)
    return artist


def plot_txm_map(data, edge=None, norm=None, ax=None, cmap='plasma',
                 origin="upper", vmin=None, vmax=None,
                 *args, **kwargs):  # pragma: no cover
    # Get a default normalizer
    if norm is None:
        norm = Normalize(vmin, vmax)
    norm.autoscale_None(data[~np.isnan(data)])
    # Create axes if necessary
    if ax is None:
        ax = new_image_axes()
        if edge is None:
            # No specific edge is given, so use up to 10 evenly space energies
            num_Es = round(min(abs(norm.vmax - norm.vmin + 1), 10))
            energies = np.linspace(norm.vmin, norm.vmax, num=num_Es)
        else:
            energies = edge.energies_in_range(norm_range=(norm.vmin, norm.vmax))
        draw_colorbar(ax=ax, cmap=cmap, norm=norm, energies=energies)
    # Do the plotting
    artist = ax.imshow(data,
                       cmap=cmap,
                       origin=origin,
                       norm=norm,
                       *args, **kwargs)
    # Add annotations and formatting stuff
    return artist


def plot_txm_histogram(data, ax=None, norm=None, bins=None,
                       cmap='plasma', add_cbar=True, *args, **kwargs):  # pragma: no cover
    """Take an array of data values and show a histogram with some
    color-coding related to normalization value.

    Returns: The matplotlib axes object used for plotting.

    Parameters
    ----------
    data : np.ndarray
      An array of values to plot on the histogram.
    ax : optional
      Matplotlib Axes instance to receive the plot. If None, a new axes
      will created.
    norm : optional
      Matplotlib Normalize instance with the colormap range.
    bins : optional
      Bins to pass to the matplotlib hist() routine. If None
      (default), we will choose based on dtype of the data: integers
      will yield 1-wide bins, anything else will give 256 bins.
    cmap : str, optional
      Matplotlib colormap for coloring the bars.
    add_cbar : bool, optional
      Boolean to decide whether to add a colorbar along the
      bottom axis or not.
    *args
      Positional arguments passed to matplotlib's `hist` call.
    *kwargs
      Keyword arguments passed to matplotlib's `hist` call.
    
    """
    if ax is None:
        ax = new_axes()
    # Flatten the data so it can be nicely plotted
    if hasattr(data, 'mask'):
        data = data[~data.mask]
    else:
        data = data.flatten()
    # Remove any Not-a-number values
    data = data[~np.isnan(data)]
    # Set normalizer
    if norm is None:
        norm = Normalize()
        norm.autoscale_None(data)
    # Clip the data so that it includes the end-members
    clip_data = np.clip(data, norm.vmin, norm.vmax)
    unique_vals = np.unique(np.array(clip_data))
    # Determine a reasonable binning parameter based on data type
    if bins is not None:
        pass
    elif np.issubdtype(data.dtype, np.integer):
        # integers, so make the bins 1-wide
        bins = np.arange(norm.vmin, norm.vmax, step=1)
    elif len(unique_vals) < 256:
        # Not many values to choose from, so use the values as bins
        bins = np.sort(unique_vals)
    else:
        bins = 256
    # Plot the histogram
    n, bins, patches = ax.hist(clip_data, bins=bins, *args, **kwargs)
    # Set colors on histogram
    for patch in patches:
        x_position = patch.get_x()
        real_cmap = cm.get_cmap(cmap)
        color = real_cmap(norm(x_position))
        patch.set_color(color)
    # Set axes decorations
    ax.set_ylabel("Pixels")
    ax.set_xlim(norm.vmin, norm.vmax)
    # ax.xaxis.set_ticks(energies)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    # Draw a colorbar along the bottom axes
    if add_cbar:
        cbar = draw_histogram_colorbar(ax=ax, cmap=cmap, norm=norm)
    else:
        cbar = None
    return (ax, cbar)


def plot_spectra(spectra, energies, ax=None):
    """Take an iterable of spectra and plot them one above the next.
    
    Parameters
    ----------
    spectra : iterable
      Each entry is an array with intensity values.
    energies : np.ndarray
      Energy values for the points in each entry of ``spectra``.
    ax : matplotlib.Axes, optional
      The Axes object to receive the plots. If omitted, a new Axes
      will be created.
    
    Returns
    -------
    artists : list
      A list of the matplotlib artists used to draw the spectra.
    
    """
    artists = []
    # Get a default axes if one was not provided
    if ax is None:
        ax = new_axes()
    # Do the plotting for each spectrum
    yoffset = 0
    for spectrum in spectra:
        artist = ax.plot(energies, spectrum + yoffset, marker='o', linestyle=':')
        artists.append(artist)
        # Calculate the appropriate step for the next round
        yoffset += np.median(spectrum) - 0.25 * (np.max(spectrum) - np.min(spectrum))
    return artists


def plot_spectra_as_map(spectra, energies, ax=None, extent=None, **kwargs):
    """Take an iterable of spectra and plot them as a heat map.
    
    This function takes energies so it can put ticks on the x-axis,
    however if they are not equally spaced, their distance on the map
    will not properly capture they're distance numerically.
    
    Parameters
    ----------
    spectra : iterable
      Each entry is an array with intensity values.
    energies : np.ndarray
      Energy values for the points in each entry of ``spectra``.
    ax : matplotlib.Axes, optional
      The Axes object to receive the plots. If omitted, a new Axes
      will be created.
    extent : Matplotlib extent.
    **kwargs : Get passed on to matplotlib imshow().
    
    Returns
    -------
    artists : list
      A list of the matplotlib artists used to draw the spectra.
    
    """
    # Get a default axes if one was not provided
    if ax is None:
        ax = new_axes()
    # Do the plotting for each spectrum
    artist = ax.imshow(spectra, origin='lower', extent=extent, aspect="auto", **kwargs)
    return [artist]


def plot_txm_intermediates(images):  # pragma: no cover
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
