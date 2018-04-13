Xanespy
=======

.. image:: https://travis-ci.org/canismarko/xanespy.svg?branch=master
   :target: https://travis-ci.org/canismarko/xanespy
   :alt: Build status
.. image:: https://coveralls.io/repos/github/m3wolf/xanespy/badge.svg?branch=master
   :target: https://coveralls.io/github/m3wolf/xanespy?branch=master
   :alt: Test coverage status
.. image:: https://readthedocs.org/projects/xanespy/badge/?version=latest
   :target: http://xanespy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


Xanespy is a library for analyzing X-ray absorbance spectroscopy data
for materials science, chemistry and similar fields. The major focus
is on X-ray microscopy frames collected at multiple energies and over
multiple timesteps. However, a design goal is for most (if not all) of
the calculation routines to be applicable to bulk XAS data as well.


Motivation & Features
---------------------

- Importing and analysis of transmission X-ray microscopy framesets
- Analysis of X-ray spectroscopy data (K-edge XANES and L-edge)
- Efficient analysis of large operando datasets


Installation
------------

Xanespy can be installed from the **python package index (PyPI) using pip**

.. code:: bash

   $ pip install xanespy


Usage
-----

Each beamline generally has its own importer in the
`xanespy.importers` module, which will process the data and create an
HDF5 file ready for the rest of xanespy to use

.. code:: python

    import xanespy
    
    # Example for importing from SSRL 6-2c
    xanespy.import_ssrl_frameset('<data_dir>',
                                 hdf_filename='imported_data.h5')
    
    # Load a pre-defined XAS edge or create your own subclass xanespy.Edge
    edge = xanespy.k_edges['Ni_NCA']
    # Now load the newly created HDF5 file and the X-ray absorbance edge
    fs = xanespy.XanesFrameset(filename='imported_data.h5', edge=edge)
    
    # Perform automatic frame alignment
    fs.align_frames(passes=5)
    # Fit the absorbance spectra and extract the edge position (SLOW!)
    fs.fit_spectra()
    
    # Inspect the result with the built-in Qt5 GUI
    fs.qt_viewer()


Beamlines
---------

We currently have importers for data from the following beamlines. If
you would like support a beamline that is not listed and are willing
to share some test data publicly, please `open an
issue`_.

.. _open an issue: https://github.com/m3wolf/xanespy/issues

- 8-BM-B - Advanced Photon Source
- 6-2c - Stanford Synchrotron Radiation Lightsource
- 5.3.2.1 - Advanced Light Source (ptychography)


Ptychography
------------

Xanespy has support for ptychography data from ALS beamline
5.3.2.1. Use ``xanespy.import_nanosurveyor_frameset`` to import the .cxi
files and ``xanespy.PtychoFrameset`` to load the data.


Extended X-ray Absorbance Fine Structure (EXAFS)
------------------------------------------------

Currently EXAFS analysis is NOT supported. While it would be nice to
someday include this, this technique generally requires heavy user
involvement and does not lend itself to batch processing; if you have
expertise in automated analysis of EXAFS data, please get in touch.


License
-------

This project is released under the `GNU General Public License version 3`_.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

.. _GNU General Public License version 3: https://www.gnu.org/licenses/gpl-3.0.en.html


Acknowledgement
---------------

This work was supported as part of the NorthEast Center for Chemical
Energy Storage (`NECCES`_), an Energy Frontier Research Center funded
by the U.S. Department of Energy, Office of Science, Basic Energy
Sciences under Award # DE-SC0012583.

.. _NECCES: http://binghamton.edu/necces/
