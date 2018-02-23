Importing Data into Xanespy
===========================

The first step in any Xanespy workflow will be to **import the raw
data into a common format**. These importer functions are written as
needed: if your preferred beamline is not here, `submit an issue`_.


APS Beamline 32-ID-C
--------------------

The ``energy_scan`` script at 32-ID-C saves source data as an HDF
file. Xanespy preserves the original ("source") file and saves
imported and processed data in a second ("desination") HDF file to be
used in later analysis. The source file can be easily imported:

.. code:: python

   import_aps32idc_xanes_file(filename='source_xanes_example_scan_001.h5',
                              hdf_filename='xanespy_destination.h5',
			      hdf_groupname'example_scan_001',
                              downsample=1, square=True)

The camera at the beamline captures micrographs of 2048x2448. In most
cases this is over-kill since the focusing power of the 60 nm
zone-plate is not strong enough to take advantage of this. The extra
pixel density can be converted into an improved contrast-to-noise
ratio by the **``downsample``** parameter. This parameter controls how
many pixels (to a power of two) are combined. In the previous example,
``downsample=1`` combines to :math:`2^1 = 2\times2` blocks and results
in a (1024, 1224) image. ``downsample=2`` combines :math:`2^2 =
4\times4` blocks to produce a (512, 612) image.

There are sometimes corner artifacts in the camera, independent of
sample position: this can cause poor alignment of the imported
data. An easy solution is to import the data with
``square=True``. This will crop the imported images to be square,
discarding the corner artifacts.

**Operando experiments** generate one HDF file per energy scan. Using
 the above approach would continually overwrite the imported data,
 leaving only the last scan in the import data store. To overcome
 this, the ``timestep`` and ``total_timesteps`` parameters should be
 used, or the related ``import_aps32idc_xanes_files`` function.

Option 1: Loop through the files explicitly:

.. code:: python

   # Populate a list of filenames, eg. with listdir() from the os module
   filenames = []
   # Loop through the files an import then one at a time
   for idx, fname in enumerate(filenames):
       import_aps_32idc_xanes_file(fname, hdf_filename='my_XANES_data.h5',
	                           hdf_groupname='operando_001',
				   timestep=idx, total_timesteps=len(filenames))

Option 2: Convenice function.

.. code::

   # Populate a list of filenames, eg. with listdir() from the os module
   filenames = []
   # The function basically does what is shown in option 1
   import_aps_32idc_xanes_files(filenames, hdf_filename='my_XANES_data.h5',
                                hdf_groupname='operando_002')

Any additional parameters given to the convenience function
:py:func:`~xanespy.importers.import_aps32idc_xanes_files` will be
passed directly to the inner
:py:func:`~xanespy.importers.import_aps32idc_xanes_file` function. If
using option 1, it is important that parameters controlling the data
shape are consistent across calls: ``total_timesteps``, ``append``,
``downsample``, ``square`` and ``exclude``.


SSRL Beamline 6-2c - Directory of XRM Files
--------------------------------------------

In-house XANES scan scripts often save a directory full of ``.xrm``
files with the metadata coding in the filenames. From the Xradia TXM
at SSRL beamline 6-2c, this XANES scan script can be generated with
the in-house generator, and the results can then be imported with
:py:func:`~xanespy.importers.import_ssrl_xanes_dir`. The list of
energies is automatically extracted from the filenames. The reference
frames will also be identified in the directory.

Example usage:

.. code:: python

    import xanespy as xp

    # First a script should be created with sector_8_xanes_script()
    # Once the script is done, import the data with this function
    xp.import_ssrl_xanes_dir("opearando_exp1/",
                             hdf_filename="operando_experiments.h5")


Ptychography from 5.3.2.1 (ALS)
-------------------------------

The output of the nanosurveyor reconstruction algorithm at 5.3.2.1
saves the data in h5
files. :py:func:`~xanespy.importers.import_nanosurveyor_frameset`
copies the reconstructed images and metadata from the individual files
and combines them into a new HDF5 file for XAS analysis. The original
CCD images are left in their original HDF5 files, so they should not
be discarded.

.. code:: python

    import xanespy as xp

    # This function copies the reconstructed images to a new file.
    xp.import_nanosurveyor_frameset('NS_160529047/')

Given the slow nature of ptychography experiments, it may be necessary
to capture an XAS scan into multiple chunks. Passing ``append=True``
to the importer allows **datasets to be combined**:

.. code:: python

    import xanespy as xp
        
    # The first data-set is imported like normal except that the
    # groupname and filename to save under are explicit.
    xp.import_nanosurveyor_frameset('NS_160529047/',
                                    hdf_filename='my_ptycho_data.h5',
                                    hdf_groupname='my_combined_experiment')

    # Now subsequent scans get the ``append=True`` argument
    xp.import_nanosurveyor_frameset('NS_160529048/',
                                    hdf_filename='my_ptycho_data.h5',
                                    hdf_groupname='my_combined_experiment',
				    append=True)
    xp.import_nanosurveyor_frameset('NS_160529049/',
                                    hdf_filename='my_ptycho_data.h5',
                                    hdf_groupname='my_combined_experiment',
				    append=True)

It may be necessary to only import a subset of the frames collected in
a given directory. For example, if the last frame drifted out of the
field-of-view and was re-collected in the next set of energies. The
arguments ``energy_range`` and ``exclude_re`` can be used to fine-tune
the list of importable files. See the documentation for
:py:func:`~xanespy.importers.import_nanosurveyor_frameset` for more
details.

.. _submit an issue: https://github.com/m3wolf/xanespy/issues

.. [#ole] If you're shopping for a container format for your new data
          storage project, I would recommend AGAINST Microsoft
          OLE. This format stores data in raw binary, meaning that you
          need to know the encoding and structure to get meaningful
          data out. Instead, try **HDF5**: a nice open-source, well
          documented, type-aware format with bindings in many
          languages. It even plays nicely with numpy out of the box.


Xradia Image Files (.xrm and .txrm)
-----------------------------------

Xradia microscopes use the Microsoft OLE container format, which is
not easily read [#ole]_. Individual scan files are generally not that
helpful anyway. But in case you need it, there are some adapters to
``.xrm`` and ``.txrm`` files, namely
:py:class:`xanespy.xradia.XRMFile` and
:py:class:`xanespy.xradia.TXRMFile`.

.. note::

   The specification for ``.xrm`` files is not public, so these
   classes are reverse-engineered and may not be (definitely aren't)
   perfect. If you encounter problems, please `submit an issue`_.

Opening xrm or txrm files is best done via the context manager:

.. code:: python

   import xanespy as xp
   import numpy as np

   # Single-image xrm file
   with xp.XRMFile('my_txm_image.xrm') as f:
       img = f.image_data()
       assert img.ndim == 2 # (row, col)

   # Multi-image txrm energy stack file
   with xp.TXRMFile('my_txm_stack.txrm') as f:
       # Get images one at a time by index
       img = f.image_data(idx=0)
       assert img.ndim == 2 # (row, col)

       # Get images all at once in one big array
       stack = f.image_stack()
       assert stack.ndim == 3 # (prj, row, col)
       assert np.array_equal(img, stack[0])

       # Get X-ray energies for the images
       energies = f.energies()
       assert len(energies) == stack.shape[0]

The :py:class:`~xanespy.xradia.XRMFile` and
:py:class:`~xanespy.xradia.TXRMFile` classes accept an optional
``flavor`` keyword argument. This option affects several pieces of
metadata. See the :py:class:`~xanespy.xradia.XRMFile` documentation
for details.


APS Beamline 8-BM-B - Energy Stack (TXRM)
-----------------------------------------

.. note:: The X-ray microscope that was temporarily at beamline 8-BM
          has been returned to NSLS-II. These functions are retained
          for compatibility with previously collected data.

The Xradia microscope can save an entire stack in one ``.txrm``
file. This file can be imported using the
:py:func:`~xanespy.importers.import_aps8bm_xanes_file` function. The
list of energies is automatically extracted from the file. The
reference frames will then reside in a different ``.txrm`` file.

Example usage:

.. code:: python

    import xanespy as xp
    
    xp.import_aps_8BM_xanes_file('exp1-sample-stack.txrm',
                                 ref_filename='exp1-reference_stack.txrm',
  			         hdf_filename='txm-data.h5',
       			         groupname='experiment1')

.. note:: Currently this function can only import one XANES stack;
	  time-resolved measurement is not implemented. If you would
	  find this feature valuable, please `submit an issue`_.
			       

APS Beamline 8-BM-B - Directory of XRM Files
--------------------------------------------

.. note:: The X-ray microscope that was temporarily at beamline 8-BM
          has been returned to NSLS-II. These functions are retained
          for compatibility with previously collected data.

In-house XANES scan scripts often save a directory full of ``.xrm``
files with the metadata coding in the filenames. From the Xradia TXM
at sector 8-BM-B, this XANES scan script can be generated with
:py:func:`~xanespy.beamlines.sector8_xanes_script`, and the results
can then be imported with
:py:func:`~xanespy.importers.import_aps8bm_xanes_dir`. The list of
energies is automatically extracted from the filenames. The reference
frames will also be identified in the directory.

Example usage:

.. code:: python

    import xanespy as xp

    # First a script should be created with sector_8_xanes_script()
    # Once the script is done, import the data with this function
    xp.import_aps_8BM_xanes_dir("opearando_exp1/",
                                hdf_filename="operando_experiments.h5")
