Importing Data into Xanespy
===========================

The first step in any Xanespy workflow will be to **import the raw
data into a common format**. These importer functions are written as
needed: if your preferred beamline is not here, `submit an issue`_.


APS Beamline 8-BM-B - Energy Stack (TXRM)
-----------------------------------------

.. note:: Currently this function can only import one XANES
   stack. Time-resolved measurement is not implemented. If you would
   find this feature valuable, please `submit an issue`_.

The Xradia microscope can save an entire stack in one ``.txrm``
file. This file can be imported using the
:py:func:`~xanespy.importers.import_aps_8BM_xanes_file` function. The
list of energies is automatically extracted from the file. The
reference frames will then reside in a different ``.txrm`` file.

Example usage:

.. code:: python

    import xanespy as xp
    
    xp.import_aps_8BM_xanes_file('exp1-sample-stack.txrm',
                                 ref_filename='exp1-reference_stack.txrm',
  			         hdf_filename='txm-data.h5',
       			         groupname='experiment1')
			       

APS Beamline 8-BM-B - Directory of XRM Files
--------------------------------------------

In-house XANES scan scripts often save a directory full of ``.xrm``
files with the metadata coding in the filenames. From the Xradia TXM
at sector 8-BM-B, this XANES scan script can be generated with
:py:func:`~xanespy.beamlines.sector8_xanes_script`, and the results
can then be imported with
:py:func:`~xanespy.importers.import_aps_8BM_xanes_dir`. The list of
energies is automatically extracted from the filenames. The reference
frames will also be identified in the directory.

Example usage:

.. code:: python

    import xanespy as xp

    # First a script should be created with sector_8_xanes_script()
    # Once the script is done, import the data with this function
    xp.import_aps_8BM_xanes_dir("opearando_exp1/",
                                hdf_filename="operando_experiments.h5")


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
files. :py:func:`~xanespy.importers.import_nanosurveyor_frameset``
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
