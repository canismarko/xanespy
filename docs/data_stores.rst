Accessing the Data Directly
===========================

While the :py:class:`~xanespy.xanes_frameset.XanesFrameset` class has
methods for common tasks, sometimes it is necessary to access the data
directly, as either numpy arrays or h5py datasets. The xanes_frameset
has a :py:meth:`~xanespy.xanes_frameset.XanesFrameset.store` method
that returns an interface (``TXMStore``) to the underlying HDF5 file.

.. warning:: The ``TXMStore`` created by ``xanes_frameset.store()`` is
   attached to an open HDF5 file. It is strongly recommended to use
   the ``with`` statement described below. Otherwise make sure to call
   the store's ``close()`` method in a ``try...except`` block. File
   corruption is likely if not opened in this manner.

Call the following to get access to the associated datasets. Properties of the interface will
return an HDF5 dataset in most cases.::

  import xanespy as xp
  frameset = xp.XanesFrameset(...)

  # Open the TXMStore interface
  with frameset.store() as store:
      # For example, the images are in (timestep, energy, row, column) order
      assert store.absorbances.shape == (10, 62, 1024, 1024)
      # Energies are in (timestep, energy) order
      assert store.energies.shape == (10, 62)
