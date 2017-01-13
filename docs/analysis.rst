Analyzing the Data
==================

Frame Alignment
---------------

In order to acquire reliable spectra, **it is important that the
frames be aligned properly**. Thermal expansion, motor slop, sample
damage and imperfect microscope alignment can all cause frames to be
misaligned. **It is often necessary to align the frames before
performing any of the subsequent steps.**

This is done with the ``xanespy.XanesFrameset().align_frames()`` method::

  import xanespy
  # Select an imported hdf file to use
  frameset = xanespy.XanesFrameset(hdf_filename="...")
  frameset.fork_data_group('aligned')
  # Run through five passes of the default phase correlation
  frameset.align_frames(passes=5, plot_results=True)

.. figure:: images/alignment-boxplot.svg
   :alt: Box and whisker plot of translations.

   With the ``plot_results`` argument, a box and whisker plot is
   generated showing the distribution of corrections needed for
   aligning each frame. Several passes help reduce the error.

The alignments are generally done with subpixel resolution, which
gives improved accuracy, but requires interpolation. To avoid problems
with accumulated error, a cumulative translation matrix is kept and
applied at the end to the original data. You can add your own
translation manually using the ``stage_transformations()`` method. If
``align_frames()`` is called with ``commit=False``, then the alignment
parameters are added to ``stage_transformations`` but not
applied. Once all transformations are staged, the
``apply_transformations()`` method will apply the cumulative
transformation matrix and (by default) save the result to disk.

If the starting alignment is particularly sporadic, a false minimum
can result in an exception or a very small image that doesn't provide
useful information. In these cases, it may be necessary to first stage
a template registration then perform several passes of phase
correlation::

  fs = XanesFrameset(hdf_filename="...")
  # Eg. use the 22nd energy and a range of the image as the template
  template = fs.frames()[21, 110:425, 150:450]
  plt.imshow(template, cmap="gray")

  fs.fork_data_group('aligned')

  fs.align_frames(method="template_match", template=template, commit=False)
  fs.align_frames(passes=5, commit=True)


Subtracting Surroundings
------------------------

Some microscopes show differences in the absorbance of the whole
frame, including background material. This can be removed from each
frame, giving a better spectrum::

  fs = XanesFrameset(hdf_filename="...")
  fs.subtract_surroundings()

.. figure:: images/subtract-surroundings.svg
   :alt: Spectrum showing before and after subtract_surroundings

   The effect of the ``subtract_surroundings()`` method.

Spectrum Fitting - K-Edge
-------------------------

Spectrum Fitting - L-Edge
-------------------------
