import numpy as np

from skimage import transform, feature


def direct_whitelines(spectra, energies, edge):
    """Takes an array of X-ray absorbance spectra and calculates the
    positions of maximum intensities over the near-edge region.

    Arguments
    ---------

    - spectra : 2D numpy array of absorbance spectra where the last
      dimension is energy.

    - energies : 1D array of X-ray energies in electron-volts.

    - edge : An XAS Edge object that describes the absorbance edge in
      question.
    """
    # Cut down to only those values on the edge
    edge_mask = (energies > edge.edge_range[0]) & (energies < edge.edge_range[1])
    spectra = spectra[...,edge_mask]
    energies = energies[edge_mask]
    # Calculate the whiteline positions
    whiteline_indices = np.argmax(spectra, axis=-1)
    map_energy = np.vectorize(lambda idx: energies[idx],
                              otypes=[np.float])
    whitelines = map_energy(whiteline_indices)
    # Return results
    return whitelines

def transform_images(data, translations=None, rotations=None,
                     scales=None, out=None, mode='constant'):
    """Takes an array of images and applies each translation, rotation and
    scale. It is assumed that the first dimension of data is the same
    as the length of translations, rotations and scales. Data will be
    written to `out` if given, otherwise returned as a new array.
    """
    # Create a new array if one is not given
    if out is None:
        out = np.zeros_like(data)
    # Loop through the images and apply each transformation
    for imidx in range(data.shape[0]):
        # Get transformation parameters if given
        scale = scales[imidx] if scales is not None else None
        translation = translations[imidx] if translations is not None else None
        rot = rotations[imidx] if rotations is not None else None
        # Prepare and execute the transformation
        transformation = transform.SimilarityTransform(
            scale=scales[imidx] if scales is not None else None,
            translation=translations[imidx] if translations is not None else None,
            rotation=rotations[imidx] if rotations is not None else None,
        )
        out[imidx] = transform.warp(data[imidx], transformation,
                                    order=3, mode=mode)
    return out


def register_correlations(frames, reference, upsample_factor=10):
    """Calculate the relative translation between the reference image and
    each image in `frames` using a modified cross-correlation algorithm.

    Returns: Array with same dimensions as 0th axis of `frames`
    containing (x, y) translations for each frame.

    """
    translations = np.zeros(shape=(frames.shape[0], 2))
    for imidx in range(frames.shape[0]):
        results = feature.register_translation(reference,
                                               frames[imidx],
                                               upsample_factor=upsample_factor)
        shift, error, diffphase = results
        # Convert (row, col) to (x, y)
        translations[imidx] = (shift[1], shift[0])
    # Negative in order to properly register with transform_images method
    translations = -translations
    return translations


def register_template(frames, reference, template):
    """Calculate the relative translation between the reference image and
    each image in `frames` using a template matching algorithm.

    Returns: Array with same dimensions as 0th axis of `frames`
    containing (x, y) translations for each frame.
    """
    return None
