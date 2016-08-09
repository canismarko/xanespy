import numpy as np

from skimage import transform

def transform_images(data, translations=None, rotations=None,
                         scales=None, out=None):
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
        scale = scales[imidx] if scales is not None else None
        print(scale)
        translation = translations[imidx] if translations is not None else None
        print(translation)
        rot = rotations[imidx] if rotations is not None else None
        print(rot)
        transformation = transform.SimilarityTransform(
            scale=scales[imidx] if scales is not None else None,
            translation=translations[imidx] if translations is not None else None,
            rotation=rotations[imidx] if rotations is not None else None,
        )
        out[imidx] = transform.warp(data[imidx], transformation, order=3)
    return out
