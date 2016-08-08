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

from collections import namedtuple

import numpy as np

from utilities import xycoord, Pixel

"""
Describes a single particle detected by image processing (skimage).
"""

BoundingBox = namedtuple('BoundingBox', ('bottom', 'left', 'top', 'right'))


class Particle():
    """A single secondary particle detected by image
    processing. Properties determine by regionprops routine from
    skimage.
    """

    def __init__(self, regionprops, frame):
        """regionprops as output from skimage.measure.regionprops."""
        self.regionprops = regionprops
        self.frame = frame

    def relative_position(self):
        """Convert centroid in pixels to relative sample position in x, y
        (µm)."""
        frame_center_pos = self.frame.relative_position
        frame_center_pix = Pixel(self.frame.image_data.shape[0] / 2,
                                 self.frame.image_data.shape[1] / 2)
        pixel_distance_h = self.centroid().horizontal - frame_center_pix.horizontal
        pixel_distance_v = self.centroid().vertical - frame_center_pix.vertical
        px_size = self.frame.pixel_size
        px_unit = px_size.unit
        new_center = xycoord(
            x=px_unit(frame_center_pos.x) + pixel_distance_h * px_size,
            y=px_unit(frame_center_pos.y) + pixel_distance_v * px_size
        )
        return new_center

    def sample_position(self):
        """Convert centroid in pixels to sample position in x, y (µm)."""
        frame_center_pos = self.frame.sample_position
        frame_center_pix = Pixel(self.frame.image_data.shape[0] / 2,
                                 self.frame.image_data.shape[1] / 2)
        pixel_distance_h = self.centroid().horizontal - frame_center_pix.horizontal
        pixel_distance_v = self.centroid().vertical - frame_center_pix.vertical
        um_per_pixel = self.frame.um_per_pixel
        new_center = xycoord(
            x=frame_center_pos.x + pixel_distance_h * um_per_pixel.horizontal,
            y=frame_center_pos.y + pixel_distance_v * um_per_pixel.vertical
        )
        return new_center

    def centroid(self):
        center = self.regionprops.centroid
        return Pixel(vertical=center[0], horizontal=center[1])

    def area(self):
        area = self.regionprops.area
        return area

    def bbox(self):
        return BoundingBox(*self.regionprops.bbox)

    def convex_area(self):
        return self.regionprops.convex_area

    def image(self):
        bbox = self.bbox()
        image = self.frame.image_data.value
        cropped_image = image[bbox.top:bbox.bottom, bbox.left:bbox.right]
        return cropped_image

    def full_mask(self):
        """Return a mask the same size as frame data with only this particle
        exposed."""
        data = self.frame.image_data
        mask = np.zeros_like(data)
        bbox = self.bbox()
        mask[bbox.top:bbox.bottom, bbox.left:bbox.right] = self.mask()
        return np.logical_not(mask)

    def masked_frame_image(self):
        """Return a masked array for the whole frame with only this particle
        only marked as valid."""
        data = self.frame.image_data
        mask = self.full_mask()
        return np.ma.array(data, mask=mask)

    def mask(self):
        return self.regionprops.image

    def plot_image(self, show_particles=False, *args, **kwargs):
        """Calls the regular plotting routine but with cropped data."""
        return self.frame.plot_image(data=self.image(),
                                     show_particles=show_particles,
                                     *args, **kwargs)
