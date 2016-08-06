# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Scimap. If not, see <http://www.gnu.org/licenses/>.

from matplotlib import animation


class FrameAnimation(animation.ArtistAnimation):
    def __init__(self, fig, artists, *args, **kwargs):
        self.fig = fig
        self.artists = artists
        self._framedata = artists
        ret = super().__init__(*args, fig=fig, artists=artists, **kwargs)
        return ret

    @property
    def artists(self):
        return self._framedata

    @artists.setter
    def artists(self, value):
        self._framedata = value

    def _step(self, current_idx):
        artists = self._framedata[current_idx]
        self._draw_next_frame(artists, self._blit)
        return True

    def stop(self):
        return self._stop()
