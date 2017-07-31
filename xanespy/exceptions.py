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

"""Define classes for more fine-grained control over exception
handling."""


#########################
# X-ray microscopy errors
#########################
class DataNotFoundError(FileNotFoundError):
    """Expected a directory containing data but found none."""
    pass


class FrameFileNotFound(IOError):
    """Expected to load a TXM frame file but it doesn't exist."""
    pass


class GroupKeyError(KeyError):
    """Tried to load or create an HDF group but failed. Examples include:
    the group doesn't exist, is ambiguous or already exists when being
    created.
    """
    pass


class FrameSourceError(KeyError):
    """The frame-source attribute is not present."""
    pass


class DataFormatError(RuntimeError):
    """The raw data are arranged in a way that the importers or TXM classes do
    not understand.
    """
    pass


class HDFScopeError(ValueError):
    """Tried to pass an HDF scope that is not recognized."""
    pass


class FileExistsError(IOError):
    """Tried to import a TXM frameset but the corresponding HDF file
    already exists."""
    pass


class CreateGroupError(ValueError):
    """Tried to import a TXM frameset into a group but the corresponding
    HDF group already exists or is otherwise inaccessible.
    """
    pass


class FilenameParseError(ValueError):
    """The parameters in the filename do not match the naming scheme
    associated with this flavor."""
    pass


class DatasetExistsError(RuntimeError):
    """Trying to save a new dataset but one already exists with the given
    path."""
    pass


class NoParticleError(Exception):
    pass


class RefinementError(RuntimeError):
    pass


class XanesMathError(RuntimeError):
    pass
