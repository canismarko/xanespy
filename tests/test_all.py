#!/usr/bin/env python
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

# flake8: noqa

"""Loader for running all the tests for xanespy. Discovers any file
matching "test*.py"."""

import unittest
import os


if __name__ == '__main__':
    start_dir = os.path.dirname(__file__)
    tests = unittest.defaultTestLoader.discover(start_dir)
    runner = unittest.runner.TextTestRunner()
    runner.run(tests)
