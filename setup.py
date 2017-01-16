#!/usr/bin/env python

from distutils.core import setup

setup(name="xanespy",
      version="0.1",
      description="Tools for analyzing X-ray absorbance spectroscopy data",
      author="Mark Wolf",
      author_email="mark.wolf.music@gmail.com",
      url="https://github.com/m3wolf/xanespy",
      packages=['xanespy',],
      package_data={
          'xanespy': ['qt_map_window.ui', 'qt_frame_window.ui']
      },
)
