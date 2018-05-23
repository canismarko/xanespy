#!/usr/bin/env python

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="xanespy",
      version=read("VERSION"),
      description="Tools for analyzing X-ray absorbance spectroscopy data",
      long_description=read('README.rst'),
      long_description_content_type='text/x-rst',
      author="Mark Wolfman",
      author_email="canismarko@gmail.com",
      url="https://github.com/canismarko/xanespy",
      keywords="XANES X-ray microscopy operando",
      install_requires=[
          'pytz>=2013b', 'h5py', 'pandas', 'olefile', 'matplotlib', 'scikit-image',
          'scikit-learn', 'numpy', 'tqdm', 'jinja2', 'pandas', 'scikit-image',
      ],
      packages=['xanespy',],
      package_data={
          'xanespy': ['qt_map_window.ui', 'qt_frame_window.ui']
      },
      entry_points={
          'gui_scripts': [
              'xanespy-viewer = xanespy.xanes_viewer:main'
          ]
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
      ]
)
