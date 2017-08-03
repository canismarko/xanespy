#!/usr/bin/env python

from distutils.core import setup

setup(name="xanespy",
      version="0.1.2",
      description="Tools for analyzing X-ray absorbance spectroscopy data",
      author="Mark Wolfman",
      author_email="canismarko@gmail.com",
      url="https://github.com/canismarko/xanespy",
      keywords="XANES X-ray microscopy operando",
      install_requires=['pytz>=2013b', 'h5py', 'pandas', 'olefile',
                        'matplotlib', 'scikit-image', 'scikit-learn'],
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
