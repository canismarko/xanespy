#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Mark Wolfman
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

import logging
import os
import shutil
from unittest import mock

import numpy as np
import h5py

from cases import XanespyTestCase
from xanespy import exceptions
from xanespy.txmstore import TXMStore, TXMDataset, merge_stores

TEST_DIR = os.path.dirname(__file__)
SSRL_DIR = os.path.join(TEST_DIR, 'txm-data-ssrl')
APS_DIR = os.path.join(TEST_DIR, 'txm-data-aps')
PTYCHO_DIR = os.path.join(TEST_DIR, 'ptycho-data-als/NS_160406074')

class TXMStoreTest(XanespyTestCase):
    hdfname = os.path.join(TEST_DIR, 'txmstore-test-temp.h5')
    
    def setUp(self):
        # Delete temporary HDF5 files
        if os.path.exists(self.hdfname):
            os.remove(self.hdfname)
        src_hdfname = os.path.join(TEST_DIR, 'txmstore-test.h5')
        shutil.copy2(src_hdfname, self.hdfname)
    
    def tearDown(self):
        # Delete temporary HDF5 files
        if os.path.exists(self.hdfname):
            os.remove(self.hdfname)
    
    def store(self, mode='r'):
        store = TXMStore(hdf_filename=self.hdfname,
                         parent_name='ssrl-test-data',
                         data_name='imported',
                         mode=mode)
        return store
    
    def test_getters(self):
        store = self.store()
        self.assertEqual(store.intensities.shape, (1, 2, 1024, 1024))
        self.assertEqual(store.references.shape, (1, 2, 1024, 1024))
        self.assertEqual(store.optical_depths.shape, (1, 2, 1024, 1024))
        self.assertEqual(store.pixel_sizes.shape, (1, 2,))
        self.assertEqual(store.energies.shape, (1, 2,))
        self.assertEqual(store.timestamps.shape, (1, 2, 2))
        self.assertEqual(store.original_positions.shape, (1, 2, 3))
        # Raises exception for non-existent datasets
        with self.assertRaises(exceptions.GroupKeyError):
            store.get_dataset('madeup_data')
        with self.assertRaises(exceptions.GroupKeyError):
            store.get_dataset(None)
        # Check that `get_frames()` returns the frames associated with a map
    
    def test_map_names(self):
        # Prepare the store with at least 1 map
        store = self.store(mode='r+')
        store.data_group()['intensities'].attrs['context'] = 'map'
        # Make sure the map name contains only the one map
        map_names = store.map_names()
        self.assertEqual(map_names, ['intensities'])
    
    def test_frameset_names(self):
        # Prepare the store with at least 1 map
        store = self.store(mode='r+')
        # Make sure the map name contains only the one map
        map_names = store.frameset_names()
        self.assertEqual(map_names, ['intensities', 'optical_depths', 'references'])
    
    def test_data_group(self):
        store = self.store()
        self.assertEqual(store.parent_group().name, '/ssrl-test-data')
        self.assertEqual(store.data_group().name, '/ssrl-test-data/imported')
        # Check that data_group throws an exception if the group doesn't exist
        store.data_name = 'nonexistent_group'
        with self.assertRaises(exceptions.CreateGroupError):
            store.data_group()
    
    def test_fork_group(self):
        store = self.store('r+')
        raises = self.assertRaises(exceptions.CreateGroupError)
        logs = self.assertLogs(level=logging.CRITICAL)
        with raises, logs:
            store.fork_data_group(dest=store.data_name, src=store.data_name)
        # Set a marker to see if it changes
        store.parent_group().create_group('new_group')
        store.data_name = 'new_group'
        store.data_group().attrs['test_val'] = 'Hello'
        # Now verify that the previous group was overwritten
        store.data_name = 'imported'
        store.fork_data_group(dest='new_group')
        self.assertNotIn('test_val', list(store.data_group().attrs.keys()))
        # Check that the new group is registered as the "latest"
        self.assertEqual(store.latest_data_name, 'new_group')
        # Check that we can easily fork a non-existent group
        store.fork_data_group(dest='brand_new')
        store.close()
    
    def test_data_tree(self):
        """Check that a data tree can be created showing the possible groups
        to choose from.
        
        """
        store = self.store()
        f = h5py.File(self.hdfname)
        # Check that all top-level groups are accounted for
        tree = store.data_tree()
        self.assertEqual(len(f.keys()), len(tree))
        # Check properties of a specific entry (optical depth data)
        abs_dict = [d for d in tree[0]['children'][0]['children'] if 'optical_depths' in d['path']][0]
        self.assertEqual(abs_dict['level'], 2)
        self.assertEqual(abs_dict['context'], 'frameset')
    
    def test_data_name(self):
        store = self.store('r+')
        store.data_name = 'imported'
        self.assertEqual(store.data_name, 'imported')
        store.close()
    
    def test_setters(self):
        store = self.store('r+')
        # Check that the "type" attribute is set
        store.optical_depths = np.zeros((2, 1024, 1024))
        self.assertEqual(store.optical_depths.attrs['context'], 'frameset')
        # Check that new data are saved in HDF file
        store.intensities = np.ones((256, 256))
        ds = store.parent_group()[store.data_name]['intensities']
        self.assertEqual(ds.shape, (256, 256))
    
    def test_delete(self):
        store = self.store('r+')
        del store.intensities
        group = store.parent_group()[store.data_name]
        self.assertNotIn('intensities', group.keys())
        # Now check that it throws the right exception on the getter
        with self.assertRaises(exceptions.GroupKeyError):
            store.intensities
    
    def test_merge_stores(self):
        # Merge some mock data that doesn't overlap
        ptycho_store = mock.Mock()
        stxm_store = mock.Mock()
        stxm_store.timestep_names = ['ex-situ']
        dest_store = mock.Mock()
        ptycho_store.energies = [[856]]
        ptycho_store.intensities = np.ones(shape=(1, 1, 32, 32))
        ptycho_store.optical_depths = np.ones(shape=(1, 1, 32, 32))
        ptycho_store.pixel_sizes = np.array([[4]])
        ptycho_store.filenames = [['file1.xim']]
        stxm_store.energies = np.array([[853, 859]])
        stxm_store.intensities = np.ones(shape=(1, 2, 24, 24))
        stxm_store.optical_depths = np.ones(shape=(1, 2, 24, 24))
        stxm_store.pixel_sizes = np.array([[35, 35]])
        stxm_store.filenames = [['file1.xim', 'file2.xim']]
        merge_stores(stxm_store, ptycho_store, destination=dest_store)
        # Check that the right data was set
        self.assertEqual(dest_store.energies, [[853, 856, 859]])
        np.testing.assert_equal(dest_store.intensities, np.ones((1, 3, 32, 32)))
        np.testing.assert_equal(dest_store.optical_depths, np.ones((1, 3, 32, 32)))
        # Check what happens if the same energy is present in both sets
        dest_store = mock.Mock()
        ptycho_store.energies = [[859]]
        stxm_store.energies = np.array([[853, 859]])
        merge_stores(stxm_store, ptycho_store, destination=dest_store)
        self.assertEqual(dest_store.energies, [[853, 859]])
        np.testing.assert_equal(dest_store.intensities, np.ones((1, 2, 32, 32)))
    
    def test_validate_parent_group(self):
        # Prepare an HDF5 file
        temp_filename = 'temp_data_file.h5'
        temp_file = h5py.File(temp_filename, mode='a')
        try:
            temp_file.create_group('default_group_1')
        except:
            os.remove(temp_filename)
        finally:
            temp_file.close()
        # Test the validate_parent_group function
        try:
            store = TXMStore(temp_filename, parent_name=None, data_name='data')
            validated_name = store.validate_parent_group(None)
            self.assertEqual(validated_name, 'default_group_1')
        except:
            os.remove(temp_filename)
        else:
            store.close()
        # Make it look like an APS 32-ID-C data file
        temp_file = h5py.File(temp_filename, mode='a')
        try:
            del temp_file['default_group_1']
            temp_file.create_group('xanespy')
            temp_file.create_group('exchange')
        except:
            os.remove(temp_filename)
        finally:
            temp_file.close()
        # Test the updated store's *validate_parent_group* function
        try:
            store = TXMStore(temp_filename, parent_name=None, data_name='data')
            validated_name = store.validate_parent_group(None)
            self.assertEqual(validated_name, 'xanespy')
        finally:
            os.remove(temp_filename)
