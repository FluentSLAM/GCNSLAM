# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 20:07:48 2023

@author: SomgBird
"""

from typing import Type, Callable
from torch.utils.data import Dataset

from .AbstractDataEntry import AbstractDataEntry
from .DataIO import DataReader

# TODO: create csv file with annotations and read them one by one
class GNNDataset(Dataset):
    def __init__(self, root_dir, dtype : Type[AbstractDataEntry], transform : Callable = None):
        self._root_dir = root_dir
        self._dtype = dtype
        self._transform = transform
        self._data_reader = DataReader()
        self._dataset = self._data_reader.read_dir(root_dir)


    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, index):
        if self._transform:
            self._transform(self._dataset[index])
        return self._dataset[index]
    
