# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 20:07:48 2023

@author: SomgBird
"""

from typing import Type, Callable
from torch.utils.data import Dataset
from dataclasses import fields

from .AbstractDataEntry import AbstractDataEntry
from .DataIO import DataReader
from .DataEntry import DataEntry


# TODO: create csv file with annotations and read them one by one
class GNNDataset(Dataset):
    def __init__(self, root_dir, transform : Callable = None):
        self._root_dir = root_dir
        self._transform = transform
        self._data_reader = DataReader()
        self._dataset = self._data_reader.read_dir(root_dir)


    def __len__(self):
        return len(self._dataset)


    # TODO: rework transforms
    def __getitem__(self, index):
        data_entry = self._dataset[index]
        if self._transform:
            laplacian_matrix = self._transform(data_entry.laplacian_matrix)
        else:
            laplacian_matrix = data_entry.laplacian_matrix
        return data_entry.measurements, laplacian_matrix, data_entry.agent
    
    
    def __copy_entry(self, data_entry : DataEntry) -> DataEntry:
        r = DataEntry()
        
        for field in fields(data_entry):
            setattr(r, field.name, getattr(data_entry, field.name))
        
        return r
    
    
    def __split_data_entry(self, data_entry : DataEntry):
        return data_entry.measurements, data_entry.laplacian_matrix, data_entry.agent

