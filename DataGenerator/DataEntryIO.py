# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:46:22 2023

@author: SomgBird
"""

import uuid
import torch
import dataclasses

from pathlib import Path
from DataGenerator.DataEntry import DataEntry
from typing import List


# TODO: rewrite it to ZipFile, separate files inside the archive instead of blindly parsing tensors.

class DataEntryWriter:
    def write(self, data_entry : DataEntry, path : Path):
        file_name = uuid.uuid4().hex + '.data'
        
        tensors = []
        for f in dataclasses.fields(data_entry):
            tensors.append(getattr(data_entry, f.name))
        
        torch.save(tensors, path/file_name)
        
    
    def write_all(self, data_entries : List[DataEntry], path : Path):
        for d in data_entries:
            self.write(d, path)


class DataEntryReader:
    def read(self, path : Path) -> DataEntry:
        tensors = torch.load(path)
        
        if len(tensors) != len(dataclasses.fields(DataEntry)):
            raise IndexError('Amount of tensors doesn\'t fit DataEntry format')
            
        return DataEntry(tensors[0], tensors[1], tensors[2], tensors[3]) # TODO: change to file names

    
    def read_all(self, paths : List[Path]) -> List[DataEntry]:
        return [self.read(p) for p in paths]


    def read_dir(self, path : Path) -> List[DataEntry]:
        all_paths = path.glob('*.data')
        return self.read_all(all_paths)