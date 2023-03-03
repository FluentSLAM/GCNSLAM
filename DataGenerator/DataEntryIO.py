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


class DataEntryWriter:
    def write(self, data_entry : DataEntry, path : Path):
        file_name = uuid.uuid4().hex
        
        tensors = []
        for f in dataclasses.fields(data_entry):
            tensors.append(getattr(data_entry, f.name))
        
        torch.save(tensors, path/file_name)
        
    
    def write_all(self, data_entries : List[DataEntry], path : Path):
        for d in data_entries:
            self.write(d, path)


class DataEntryReader:
    def read(self, path : Path) -> DataEntry:
        pass

    
    def write(self, data_entries : List[DataEntry], path : Path) -> List[DataEntry]:
        pass
