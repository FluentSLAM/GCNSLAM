# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:46:22 2023

@author: SomgBird
"""

import uuid
import torch
import dataclasses

from pathlib import Path
from .DataEntry import DataEntry
from typing import List


# TODO: rewrite it to ZipFile, separate files inside the archive instead of blindly parsing tensors.

class DataWriter:
    def write(self, data_entry : DataEntry, path : Path):
        file_name = uuid.uuid4().hex + '.data'
        torch.save(data_entry, path/file_name)
        
    
    def write_all(self, data_entries : List[DataEntry], path : Path):
        for d in data_entries:
            self.write(d, path)


class DataReader:
    def read(self, path : Path) -> DataEntry:
        return torch.load(path)

    
    def read_all(self, paths : List[Path]) -> List[DataEntry]:
        return [self.read(p) for p in paths]


    def read_dir(self, path : Path) -> List[DataEntry]:
        all_paths = path.glob('*.data')
        return self.read_all(all_paths)

