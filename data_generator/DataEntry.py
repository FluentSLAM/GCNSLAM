# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:46:44 2023

@author: SomgBird
"""

from dataclasses import dataclass
from torch import Tensor

from .AbstractDataEntry import AbstractDataEntry


@dataclass
class DataEntry(AbstractDataEntry):
    adjacency_matrix : Tensor
    laplacian_matrix : Tensor
    agent : Tensor
    measurements : Tensor

