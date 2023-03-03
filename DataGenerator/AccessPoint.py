# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:44:00 2023

@author: SomgBird
"""

import networkx as nx
import numpy as np


class AccessPoint:
    def __init__(self, node : int, signal_range : float):
        self._node = node
        self._signal_range = signal_range
    
    
    def get_node(self) -> int:
        return self._node
    
    
    def get_range(self) -> float:
        return self._signal_range
    
    
    def measure_distance(self, pos : int, graph : nx.Graph):
        dist = nx.shortest_path(graph, self._node, pos, weight='weight')
        
        if(dist <= self._signal_range):
            return dist
        return np.Inf

