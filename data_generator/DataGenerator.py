# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 01:02:47 2023

@author: SomgBird
"""

import networkx as nx
import random
import torch

from typing import Callable, List
from torch import Tensor

from .GraphGenerator import GraphGenerator
from .DataEntry import DataEntry
from .AccessPoint import AccessPoint


class DataGenerator:
    def __init__(self, weight_min, weight_max):
        self._weight_min = weight_min
        self._weight_max = weight_max
        self._graph_generator = GraphGenerator(weight_min, weight_max)
    
    
    
    def generate_data(self,
                      N : int,
                      nodes_number : int,
                      ap_number : int, 
                      signal_range : float, 
                      noise_source : Callable[[], float]) -> List[DataEntry]:
        return [self.generate_data_entry(nodes_number, ap_number, signal_range, noise_source) for i in range(N)]
        
    
    
    def generate_data_entry(self,
                            nodes_number : int,
                            ap_number : int, 
                            signal_range : float, 
                            noise_source : Callable[[], float]) -> DataEntry:
        graph, nodes_list = self._graph_generator.generate_graph(nodes_number)
        agent_pos = self.__generate_agent_pos(graph)
        agent = self.__create_agent_tensor(graph, agent_pos)
        access_points = self.__generate_access_points(graph, ap_number, signal_range)
        measurements = self.__generate_measurements(graph, agent_pos, access_points, noise_source)
        
        return DataEntry(
            torch.from_numpy(nx.adjacency_matrix(graph, nodes_list).todense()),
            torch.from_numpy(nx.laplacian_matrix(graph, nodes_list).todense()),
            agent, 
            measurements)
    
    
    def __generate_access_points(
            self, 
            graph : nx.Graph, 
            ap_number : int,
            signal_range : float) -> List[AccessPoint]:
        return [AccessPoint(node, signal_range) for node in self.__generate_ap_nodes(graph, ap_number)]
    
    
    def __generate_ap_nodes(self, graph : nx.Graph, ap_number: int) -> List[int]:
        if ap_number > len(graph):
            raise ValueError("Number of access points should not be higher than number of nodes")
        
        nodes = []
        while len(nodes) != ap_number:
            node = random.randint(0, len(graph) - 1)
            if node not in nodes:
                nodes.append(node)
        
        return nodes
        
    
    def __create_agent_tensor(self, graph : nx.Graph, agent_pos : int) -> Tensor:
        tensor = torch.zeros(len(graph))
        tensor[agent_pos] = 1
        return tensor

    
    def __generate_agent_pos(self, graph : nx.Graph) -> int:
        return random.randint(0, len(graph) - 1)
    
    
    def __generate_measurements(
            self, 
            graph : nx.Graph, 
            agent_pos : int, 
            access_points : List[AccessPoint], 
            noise_source : Callable[[], float]) -> Tensor:
        number = len(access_points)
        tensor = Tensor(number)
        for i in range(0, number):
            tensor[i] = access_points[i].measure_distance(agent_pos, graph) # TODO: add noise
        return tensor
            




