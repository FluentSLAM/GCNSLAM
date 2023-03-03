# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:44:45 2023

@author: SomgBird
"""

import random
import networkx as nx
from typing import Tuple, List


class GraphGenerator:
    def __init__(self, weight_min, weight_max):
        self._weight_min = weight_min
        self._weight_max = weight_max
    
    
    def generate_graph(self, nodes_number : int) -> Tuple[nx.Graph, List[int]]:
        graph = nx.Graph()
        nodes_list = self.__generate_nodes(graph, nodes_number)
        self.__generate_edges(graph, nodes_list)
        return graph, nodes_list
        
    
    def __generate_nodes(self, graph : nx.Graph, nodes_number : int) -> List[int]:
        nodes_list = [node for node in range(0, nodes_number)]
        graph.add_nodes_from(nodes_list)
        return nodes_list


    def __generate_edges(self, graph : nx.Graph, nodes_list : List[int]):
        while True:
            for u in nodes_list:
                self.__add_new_edge(u, graph, nodes_list)
                if nx.is_connected(graph):
                    return


    def __add_new_edge(self, u : int, graph : nx.Graph, nodes_list : List[int]):
        if len(graph.edges(u)) >= len(graph) - 1:
            return
        
        while True:
            v = random.choice(nodes_list)
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v, weight=self.__get_random_weight())
                return


    def __get_random_weight(self):
        return random.randint(self._weight_min, self._weight_max)

