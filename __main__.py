# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 01:24:44 2023

@author: SomgBird
"""

import networkx as nx
from torch import Tensor
from DataGenerator.GraphGenerator import GraphGenerator

GG = GraphGenerator(1, 5)
graph, nodes = GG.generate_graph(5)
nx.draw_networkx(graph)
A = nx.adjacency_matrix(graph, nodes)
print(A.todense())
print(Tensor(A.todense()))
print(nx.shortest_path(graph, 0 , 2, weight='weight'))
print(nx.shortest_path_length(graph, 0 , 2, weight='weight'))