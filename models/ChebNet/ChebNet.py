# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:54:06 2023

@author: SomgBird
"""


import torch
import torch.nn as nn

from .ChebConvLayer import ChebConvLayer


class ChebNet(nn.Module):
    """
    https://arxiv.org/pdf/1606.09375v3.pdf
    """
    def __init__(
            self, 
            input_size : int, 
            hidden_size : int, 
            output_size : int, 
            enable_bias : bool, 
            poly_order : int, 
            hidden_layers_number : int, 
            droprate : float):
        """
        

        Parameters
        ----------
        input_size : int
            DESCRIPTION.
        hidden_size : int
            DESCRIPTION.
        output_size : int
            DESCRIPTION.
        enable_bias : bool
            DESCRIPTION.
        poly_order : int
            DESCRIPTION.
        hidden_layers_number : int
            DESCRIPTION.
        droprate : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(ChebNet, self).__init__()
        
        self.cheb_graph_convs = nn.ModuleList()
        self.poly_order = poly_order
        self.hidden_layers_number = hidden_layers_number
        
        self.cheb_graph_convs.append(ChebConvLayer(poly_order, input_size, hidden_size, enable_bias))
        for k in range(hidden_layers_number):
            self.cheb_graph_convs.append(ChebConvLayer(poly_order, hidden_size, hidden_size, enable_bias))
        self.cheb_graph_convs.append(ChebConvLayer(poly_order, hidden_size, output_size, enable_bias))
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x, gso):
        for k in range(self.hidden_layers_number-1):
            x = self.cheb_graph_convs[k](x, gso)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        x = self.cheb_graph_convs[-1](x, gso)
        x = self.log_softmax(x)

        return x

