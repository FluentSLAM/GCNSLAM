# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:33:18 2023

@author: SomgBird
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from torch import Tensor, FloatTensor
from typing import List

class ChebConvLayer(nn.Module):
    """
    https://arxiv.org/abs/1606.09375v3
    """
    
    def __init__(self, order : int, input_size : int, output_size : int, bias_enabled : bool):
        """
        

        Parameters
        ----------
        order : int
            Order of Chebyshev polinomials.
        input_size : int
            Number of input features.
        output_size : int
            Number of output featires.
        bias_enabled : bool
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(ChebConvLayer, self).__init__()
        
        self.order = order
        self.weight = nn.Parameter(FloatTensor(order, input_size, output_size))
        
        if bias_enabled:
            self.bias = nn.Parameter(FloatTensor(input_size, output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
       
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
            
    def forward(self, x : Tensor, gso :Tensor) -> Tensor:
        convolution = self.__compute_chebyshev_convolution(x, gso)
        
        if self.bias is not None:
            convolution = torch.add(input=convolution, other=self.bias, alpha=1)
        
        return convolution

    
    def __compute_chebyshev_convolution(self, x: Tensor, gso: Tensor) -> Tensor:
        """
        Computes Chebyshev polynomial of the first kind convolution for Tensor 
        x with graph shift operator gso.

        Parameters
        ----------
        x : Tensor
            Chebyshev polynomial input Tensor.
        gso : Tensor
            Graph shift operator.

        Raises
        ------
        ValueError
            if Chebyshev polinomial order is lower than 0.

        Returns
        -------
        Tensor
            Chebyshev polynomial convolution as a PyTorch Tensor.

        """
        cheb_poly_parts = []
        
        if self.order < 0:
            raise ValueError('Chebyshev polinomial order shold not be less than 0')
        elif self.order == 0:
            cheb_poly_parts.append(x)
        else:
            cheb_poly_parts.append(x)
            if gso.is_sparse:
                self.__sparce_cheb_polynomial(cheb_poly_parts, gso, x)
            else:
                self.__cheb_polynomial(cheb_poly_parts, gso, x)
        
        feature = torch.stack(cheb_poly_parts, dim=0)
        
        if feature.is_sparse:
            feature = feature.to_dense()
        return torch.einsum('bij,bjk->ik', feature, self.weight)


    def __sparce_cheb_polynomial(self, cheb_poly_parts : List[Tensor], gso : Tensor, x : Tensor):
        """
        Computes parts Chebyshev polynomial for k > 0 for sparce gso.

        """
        cheb_poly_parts.append(torch.sparce.mm(gso, x))
        
        for k in range(2, self.order):
            cheb_poly_parts.append(torch.sparse.mm(2 * gso, cheb_poly_parts[k - 1]) - cheb_poly_parts[k - 2])


    def __cheb_polynomial(self, cheb_poly_parts : List[Tensor], gso : Tensor, x : Tensor):
        """
        Computes parts Chebyshev polynomial for k > 0 for dense gso.

        """
        if x.is_sparse:
            x = x.to_dense()
        cheb_poly_parts.append(torch.mm(gso, x))
                
        for k in range(2, self.order):
            cheb_poly_parts.append(torch.mm(2 * gso, cheb_poly_parts[k - 1]) - cheb_poly_parts[k - 2])        

