# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 00:17:10 2023

@author: SomgBird
"""


import torch
import torch.nn as nn
import numpy as np

from scipy.linalg import eigvals
from pathlib import Path
from torch import Tensor

from models import ChebNet
from data_generator import DataGenerator
from data_generator import DataEntry
from data_generator import DataWriter
from data_generator import GNNDataset

min_weight = 5
max_weight = 10
N = 1000
node_number = 10
ap_number = 4
signal_range = np.Inf

dir_path = Path('K:/Dev/PyTorch projects/dataset_chebnet_10n_4ap_infsr_5-10w_noise0.5')

# DG = DataGenerator(min_weight, max_weight)

# data = DG.generate_data(N, node_number, ap_number, signal_range, lambda x: x + np.random.normal(0, 1.0))


# DEW = DataWriter()
# DEW.write_all(data, dir_path)


def norm_laplacian(laplacian_matrix : Tensor):
    eigen_max = max(eigvals(a=laplacian_matrix).real)
    node_number = laplacian_matrix.size()[0]
    return 2 * laplacian_matrix / eigen_max - np.identity(node_number)


def accuracy(model, dataset, threshold, func = None):
    n_correct = 0
    n_wrong = 0
    
    for inputs, gso, labels in dataset:
        inputs = inputs.to(device)
        gso = gso.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs, gso.float())
            
            if func:
                outputs = func(outputs)
                
            c = []
            for j in range(len(outputs)):
                if outputs[j][0] > threshold:
                    c.append(j)
            
            if len(c) != 1:
                n_wrong += 1
            elif labels[c[0]][0] == 1:
                n_correct += 1
            else:
                n_wrong += 1
    return n_correct / (n_correct + n_wrong) 
                
        
num_epochs = 20
learning_rate = 0.005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:\t', device)

dataset = GNNDataset(dir_path, norm_laplacian)


model = ChebNet(
    input_size=1, 
    hidden_size=15, 
    output_size=2, 
    enable_bias=True, 
    poly_order=3, 
    hidden_layers_number=7,
    droprate=0.0).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0001, amsgrad=False)
torch.set_printoptions(sci_mode=False)
soft_max = nn.Softmax(dim=1)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train, test = torch.utils.data.random_split(dataset, [train_size, test_size])


# Train model
for epoch in range(num_epochs):
    for inputs, gso, labels in train:
        inputs = inputs.to(device)
        gso = gso.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        model.train()
        outputs = model(inputs, gso.float())
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_acc = accuracy(model, train, 0.5, soft_max)
    print('Epoch [{}/{}], Loss: {:.6f}, Acc: {:.6f}'.format(epoch+1, num_epochs, loss.item(), val_acc))

model.eval()
test_acc = accuracy(model, test, 0.5, soft_max)
print('Test Acc: {:.6f}'.format(test_acc))
