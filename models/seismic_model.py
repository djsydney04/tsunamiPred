import numpy as np

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import sklearn.datasets as ds
import random 

class SeismicTsunamiEventLinkageModel(nn.Module):
    def __init__self(self, input_size, hidden_size, output_size):
        super(SeismicTsunamiEventLinkageModel, self).__init__()
        #Linear input layer to first hidden layer   
        self.fc1 = nn.Linear(input_size, hidden_size)
        #Linear first hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #Linear second hidden layer to third hidden layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        #Linear third hidden layer to output layer
        self.fc4 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        #first hidden layer
        x = torch.relu(self.fc1(x))
        #second hidden layer
        x = torch.relu(self.fc2(x))
        #third hidden layer
        x = torch.gelu(self.fc3(x))
        #output layer
        x = self.fc4(x)
        return x
