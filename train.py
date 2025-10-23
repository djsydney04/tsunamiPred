import torch
import torch.nn as nn
import torch.optim as optim
import random 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from models.seismic_model import SeismicTsunamiEventLinkageModel
from utils.preprocess import preprocess

#setting random seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#loading data
x_train,y_train = preprocess()

#creating model
model = SeismicTsunamiEventLinkageModel(input_size=10, hidden_size=128, hidden_size2=64, hidden_size3=32, output_size=1)

#defining loss function
loss_function = nn.BCELoss()

#defining optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#defining number of epochs
num_epochs = 100

#training model
for epoch in range(num_epochs):
    #forward pass
    predictions = model(x_train)
    #calculating loss
    loss = loss_function(predictions, y_train)
    #backward pass
    loss.backward()
    #optimizer step
    optimizer.step()    
    #printing loss
    if epoch % 100 ==0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")