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
x_train, x_test, y_train, y_test = preprocess()

#creating model
model = SeismicTsunamiEventLinkageModel(input_size=10, hidden_size=128, hidden_size2=64, hidden_size3=32, output_size=1)

#defining loss function
loss_function = nn.BCELoss()

#defining optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#defining number of epochs
num_epochs = 10000

#training model
for epoch in range(num_epochs):
    optimizer.zero_grad()
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
    
    #early stopping
    #best loss and best epoch and patience (how many epochs to wait before stopping)
    best_auc = 0.0
    patience = 5
    best_epoch = 0
    #if loss is less than best loss, update best loss and best epoch
    if loss.item() < best_auc:
        best_loss = loss.item()
        best_epoch = epoch
        patience = 5
    elif loss.item() > best_auc:
        patience -= 1
        if patience == 0:
            print(f"Early stopping triggered at epoch {epoch}")
            break


torch.save(model.state_dict(), "models/tsunami_model.pth")