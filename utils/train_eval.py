import torch
import torch.nn as nn
import numpy as np
from train.py import loss_function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        test_loss = loss_function(predictions, y_test)
        return test_loss