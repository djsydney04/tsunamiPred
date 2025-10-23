import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate(model, x_test, y_test, loss_function, threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        #forward pass
        probs = model(x_test)
        test_loss = loss_function(probs, y_test)  # Fixed: use 'probs' instead of 'predictions'
        preds = (probs >= threshold).float()
        
        # Convert to NumPy for sklearn
        y_true = y_test.cpu().numpy().ravel()
        y_hat  = preds.cpu().numpy().ravel()
        y_prob = probs.cpu().numpy().ravel()

    # Compute metrics (moved outside the with block)
    acc  = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    f1   = f1_score(y_true, y_hat, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_hat)

    return {
        "loss": test_loss.item(),  # Fixed: convert tensor to scalar
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }