import torch
import torch.nn as nn
import numpy as np
from models.seismic_model import SeismicTsunamiEventLinkageModel
from utils.preprocess import preprocess
from utils.train_eval import evaluate

def main():
    print("Tsunami Prediction Model - Test Evaluation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load test data
    print("\n1. Loading test data...")
    x_train, x_test, y_train, y_test = preprocess()
    
    print(f"   Training samples: {len(x_train)}")
    print(f"   Test samples: {len(x_test)}")
    print(f"   Feature dimensions: {x_train.shape[1]}")
    
    # Load the trained model
    print("\n2. Loading trained model...")
    model = SeismicTsunamiEventLinkageModel(
        input_size=10, 
        hidden_size=128, 
        hidden_size2=64, 
        hidden_size3=32, 
        output_size=1
    )
    
    try:
        model.load_state_dict(torch.load("models/tsunami_model.pth"))
        print("   Model loaded successfully!")
    except FileNotFoundError:
        print("   ERROR: Model file not found. Please train the model first using train.py")
        return
    
    # Define loss function (same as training)
    loss_function = nn.BCELoss()
    
    # Evaluate on test data
    print("\n3. Evaluating on test data...")
    metrics = evaluate(model, x_test, y_test, loss_function)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Test Loss:     {metrics['loss']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Show some sample predictions
    print("\n4. Sample predictions:")
    print("-" * 30)
    
    model.eval()
    with torch.no_grad():
        # Get predictions for first 5 test samples
        sample_predictions = model(x_test[:5])
        sample_probs = sample_predictions.squeeze().cpu().numpy()
        sample_true = y_test[:5].squeeze().cpu().numpy()
        
        for i in range(5):
            true_label = "Tsunami" if sample_true[i] == 1 else "No Tsunami"
            pred_label = "Tsunami" if sample_probs[i] >= 0.5 else "No Tsunami"
            confidence = max(sample_probs[i], 1 - sample_probs[i])
            
            print(f"\nSample {i+1}:")
            print(f"  True: {true_label}")
            print(f"  Predicted: {pred_label}")
            print(f"  Probability: {sample_probs[i]:.4f}")
            print(f"  Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()