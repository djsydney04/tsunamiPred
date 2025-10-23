# Tsunami Predictor

A deep learning model that predicts tsunami occurrence based on seismic data. The model uses a multi-layer neural network to analyze earthquake characteristics and assess tsunami risk.

## Features

- Deep learning-based tsunami prediction
- Real-time single earthquake predictions
- Model training with early stopping
- Comprehensive model evaluation
- Command-line interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/djsydney04/tsunamiPred.git
cd Tsunami\ Predictor
```

2. Install dependencies:
```bash
pip install torch pandas numpy matplotlib scikit-learn datasets
```

## Usage

### Training the Model

Train a new model with default parameters:
```bash
./tsunami train
```

Customize training parameters:
```bash
./tsunami train --epochs 5000 --patience 100 --learning-rate 0.001
```

Parameters:
- `--epochs`: Number of training iterations (default: 10000)
- `--patience`: Early stopping patience (default: 100)
- `--learning-rate`: Learning rate (default: 0.001)
- `--save-path`: Where to save model (default: 'models/tsunami_model.pth')
- `--verbose`: Show detailed training progress

### Testing the Model

Evaluate model performance on test data:
```bash
./tsunami test
```

With additional metrics:
```bash
./tsunami test --threshold 0.7 --verbose
```

Parameters:
- `--threshold`: Classification threshold (default: 0.5)
- `--verbose`: Show detailed metrics
- `--model-path`: Model to test (default: 'models/tsunami_model.pth')

### Making Predictions

Predict tsunami probability for a single earthquake:
```bash
./tsunami predict \
    --magnitude 7.2 \
    --depth 10.5 \
    --latitude 35.6 \
    --longitude 139.7
```

Required parameters:
- `--magnitude`: Earthquake magnitude (Richter scale)
- `--depth`: Earthquake focal depth (km)
- `--latitude`: Epicenter latitude
- `--longitude`: Epicenter longitude

Optional parameters:
- `--cdi`: Community Decimal Intensity
- `--mmi`: Modified Mercalli Intensity
- `--sig`: Event significance score
- `--nst`: Number of stations
- `--dmin`: Distance to nearest station
- `--gap`: Azimuthal gap
- `--threshold`: Classification threshold (default: 0.5)

## Model Architecture

The model uses a feed-forward neural network with the following architecture:

- Input Layer: 10 features
  - Earthquake magnitude
  - Community Decimal Intensity (CDI)
  - Modified Mercalli Intensity (MMI)
  - Significance score
  - Number of stations
  - Distance to nearest station
  - Azimuthal gap
  - Depth
  - Latitude
  - Longitude

- Hidden Layers:
  1. 128 neurons (ReLU activation)
  2. 64 neurons (ReLU activation)
  3. 32 neurons (ReLU activation)

- Output Layer:
  - 1 neuron (Sigmoid activation)
  - Outputs tsunami probability (0-1)

## Training Process

1. Data Preprocessing:
   - Loads seismic-tsunami event linkage dataset
   - Splits into training (80%) and test (20%) sets
   - Normalizes numerical features

2. Training:
   - Binary Cross-Entropy Loss
   - Adam optimizer
   - Early stopping to prevent overfitting
   - Saves best model based on validation loss

3. Early Stopping:
   - Monitors training loss
   - Stops when no improvement for specified patience
   - Saves best model weights

## Performance Metrics

The model evaluates performance using:
- Binary Cross-Entropy Loss
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## Dataset

Uses the "seismic-tsunami event linkage" dataset from Hugging Face, containing:
- Historical earthquake data
- Associated tsunami occurrences
- Seismic measurements
- Geographical information

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests if applicable
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

Guidelines:
- Keep code style consistent with the project
- Add comments for complex logic
- Update documentation for any changes
- Test your changes thoroughly
- Keep pull requests focused on a single feature/fix

## License

MIT License

Copyright (c) 2025 Dylan Mitic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

Dylan Mitic 
X/Twitter: Dylanmitic

## Acknowledgments

- Dataset: mnemoraorg/seismic-tsunami-event-linkage
- PyTorch framework
- [Other acknowledgments]
