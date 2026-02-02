# Hybrid Stacking for Font Classification: Project Walkthrough

## 1. Project Overview

This project implements a **Hybrid Stacking Ensemble** for fine-grained font classification, specifically targeting visually similar fonts (Ranks 20-30 of the UCI dataset).

The pipeline combines:

1.  **Deep Learning (CNN)**: Extracting high-level "style embeddings" from 20x20 grayscale images.
2.  **Ensemble Learning (Stacking)**: Using Random Forests, Gradient Boosting, and MLPs to classify these embeddings, significantly outperforming the standard CNN baseline.

## 2. Directory Structure

| File / Folder              | Description                                                                                               |
| :------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `01_data_preprocessing.py` | **Step 1:** Ingests raw CSVs, filters classes (Rank 10-30), reshapes to 20x20 images, and saves to `.pt`. |
| `02_cnn_model_training.py` | **Step 2:** Trains the custom `FontCNN` to learn features. Saves the best model to `cnn_model_output/`.   |
| `03_ensemble_stacking.py`  | **Step 3:** Extracts CNN features, applies Stacking (with and without KPCA), and generates reports.       |
| `src/cnn_architecture.py`  | Defines `FontCNN` with a custom **Spatial Attention** module.                                             |
| `src/stacking_ensemble.py` | Implementation of the `HybridStackingEnsemble` with RF, HistGradientBoosting, and MLP base learners.      |
| `src/data_loader.py`       | Utilities for loading the processed `.pt` data into PyTorch DataLoaders.                                  |
| `src/kernel_methods.py`    | Helper functions for Kernel PCA (KPCA) dimensionality reduction.                                          |
| `data/`                    | Stores input inputs and processed tensors.                                                                |
| `results/`                 | Output folder for experiment reports and confusion matrices.                                              |

## 3. How to Run

### Prerequisites

Ensure you have the necessary dependencies installed. A `requirements.txt` file is provided in the root directory.

```bash
pip install -r requirements.txt
```

### Execution Flow

#### Step 1: Data Preprocessing

Reads raw pixel CSVs and prepares the dataset.

```bash
python 01_data_preprocessing.py
```

- **Input**: `data/inputs/*.csv` - The fonts can be found on `https://archive.ics.uci.edu/dataset/417/character+font+images`
- **Output**: `data/processed_fonts.pt`
- **Key Logic**: Selects fonts ranked 20-30 by size, treating everything else as "Other".

#### Step 2: Train CNN Backbone

Trains the Feature Extractor.

```bash
python 02_cnn_model_training.py
```

- **Output**: `cnn_model_output/best_cnn.pth`
- **Key Logic**: 20 Epochs, Adam Optimizer, Spatial Attention Mechanism.

#### Step 3: Run Ensemble Experiment

Runs the comparative analysis (CNN vs. KPCA-Stacking vs. Raw-Stacking).

```bash
python 03_ensemble_stacking.py
```

- **Output**: `results/experiment_report.txt` and `results/comparison_matrix.png`
- **Key Logic**: 5-Fold Cross-Validation comparing the baseline CNN against Stacking Ensembles.

## 4. Key Components Deep Dive

### The CNN Architecture (`src/cnn_architecture.py`)

The `FontCNN` is a custom backbone designed to learn latent style representations from low-resolution 20x20 text images.

- **Convolutional Features**:
  - Block 1: `Conv2d(1 -> 32)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)`.
  - Block 2: `Conv2d(32 -> 64)` + `BatchNorm` + `ReLU`.
- **Spatial Attention Module**:
  - Unlike standard CNNs, this module computes a spatial importance map.
  - It aggregates features using both **Average Pooling** and **Max Pooling** across channels.
  - These are concatenated and passed through a **7x7 Convolution** followed by a **Sigmoid** activation.
  - The resulting mask highlights information-rich regions (like stroke terminations) and suppresses background noise.
- **Heads**:
  - **Embedding Head (`fc1`)**: Compresses the features into a dense **128-dimensional vector**. This embedding is extracted and used as the input for the Stacking Ensemble.
  - **Classification Head (`fc2`)**: A final linear layer used solely for training the backbone via Cross-Entropy Loss.

### The Stacking Ensemble (`src/stacking_ensemble.py`)

Because the linear classifier of the CNN is insufficient for fine-grained font distinction, we replace it with a **Hybrid Stacking Ensemble**.

- **Level-0: Heterogeneous Base Learners**
  These models receive the 128-dimensional embeddings from the CNN and independently predict class probabilities.
  1.  **Random Forest**: Configured with 200 trees and balanced class weights. It provides high-variance, low-bias predictions robust to noise.
  2.  **Histogram Gradient Boosting**: A highly efficient implementation of GBDT (similar to LightGBM). It runs for 100 iterations and excels at capturing non-linear interactions in the feature space.
  3.  **MLP Classifier**: A Multi-Layer Perceptron (Neural Network) with hidden layers of size (64, 32). It adds a different inductive bias compared to the tree-based models.

- **Level-1: Meta Learner**
  - **Logistic Regression**: The outputs (probabilities) of the three base learners are concatenated. The meta-learner allows the system to weigh the confidence of each base model dynamically to produce the final prediction.

## 5. Customization Guide

Here are the key locations in the code to adjust parameters.

### Changing the Embedding Dimension (fc1)

To change the size of the latent feature vector (currently 512), modify `src/cnn_architecture.py`.
**Important:** If you change this, you must retain the CNN (`02_cnn_model_training.py`) as the weights will change.

```python
# src/cnn_architecture.py

class FontCNN(nn.Module):
    def __init__(self, num_classes, input_dim=20):
        # ...
        # Change 512 to your desired dimension (e.g., 128, 256)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
```

### Adjusting Training Hyperparameters

Modify `02_cnn_model_training.py` to change batch size, learning rate, or epochs.

```python
# 02_cnn_model_training.py

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
```

### Modifying Ensemble Classifiers

To change the base learners (e.g., number of trees in Random Forest), edit `src/stacking_ensemble.py`.

```python
# src/stacking_ensemble.py

self.base_models = {
    "rf": RandomForestClassifier(
        n_estimators=200,
        # ...
    ),
    # ...
}
```
