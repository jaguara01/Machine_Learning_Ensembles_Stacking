"""
src/cnn_architecture.py
=======================

Description:
    This module defines the PyTorch neural network architecture used for font classification.
    It implements a custom Convolutional Neural Network (CNN) enhanced with a Spatial
    Attention mechanism to focus on fine-grained stroke details.

Components:
    1.  **SpatialAttention**: A custom module that computes a 2D attention map using
        both Average and Max pooling statistics. It helps the network prioritize
        informative regions (e.g., serifs, edges) over background noise.
    2.  **FontCNN**: The main backbone architecture.
        - **Input**: 1-channel, 20x20 grayscale images.
        - **Structure**: 2 Conv Blocks -> Spatial Attention -> Flatten -> FC Layers.
        - **Embedding Head (fc1)**: Produces the 128-dimensional style embedding used
          for Stacking.
        - **Classification Head (fc2)**: Used for end-to-end training of the backbone.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F


# --- 1. Spatial Attention Module ---
class SpatialAttention(nn.Module):
    """
    Focuses on specific stroke patterns (serifs, ligatures)
    and filters out background noise.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Input channels = 2 (1 MaxPool + 1 AvgPool)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)


# --- 2. Main CNN Architecture ---
class FontCNN(nn.Module):
    def __init__(self, num_classes, input_dim=20):
        super(FontCNN, self).__init__()

        # --- REPRODUCIBILITY ---
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Pooling (Reduces size by half)
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional Block 2 (Filters double here)
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)

        # Attention Mechanism
        self.attention = SpatialAttention()

        # Dynamic Flatten Size Calculation
        # Image size: input_dim -> pool(2) -> input_dim // 2
        final_dim = input_dim // 2
        self.flatten_size = (base_filters * 2) * final_dim * final_dim

        self.fc1 = nn.Linear(self.flatten_size, 512)  # Latent Space (Embedding)
        self.fc2 = nn.Linear(512, num_classes)  # Classifier

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, m_label_idx=None):
        # Feature Extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))

        # Apply Attention
        x = self.attention(x)

        # Flatten
        x = x.view(-1, self.flatten_size)

        # Embedding (Used for Phase 2: kPCA)
        embedding = F.relu(self.fc1(x))

        # Classification (Used for Phase 1 Training)
        out = self.dropout_layer(fused)
        logits = self.fc2(out)

        return logits, embedding
