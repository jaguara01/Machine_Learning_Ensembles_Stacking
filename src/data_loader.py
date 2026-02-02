"""
src/data_loader.py
==================

Description:
    This module prepares PyTorch DataLoaders for the font classification task.
    It handles data loading, optional filtering, normalization, and critically,
    addresses the dataset's class imbalance.

Key Operations:
    1.  **Loading**: Reads the serialized tensors from `data/processed_fonts.pt`.
    2.  **Filtering**: Optionally excludes the "Other" class to create a closed-set
        classification problem (Ranks 10-30).
    3.  **Splitting**: Performs Stratified splitting to generate Train, Validation,
        and Test sets (preserves class ratios).
    4.  **Imbalance Handling (Weighted Random Sampling)**:
        - compute inverse class frequency weights from the *training* split.
        - Creates a `WeightedRandomSampler` that oversamples minority classes.
        - Ensures that every training batch has a roughly balanced distribution of fonts.

Usage:
    from src.data_loader import get_dataloaders
    train_dl, val_dl, test_dl, num_classes, dim = get_dataloaders(...)
"""

import os

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


def get_dataloaders(
    data_path="data/processed_fonts.pt",
    batch_size=64,
    include_other=False,
):
    """
    Args:
        include_other (bool): If False, removes the 'Other' class entirely.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Run 'preprocess.py' first! Cannot find {data_path}")

    print(f"⚡ Loading data from {data_path}...")
    data = torch.load(data_path)
    X = data["X"]
    y = data["y"]
    m_labels = data.get("m_labels", None)
    if m_labels is None:
        raise KeyError("m_labels not found in processed data; re-run preprocessing script that saves m_labels")

    # Identify the "Other" label index
    original_num_classes = len(data["class_names"])
    other_label_idx = original_num_classes - 1

    # --- FILTERING LOGIC ---
    if not include_other:
        print(f"✂️ Excluding 'Other' class (Index {other_label_idx})...")
        mask = y != other_label_idx
        X = X[mask]
        y = y[mask]
        m_labels = m_labels[mask]
        # Reduce class count by 1
        num_classes = original_num_classes - 1
        print(f"   Samples remaining: {len(y)}")
    else:
        num_classes = original_num_classes

    # Normalization (0-255 -> 0 to 1)
    # Note: Standard CNNs often prefer [0, 1]. Your previous code did [-1, 1].
    # Sticking to [0, 1] is usually safer for ReLUs unless you use Tanh.
    if X.max() > 1.0:
        X = X / 255.0

    # Standard Stratified Split
    # 1. Split Train+Val vs Test
    X_train_val, X_test, y_train_val, y_test, m_train_val, m_test = train_test_split(
        X, y, m_labels, test_size=0.10, stratify=y, random_state=42
    )
    # 2. Split Train vs Val
    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_train_val, y_train_val, m_train_val, test_size=0.11, stratify=y_train_val, random_state=42
    )

    # --- SOLUTION 2: WEIGHTED SAMPLING (The Fix) ---
    print("⚖️  Calculating class weights for Imbalance Handling...")

    # 1. Count samples per class in the TRAINING set
    # We use torch.bincount which is fast for integer labels
    class_counts = torch.bincount(y_train)

    # 2. Compute weight for each class (Inverse Frequency)
    # Rare class = Low Count = High Weight
    class_weights = 1.0 / class_counts.float()

    # 3. Assign a weight to every single training sample
    # If sample_i is 'Consolas', it gets the high Consolas weight.
    sample_weights = class_weights[y_train]

    # 4. Create the Sampler
    # replacement=True is MANDATORY for oversampling rare classes
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create Loaders
    # IMPORTANT: shuffle MUST be False when using a sampler!
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, m_train),
        batch_size=batch_size,
        sampler=sampler,  # <--- Apply the fix here
        shuffle=False,  # <--- Must be False
    )

    # Val and Test don't need sampling (we want to evaluate on real distribution)
    val_loader = DataLoader(TensorDataset(X_val, y_val, m_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test, m_test), batch_size=batch_size)

    num_m_labels = int(m_labels.max().item() + 1)
    return train_loader, val_loader, test_loader, num_classes, data["img_dim"], num_m_labels
